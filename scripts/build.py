"""Cold start: build the opening 15 with the league's cards face up.

Normally the opening squad is picked blind. You're entering at GW2, which
costs you a gameweek but hands you something nobody who entered on time
gets: every rival's GW1 squad is already public. So this isn't "pick a good
team", it's "pick the active weight vector that maximises P(win) against
these N known squads, from one gameweek down".

The method is an efficient frontier rather than a single answer. We solve
the standard FPL integer program

    max  SUM_p ep(p)*x(p) - lam * SUM_p EO(p)*x(p)
    s.t. 15 players, 2/5/5/3 by position, <= 3 per club, <= 100.0m

for a sweep of lambda. lam=0 gives the raw points-optimal squad. Raising
lambda penalises duplicating the league's effective ownership, buying
variance at a known, explicit cost in expected points.

That cost is the number that matters. "This differential squad costs you
2.1 expected points over six gameweeks and cuts template overlap from 78%
to 54%" is a decision. "Trust me, he's a great differential" is not.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import pulp

from . import fpl
from .eo import effective_ownership
from .project import project

POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD = {1: 2, 2: 5, 3: 5, 4: 3}
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}


# Weight on a bench player's expected points in the objective.
#
# The naive formulation maximises ep over all 15 squad members equally, which
# is wrong: bench points are only scored via autosubs, so budget spent on
# bench ep is mostly burned. Left uncorrected the solver happily buys a
# £5.5m second keeper who will never play, and stacks £5.5m defenders on the
# bench, starving the XI.
#
# 0.12 is roughly the share of a bench player's points you actually realise
# through autosubs and rotation over a horizon. Raise it toward 1.0 and you
# recover the old behaviour; drop it to 0.0 and the solver buys the cheapest
# legal bench it can find, which is too aggressive — bench players do come on.
BENCH_WEIGHT = 0.12


def solve(proj: pd.DataFrame, eo: pd.Series, lam: float,
          budget: int = 1000, locks: list[int] | None = None,
          bans: list[int] | None = None,
          bench_weight: float = BENCH_WEIGHT) -> list[int] | None:
    """Pick 15, but value the 11 who actually play far above the 4 who don't.

    Two sets of binaries: `x` = in the 15, `s` = in the starting XI. The
    objective credits full ep to starters and only `bench_weight` of it to
    the rest, so the ILP has a reason to spend money on the XI and buy cheap
    cover. The EO/variance penalty stays on `x` — a player you own but bench
    still shifts your active weight, just less of it.
    """
    ids = proj["id"].tolist()
    x = pulp.LpVariable.dicts("x", ids, cat="Binary")   # in the 15
    s = pulp.LpVariable.dicts("s", ids, cat="Binary")   # in the XI
    prob = pulp.LpProblem("fpl", pulp.LpMaximize)

    ep = dict(zip(proj["id"], proj["ep_horizon"]))
    cost = dict(zip(proj["id"], proj["now_cost"]))
    pos = dict(zip(proj["id"], proj["element_type"]))
    team = dict(zip(proj["id"], proj["team"]))

    # starters earn full ep; squad members earn the bench share on top of it
    # being in the squad at all. ep[i]*(bench_weight*x + (1-bench_weight)*s)
    # => a starter gets ep, a benched player gets bench_weight * ep.
    prob += pulp.lpSum(
        ep[i] * (bench_weight * x[i] + (1.0 - bench_weight) * s[i])
        - lam * float(eo.get(i, 0.0)) * x[i]
        for i in ids
    )

    prob += pulp.lpSum(x[i] for i in ids) == 15
    prob += pulp.lpSum(s[i] for i in ids) == 11
    prob += pulp.lpSum(cost[i] * x[i] for i in ids) <= budget

    for i in ids:                      # can only start if you're in the squad
        prob += s[i] <= x[i]

    for p, n in SQUAD.items():
        prob += pulp.lpSum(x[i] for i in ids if pos[i] == p) == n

    # legal XI shape: exactly 1 GKP, and position minimums for the outfield
    prob += pulp.lpSum(s[i] for i in ids if pos[i] == 1) == 1
    for p, n in XI_MIN.items():
        if p == 1:
            continue
        prob += pulp.lpSum(s[i] for i in ids if pos[i] == p) >= n

    for t in set(team.values()):
        prob += pulp.lpSum(x[i] for i in ids if team[i] == t) <= 3
    for i in locks or []:
        prob += x[i] == 1
    for i in bans or []:
        prob += x[i] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return [i for i in ids if x[i].value() > 0.5]


def best_xi(squad: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Highest-EP legal starting XI; the rest is the bench, ordered by EP."""
    best, best_ep = None, -1
    for d in range(3, 6):
        for m in range(2, 6):
            for f in range(1, 4):
                if 1 + d + m + f != 11:
                    continue
                xi = []
                for p, n in [(1, 1), (2, d), (3, m), (4, f)]:
                    xi += squad[squad["element_type"] == p].nlargest(
                        n, "ep_horizon")["id"].tolist()
                if len(xi) != 11:
                    continue
                ep = squad[squad["id"].isin(xi)]["ep_horizon"].sum()
                if ep > best_ep:
                    best, best_ep = xi, ep
    bench = squad[~squad["id"].isin(best)].sort_values(
        "ep_horizon", ascending=False)["id"].tolist()
    return best, bench


def frontier(proj: pd.DataFrame, eo: pd.Series, lams: list[float],
             **kw) -> pd.DataFrame:
    rows, seen = [], {}
    for lam in lams:
        sq = solve(proj, eo, lam, **kw)
        if not sq:
            continue
        key = tuple(sorted(sq))
        if key in seen:
            continue
        seen[key] = True
        s = proj[proj["id"].isin(sq)]
        xi, _ = best_xi(s)
        xi_df = proj[proj["id"].isin(xi)]
        rows.append({
            "lam": lam,
            "squad": sq,
            "xi": xi,
            "ep_squad": round(float(s["ep_horizon"].sum()), 1),
            "ep_xi": round(float(xi_df["ep_horizon"].sum()), 1),
            "cost": round(float(s["now_cost"].sum()) / 10, 1),
            # overlap: how much of your XI the league already effectively owns
            "overlap": round(float(sum(min(1.0, eo.get(i, 0.0)) for i in xi) / 11), 3),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ep_cost_vs_optimal"] = (df["ep_xi"].max() - df["ep_xi"]).round(1)
    return df


def render(proj: pd.DataFrame, eo: pd.Series, row) -> str:
    s = proj[proj["id"].isin(row["squad"])].copy()
    s["eo"] = s["id"].map(lambda i: eo.get(i, 0.0))
    s["xi"] = s["id"].isin(row["xi"])
    out = []
    for p in [1, 2, 3, 4]:
        grp = s[s["element_type"] == p].sort_values("ep_horizon", ascending=False)
        out.append(f"  {POS[p]}")
        for _, r in grp.iterrows():
            mark = " " if r["xi"] else "b"
            out.append(f"   {mark} {r['web_name']:<18} £{r['now_cost']/10:>4.1f}m  "
                       f"ep {r['ep_horizon']:>5.1f}   leagueEO {r['eo']:.2f}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, required=True, help="first gameweek you'll play")
    ap.add_argument("--ref-gw", type=int, default=None,
                    help="snapshot gameweek to read league EO from (default gw-1)")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--deep", action="store_true",
                    help="fetch last-season rates (slower, much better cold start)")
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--lock", type=int, nargs="*", default=[])
    ap.add_argument("--ban", type=int, nargs="*", default=[])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = fpl.load_config()
    horizon = args.horizon or cfg["strategy"]["horizon_gws"]
    ref = args.ref_gw or (args.gw - 1)
    root = fpl.repo_root() / cfg["data"]["root"] / f"gw{ref:02d}"
    if not root.exists():
        raise SystemExit(f"No snapshot at GW{ref}. Run: python -m scripts.ingest snapshot --gw {ref}")

    players = pd.read_parquet(root / "players.parquet")
    teams = pd.read_parquet(root / "teams.parquet")
    fixtures = pd.read_parquet(root / "fixtures.parquet")
    picks = pd.read_parquet(root / f"picks_{cfg['leagues']['focus']}.parquet")

    eo_df = effective_ownership(picks)
    eo = pd.Series(eo_df["eo"].values, index=eo_df["element"].values)
    n_mgrs = picks["entry"].nunique()

    print(f"  projecting {len(players)} players over GW{args.gw}-{args.gw+horizon-1}"
          f"{' (deep)' if args.deep else ''}...")
    proj = project(players, teams, fixtures, args.gw, horizon, deep=args.deep)
    proj = proj[proj["status"] != "u"]

    print(f"  solving frontier against {n_mgrs} known squads...")
    fr = frontier(proj, eo, [0, 2, 4, 7, 10, 14, 19, 25, 32, 40],
                  budget=int(args.budget * 10), locks=args.lock, bans=args.ban)
    if fr.empty:
        raise SystemExit("no feasible squad — check budget/locks")

    if args.json:
        print(json.dumps(fr.drop(columns=["squad", "xi"]).to_dict("records"), indent=1))
        return 0

    print(f"\nCold-start frontier — GW{args.gw}, {horizon}-GW horizon, "
          f"league EO from GW{ref} ({n_mgrs} managers)\n")
    print(f"  {'lam':>4}{'ep(XI)':>9}{'cost':>7}{'overlap':>9}{'ep given up':>13}")
    for _, r in fr.iterrows():
        print(f"  {r['lam']:>4.0f}{r['ep_xi']:>9.1f}{r['cost']:>7.1f}"
              f"{r['overlap']:>8.0%}{r['ep_cost_vs_optimal']:>13.1f}")

    target = cfg["strategy"]["target_overlap"]
    pick = fr.iloc[(fr["overlap"] - target).abs().argmin()]
    print(f"\n  --- squad at overlap {pick['overlap']:.0%} "
          f"(target {target:.0%}, costs {pick['ep_cost_vs_optimal']:.1f} ep) ---")
    print(render(proj, eo, pick))
    print("\n  Read the frontier, don't just take the highlighted row. Every step "
          "\n  down in overlap is variance bought at a stated price in expected "
          "\n  points. How much you should buy depends on your gap — run "
          "\n  scripts.threat once you have a squad.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
