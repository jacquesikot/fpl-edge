"""Who can actually beat you, and how much variance you should be taking.

Two ideas do all the work here.

1. Threat is pairwise, not versus the field. A rival 40 points back running
   your squad is nearly harmless — your tracking error against him is tiny,
   so the gap is stable. A rival 15 back with six different starters and an
   unused Wildcard is far more dangerous than the raw gap suggests.

2. Optimal variance depends on your position. Model the gap to a rival over
   the remaining season as ~ N(mu*G, sigma^2*G):

       P(overtake) = Phi( (mu*G - g) / (sigma * sqrt(G)) )

   Differentiate w.r.t. sigma and you get a switch rule:

       mu*G > g  ->  your edge alone closes the gap. MINIMISE tracking
                     error. Copy the field. Shield the lead.
       mu*G < g  ->  it doesn't. MAXIMISE tracking error, even at a cost
                     in expected points.

   That second branch is the uncomfortable one: when you're behind and out
   of runway, the lower-EV differential is the correct pick, because the
   objective is P(win), not E(points).

mu is your honest per-gameweek edge. It is 0.0 until the journal has enough
logged decisions to measure it. Do not flatter yourself here.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import fpl
from .eo import load_gw

CHIP_VARIANCE = {"wildcard": 1.15, "freehit": 1.10, "bboost": 1.05, "3xc": 1.20}


def squad_vectors(picks: pd.DataFrame) -> pd.DataFrame:
    """entry x element matrix of multipliers."""
    return picks.pivot_table(
        index="entry", columns="element", values="multiplier",
        aggfunc="first", fill_value=0,
    )


def pairwise(picks: pd.DataFrame, players: pd.DataFrame, me: int,
             chips: pd.DataFrame | None = None, current_gw: int = 1,
             sigma_player: float = 2.6) -> pd.DataFrame:
    """Gap, squad divergence and per-GW tracking error against every rival.

    sigma_player is a crude per-player-per-GW standard deviation of FPL
    points. It is a placeholder: replace with the Monte Carlo simulator
    once the projection model is in. The *ranking* of threats is robust to
    the exact value; the absolute probabilities are not.
    """
    mat = squad_vectors(picks)
    if me not in mat.index:
        raise SystemExit(f"your entry {me} is not in this league snapshot")

    totals = picks.groupby("entry")[["total", "rank", "event_total"]].first()
    my_vec = mat.loc[me]
    my_total = totals.loc[me, "total"]

    # 2026/27 splits chips into two halves: each of wildcard, freehit,
    # bboost and 3xc is available once in GW1-19 and again in GW20-38.
    # A first-half wildcard being spent does NOT remove it for the season.
    half = 1 if current_gw < 20 else 2
    chip_map = {}
    if chips is not None and not chips.empty:
        in_half = chips[
            (chips["event"] < 20) if half == 1 else (chips["event"] >= 20)
        ]
        used = in_half.groupby("entry")["chip"].apply(set).to_dict()
        all_chips = set(CHIP_VARIANCE)
        chip_map = {e: all_chips - used.get(e, set()) for e in mat.index}

    rows = []
    for eid in mat.index:
        if eid == me:
            continue
        diff = (mat.loc[eid] - my_vec)
        # per-GW tracking error: independent-player approximation on the
        # difference vector. Overstates sigma when squads share a team block
        # (correlated clean sheets) — the simulator fixes that.
        sigma = float(np.sqrt((diff.abs() ** 2).sum())) * sigma_player
        spare = chip_map.get(eid, set())
        chip_mult = float(np.prod([CHIP_VARIANCE[c] for c in spare])) if spare else 1.0
        rows.append({
            "entry": eid,
            "manager": picks[picks["entry"] == eid]["player_name"].iloc[0],
            "team_name": picks[picks["entry"] == eid]["entry_name"].iloc[0],
            "rank": int(totals.loc[eid, "rank"]),
            "total": int(totals.loc[eid, "total"]),
            # positive = they are ahead of you
            "gap": int(totals.loc[eid, "total"] - my_total),
            "differing_starters": int((diff != 0).sum()),
            "sigma_per_gw": round(sigma, 2),
            "sigma_chip_adj": round(sigma * chip_mult, 2),
            "chips_left": ",".join(sorted(spare)) if spare else "",
        })
    return pd.DataFrame(rows)


def p_overtake(gap: float, sigma: float, gws: int, edge: float = 0.0) -> float:
    """P(the trailing side ends ahead). gap>0 means they lead you."""
    if gws <= 0 or sigma <= 0:
        return float(gap < 0)
    return float(norm.cdf((edge * gws - gap) / (sigma * np.sqrt(gws))))


def variance_directive(threats: pd.DataFrame, gws: int, edge: float) -> dict:
    """The switch rule, applied to your single most dangerous rival."""
    ahead = threats[threats["gap"] > 0]
    if ahead.empty:
        # You lead. Risk of being caught, weighted by pairwise volatility.
        risk = threats.assign(
            p=lambda d: [
                p_overtake(-g, s, gws, -edge)
                for g, s in zip(d["gap"], d["sigma_chip_adj"])
            ]
        )
        worst = risk.nlargest(1, "p").iloc[0]
        return {
            "position": "leading",
            "directive": "MINIMISE variance",
            "reason": (
                f"You lead. Closest live threat is {worst['manager']} at "
                f"{abs(worst['gap'])} back with P(catch)={worst['p']:.0%}. "
                "Tracking error is now your enemy — moving toward the league "
                "template protects the lead even at a small cost in points."
            ),
            "p_worst_case": round(float(worst["p"]), 3),
        }

    lead = ahead.nlargest(1, "gap").iloc[0]
    gap, sigma = float(lead["gap"]), float(lead["sigma_chip_adj"])
    p_now = p_overtake(gap, sigma, gws, edge)
    closable = edge * gws > gap
    return {
        "position": "trailing",
        "gap_to_leader": int(gap),
        "gws_remaining": gws,
        "edge_can_close": closable,
        "p_overtake_now": round(p_now, 3),
        "directive": "MINIMISE variance" if closable else "MAXIMISE variance",
        "reason": (
            f"Edge of {edge:+.1f}/GW over {gws} GWs is worth {edge*gws:+.0f}, "
            f"vs a {gap:.0f} point gap. "
            + (
                "Your edge alone closes this. Don't gamble — grind."
                if closable
                else "Your edge does not close this. You need variance: take "
                "differentials even where they cost expected points, because "
                "the objective is P(win), not E(points)."
            )
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--league", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = fpl.load_config()
    picks, players = load_gw(cfg, args.gw, args.league)
    lid = args.league or cfg["leagues"]["focus"]
    cpath = fpl.repo_root() / cfg["data"]["root"] / f"gw{args.gw:02d}" / f"chips_{lid}.parquet"
    chips = pd.read_parquet(cpath) if cpath.exists() else None

    t = pairwise(picks, players, cfg["entry_id"], chips, current_gw=args.gw)
    gws = 38 - args.gw
    edge = float(cfg["strategy"]["edge_per_gw"])
    t["p_they_finish_ahead"] = [
        p_overtake(g, s, gws, -edge) if g > 0 else 1 - p_overtake(-g, s, gws, edge)
        for g, s in zip(t["gap"], t["sigma_chip_adj"])
    ]
    t = t.sort_values("p_they_finish_ahead", ascending=False)
    directive = variance_directive(t, gws, edge)

    if args.json:
        print(json.dumps({"directive": directive,
                          "threats": t.round(3).to_dict("records")}, indent=1))
        return 0

    print(f"\nGW{args.gw} threat board — {gws} gameweeks remain, "
          f"assumed edge {edge:+.2f}/GW\n")
    print(f"  {'manager':<22}{'gap':>6}{'diff':>6}{'sigma':>8}{'P(ahead)':>10}  chips left")
    for _, r in t.head(10).iterrows():
        print(f"  {r['manager'][:21]:<22}{r['gap']:>+6}{r['differing_starters']:>6}"
              f"{r['sigma_chip_adj']:>8.1f}{r['p_they_finish_ahead']:>9.0%}  {r['chips_left']}")
    print(f"\n  DIRECTIVE: {directive['directive']}")
    print(f"  {directive['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
