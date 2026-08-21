"""Exact league effective ownership and your active weight vector.

The whole engine rests on one decomposition:

    delta = SUM_p [ m_you(p) - EO(p) ] * points(p)

Your rank movement inside a mini league is that active weight vector dotted
with actual points. Nothing else matters. A player at EO 1.0 in your league
is a no-op no matter how many points he scores.

Unlike global EO (an estimate broadcast to 11m managers), this is exact:
we have all N squads, and `multiplier` is literally the weight.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import fpl


def load_gw(cfg: dict, gw: int, league_id: int | None = None):
    root = fpl.repo_root() / cfg["data"]["root"] / f"gw{gw:02d}"
    if not root.exists():
        raise SystemExit(f"No snapshot for GW{gw}. Run: python -m scripts.ingest snapshot --gw {gw}")
    lid = league_id or cfg["leagues"]["focus"]
    picks = pd.read_parquet(root / f"picks_{lid}.parquet")
    players = pd.read_parquet(root / "players.parquet")
    return picks, players


def effective_ownership(picks: pd.DataFrame) -> pd.DataFrame:
    """EO(p) = mean multiplier across all managers in the league."""
    n = picks["entry"].nunique()
    agg = (
        picks.groupby("element")
        .agg(
            eo=("multiplier", lambda s: s.sum() / n),
            owned_by=("entry", "nunique"),
            starters=("multiplier", lambda s: (s >= 1).sum()),
            captains=("is_captain", "sum"),
            benched=("multiplier", lambda s: (s == 0).sum()),
        )
        .reset_index()
    )
    agg["owned_pct"] = 100 * agg["owned_by"] / n
    agg["captain_pct"] = 100 * agg["captains"] / n
    return agg.sort_values("eo", ascending=False)


def active_weights(picks: pd.DataFrame, players: pd.DataFrame, entry_id: int) -> pd.DataFrame:
    """Your signed bet against the league, including what you've faded.

    Negative active weight (not owning something the league loads up on)
    is a real position and is usually the largest bet in a squad. Most
    managers only ever look at their positive differentials.
    """
    eo = effective_ownership(picks)
    mine = picks[picks["entry"] == entry_id][["element", "multiplier"]]
    if mine.empty:
        raise SystemExit(f"entry {entry_id} has no picks in this league snapshot")

    df = eo.merge(mine, on="element", how="outer").fillna({"multiplier": 0, "eo": 0})
    df["active_weight"] = df["multiplier"] - df["eo"]
    df = df.merge(
        players[["id", "web_name", "team", "element_type", "now_cost", "total_points"]],
        left_on="element", right_on="id", how="left",
    ).drop(columns=["id"])
    df["now_cost"] = df["now_cost"] / 10
    return df.sort_values("active_weight")


def summarise(df: pd.DataFrame, n: int = 8) -> dict:
    owned = df[df["multiplier"] > 0]
    template_overlap = float(
        (df[["multiplier", "eo"]].min(axis=1).sum()) / max(df["multiplier"].sum(), 1)
    )
    return {
        "template_overlap": round(template_overlap, 3),
        "active_share": round(1 - template_overlap, 3),
        "gross_active_weight": round(float(df["active_weight"].abs().sum()), 2),
        "top_longs": df.nlargest(n, "active_weight")[
            ["web_name", "multiplier", "eo", "active_weight"]
        ].round(2).to_dict("records"),
        "top_shorts": df.nsmallest(n, "active_weight")[
            ["web_name", "multiplier", "eo", "active_weight"]
        ].round(2).to_dict("records"),
        # Dead weight is near-ZERO active weight, not high EO. Owning a
        # player the league captains is a large short, not a neutral hold.
        "dead_weight": owned[owned["active_weight"].abs() < 0.15][
            ["web_name", "eo", "multiplier", "now_cost"]
        ].round(2).to_dict("records"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--league", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = fpl.load_config()
    picks, players = load_gw(cfg, args.gw, args.league)
    df = active_weights(picks, players, cfg["entry_id"])
    s = summarise(df)

    if args.json:
        print(json.dumps(s, indent=1))
        return 0

    print(f"\nGW{args.gw} — active position vs focus league "
          f"({picks['entry'].nunique()} managers)")
    print(f"  template overlap : {s['template_overlap']:.0%}")
    print(f"  active share     : {s['active_share']:.0%}  (this is your entire edge)")
    print("\n  LONGS  (you own more than the league)")
    for r in s["top_longs"]:
        print(f"    {r['web_name']:<18} you {r['multiplier']:.0f}  league {r['eo']:.2f}"
              f"   -> {r['active_weight']:+.2f}")
    print("\n  SHORTS (the league owns, you don't — these count too)")
    for r in s["top_shorts"]:
        print(f"    {r['web_name']:<18} you {r['multiplier']:.0f}  league {r['eo']:.2f}"
              f"   -> {r['active_weight']:+.2f}")
    if s["dead_weight"]:
        print("\n  DEAD WEIGHT (EO ~1.0 — cannot move you, in either direction)")
        for r in s["dead_weight"]:
            print(f"    {r['web_name']:<18} £{r['now_cost']:.1f}m  you {r['multiplier']:.0f}"
                  f"  league {r['eo']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
