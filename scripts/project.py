"""Expected points per player over a horizon. THE PART YOU SHOULD IMPROVE.

Everything else in this repo — effective ownership, threat modelling,
variance targeting, the build frontier — is structural and correct
regardless of what goes in here. This module is the weakest link and it is
deliberately isolated behind one function so you can replace it without
touching anything else:

    project(players, teams, fixtures, from_gw, horizon) -> DataFrame[element, ep]

v1 is a cold-start model, because at GW2 this season has essentially no
data. It blends:

  * FPL's own `ep_next` (Opta-informed, but only one gameweek deep)
  * last season's points-per-90 via element-summary history (--deep)
  * a price-implied prior (FPL prices encode last season's output well)
  * minutes availability from status / chance_of_playing
  * fixture difficulty over the horizon, home/away adjusted

Known weaknesses, in rough order of how much they cost you:
  1. No explicit minutes model — rotation risk is only crudely handled.
  2. Fixture adjustment is FDR-based, not xG-based team strength.
  3. No set-piece / penalty premium beyond what price already encodes.
  4. Bonus points are folded into the historical rate, not modelled.
Replace with a proper Poisson goal model once you have 6-8 gameweeks of
this season's xG in the snapshots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import fpl

# rough league-average points per 90 by position, used as the price anchor
POS_BASE = {1: 3.4, 2: 3.3, 3: 3.6, 4: 3.8}


def availability(players: pd.DataFrame) -> pd.Series:
    """P(plays a meaningful share of minutes). Crude but keeps the ILP honest."""
    chance = players["chance_of_playing_next_round"]
    avail = np.where(chance.notna(), chance.fillna(100) / 100.0, 1.0)
    avail = np.where(players["status"].isin(["i", "s", "u", "n"]), 0.0, avail)
    avail = np.where(players["status"] == "d", np.minimum(avail, 0.6), avail)
    return pd.Series(avail, index=players.index).clip(0, 1)


def fixture_multiplier(teams: pd.DataFrame, fixtures: pd.DataFrame,
                       from_gw: int, horizon: int) -> pd.Series:
    """Mean fixture ease per team over the horizon, centred on 1.0.

    FDR runs 1 (easiest) to 5 (hardest); we map to a multiplier in roughly
    [0.82, 1.18] and average across the horizon. Blank gameweeks pull a
    team's multiplier down, which is the intended behaviour.
    """
    fx = fixtures[
        (fixtures["event"] >= from_gw) & (fixtures["event"] < from_gw + horizon)
    ]
    acc: dict[int, list[float]] = {t: [] for t in teams["id"]}
    for _, f in fx.iterrows():
        acc[f["team_h"]].append(1.0 + (3 - f["team_h_difficulty"]) * 0.09)
        acc[f["team_a"]].append(1.0 + (3 - f["team_a_difficulty"]) * 0.09 - 0.04)
    return pd.Series(
        {t: (np.mean(v) * min(len(v) / horizon, 1.25) if v else 0.0)
         for t, v in acc.items()}
    )


def _price_prior(players: pd.DataFrame) -> pd.Series:
    """Price-implied points per 90. FPL prices are a strong cold-start signal."""
    base = players["element_type"].map(POS_BASE).astype(float)
    # each £1.0m above the 4.5 floor buys roughly 0.45 pts/90
    return base + (players["now_cost"] / 10.0 - 4.5) * 0.45


def _last_season_rate(ids: list[int]) -> pd.Series:
    """Points per 90 last season from element-summary history_past."""
    out = {}
    for i, pid in enumerate(ids):
        try:
            s = fpl._fetch(f"element-summary/{pid}/", ttl=86400)
            past = s.get("history_past") or []
            if past:
                last = past[-1]
                mins = last.get("minutes") or 0
                if mins >= 500:
                    out[pid] = 90.0 * last["total_points"] / mins
        except Exception:  # noqa: BLE001
            continue
        if i and i % 50 == 0:
            print(f"    ...{i}/{len(ids)} player histories", flush=True)
    return pd.Series(out, dtype=float)


def project(players: pd.DataFrame, teams: pd.DataFrame, fixtures: pd.DataFrame,
            from_gw: int, horizon: int = 6, deep: bool = False) -> pd.DataFrame:
    p = players.copy()
    for c in ["ep_next", "form", "points_per_game", "selected_by_percent"]:
        if c in p:
            p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)

    prior = _price_prior(p)
    rate = prior.copy()

    if deep:
        # Only bother with plausible picks: cuts ~600 calls to ~300.
        cand = p[(p["now_cost"] >= 40) & (p["status"] != "u")]["id"].tolist()
        print(f"  fetching last-season rates for {len(cand)} players...")
        hist = _last_season_rate(cand)
        mapped = p["id"].map(hist)
        # shrink toward the price prior — last season is evidence, not truth
        rate = np.where(mapped.notna(), 0.65 * mapped.fillna(0) + 0.35 * prior, prior)
        rate = pd.Series(rate, index=p.index)

    # blend in FPL's own next-GW expectation where it's meaningful
    if "ep_next" in p:
        ep_rate = p["ep_next"] * (90 / 75.0)
        rate = np.where(p["ep_next"] > 0, 0.7 * rate + 0.3 * ep_rate, rate)
        rate = pd.Series(rate, index=p.index)

    fm = fixture_multiplier(teams, fixtures, from_gw, horizon)
    avail = availability(p)
    minutes_share = np.where(p["now_cost"] >= 55, 0.85, 0.72)

    p["ep_per_gw"] = rate * avail * minutes_share * p["team"].map(fm).fillna(1.0)
    p["ep_horizon"] = p["ep_per_gw"] * horizon
    p["value"] = p["ep_horizon"] / (p["now_cost"] / 10.0)
    return p[["id", "web_name", "team", "element_type", "now_cost", "status",
              "selected_by_percent", "ep_per_gw", "ep_horizon", "value"]]
