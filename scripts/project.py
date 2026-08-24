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
  1. Fixture adjustment is FDR-based, not xG-based team strength.
  2. No set-piece / penalty premium beyond what price already encodes.
  3. Bonus points are folded into the historical rate, not modelled.
  4. `start_probability` predicts a STARTING role, not rotation within a
     known starter group, and cannot see press conferences — the `lock`
     step still matters.
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
    """P(fit and permitted to play) — INJURY/SUSPENSION ONLY.

    This deliberately says nothing about whether a fit player is actually in
    the manager's XI. A fully fit bench player scores 1.0 here, exactly like
    a nailed starter. That is not a bug in this function; role is handled
    separately by `start_probability`, and the two are multiplied.
    """
    chance = players["chance_of_playing_next_round"]
    avail = np.where(chance.notna(), chance.fillna(100) / 100.0, 1.0)
    avail = np.where(players["status"].isin(["i", "s", "u", "n"]), 0.0, avail)
    avail = np.where(players["status"] == "d", np.minimum(avail, 0.6), avail)
    return pd.Series(avail, index=players.index).clip(0, 1)


# minutes in a full season if you start every game
_SEASON_MINUTES = 38 * 90

# Bounds on expected share of minutes. A rotation risk is discounted, never
# written off; a nailed starter is credited, never assumed to play 100%.
# Tighten these and rotation stops mattering; widen them and the optimiser
# starts preferring availability over scoring ability.
MIN_SHARE = 0.55
MAX_SHARE = 0.92


def _last_season_minutes(ids: list[int]) -> pd.Series:
    """Minutes played last season, from element-summary history_past."""
    out = {}
    for pid in ids:
        try:
            s = fpl._fetch(f"element-summary/{pid}/", ttl=86400)
            past = s.get("history_past") or []
            if past:
                out[pid] = float(past[-1].get("minutes") or 0)
        except Exception:  # noqa: BLE001
            continue
    return pd.Series(out, dtype=float)


def start_probability(players: pd.DataFrame, from_gw: int,
                      last_minutes: pd.Series | None = None) -> pd.Series:
    """P(starts a given gameweek), independent of injury status.

    This is the fix for the model's largest documented weakness: before it,
    expected minutes were a function of PRICE (>=5.5m got 0.85, else 0.72),
    so a fit, expensive, permanently-benched player was indistinguishable
    from a nailed starter.

    Three signals, blended by how much each deserves to be trusted:

      role prior   last season's minutes / (38*90). At GW2 this is nearly
                   all we have, and it is a genuinely good predictor of
                   whether a player is a starter or a squad option.
      current form this season's starts / matches played. Strong evidence
                   but tiny sample early, so its weight grows with the
                   number of matches played, reaching parity around GW8.
      price rank   percentile of price within the player's own club. Weak,
                   used only as a tiebreak and to rescue players with no
                   history at all (promoted clubs, new signings, youth).

    The weight on current form is `n / (n + 4)`, so one match carries 20%
    and eight matches carry 67%. That deliberately stops a single start
    from certifying anyone as nailed.

    A floor of 0.25 applies to players whose current starts rate exceeds
    their historical rate — a young player breaking into the XI must not be
    permanently condemned by last season's bench minutes.
    """
    idx = players.index
    n_played = pd.to_numeric(players.get("starts"), errors="coerce").fillna(0.0)
    minutes_now = pd.to_numeric(players["minutes"], errors="coerce").fillna(0.0)

    matches = max(int(from_gw) - 1, 0)

    price = players["now_cost"].astype(float)
    club_rank = price.groupby(players["team"]).rank(pct=True)

    if last_minutes is not None and len(last_minutes):
        raw = players["id"].map(last_minutes).astype(float)
        # Normalise WITHIN POSITION. Raw minutes are not comparable across
        # positions: a keeper on 2200 minutes is a backup, an attacker on
        # 2200 is nailed. Without this, goalkeepers and centre-backs sweep
        # the top of the ranking and the optimiser buys availability
        # instead of points.
        share = (raw / _SEASON_MINUTES).clip(0, 1)
        pos_med = share.groupby(players["element_type"]).transform("median")
        pos_med = pos_med.replace(0, np.nan).fillna(0.5)
        # 0.72 is roughly "median starter"; scale each position onto that
        prior = (share / pos_med) * 0.72

        # Injured-elite rescue: a player who is expensive relative to his
        # club but has few minutes was hurt, not benched. Price is the
        # better signal of intended role in that case.
        expensive = club_rank >= 0.80
        low_mins = share < 0.55
        rescue = expensive & low_mins & raw.notna()
        prior = prior.where(~rescue, np.maximum(prior, 0.55 + 0.35 * club_rank))
    else:
        prior = pd.Series(np.nan, index=idx, dtype=float)

    prior = prior.fillna(0.35 + 0.45 * club_rank)
    prior = prior.clip(0.05, 0.98)

    if matches > 0:
        starts_rate = (n_played / matches).clip(0, 1)
        sub_only = (n_played <= 0) & (minutes_now > 0)
        starts_rate = starts_rate.where(~sub_only, 0.15)
        w = matches / (matches + 4.0)
        p_start = (1 - w) * prior + w * starts_rate
        breaking_out = starts_rate > prior
        p_start = p_start.where(~breaking_out, np.maximum(p_start, 0.25))
    else:
        p_start = prior

    return pd.Series(p_start, index=idx).clip(0.0, 1.0)


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
    last_min: pd.Series | None = None

    if deep:
        # Only bother with plausible picks: cuts ~600 calls to ~300.
        cand = p[(p["now_cost"] >= 40) & (p["status"] != "u")]["id"].tolist()
        print(f"  fetching last-season rates for {len(cand)} players...")
        hist = _last_season_rate(cand)
        # same cached element-summary calls, so this is effectively free
        last_min = _last_season_minutes(cand)
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

    # Expected minutes = P(starts), NOT price. But the spread is deliberately
    # COMPRESSED into [MIN_SHARE, MAX_SHARE] rather than [0.18, 0.85].
    #
    # Why: ep = rate * minutes_share, and an uncapped share varies by ~4.7x
    # while the underlying points rate varies far less. Minutes then dominate
    # the objective and the ILP buys guaranteed-90-minutes defenders over
    # players who actually score. Rotation risk should DISCOUNT a player, not
    # disqualify him. Keep the ordering, shrink the leverage.
    p_start = start_probability(p, from_gw, last_minutes=last_min)
    minutes_share = MIN_SHARE + (MAX_SHARE - MIN_SHARE) * p_start
    p["p_start"] = p_start

    p["ep_per_gw"] = rate * avail * minutes_share * p["team"].map(fm).fillna(1.0)
    p["ep_horizon"] = p["ep_per_gw"] * horizon
    p["value"] = p["ep_horizon"] / (p["now_cost"] / 10.0)
    return p[["id", "web_name", "team", "element_type", "now_cost", "status",
              "selected_by_percent", "p_start", "ep_per_gw", "ep_horizon",
              "value"]]
