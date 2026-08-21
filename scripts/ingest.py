"""Snapshot the world. Dumb by design — fetches and writes, never decides.

Run modes:
  python -m scripts.ingest watch     # cheap: exit unless there's work to do
  python -m scripts.ingest snapshot  # force a snapshot of the current GW

The FPL API only ever exposes *current* state. Ownership, prices, transfer
counts and rival squads are overwritten in place and gone forever. This
script is the only reason you'll have a season of history to model on.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import fpl

PLAYER_COLS = [
    "id", "web_name", "first_name", "second_name", "team", "element_type",
    "now_cost", "status", "news", "chance_of_playing_next_round",
    "selected_by_percent", "transfers_in_event", "transfers_out_event",
    "total_points", "event_points", "points_per_game", "form", "minutes",
    "starts", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "saves", "bonus", "bps", "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "expected_goals_per_90", "expected_assists_per_90",
    "expected_goal_involvements_per_90", "expected_goals_conceded_per_90",
    "defensive_contribution", "defensive_contribution_per_90",
    "tackles", "recoveries", "clearances_blocks_interceptions",
    "ep_this", "ep_next", "penalties_order", "corners_and_indirect_freekicks_order",
    "direct_freekicks_order", "price_change_percent", "cost_change_event",
]


def _dirs(cfg: dict, gw: int) -> Path:
    d = fpl.repo_root() / cfg["data"]["root"] / f"gw{gw:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def snapshot_players(boot: dict, out: Path) -> int:
    df = pd.DataFrame(boot["elements"])
    cols = [c for c in PLAYER_COLS if c in df.columns]
    df = df[cols]
    df.to_parquet(out / "players.parquet", index=False, compression="zstd")

    pd.DataFrame(boot["teams"])[
        ["id", "name", "short_name", "strength",
         "strength_attack_home", "strength_attack_away",
         "strength_defence_home", "strength_defence_away"]
    ].to_parquet(out / "teams.parquet", index=False, compression="zstd")
    return len(df)


def snapshot_fixtures(out: Path) -> int:
    fx = pd.DataFrame(fpl.fixtures())
    keep = ["id", "event", "team_h", "team_a", "team_h_score", "team_a_score",
            "team_h_difficulty", "team_a_difficulty", "kickoff_time",
            "finished", "started"]
    fx[[c for c in keep if c in fx.columns]].to_parquet(
        out / "fixtures.parquet", index=False, compression="zstd"
    )
    return len(fx)


def snapshot_league(cfg: dict, league_id: int, gw: int, out: Path, tag: str) -> dict:
    """Pull standings + every member's squad + chip state for this gameweek."""
    st = fpl.league_standings(league_id)
    members = st["results"]
    print(f"    {tag} league {league_id}: {len(members)} managers", flush=True)

    rows, chips, missing = [], [], 0
    for m in members:
        eid = m["entry"]
        picks = fpl.entry_picks(eid, gw)
        if not picks:
            missing += 1
            continue
        chip = picks.get("active_chip")
        for p in picks["picks"]:
            rows.append({
                "entry": eid,
                "entry_name": m["entry_name"],
                "player_name": m["player_name"],
                "rank": m["rank"],
                "total": m["total"],
                "event_total": m["event_total"],
                "element": p["element"],
                "position": p["position"],
                # multiplier IS the effective-ownership weight:
                # 0 bench, 1 starter, 2 captain, 3 triple captain,
                # and bench slots become 1 under Bench Boost.
                "multiplier": p["multiplier"],
                "is_captain": p["is_captain"],
                "is_vice_captain": p["is_vice_captain"],
                "active_chip": chip,
            })
        try:
            hist = fpl.entry_history(eid)
            for c in hist.get("chips", []):
                chips.append({"entry": eid, "chip": c["name"], "event": c["event"]})
        except Exception:  # noqa: BLE001
            pass

    if not rows:
        print(f"    {tag}: no picks visible yet (deadline not passed?)")
        return {"league_id": league_id, "managers": len(members), "picks": 0}

    pd.DataFrame(rows).to_parquet(
        out / f"picks_{league_id}.parquet", index=False, compression="zstd"
    )
    pd.DataFrame(chips).to_parquet(
        out / f"chips_{league_id}.parquet", index=False, compression="zstd"
    )
    (out / f"standings_{league_id}.json").write_text(
        json.dumps({"league": st["league"], "results": members}, indent=1)
    )
    return {
        "league_id": league_id,
        "name": st["league"]["name"],
        "managers": len(members),
        "picks_captured": len(rows) // 15,
        "missing": missing,
        "role": tag,
    }


def do_snapshot(cfg: dict, gw: int) -> dict:
    boot = fpl.bootstrap(ttl=0)
    out = _dirs(cfg, gw)
    meta = {
        "gw": gw,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "players": snapshot_players(boot, out),
        "fixtures": snapshot_fixtures(out),
        "leagues": [],
    }
    lg = cfg["leagues"]
    meta["leagues"].append(snapshot_league(cfg, lg["focus"], gw, out, "focus"))
    for lid in lg.get("secondary") or []:
        meta["leagues"].append(snapshot_league(cfg, lid, gw, out, "secondary"))

    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  snapshot written -> {out}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["watch", "snapshot", "state"])
    ap.add_argument("--gw", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = fpl.load_config()
    state = fpl.gw_state()
    print(f"[fpl-edge] {json.dumps(state)}")

    if args.mode == "state":
        return 0

    gw = args.gw or state["current_gw"]
    if gw is None:
        print("  no current gameweek — season not started or finished")
        return 0

    marker = _dirs(cfg, gw) / "meta.json"

    if args.mode == "watch":
        if not state["deadline_passed"]:
            print(f"  GW{gw} deadline not passed — nothing to do")
            return 0
        if marker.exists() and not args.force:
            # Already have the locked squads. Re-snapshot once more after the
            # gameweek finishes to capture final points and autosubs.
            done = json.loads(marker.read_text()).get("final")
            if done or not state["current_finished"]:
                print(f"  GW{gw} already captured — nothing to do")
                return 0
            print(f"  GW{gw} finished — capturing final state")

    meta = do_snapshot(cfg, gw)
    if state["current_finished"]:
        meta["final"] = True
        (_dirs(cfg, gw) / "meta.json").write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
