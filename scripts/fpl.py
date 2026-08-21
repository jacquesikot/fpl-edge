"""Thin, polite client for the public Fantasy Premier League API.

No auth, no key. Everything here is read-only and public.
Responses are cached on disk for the session so repeated calls in one
run (e.g. 20 rival squads) don't hammer the API.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

BASE = "https://fantasy.premierleague.com/api"
UA = "fpl-edge/0.1 (+github.com/jacquesikot/fpl-edge)"
CACHE = Path(os.environ.get("FPL_CACHE", "/tmp/fpl-cache"))
CACHE.mkdir(parents=True, exist_ok=True)

_REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- config


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else _REPO / "config.yaml"
    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    if not cfg.get("entry_id"):
        raise SystemExit(
            "config.yaml: entry_id is not set.\n"
            "Create your FPL team, then find the number in the URL of your "
            "Points tab (/entry/<NUMBER>/event/1) and put it in config.yaml."
        )
    if not (cfg.get("leagues") or {}).get("focus"):
        raise SystemExit(
            "config.yaml: leagues.focus is not set.\n"
            "Open your mini league on the FPL site; the id is in the URL."
        )
    return cfg


def repo_root() -> Path:
    return _REPO


# --------------------------------------------------------------------------- fetch


def _fetch(path: str, ttl: int = 900) -> Any:
    key = CACHE / (path.strip("/").replace("/", "_").replace("?", "_") + ".json")
    if key.exists() and (time.time() - key.stat().st_mtime) < ttl:
        return json.loads(key.read_text())

    req = urllib.request.Request(f"{BASE}/{path}", headers={"User-Agent": UA})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            key.write_text(json.dumps(data))
            return data
        except Exception as exc:  # noqa: BLE001 - transient network/5xx
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"FPL API failed for {path}: {last}")


def bootstrap(ttl: int = 900) -> dict:
    """Players, teams, gameweeks, scoring rules, prices, ownership."""
    return _fetch("bootstrap-static/", ttl)


def fixtures(event: int | None = None) -> list[dict]:
    return _fetch(f"fixtures/?event={event}" if event else "fixtures/", ttl=3600)


def entry(entry_id: int) -> dict:
    return _fetch(f"entry/{entry_id}/")


def entry_history(entry_id: int) -> dict:
    """Includes the `chips` array — which chips each rival has burned."""
    return _fetch(f"entry/{entry_id}/history/")


def entry_picks(entry_id: int, gw: int) -> dict | None:
    """Squad for a gameweek. Returns None before that gameweek's deadline."""
    try:
        return _fetch(f"entry/{entry_id}/event/{gw}/picks/")
    except RuntimeError:
        return None


def entry_transfers(entry_id: int) -> list[dict]:
    return _fetch(f"entry/{entry_id}/transfers/", ttl=300)


def league_standings(league_id: int, max_pages: int = 20) -> dict:
    """Classic league standings, paginated (50 per page)."""
    page, results, meta = 1, [], None
    while page <= max_pages:
        d = _fetch(f"leagues-classic/{league_id}/standings/?page_standings={page}")
        meta = meta or d["league"]
        results.extend(d["standings"]["results"])
        if not d["standings"].get("has_next"):
            break
        page += 1
    return {"league": meta, "results": results}


def live(gw: int) -> dict:
    """Live per-player stats mid-gameweek, including provisional bonus."""
    return _fetch(f"event/{gw}/live/", ttl=120)


# --------------------------------------------------------------------------- gw state


def gw_state(boot: dict | None = None) -> dict:
    """Which gameweek we're in, and whether its deadline has passed.

    This is what drives the cron. Never hardcode a schedule — deadlines
    move between Friday, Saturday, Sunday and Wednesday across the season.
    """
    boot = boot or bootstrap()
    events = boot["events"]
    current = next((e for e in events if e.get("is_current")), None)
    nxt = next((e for e in events if e.get("is_next")), None)
    now = time.time()

    def ts(e):
        return time.mktime(time.strptime(e["deadline_time"], "%Y-%m-%dT%H:%M:%SZ"))

    return {
        "current_gw": current["id"] if current else None,
        "next_gw": nxt["id"] if nxt else None,
        "current_deadline": current["deadline_time"] if current else None,
        "next_deadline": nxt["deadline_time"] if nxt else None,
        "deadline_passed": bool(current) and now > ts(current) - time.timezone,
        "current_finished": bool(current and current.get("finished")),
        "hours_to_next_deadline": round((ts(nxt) - time.timezone - now) / 3600, 1)
        if nxt
        else None,
    }
