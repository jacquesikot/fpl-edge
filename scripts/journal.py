"""Decision records, written before the outcome is known.

One rule makes this work: `plan` writes the entry BEFORE the deadline, and
`review` only ever appends. A journal written after the fact is contaminated
by hindsight and teaches you nothing.

The review then splits every decision into:

  * process error    — a different option was better given information
                       available at the time. Fixable. Learn from it.
  * outcome variance — the choice was right and it didn't land. Not
                       fixable. Do NOT learn from it.

Most managers can't tell these apart, so they thrash: overreacting to
variance while never correcting real process errors.

This is also the estimator for `mu`, your per-gameweek edge. The variance
switch rule in threat.py turns on that number, and you don't know it until
roughly 10 logged gameweeks. Until then it stays 0.0 and the engine plays
close to template.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import fpl

TEMPLATE = """# GW{gw} — decision record

*Written {ts} — BEFORE the deadline. Do not edit below the line after kickoff.*

## Position going in
- League rank: {rank} of {n}
- Gap to leader: {gap:+d}
- Gameweeks remaining: {gws}
- Assumed edge (mu): {edge:+.2f}/GW
- **Directive: {directive}**
- Template overlap: {overlap}

## Options considered
| option | ep | delta P(win) | chosen |
|---|---|---|---|
{options}

## Reasoning
{reasoning}

## What I am betting on
{bets}

---
<!-- REVIEW BELOW — appended by `python -m scripts.journal review --gw {gw}` -->
"""

REVIEW = """
## Review (appended {ts})

- Points scored: {points}
- League rank: {rank_before} -> {rank_after}
- Gap to leader: {gap_before:+d} -> {gap_after:+d}
- Active weight contribution: {contrib:+.1f}

### Rejected options, scored
{rejected}

### Classification
{classification}
"""


def path_for(cfg: dict, gw: int) -> Path:
    d = fpl.repo_root() / cfg["data"]["journal_root"]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"gw{gw:02d}.md"


def write_plan(cfg: dict, gw: int, *, rank: int, n: int, gap: int, gws: int,
               directive: str, overlap: str, options: list[dict],
               reasoning: str, bets: str) -> Path:
    p = path_for(cfg, gw)
    if p.exists():
        raise SystemExit(
            f"{p} already exists. The plan is written once, before the "
            "deadline, and never rewritten — that's the whole point."
        )
    rows = "\n".join(
        f"| {o['name']} | {o.get('ep', '')} | {o.get('dp_win', '')} | "
        f"{'**YES**' if o.get('chosen') else ''} |" for o in options
    )
    p.write_text(TEMPLATE.format(
        gw=gw, ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        rank=rank, n=n, gap=gap, gws=gws,
        edge=float(cfg["strategy"]["edge_per_gw"]),
        directive=directive, overlap=overlap, options=rows,
        reasoning=reasoning, bets=bets,
    ))
    return p


def append_review(cfg: dict, gw: int, body: str) -> Path:
    p = path_for(cfg, gw)
    if not p.exists():
        raise SystemExit(f"No plan for GW{gw}. Nothing to review.")
    txt = p.read_text()
    if "## Review (appended" in txt:
        raise SystemExit(f"GW{gw} already reviewed.")
    p.write_text(txt + body)
    return p


def estimate_edge(cfg: dict) -> dict:
    """mu = mean per-GW points scored minus the focus league mean.

    This is the only honest source for the number that drives the variance
    switch rule. Needs ~10 gameweeks before it means much; the standard
    error is reported so you can see whether it does.
    """
    root = fpl.repo_root() / cfg["data"]["root"]
    lid = cfg["leagues"]["focus"]
    me = cfg["entry_id"]
    diffs = []
    for d in sorted(root.glob("gw*")):
        f = d / f"picks_{lid}.parquet"
        if not f.exists():
            continue
        pk = pd.read_parquet(f)
        tot = pk.groupby("entry")["event_total"].first()
        if me not in tot.index or len(tot) < 2:
            continue
        diffs.append(float(tot.loc[me] - tot.drop(index=me).mean()))
    if len(diffs) < 3:
        return {"n_gws": len(diffs), "edge": 0.0,
                "note": "not enough logged gameweeks — keep edge at 0.0"}
    a = np.array(diffs)
    se = float(a.std(ddof=1) / np.sqrt(len(a)))
    mu = float(a.mean())
    return {
        "n_gws": len(a),
        "edge": round(mu, 2),
        "std_error": round(se, 2),
        "ci95": [round(mu - 1.96 * se, 2), round(mu + 1.96 * se, 2)],
        "significant": bool(abs(mu) > 1.96 * se),
        "note": (
            "Significant. Update strategy.edge_per_gw in config.yaml."
            if abs(mu) > 1.96 * se else
            "Not distinguishable from zero. Leave edge_per_gw at 0.0 — "
            "assuming an edge you can't demonstrate is how people talk "
            "themselves into bad differentials."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("estimate-edge")
    r = sub.add_parser("review")
    r.add_argument("--gw", type=int, required=True)
    r.add_argument("--body", type=str, required=True)
    ls = sub.add_parser("list")
    args = ap.parse_args()

    cfg = fpl.load_config()
    if args.cmd == "estimate-edge":
        print(json.dumps(estimate_edge(cfg), indent=1))
    elif args.cmd == "review":
        print(append_review(cfg, args.gw, args.body))
    else:
        d = fpl.repo_root() / cfg["data"]["journal_root"]
        for f in sorted(d.glob("gw*.md")):
            txt = f.read_text()
            done = "reviewed" if "## Review (appended" in txt else "OPEN"
            m = re.search(r"Directive: (.+)", txt)
            print(f"  {f.stem}  {done:<9} {m.group(1) if m else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
