# fpl-edge

A mini-league engine, not an FPL optimiser. It maximises **P(win your focus
league)**, which is a different objective from expected points and
occasionally the opposite one.

## Why

Inside a mini league your rank moves according to

    delta = SUM_p [ m_you(p) - EO(p) ] * points(p)

Effective ownership here is *exact*, not estimated — the API exposes every
rival's squad, and `multiplier` is literally the weight (0 bench, 1 starter,
2 captain, 3 triple captain, bench→1 under Bench Boost). A player at EO 1.0
in your league cannot move you regardless of what he scores.

## Setup

1. Create your FPL team, note the number in your Points tab URL
   (`/entry/<NUMBER>/event/1`).
2. Open your mini league; the id is in that URL.
3. Put both in `config.yaml`.
4. `pip install -r requirements.txt`

## Use

    python -m scripts.ingest state              # where are we in the season
    python -m scripts.ingest snapshot --gw 1    # capture rival squads
    python -m scripts.build --gw 2 --ref-gw 1 --deep
    python -m scripts.eo --gw 1
    python -m scripts.threat --gw 1
    python -m scripts.journal estimate-edge

## Snapshots

The FPL API only ever exposes *current* state — ownership, prices, transfer
counts and rival squads are overwritten in place and gone. `.github/workflows/snapshot.yml`
runs a deadline-aware watcher every two hours: it reads the API's own
`deadline_time` and only does work when there's work. Never cron to a fixed
weekly slot — deadlines land on Fri, Sat, Sun and Wed across a season.

Public repo ⇒ Actions minutes are free and unmetered. A gameweek snapshot is
about 110KB.

## Improve this first

`scripts/project.py`. Everything else is structural and correct whatever you
put in there.
