---
name: fpl-edge
description: Win a specific FPL mini league by managing active weight against known rival squads, not by maximising expected points. Use this skill whenever Jacques mentions FPL, Fantasy Premier League, his mini league, gameweeks, captaincy, transfers, wildcards, chips, differentials, effective ownership, "who should I captain", "should I take the hit", "am I safe", "who's catching me", or asks to build, plan, lock, track or review a gameweek. Also use it for any question about his league position, rivals, or FPL strategy, and for the weekly post-deadline review — even when phrased casually like "what should I do this week" or "how am I looking".
---

# fpl-edge

Winning a mini league is not the same problem as scoring points. Your rank
inside the league is driven entirely by one quantity:

```
delta = SUM_p [ m_you(p) - EO(p) ] * points(p)
```

The bracketed term is your **active weight** on player p. A player at
effective ownership 1.0 in your league cannot move you no matter how many
points he scores. Not owning a player the league loads up on is a large
*negative* bet, and it is usually the biggest position in a squad.

Everything in this skill follows from that. The objective function is
**P(win the focus league)**, not expected points. They are not the same, and
where they conflict, P(win) wins.

## Setup check

Before anything else, confirm `config.yaml` has `entry_id` and
`leagues.focus` set. If either is null, ask for them and stop — nothing
works without a league to compete in. `leagues.secondary` is optional and
never drives decisions; it only produces warnings.

Snapshots live in `data/gw{NN}/`. If the gameweek you need isn't there, run
`python -m scripts.ingest snapshot --gw N` before analysing.

## The five modes

Route on what Jacques is asking for. When ambiguous, ask — don't guess,
because `plan` and `review` operate on different information sets.

### `build` — cold start, once

Only for creating the opening 15.

```bash
python -m scripts.build --gw 2 --ref-gw 1 --deep
```

Always pass `--deep` for a cold start. Without it the projection falls back
to a price prior and produces a flat, undifferentiated squad.

Output is a **frontier**, not an answer: each row is a squad with its
expected points, its overlap with the league template, and the expected
points it gives up versus optimal. Present the frontier. Explain that each
step down in overlap is variance bought at a stated price. Recommend a row
based on the directive from `threat`, and say plainly what the squad is
betting on and against.

Never present a single squad as "the answer".

### `plan` — pre-deadline, the workhorse

```bash
python -m scripts.threat --gw <last completed gw>
python -m scripts.eo --gw <last completed gw>
```

Rival squads for the *upcoming* gameweek are hidden until its deadline
passes, so this runs on the last completed gameweek's squads. Rivals will
have moved. Don't pretend otherwise — treat their squads as a distribution,
and flag that uncertainty in the recommendation.

Score every candidate transfer and captain by its effect on P(win), not on
expected points. Then **write the decision record before the deadline** with
`scripts/journal.py`, including the options rejected. This is not optional;
see "The journal rule" below.

Respect `strategy.max_ep_sacrifice` — never recommend a move costing more
expected points than that, however good the variance argument sounds.

### `lock` — T-minus two hours

Press conferences are done. Re-pull and check `news`, `status` and
`chance_of_playing_next_round` for the planned XI. Confirm or override. Keep
this short — it is a safety check, not a re-plan.

### `track` — during matches

Live points and provisional bonus via `event/{gw}/live/`. Report live league
rank and, where it's informative, the counterfactual: where Jacques would
sit under each option he rejected. Do not draw lessons from a gameweek in
progress.

### `review` — Monday, and the real planning session

Rival squads for the completed gameweek are now **exact**. This is the
moment of best information all week: you know precisely what the field
owned. Choose next week's differential here and merely confirm it on Friday.

```bash
python -m scripts.ingest snapshot --gw <completed gw> --force
python -m scripts.eo --gw <gw>
python -m scripts.threat --gw <gw>
python -m scripts.journal review --gw <gw> --body "..."
```

Diff each rival against the previous gameweek: transfers made, chips burned,
rank moves. Then classify the decision — see below.

## The journal rule

**The plan is written before the outcome is known, and never rewritten.**
A journal written afterwards is contaminated by hindsight and teaches
nothing. `journal.py` refuses to overwrite an existing plan; that refusal is
a feature, don't work around it.

At review, classify every decision into exactly one of:

- **Process error** — a different option was better *given the information
  available at the time*. Fixable. Say what the missed signal was.
- **Outcome variance** — the choice was right and it didn't land. Not
  fixable. Explicitly say so and do not extract a lesson from it.

Most managers cannot tell these apart, so they overreact to variance and
never correct real process errors. Holding this line is a large part of the
skill's value. Be willing to say "that was the right call, it just didn't
come off" — and equally willing to say a decision that scored well was
process-wrong.

## The variance directive

`threat.py` emits a directive from the switch rule:

```
P(overtake) = Phi( (mu*G - g) / (sigma * sqrt(G)) )

mu*G > g  ->  edge alone closes the gap  ->  MINIMISE variance, copy the field
mu*G < g  ->  it does not               ->  MAXIMISE variance, take differentials
```

Follow it, including when it's uncomfortable. Trailing with no runway means
**the lower-expected-points pick is correct**, because P(win) is the
objective. Leading in April means deliberately buying template players you
think are overrated, because eliminating tracking error is worth more than a
few points.

`mu` is Jacques's honest per-gameweek edge. It is **0.0 until measured**.

```bash
python -m scripts.journal estimate-edge
```

Only update `strategy.edge_per_gw` when the estimate is statistically
distinguishable from zero (roughly 10+ logged gameweeks). Assuming an edge
that can't be demonstrated is exactly how people talk themselves into bad
differentials. If Jacques wants to raise it on vibes, push back.

## What this skill does not do

- **It never submits transfers.** It recommends; Jacques clicks. Auto-submit
  needs a session cookie, is fragile, and a bug costs real points.
- **It does not optimise for overall rank.** If asked for a "best team" in
  the abstract, note that it's a different objective and ask which one.
- **It does not treat secondary leagues as constraints.** Flag conflicts,
  then optimise for `focus`.

## Known weaknesses — state these when they matter

`scripts/project.py` is the weak link and is deliberately isolated. Its v1
cold-start model has no explicit minutes model, uses FDR rather than xG for
fixture strength, and folds bonus into a historical rate. The active-weight
and threat machinery is sound regardless of what goes in there, but **the
projections carry real error and recommendations should be voiced with
appropriate hedging**. Replace it with a correlated Poisson model once six
to eight gameweeks of this season's xG are in `data/`.

`threat.py` uses an independent-player approximation for tracking error,
which overstates sigma when two squads share a team block (correlated clean
sheets). The *ranking* of threats is robust; the absolute probabilities are
not. Present them as rough.

## Reference

| file | purpose |
|---|---|
| `scripts/fpl.py` | API client, config loader, gameweek state |
| `scripts/ingest.py` | snapshots — the only reason history exists |
| `scripts/eo.py` | exact league EO and active weight vector |
| `scripts/threat.py` | pairwise threats + variance directive |
| `scripts/project.py` | expected points (**replace this**) |
| `scripts/build.py` | cold-start ILP frontier |
| `scripts/journal.py` | decision records, edge estimation |
