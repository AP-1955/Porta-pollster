# Porta-Pollster

> *It's an AI project designed to project possible polls and election results based on actual data and not biases. Plus it factors in how stupid voters are!*

An AI-assisted pollster that projects polls and election results from data rather than gut feeling. By default it takes the "independent" view of the electorate: voters ignore the campaign until the final week before election day, then decide based on what's in front of them at that point.

## What it does

1. **Averages polls fairly.** Recent polls, bigger samples and likely-voter polls count more.
2. **Corrects pollster lean ("house effects").** It measures how far each pollster tends to sit from the consensus and subtracts it. A pollster with only a few polls gets the benefit of the doubt.
3. **Adds the fundamentals.** It uses incumbent approval, real income growth, unemployment and inflation. Far from election day these count for more; close to it, polls dominate.
4. **Models voters as late deciders** (the "independent" view).
   - **Final-week decision window:** voters don't settle on a choice until the last 7 days (`decision_window_days`). Before that, the race is treated as still up for grabs, so the projection ranges stay wide until the window opens, then narrow sharply.
   - **Early news doesn't stick:** if a scandal knocked a candidate down 4 points in August, voters told pollsters so at the time, but by the final week it's gone. The model removes that dip from earlier polls. Events inside the final week count in full.
   - **Final-week polls count most:** earlier polls are down-weighted because they catch voters before they've made up their minds.
   - **Economy as it is at the end:** voters judge the economy by conditions in the final window (or the latest reading), not the whole term.
   - **Not everyone has to be a late decider:** set `independent_share` below 1 (e.g. `0.3`) and the rest follow the campaign all along, slowly forgetting older news (memory half-life, default 120 days). This mixed setup lines up with research on retrospective voting ([Achen & Bartels](https://press.princeton.edu/books/paperback/9780691178240/democracy-for-realists); Healy & Lenz on "end bias").
   - **Undecideds** split in proportion to current support, or evenly.
5. **Simulates the election** 20,000 times and reports projected vote share, an 80% range and each candidate's chance of winning.
6. **AI analyst (optional):** Claude writes a nonpartisan, plain-English briefing from the forecast numbers, or answers your questions about them. It explains only what the model produced and never invents polls.

The model is symmetric: no party or candidate gets special treatment in the code. All assumptions are settings you can change.

## Quick start

```bash
pip install -e ".[ai,dev]"
```

```bash
porta-pollster forecast --polls examples/polls.csv --config examples/races.json --fundamentals examples/fundamentals.csv
```

> **The files in `examples/` are synthetic.** The candidates, pollsters, polls and events are made up (see `examples/make_sample_data.py`). Replace them with real data before drawing any conclusions.

Add an AI briefing (needs `ANTHROPIC_API_KEY` or `ant auth login`):

```bash
porta-pollster forecast --polls examples/polls.csv --config examples/races.json --explain
```

```bash
porta-pollster forecast --polls examples/polls.csv --config examples/races.json --ask "Why is the Senate race tilting the way it is?"
```

Other useful flags: `--race NAME` (repeatable), `--as-of YYYY-MM-DD` (backtest as if it were an earlier date), `--decision-window DAYS` (default 7), `--independent-share 0-1` (default 1 = every voter decides late), `--memory-half-life DAYS` (for voters who follow along), `--json`.

## Your data

**Polls** (`polls.csv`): one row per candidate per poll.

| column | example | notes |
|---|---|---|
| poll_id | `P001` | groups rows into one poll |
| end_date | `2026-10-20` | last field date |
| pollster | `Northfield Research` | |
| race | `Example Senate` | must match a race in the config |
| population | `lv` | `lv` likely voters, `rv` registered, `a` adults, `v` voters |
| sample_size | `800` | |
| candidate | `Avery Hart` | |
| pct | `46.0` | undecided = 100 − sum of the poll's rows |

**Fundamentals** (`fundamentals.csv`, optional): `date,metric,value` with metrics `incumbent_net_approval`, `real_income_growth`, `unemployment_change`, `inflation`.

**Config** (`races.json`): each race's `election_date`, optional `incumbent` (the candidate from the party holding the seat, needed for fundamentals), and optional `events` (`date`, `candidate`, `shift` in points, `label`). The top-level `voter_behavior` holds `independent_share`, `decision_window_days`, `memory_half_life_days` and `undecided_allocation`.

## Caveats

- The default fundamentals coefficients are rough, illustrative values, not fit to real history. Fit your own from past elections with `porta_pollster.fundamentals.fit_coefficients`.
- Pollster lean is measured against the median pollster. If most pollsters miss in the same direction (as in 2016 and 2020), no averaging can see it. That shared risk is in the `systematic_poll_error_sd` setting (default 2 points).
- Event shifts are estimates you enter. The model only handles how they fade.
- The late-decider settings are assumptions. Backtest them on past races with `--as-of` to see which window and share fit best.
- Races are simulated independently; there's no cross-state correlation yet.

## Layout

```
porta_pollster/
  models.py        Poll, Event, RaceConfig, VoterBehavior
  data.py          CSV/JSON loading and validation
  aggregate.py     poll weighting, house effects, event decay, undecideds
  fundamentals.py  memory-weighted approval/economy model, coefficient fitting
  forecast.py      blending and Monte Carlo simulation
  ai.py            Claude-powered briefing
  cli.py           command line
tests/             pytest suite
examples/          synthetic sample data
```

Run tests with `pytest`.
