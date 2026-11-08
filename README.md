# Porta-Pollster

An AI-assisted pollster that projects polls and election results from data rather than gut feeling. It also models how voters actually behave: they forget older news, and many tune in only at the last minute.

## What it does

1. **Averages polls fairly.** Recent polls, bigger samples and likely-voter polls count more.
2. **Corrects pollster lean ("house effects").** It measures how far each pollster tends to sit from the consensus and subtracts it. A pollster with only a few polls gets the benefit of the doubt.
3. **Adds the fundamentals.** It uses incumbent approval, real income growth, unemployment and inflation. Far from election day these count for more; close to it, polls dominate.
4. **Models voter memory and attention.**
   - **Memory half-life:** an event's effect (or an economic reading) fades by half every *N* days (default 120). Research on retrospective voting finds voters weight the last few months heavily ([Achen & Bartels](https://press.princeton.edu/books/paperback/9780691178240/democracy-for-realists); Healy & Lenz on "end bias").
   - **Event decay:** if a scandal knocked a candidate down 4 points in August, polls taken right after it overstate the damage on election day. The model moves those polls toward what voters will still remember.
   - **Low-attention share:** the bigger the share of tuned-out voters, the more the race can still move late. This widens the uncertainty.
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

Other useful flags: `--race NAME` (repeatable), `--as-of YYYY-MM-DD` (backtest as if it were an earlier date), `--memory-half-life DAYS`, `--json`.

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

**Config** (`races.json`): each race's `election_date`, optional `incumbent` (the candidate from the party holding the seat, needed for fundamentals), and optional `events` (`date`, `candidate`, `shift` in points, `label`). The top-level `voter_behavior` holds the memory and attention settings.

## Caveats

- The default fundamentals coefficients are rough, illustrative values, not fit to real history. Fit your own from past elections with `porta_pollster.fundamentals.fit_coefficients`.
- Pollster lean is measured against the median pollster. If most pollsters miss in the same direction (as in 2016 and 2020), no averaging can see it. That shared risk is in the `systematic_poll_error_sd` setting (default 2 points).
- Event shifts are estimates you enter. The model only handles how they fade.
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
