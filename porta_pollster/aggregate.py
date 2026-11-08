"""Poll averaging with recency, sample-size and population weights, plus house-effect correction."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date

from .models import POPULATION_WEIGHTS, Event, Poll, VoterBehavior


@dataclass
class PollAverage:
    shares: dict[str, float]  # candidate -> share, undecided allocated, sums to 100
    raw_shares: dict[str, float]  # before undecided allocation
    undecided: float
    house_effects: dict[str, dict[str, float]]  # pollster -> candidate -> lean
    effective_sample_size: float
    n_polls: int


def poll_weight(poll: Poll, as_of: date, half_life_days: float = 14.0) -> float:
    age = max(0, (as_of - poll.end_date).days)
    recency = 0.5 ** (age / half_life_days)
    size = math.sqrt(min(poll.sample_size, 5000) / 600.0)
    return recency * size * POPULATION_WEIGHTS[poll.population]


def event_adjustment(poll: Poll, candidate: str, events: list[Event], election_date: date, behavior: VoterBehavior) -> float:
    """How much of each event's effect in this poll will have faded (or still remain) by election day.

    A poll right after a scandal shows its full effect; by election day only the
    remembered part is left, so we move the poll toward that.
    """
    adj = 0.0
    for e in events:
        if e.candidate != candidate or e.on > poll.end_date:
            continue
        in_poll = behavior.memory_weight((poll.end_date - e.on).days)
        on_election_day = behavior.memory_weight((election_date - e.on).days)
        adj += e.shift * (on_election_day - in_poll)
    return adj


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    total = sum(w for _, w in values)
    return sum(v * w for v, w in values) / total if total else 0.0


def estimate_house_effects(polls: list[Poll], as_of: date, shrinkage: float = 3.0, iterations: int = 3) -> dict[str, dict[str, float]]:
    """Estimate each pollster's systematic lean per candidate.

    Lean = average gap between a pollster's numbers and everyone else's,
    shrunk toward zero so a pollster with one or two polls isn't judged harshly,
    then measured from the median pollster (who counts as neutral).
    """
    candidates = sorted({c for p in polls for c in p.shares})
    effects: dict[str, dict[str, float]] = {p.pollster: {c: 0.0 for c in candidates} for p in polls}
    if len({p.pollster for p in polls}) < 2:
        return effects

    for _ in range(iterations):
        new = {}
        for pollster in effects:
            new[pollster] = {}
            mine = [p for p in polls if p.pollster == pollster]
            others = [p for p in polls if p.pollster != pollster]
            for c in candidates:
                residuals = []
                for p in mine:
                    if c not in p.shares:
                        continue
                    # Compare against other pollsters' polls near the same date.
                    peers = [
                        (o.shares[c] - effects[o.pollster][c], poll_weight(o, p.end_date, half_life_days=10))
                        for o in others
                        if c in o.shares and abs((o.end_date - p.end_date).days) <= 30
                    ]
                    if peers:
                        residuals.append(p.shares[c] - _weighted_mean(peers))
                new[pollster][c] = sum(residuals) / (len(residuals) + shrinkage) if residuals else 0.0
        # Leans are only defined relative to each other. Anchor them on the median
        # pollster so one outlier is corrected instead of shifting everyone else.
        for c in candidates:
            mid = statistics.median(new[p][c] for p in new)
            for p in new:
                new[p][c] -= mid
        effects = new
    return effects


def average_polls(
    polls: list[Poll],
    as_of: date,
    election_date: date,
    behavior: VoterBehavior,
    events: list[Event] | None = None,
    half_life_days: float = 14.0,
) -> PollAverage:
    if not polls:
        raise ValueError("no polls to average")
    polls = [p for p in polls if p.end_date <= as_of]
    if not polls:
        raise ValueError("no polls on or before the as-of date")
    events = events or []
    house = estimate_house_effects(polls, as_of)
    candidates = sorted({c for p in polls for c in p.shares})

    raw: dict[str, float] = {}
    for c in candidates:
        vals = []
        for p in polls:
            if c not in p.shares:
                continue
            adjusted = p.shares[c] - house[p.pollster][c] + event_adjustment(p, c, events, election_date, behavior)
            vals.append((adjusted, poll_weight(p, as_of, half_life_days)))
        raw[c] = max(0.0, _weighted_mean(vals))

    undecided = _weighted_mean([(p.undecided, poll_weight(p, as_of, half_life_days)) for p in polls])
    shares = allocate_undecided(raw, behavior.undecided_allocation)

    # Effective sample size of the weighted average: n such that p(1-p)/n
    # matches the variance of the weighted mean of independent polls.
    weights = [poll_weight(p, as_of, half_life_days) for p in polls]
    eff_n = sum(weights) ** 2 / max(sum(w * w / p.sample_size for w, p in zip(weights, polls)), 1e-12)
    return PollAverage(shares, raw, undecided, house, eff_n, len(polls))


def allocate_undecided(raw: dict[str, float], method: str = "proportional") -> dict[str, float]:
    total = sum(raw.values())
    if total <= 0:
        return {c: 100.0 / len(raw) for c in raw}
    remainder = max(0.0, 100.0 - total)
    if method == "even":
        return {c: v + remainder / len(raw) for c, v in raw.items()}
    if method != "proportional":
        raise ValueError(f"unknown undecided_allocation: {method}")
    return {c: v * 100.0 / total for c, v in raw.items()}
