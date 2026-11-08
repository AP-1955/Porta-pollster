"""Blend polls with fundamentals and simulate election-day outcomes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np

from . import fundamentals as fund
from .aggregate import PollAverage, average_polls
from .models import Poll, RaceConfig, VoterBehavior


@dataclass
class CandidateOutlook:
    candidate: str
    projected_share: float
    low_80: float
    high_80: float
    win_probability: float


@dataclass
class ForecastResult:
    race: str
    as_of: date
    election_date: date
    days_out: int
    outlook: list[CandidateOutlook]
    poll_weight: float
    poll_average: dict[str, float] | None
    fundamentals_incumbent_share: float | None
    share_sd: float
    n_polls: int
    house_effects: dict[str, dict[str, float]] = field(default_factory=dict)
    remembered_events: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        d["election_date"] = self.election_date.isoformat()
        return d


@dataclass
class ModelSettings:
    simulations: int = 20000
    systematic_poll_error_sd: float = 2.0  # how far poll averages miss on election day, historically
    fundamentals_sd: float = 3.0
    poll_half_life_days: float = 14.0
    seed: int | None = 2026


def poll_weight_for(days_out: int, have_polls: bool, have_fundamentals: bool) -> float:
    """How much to trust polls vs fundamentals. Polls win as election day nears."""
    if not have_fundamentals:
        return 1.0
    if not have_polls:
        return 0.0
    fundamentals_weight = 0.1 + 0.5 * days_out / (days_out + 150.0)
    return 1.0 - fundamentals_weight


def forecast_race(
    race: RaceConfig,
    polls: list[Poll],
    behavior: VoterBehavior | None = None,
    fundamentals_series: dict | None = None,
    as_of: date | None = None,
    settings: ModelSettings | None = None,
    coefficients: dict[str, float] | None = None,
) -> ForecastResult:
    behavior = behavior or VoterBehavior()
    settings = settings or ModelSettings()
    as_of = as_of or min(date.today(), race.election_date)
    days_out = max(0, (race.election_date - as_of).days)
    notes: list[str] = []

    race_polls = [p for p in polls if p.race == race.name and p.end_date <= as_of]
    avg: PollAverage | None = None
    if race_polls:
        avg = average_polls(race_polls, as_of, race.election_date, behavior, race.events, settings.poll_half_life_days)

    fund_est = None
    if fundamentals_series and race.incumbent:
        fund_est = fund.estimate(fundamentals_series, race.election_date, behavior, coefficients)
    elif fundamentals_series and not race.incumbent:
        notes.append("Fundamentals ignored: no incumbent set for this race.")

    if avg is None and fund_est is None:
        raise ValueError(f"race {race.name!r}: no polls and no usable fundamentals")

    w = poll_weight_for(days_out, avg is not None, fund_est is not None)
    shares = _blend(race, avg, fund_est, w, notes)

    candidates = list(shares)
    mean = np.array([shares[c] for c in candidates])
    sd = _share_sd(avg, w, days_out, behavior, settings)
    sims = _simulate(mean, sd, settings)

    winners = np.argmax(sims, axis=1)
    outlook = []
    for i, c in enumerate(candidates):
        lo, hi = np.percentile(sims[:, i], [10, 90])
        outlook.append(
            CandidateOutlook(
                candidate=c,
                projected_share=round(float(mean[i]), 2),
                low_80=round(float(lo), 2),
                high_80=round(float(hi), 2),
                win_probability=round(float(np.mean(winners == i)), 4),
            )
        )
    outlook.sort(key=lambda o: -o.win_probability)

    if avg is not None and avg.n_polls < 3:
        notes.append(f"Only {avg.n_polls} poll(s): treat this projection as rough.")

    return ForecastResult(
        race=race.name,
        as_of=as_of,
        election_date=race.election_date,
        days_out=days_out,
        outlook=outlook,
        poll_weight=round(w, 3),
        poll_average={c: round(v, 2) for c, v in avg.shares.items()} if avg else None,
        fundamentals_incumbent_share=round(fund_est.incumbent_two_party_share, 2) if fund_est else None,
        share_sd=round(sd, 2),
        n_polls=avg.n_polls if avg else 0,
        house_effects={p: {c: round(v, 2) for c, v in e.items()} for p, e in avg.house_effects.items()} if avg else {},
        remembered_events=[
            {
                "label": e.label,
                "candidate": e.candidate,
                "initial_shift": e.shift,
                "remembered_on_election_day": round(e.shift * behavior.memory_weight((race.election_date - e.on).days), 2),
            }
            for e in race.events
        ],
        notes=notes,
    )


def _blend(race: RaceConfig, avg: PollAverage | None, fund_est, w: float, notes: list[str]) -> dict[str, float]:
    if avg is None:
        # Fundamentals only: incumbent vs a generic challenger.
        inc = fund_est.incumbent_two_party_share
        notes.append("No polls: projection is from fundamentals alone (incumbent vs. generic challenger).")
        return {race.incumbent: inc, "Challenger": 100.0 - inc}

    shares = dict(avg.shares)
    if fund_est is None or w >= 1.0:
        return shares
    if race.incumbent not in shares:
        notes.append(f"Incumbent {race.incumbent!r} not in polls; fundamentals ignored.")
        return shares

    challengers = sorted((c for c in shares if c != race.incumbent), key=lambda c: -shares[c])
    if not challengers:
        return shares
    top = challengers[0]
    pair_total = shares[race.incumbent] + shares[top]
    poll_two_party = 100.0 * shares[race.incumbent] / pair_total
    blended = w * poll_two_party + (1 - w) * fund_est.incumbent_two_party_share
    shares[race.incumbent] = blended * pair_total / 100.0
    shares[top] = pair_total - shares[race.incumbent]
    return shares


def _share_sd(avg: PollAverage | None, w: float, days_out: int, behavior: VoterBehavior, settings: ModelSettings) -> float:
    """Total uncertainty on each candidate's share, in points."""
    drift = behavior.base_daily_drift * math.sqrt(days_out) * (1.0 + behavior.low_attention_share)
    parts = [drift**2]
    if avg is not None:
        sampling = 100.0 * math.sqrt(0.25 / max(avg.effective_sample_size, 1.0))
        parts.append((w * math.hypot(sampling, settings.systematic_poll_error_sd)) ** 2)
    parts.append(((1 - w) * settings.fundamentals_sd) ** 2)
    return math.sqrt(sum(parts))


def _simulate(mean: np.ndarray, sd: float, settings: ModelSettings) -> np.ndarray:
    rng = np.random.default_rng(settings.seed)
    k = len(mean)
    noise = rng.normal(0.0, sd, size=(settings.simulations, k))
    if k > 1:
        # Votes gained by one candidate come from others: make errors sum to zero
        # while keeping each candidate's error at `sd`.
        noise = (noise - noise.mean(axis=1, keepdims=True)) * math.sqrt(k / (k - 1))
    sims = np.clip(mean + noise, 0.0, None)
    return sims * 100.0 / sims.sum(axis=1, keepdims=True)
