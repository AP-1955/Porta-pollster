"""Plain data types shared across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# Likely voters are the best predictor of an election; adult samples the worst.
POPULATION_WEIGHTS = {"lv": 1.0, "rv": 0.85, "a": 0.6, "v": 1.0}


@dataclass(frozen=True)
class Poll:
    """One poll of one race. `shares` maps candidate -> percent (undecided is the remainder)."""

    poll_id: str
    end_date: date
    pollster: str
    race: str
    population: str
    sample_size: int
    shares: dict[str, float]

    @property
    def undecided(self) -> float:
        return max(0.0, 100.0 - sum(self.shares.values()))


@dataclass(frozen=True)
class Event:
    """Something that moved opinion (scandal, debate, crisis).

    `shift` is the immediate effect in vote-share points for `candidate`
    (negative = hurt them). Its effect fades with voter memory.
    """

    on: date
    candidate: str
    shift: float
    label: str = ""


@dataclass
class VoterBehavior:
    """How the electorate remembers and pays attention.

    Grounded in political-science findings that voters weight recent
    conditions far more than older ones (e.g. Achen & Bartels, Healy & Lenz
    "end bias") and that low-attention voters move late and unpredictably.
    """

    # Days until a past event / economic reading counts half as much.
    memory_half_life_days: float = 120.0
    # Share of the electorate that pays little attention (0-1). Raises late volatility.
    low_attention_share: float = 0.3
    # How undecideds break: "proportional" (to current support) or "even".
    undecided_allocation: str = "proportional"
    # Daily opinion drift (share points per sqrt(day)) for a fully attentive electorate.
    base_daily_drift: float = 0.18

    def memory_weight(self, days_ago: float) -> float:
        """Fraction of an effect still remembered after `days_ago` days."""
        if days_ago <= 0:
            return 1.0
        return 0.5 ** (days_ago / self.memory_half_life_days)


@dataclass
class RaceConfig:
    name: str
    election_date: date
    incumbent: str | None = None  # candidate of the party holding the office, if any
    events: list[Event] = field(default_factory=list)
