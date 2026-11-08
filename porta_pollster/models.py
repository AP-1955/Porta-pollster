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

    By default every voter is a late decider. The electorate can be split
    into two groups with `independent_share`:

    - Partisans follow the campaign all along and slowly forget older news
      (memory half-life). Voters weight recent conditions far more than older
      ones (e.g. Achen & Bartels, Healy & Lenz "end bias").
    - Independents ignore the campaign until the final `decision_window_days`
      before the election. Only what happens in that window moves them, and
      until it opens, their final choice is wide open.
    """

    # Days until a past event / economic reading counts half as much (partisans).
    memory_half_life_days: float = 120.0
    # Share of the electorate that only tunes in during the final window (0-1).
    # Default: everyone. Lower it to keep some voters following the campaign all along.
    independent_share: float = 1.0
    # Independents tune in only this many days before the election.
    decision_window_days: float = 7.0
    # How far the late deciders' split can swing before they decide (points of their vote).
    independent_swing_sd: float = 6.0
    # How undecideds break: "proportional" (to current support) or "even".
    undecided_allocation: str = "proportional"
    # Daily opinion drift (share points per sqrt(day)) among partisans.
    base_daily_drift: float = 0.18

    def __post_init__(self):
        if not 0.0 <= self.independent_share <= 1.0:
            raise ValueError("independent_share must be between 0 and 1")
        if self.decision_window_days < 0 or self.memory_half_life_days <= 0:
            raise ValueError("decision_window_days must be >= 0 and memory_half_life_days > 0")

    def memory_weight(self, days_ago: float) -> float:
        """Fraction of an effect partisans still remember after `days_ago` days."""
        if days_ago <= 0:
            return 1.0
        return 0.5 ** (days_ago / self.memory_half_life_days)

    def in_decision_window(self, day: date, election_date: date) -> bool:
        return (election_date - day).days <= self.decision_window_days

    def independents_deciding(self, as_of: date, election_date: date) -> bool:
        return self.in_decision_window(as_of, election_date)

    def event_effect(self, event: "Event", on: date, election_date: date) -> float:
        """Share of an event's shift that shows up in opinion on day `on`.

        Before the final window, everyone reacts to news when a pollster asks,
        and that reaction fades with memory. Once late deciders make up their
        minds in the final window, they go only on what happened inside it.
        """
        if event.on > on:
            return 0.0
        reaction = self.memory_weight((on - event.on).days)
        if not self.in_decision_window(on, election_date):
            return reaction
        late = 1.0 if self.in_decision_window(event.on, election_date) else 0.0
        s = self.independent_share
        return (1 - s) * reaction + s * late


@dataclass
class RaceConfig:
    name: str
    election_date: date
    incumbent: str | None = None  # candidate of the party holding the office, if any
    events: list[Event] = field(default_factory=list)
