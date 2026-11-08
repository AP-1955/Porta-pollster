"""Porta-Pollster: data-driven poll aggregation and election projection."""

from .models import Poll, Event, RaceConfig, VoterBehavior
from .forecast import forecast_race, ForecastResult

__all__ = [
    "Poll",
    "Event",
    "RaceConfig",
    "VoterBehavior",
    "forecast_race",
    "ForecastResult",
]
__version__ = "0.1.0"
