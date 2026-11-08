"""Fundamentals model: approval and the economy, as voters remember them.

Voters judge incumbents mostly on recent conditions. Each reading is weighted
by how much of it the electorate still remembers on election day, using the
memory half-life from `VoterBehavior`.

The default coefficients are rough, illustrative values in the spirit of
published "fundamentals" models. They are not fit to any specific dataset.
Use `fit_coefficients` with historical results to replace them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .models import VoterBehavior

# Points of incumbent two-party share per unit of each metric (centered at 0).
DEFAULT_COEFFICIENTS = {
    "intercept": 50.0,
    "incumbent_net_approval": 0.10,  # net approval points
    "real_income_growth": 0.60,  # annualized %
    "unemployment_change": -0.80,  # change in rate, points
    "inflation": -0.35,  # annual %, above a 2% baseline
}
INFLATION_BASELINE = 2.0


@dataclass
class FundamentalsEstimate:
    incumbent_two_party_share: float
    remembered_values: dict[str, float]
    coefficients: dict[str, float] = field(default_factory=dict)


def remembered_value(series: list[tuple[date, float]], election_date: date, behavior: VoterBehavior) -> float | None:
    """Memory-weighted average of a series as of election day."""
    usable = [(d, v) for d, v in series if d <= election_date]
    if not usable:
        return None
    weights = [behavior.memory_weight((election_date - d).days) for d, _ in usable]
    return sum(w * v for w, (_, v) in zip(weights, usable)) / sum(weights)


def estimate(
    series: dict[str, list[tuple[date, float]]],
    election_date: date,
    behavior: VoterBehavior,
    coefficients: dict[str, float] | None = None,
) -> FundamentalsEstimate | None:
    coef = coefficients or DEFAULT_COEFFICIENTS
    remembered = {}
    share = coef.get("intercept", 50.0)
    for metric, beta in coef.items():
        if metric == "intercept" or metric not in series:
            continue
        val = remembered_value(series[metric], election_date, behavior)
        if val is None:
            continue
        remembered[metric] = val
        if metric == "inflation":
            val -= INFLATION_BASELINE
        share += beta * val
    if not remembered:
        return None
    return FundamentalsEstimate(float(np.clip(share, 30.0, 70.0)), remembered, dict(coef))


def fit_coefficients(history: list[dict[str, float]], target: str = "incumbent_two_party_share") -> dict[str, float]:
    """Ordinary least squares fit from past elections.

    Each row holds the remembered metric values for one past election plus the
    actual incumbent two-party share. Small samples overfit: treat with care.
    """
    metrics = sorted({k for row in history for k in row} - {target})
    if len(history) <= len(metrics) + 1:
        raise ValueError(f"need more than {len(metrics) + 1} elections to fit {len(metrics)} metrics")
    X = np.array([[1.0] + [row.get(m, 0.0) - (INFLATION_BASELINE if m == "inflation" else 0.0) for m in metrics] for row in history])
    y = np.array([row[target] for row in history])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"intercept": float(beta[0]), **{m: float(b) for m, b in zip(metrics, beta[1:])}}
