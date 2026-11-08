"""Loading polls, fundamentals and race configs from plain files."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from .models import POPULATION_WEIGHTS, Event, Poll, RaceConfig, VoterBehavior

POLL_COLUMNS = {"poll_id", "end_date", "pollster", "race", "population", "sample_size", "candidate", "pct"}


class DataError(ValueError):
    pass


def load_polls(path: str | Path) -> list[Poll]:
    """Read polls in long format: one row per (poll, candidate).

    Columns: poll_id,end_date,pollster,race,population,sample_size,candidate,pct
    """
    rows_by_poll: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = POLL_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise DataError(f"{path}: missing columns {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            rows_by_poll[row["poll_id"].strip()].append({**row, "_line": line})

    polls = []
    for poll_id, rows in rows_by_poll.items():
        first = rows[0]
        population = first["population"].strip().lower()
        if population not in POPULATION_WEIGHTS:
            raise DataError(f"line {first['_line']}: population must be one of {sorted(POPULATION_WEIGHTS)}")
        try:
            shares = {r["candidate"].strip(): float(r["pct"]) for r in rows}
            sample_size = int(first["sample_size"])
            end_date = date.fromisoformat(first["end_date"].strip())
        except ValueError as e:
            raise DataError(f"poll {poll_id} (line {first['_line']}): {e}") from e
        if sample_size <= 0:
            raise DataError(f"poll {poll_id}: sample_size must be positive")
        if any(p < 0 for p in shares.values()) or sum(shares.values()) > 100.5:
            raise DataError(f"poll {poll_id}: shares must be >= 0 and total <= 100")
        polls.append(
            Poll(
                poll_id=poll_id,
                end_date=end_date,
                pollster=first["pollster"].strip(),
                race=first["race"].strip(),
                population=population,
                sample_size=sample_size,
                shares=shares,
            )
        )
    return sorted(polls, key=lambda p: p.end_date)


def load_fundamentals(path: str | Path) -> dict[str, list[tuple[date, float]]]:
    """Read economic / approval series. Columns: date,metric,value

    Metrics understood by the default model: incumbent_net_approval,
    real_income_growth, unemployment_change, inflation. Others are kept but ignored.
    """
    series: dict[str, list[tuple[date, float]]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not {"date", "metric", "value"} <= set(reader.fieldnames or []):
            raise DataError(f"{path}: needs columns date,metric,value")
        for row in reader:
            series[row["metric"].strip()].append((date.fromisoformat(row["date"].strip()), float(row["value"])))
    return {k: sorted(v) for k, v in series.items()}


def load_config(path: str | Path) -> tuple[dict[str, RaceConfig], VoterBehavior]:
    """Read a JSON config with `races` and optional `voter_behavior`."""
    raw = json.loads(Path(path).read_text())
    behavior = VoterBehavior(**raw.get("voter_behavior", {}))
    races = {}
    for name, r in raw.get("races", {}).items():
        events = [
            Event(
                on=date.fromisoformat(e["date"]),
                candidate=e["candidate"],
                shift=float(e["shift"]),
                label=e.get("label", ""),
            )
            for e in r.get("events", [])
        ]
        races[name] = RaceConfig(
            name=name,
            election_date=date.fromisoformat(r["election_date"]),
            incumbent=r.get("incumbent"),
            events=events,
        )
    return races, behavior
