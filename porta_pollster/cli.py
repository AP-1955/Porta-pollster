"""Command line: porta-pollster forecast --polls polls.csv --config races.json [--explain]"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from .data import DataError, load_config, load_fundamentals, load_polls
from .forecast import ForecastResult, ModelSettings, forecast_race


def _bar(p: float, width: int = 30) -> str:
    n = round(p * width)
    return "#" * n + "." * (width - n)


def render(result: ForecastResult) -> str:
    lines = [
        f"== {result.race} ==  election {result.election_date}  ({result.days_out} days out, as of {result.as_of})",
        f"   {result.n_polls} polls, poll weight {result.poll_weight:.0%}, "
        f"fundamentals: {'incumbent ' + format(result.fundamentals_incumbent_share, '.1f') + '% two-party' if result.fundamentals_incumbent_share is not None else 'none'}",
        "",
        f"   {'Candidate':<22}{'Proj.':>7}{'80% range':>16}   Win chance",
    ]
    for o in result.outlook:
        lines.append(
            f"   {o.candidate:<22}{o.projected_share:>6.1f}%{o.low_80:>8.1f}-{o.high_80:<5.1f}  "
            f"{_bar(o.win_probability)} {o.win_probability:6.1%}"
        )
    leans = [(p, c, v) for p, e in result.house_effects.items() for c, v in e.items() if abs(v) >= 1.0]
    if leans:
        lines.append("\n   Notable pollster leans (corrected for):")
        for p, c, v in sorted(leans, key=lambda x: -abs(x[2])):
            lines.append(f"     {p}: {v:+.1f} for {c}")
    ind = result.independents
    if ind:
        status = "deciding now" if ind["deciding_now"] else "not tuned in yet, so the race is still up for grabs"
        lines.append(
            f"\n   Late deciders: {ind['share']:.0%} of voters, decide in the final "
            f"{ind['decision_window_days']:g} days ({status})"
        )
    if result.remembered_events:
        lines.append("\n   Events, and their effect on election day:")
        for e in result.remembered_events:
            who = "inside the final week" if e["independents_notice"] else "before the final week, so late deciders never noticed"
            detail = who if ind.get("share", 1.0) >= 1.0 else f"engaged voters remember {e['partisans_remember']:+.1f}; {who}"
            lines.append(
                f"     {e['label'] or 'event'} ({e['candidate']}): {e['initial_shift']:+.1f} -> "
                f"{e['remembered_on_election_day']:+.1f}  ({detail})"
            )
    for n in result.notes:
        lines.append(f"   note: {n}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="porta-pollster", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("forecast", help="Project polls and results for races in a config file")
    f.add_argument("--polls", required=True, help="CSV of polls (long format)")
    f.add_argument("--config", required=True, help="JSON file with races and voter behavior")
    f.add_argument("--fundamentals", help="CSV of approval / economic series")
    f.add_argument("--race", action="append", help="Only forecast these races (repeatable)")
    f.add_argument("--as-of", type=date.fromisoformat, help="Pretend today is this date (YYYY-MM-DD) for backtesting")
    f.add_argument("--memory-half-life", type=float, help="Override voter memory half-life in days")
    f.add_argument("--independent-share", type=float, help="Share of voters who only tune in during the final week (0-1, default 1 = everyone)")
    f.add_argument("--decision-window", type=float, help="Days before the election when independents tune in")
    f.add_argument("--simulations", type=int, default=20000)
    f.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    f.add_argument("--explain", action="store_true", help="Add a plain-English AI briefing (needs Anthropic API access)")
    f.add_argument("--ask", help="Ask the AI analyst a specific question about the forecast")
    args = parser.parse_args(argv)

    try:
        polls = load_polls(args.polls)
        races, behavior = load_config(args.config)
        series = load_fundamentals(args.fundamentals) if args.fundamentals else None
    except (DataError, OSError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.memory_half_life:
        behavior.memory_half_life_days = args.memory_half_life
    if args.independent_share is not None:
        behavior.independent_share = args.independent_share
    if args.decision_window is not None:
        behavior.decision_window_days = args.decision_window
    try:
        behavior.__post_init__()
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    selected = [races[r] for r in args.race] if args.race else list(races.values())
    settings = ModelSettings(simulations=args.simulations)

    results = []
    for race in selected:
        try:
            results.append(forecast_race(race, polls, behavior, series, as_of=args.as_of, settings=settings))
        except ValueError as e:
            print(f"skipping {race.name}: {e}", file=sys.stderr)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print("\n\n".join(render(r) for r in results))

    if (args.explain or args.ask) and results:
        from .ai import explain

        print("\n== AI analyst ==\n")
        try:
            print(explain(results, question=args.ask))
        except Exception as e:  # keep the numeric forecast usable even if the AI call fails
            print(f"AI briefing unavailable: {e}", file=sys.stderr)
            return 1
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
