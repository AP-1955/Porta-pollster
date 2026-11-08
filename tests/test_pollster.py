import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from porta_pollster import Event, Poll, RaceConfig, VoterBehavior, forecast_race
from porta_pollster.aggregate import allocate_undecided, average_polls, estimate_house_effects, event_adjustment
from porta_pollster.cli import main
from porta_pollster.data import DataError, load_polls
from porta_pollster.forecast import ModelSettings
from porta_pollster import fundamentals as fund

EXAMPLES = Path(__file__).parent.parent / "examples"
ELECTION = date(2026, 11, 3)


def make_poll(i, days_before, pollster, shares, n=800, race="R", pop="lv"):
    return Poll(f"p{i}", ELECTION - timedelta(days=days_before), pollster, race, pop, n, shares)


def test_memory_weight_halves_each_half_life():
    b = VoterBehavior(memory_half_life_days=100)
    assert b.memory_weight(0) == 1.0
    assert b.memory_weight(100) == pytest.approx(0.5)
    assert b.memory_weight(200) == pytest.approx(0.25)


def test_house_effect_detects_biased_pollster():
    polls = []
    for i in range(20):
        polls.append(make_poll(i, 5 + i, "Fair A", {"X": 45, "Y": 45}))
        polls.append(make_poll(100 + i, 5 + i, "Fair B", {"X": 45, "Y": 45}))
        polls.append(make_poll(200 + i, 5 + i, "Biased", {"X": 49, "Y": 41}))
    effects = estimate_house_effects(polls, ELECTION)
    assert effects["Biased"]["X"] > 2.5
    assert effects["Biased"]["Y"] < -2.5
    assert abs(effects["Fair A"]["X"]) < 2.0


def test_average_corrects_house_effect():
    # Lean is measured against the consensus, so it takes 3+ pollsters to single one out.
    polls = [make_poll(i, i + 1, "Fair A", {"X": 45, "Y": 45}) for i in range(10)]
    polls += [make_poll(30 + i, i + 1, "Fair B", {"X": 45, "Y": 45}) for i in range(10)]
    polls += [make_poll(60 + i, i + 1, "Biased", {"X": 54, "Y": 36}) for i in range(10)]
    avg = average_polls(polls, ELECTION, ELECTION, VoterBehavior())
    # Uncorrected, X would lead by 6; after correction the race is much closer.
    assert abs(avg.raw_shares["X"] - avg.raw_shares["Y"]) < 3.0


def test_event_effect_fades_by_election_day():
    b = VoterBehavior(memory_half_life_days=60, independent_share=0.0)
    event = Event(ELECTION - timedelta(days=120), "X", -6.0)
    poll = make_poll(1, 118, "A", {"X": 40, "Y": 50})  # right after the event
    # 6 * (~0.98 - 0.25): most of the damage is forgotten, so the poll moves up for X.
    assert event_adjustment(poll, "X", [event], ELECTION, b) == pytest.approx(6 * (0.98 - 0.25), abs=0.1)
    assert event_adjustment(poll, "Y", [event], ELECTION, b) == 0.0


def test_independents_never_notice_early_events():
    only_partisans = VoterBehavior(independent_share=0.0)
    with_independents = VoterBehavior(independent_share=0.4)
    early = Event(ELECTION - timedelta(days=60), "X", -5.0)
    # Independents never saw it, so less of the hit survives to election day.
    assert with_independents.event_effect(early, ELECTION, ELECTION) == pytest.approx(
        0.6 * only_partisans.event_effect(early, ELECTION, ELECTION)
    )


def test_independents_feel_final_week_events_fully():
    b = VoterBehavior(independent_share=0.4, decision_window_days=7)
    late = Event(ELECTION - timedelta(days=3), "X", -5.0)
    assert b.event_effect(late, ELECTION, ELECTION) == pytest.approx(0.6 * b.memory_weight(3) + 0.4)
    # A poll taken before the event can't show it; the adjustment adds its election-day effect.
    poll = make_poll(1, 5, "A", {"X": 45, "Y": 45})
    assert event_adjustment(poll, "X", [late], ELECTION, b) == pytest.approx(-5.0 * b.event_effect(late, ELECTION, ELECTION))


def test_final_week_polls_count_more():
    b = VoterBehavior(independent_share=0.3, decision_window_days=7)
    early = make_poll(1, 20, "A", {"X": 45, "Y": 45})
    late = make_poll(2, 3, "A", {"X": 45, "Y": 45})
    as_of = ELECTION
    from porta_pollster.aggregate import poll_weight

    ratio = poll_weight(early, as_of, 1e9, ELECTION, b) / poll_weight(late, as_of, 1e9, ELECTION, b)
    assert ratio == pytest.approx(0.7)


def test_uncertainty_drops_once_independents_tune_in():
    polls = [make_poll(i, 10 + i, "A", {"X": 50, "Y": 46}) for i in range(10)]
    race = RaceConfig("R", ELECTION)
    b = VoterBehavior(decision_window_days=7)
    before = forecast_race(race, polls, b, as_of=ELECTION - timedelta(days=8))
    after = forecast_race(race, polls, b, as_of=ELECTION - timedelta(days=1))
    assert before.share_sd > after.share_sd + 1.0
    assert not before.independents["deciding_now"] and after.independents["deciding_now"]


def test_more_independents_means_more_uncertainty_before_window():
    polls = [make_poll(i, 60 + i, "A", {"X": 50, "Y": 46}) for i in range(10)]
    race = RaceConfig("R", ELECTION)
    as_of = ELECTION - timedelta(days=59)
    few = forecast_race(race, polls, VoterBehavior(independent_share=0.1), as_of=as_of)
    many = forecast_race(race, polls, VoterBehavior(independent_share=0.5), as_of=as_of)
    assert many.share_sd > few.share_sd


def test_independent_share_validated():
    with pytest.raises(ValueError):
        VoterBehavior(independent_share=1.5)


def test_undecided_allocation():
    assert allocate_undecided({"X": 40, "Y": 40}) == {"X": 50, "Y": 50}
    assert allocate_undecided({"X": 45, "Y": 35}, "even") == {"X": 55, "Y": 45}
    with pytest.raises(ValueError):
        allocate_undecided({"X": 1}, "nonsense")


def test_forecast_probabilities_and_leader():
    polls = [make_poll(i, i + 1, f"P{i % 3}", {"X": 52, "Y": 44}, n=1000) for i in range(15)]
    r = forecast_race(RaceConfig("R", ELECTION), polls, as_of=ELECTION)
    assert sum(o.win_probability for o in r.outlook) == pytest.approx(1.0, abs=1e-3)
    assert r.outlook[0].candidate == "X"
    assert r.outlook[0].win_probability > 0.9


def test_more_time_means_more_uncertainty_when_some_voters_follow_along():
    # Only voters who follow the campaign drift over time; late deciders don't move until the final week.
    b = VoterBehavior(independent_share=0.3)
    polls = [make_poll(i, 100 + i, "A", {"X": 50, "Y": 46}) for i in range(10)]
    race = RaceConfig("R", ELECTION)
    near = forecast_race(race, polls, b, as_of=ELECTION - timedelta(days=99))
    polls_far = [make_poll(i, 300 + i, "A", {"X": 50, "Y": 46}) for i in range(10)]
    far = forecast_race(race, polls_far, b, as_of=ELECTION - timedelta(days=299))
    assert far.share_sd > near.share_sd


def test_forecast_is_reproducible_with_seed():
    polls = [make_poll(i, i + 1, "A", {"X": 48, "Y": 47}) for i in range(5)]
    race = RaceConfig("R", ELECTION)
    a = forecast_race(race, polls, as_of=ELECTION, settings=ModelSettings(seed=1))
    b = forecast_race(race, polls, as_of=ELECTION, settings=ModelSettings(seed=1))
    assert a.outlook == b.outlook


def test_fundamentals_weight_recent_data_more():
    b = VoterBehavior(memory_half_life_days=90, independent_share=0.0)
    series = [(ELECTION - timedelta(days=360), -20.0), (ELECTION - timedelta(days=10), 10.0)]
    remembered = fund.remembered_value(series, ELECTION, b)
    assert remembered > 5.0  # the recent good reading dominates the old bad one


def test_independents_judge_economy_on_latest_reading():
    series = [(ELECTION - timedelta(days=300), -20.0), (ELECTION - timedelta(days=40), 10.0)]
    all_independents = VoterBehavior(independent_share=1.0)
    assert fund.remembered_value(series, ELECTION, all_independents) == 10.0


def test_fundamentals_only_forecast():
    series = {"incumbent_net_approval": [(ELECTION - timedelta(days=30), 20.0)]}
    r = forecast_race(RaceConfig("R", ELECTION, incumbent="X"), [], fundamentals_series=series, as_of=ELECTION)
    assert r.outlook[0].candidate == "X"
    assert r.poll_weight == 0.0


def test_fit_coefficients_recovers_slope():
    history = [{"incumbent_net_approval": a, "incumbent_two_party_share": 50 + 0.2 * a} for a in range(-20, 21, 5)]
    coef = fund.fit_coefficients(history)
    assert coef["incumbent_net_approval"] == pytest.approx(0.2)
    assert coef["intercept"] == pytest.approx(50.0)


def test_load_polls_validates(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("poll_id,end_date,pollster,race,population,sample_size,candidate,pct\n"
                   "1,2026-10-01,A,R,martians,500,X,50\n")
    with pytest.raises(DataError):
        load_polls(bad)


def test_cli_on_examples(capsys):
    code = main(["forecast", "--polls", str(EXAMPLES / "polls.csv"), "--config", str(EXAMPLES / "races.json"),
                 "--fundamentals", str(EXAMPLES / "fundamentals.csv"), "--json", "--simulations", "2000"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert {r["race"] for r in out} == {"Example Senate", "Example Governor"}


def test_ai_explain_uses_forecast_and_returns_text():
    from porta_pollster import ai

    captured = {}

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="X is favored.")])

    class FakeMessages:
        def stream(self, **kwargs):
            captured.update(kwargs)
            return FakeStream()

    fake = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()))
    polls = [make_poll(i, i + 1, "A", {"X": 52, "Y": 44}) for i in range(5)]
    result = forecast_race(RaceConfig("R", ELECTION), polls, as_of=ELECTION, settings=ModelSettings(simulations=500))
    assert ai.explain([result], client=fake) == "X is favored."
    assert captured["model"] == ai.MODEL
    assert '"race": "R"' in captured["messages"][0]["content"]


def test_default_is_every_voter_a_late_decider():
    assert VoterBehavior().independent_share == 1.0


def test_all_late_deciders_with_only_early_polls_still_works():
    polls = [make_poll(i, 30 + i, f"P{i % 3}", {"X": 53, "Y": 43}) for i in range(10)]
    r = forecast_race(RaceConfig("R", ELECTION), polls, VoterBehavior(independent_share=1.0), as_of=ELECTION - timedelta(days=29))
    assert r.outlook[0].candidate == "X"
    assert r.poll_average["X"] > 54  # undecideds allocated, not a 50/50 fallback


def test_early_event_forgotten_entirely_when_everyone_decides_late():
    b = VoterBehavior(independent_share=1.0)
    early = Event(ELECTION - timedelta(days=60), "X", -5.0)
    assert b.event_effect(early, ELECTION, ELECTION) == 0.0


def test_poll_dip_from_early_event_is_undone_when_everyone_decides_late():
    b = VoterBehavior(independent_share=1.0)
    scandal = Event(ELECTION - timedelta(days=60), "X", -5.0)
    poll = make_poll(1, 55, "A", {"X": 40, "Y": 50})  # taken during the dip
    # The poll shows most of the -5 hit; on election day it's gone, so X moves back up.
    assert event_adjustment(poll, "X", [scandal], ELECTION, b) == pytest.approx(5.0 * b.memory_weight(5))
