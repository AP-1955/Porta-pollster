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
    b = VoterBehavior(memory_half_life_days=60)
    event = Event(ELECTION - timedelta(days=120), "X", -6.0)
    poll = make_poll(1, 118, "A", {"X": 40, "Y": 50})  # right after the event
    # 6 * (0.25 - ~0.98): most of the damage is forgotten, so poll moves up for X.
    assert event_adjustment(poll, "X", [event], ELECTION, b) == pytest.approx(6 * (0.98 - 0.25), abs=0.1)
    assert event_adjustment(poll, "Y", [event], ELECTION, b) == 0.0


def test_undecided_allocation():
    assert allocate_undecided({"X": 40, "Y": 40}) == {"X": 50, "Y": 50}
    assert allocate_undecided({"X": 45, "Y": 35}, "even") == {"X": 55, "Y": 45}
    with pytest.raises(ValueError):
        allocate_undecided({"X": 1}, "nonsense")


def test_forecast_probabilities_and_leader():
    polls = [make_poll(i, i + 1, f"P{i % 3}", {"X": 52, "Y": 44}, n=1000) for i in range(15)]
    r = forecast_race(RaceConfig("R", ELECTION), polls, as_of=ELECTION)
    assert sum(o.win_probability for o in r.outlook) == pytest.approx(1.0, abs=1e-6)
    assert r.outlook[0].candidate == "X"
    assert r.outlook[0].win_probability > 0.9


def test_more_time_means_more_uncertainty():
    polls = [make_poll(i, 100 + i, "A", {"X": 50, "Y": 46}) for i in range(10)]
    race = RaceConfig("R", ELECTION)
    near = forecast_race(race, polls, as_of=ELECTION - timedelta(days=99))
    polls_far = [make_poll(i, 300 + i, "A", {"X": 50, "Y": 46}) for i in range(10)]
    far = forecast_race(race, polls_far, as_of=ELECTION - timedelta(days=299))
    assert far.share_sd > near.share_sd


def test_low_attention_increases_uncertainty():
    polls = [make_poll(i, 60 + i, "A", {"X": 50, "Y": 46}) for i in range(10)]
    race = RaceConfig("R", ELECTION)
    as_of = ELECTION - timedelta(days=59)
    calm = forecast_race(race, polls, VoterBehavior(low_attention_share=0.0), as_of=as_of)
    jumpy = forecast_race(race, polls, VoterBehavior(low_attention_share=0.8), as_of=as_of)
    assert jumpy.share_sd > calm.share_sd


def test_forecast_is_reproducible_with_seed():
    polls = [make_poll(i, i + 1, "A", {"X": 48, "Y": 47}) for i in range(5)]
    race = RaceConfig("R", ELECTION)
    a = forecast_race(race, polls, as_of=ELECTION, settings=ModelSettings(seed=1))
    b = forecast_race(race, polls, as_of=ELECTION, settings=ModelSettings(seed=1))
    assert a.outlook == b.outlook


def test_fundamentals_weight_recent_data_more():
    b = VoterBehavior(memory_half_life_days=90)
    series = [(ELECTION - timedelta(days=360), -20.0), (ELECTION - timedelta(days=10), 10.0)]
    remembered = fund.remembered_value(series, ELECTION, b)
    assert remembered > 5.0  # the recent good reading dominates the old bad one


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
