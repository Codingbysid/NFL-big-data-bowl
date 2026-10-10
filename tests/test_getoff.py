"""Sensor transit time: clocks, geometry, and the closed 2025 endpoint."""

import numpy as np
import pandas as pd
import pytest

from bdb2027.config import load_config
from bdb2027.getoff import assert_development_years, measure_attempt


def _spec():
    return load_config("config/getoff_v1.yaml")


def _straight(offset_s=0.0, shift_x=0.0, flip=False):
    t = np.arange(0.0, 6.0, 0.1) + offset_s
    # Two stationary frames, then a constant 8 yd/s run along +x.
    x = np.zeros(len(t))
    speed = np.zeros(len(t))
    for i, ti in enumerate(t - offset_s):
        if ti < 0.2:
            x[i] = 10.0
            speed[i] = 0.4
        else:
            x[i] = 10.0 + 8.0 * (ti - 0.2)
            speed[i] = 8.0
    y = np.full(len(t), -0.5)
    if flip:
        x = -x
    x = x + shift_x
    return pd.DataFrame({"time": pd.to_datetime(t, unit="s"), "x": x, "y": y, "s": speed})


def test_time_offset_and_translation_do_not_change_transit():
    spec = _spec()
    base = measure_attempt(_straight(), spec)
    shifted = measure_attempt(_straight(offset_s=4.0, shift_x=12.0), spec)
    assert base["valid"] and shifted["valid"]
    assert base["early_transit_1_5_s"] == pytest.approx(shifted["early_transit_1_5_s"], abs=1e-8)
    # 4 yards at 8 yd/s is 0.5 seconds.
    assert base["early_transit_1_5_s"] == pytest.approx(0.5, abs=1e-6)


def test_reflection_keeps_the_transit():
    spec = _spec()
    base = measure_attempt(_straight(), spec)
    reflected = measure_attempt(_straight(flip=True), spec)
    assert reflected["valid"]
    assert reflected["early_transit_1_5_s"] == pytest.approx(base["early_transit_1_5_s"], abs=1e-8)


def test_incomplete_crossing_is_missing():
    spec = _spec()
    short = _straight().iloc[:8].copy()
    measured = measure_attempt(short, spec)
    assert not measured["valid"]
    assert measured["reason"] in {"incomplete_crossing", "incomplete_attempt", "implausible_span"}
    assert not np.isfinite(measured["early_transit_1_5_s"]) or measured["reason"] != "ok"


def test_gap_at_the_crossing_is_missing():
    spec = _spec()
    frame = _straight()
    frame = frame.drop(index=range(8, 20)).copy()
    measured = measure_attempt(frame, spec)
    assert not measured["valid"]
    assert measured["reason"] in {"gap_at_crossing", "incomplete_crossing"}


def test_one_slow_frame_is_an_ambiguous_start():
    spec = _spec()
    frame = _straight()
    frame.loc[frame.index[1], "s"] = 3.0
    measured = measure_attempt(frame, spec)
    assert measured["reason"] == "ambiguous_start"
    assert not measured["valid"]


def test_shared_penalty_diagnostic_matches_the_reviewed_errors():
    from pathlib import Path

    import pandas as pd

    from bdb2027.getoff import common_penalty_diagnostic

    ledger = Path("outputs/getoff_v1/player_ledger.csv")
    if not ledger.exists():
        pytest.skip("player ledger is produced by the development run")
    result = common_penalty_diagnostic(pd.read_csv(ledger))
    assert result["label"].startswith("post-evaluation")
    assert result["mae_m1_alpha_3"] == pytest.approx(0.0594126898, abs=1e-9)
    assert result["mae_m2_alpha_3"] == pytest.approx(0.0607088708, abs=1e-9)


def test_2025_getoff_is_refused():
    with pytest.raises(PermissionError):
        assert_development_years([2023, 2025])
