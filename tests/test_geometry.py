"""Numerical checks for headings, rotation, and the synthetic turn."""

from __future__ import annotations

import numpy as np
import pytest

from bdb2027.config import load_config
from bdb2027.cuts import detect_cuts, detection_params
from bdb2027.tracking import (
    heading_deg,
    reflect_x,
    rotate_xy,
    to_offense_direction,
    unit_from_heading,
    wrap_deg,
)


def _params(window=5):
    cfg = load_config()
    return detection_params(cfg, sg_window=window)


def _integrate(headings, speeds, dt=0.1):
    n = len(headings)
    t = np.round(np.arange(n) * dt, 10)
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        rad = np.radians(headings[i])
        x[i] = x[i - 1] + speeds[i] * dt * np.sin(rad)
        y[i] = y[i - 1] + speeds[i] * dt * np.cos(rad)
    return t, x, y


def _sharp_corner(angle=90.0, t_corner=1.0, t_end=2.2, speed_in=6.0, speed_out=5.0, dt=0.1):
    t = np.round(np.arange(0.0, t_end + 1e-9, dt), 10)
    headings = np.where(t <= t_corner, 0.0, angle)
    speeds = np.where(t <= t_corner, speed_in, speed_out)
    t, x, y = _integrate(headings, speeds, dt=dt)
    return t, x, y


def _smooth_turn(angle=90.0, turn_s=0.3):
    dt = 0.1
    t = np.round(np.arange(0.0, 2.4 + 1e-9, dt), 10)
    headings = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < 1.0:
            headings[i] = 0.0
        elif ti < 1.0 + turn_s:
            headings[i] = angle * (ti - 1.0) / turn_s
        else:
            headings[i] = angle
    speeds = np.full_like(t, 6.0)
    return _integrate(headings, speeds, dt=dt)


def test_wrap_and_heading_convention():
    assert wrap_deg(190) == pytest.approx(-170)
    assert wrap_deg(-190) == pytest.approx(170)
    assert wrap_deg(0) == pytest.approx(0)
    # 0 degrees is +y, 90 degrees is +x.
    assert heading_deg(0.0, 5.0) == pytest.approx(0.0)
    assert heading_deg(5.0, 0.0) == pytest.approx(90.0)
    ux, uy = unit_from_heading(90.0)
    assert ux == pytest.approx(1.0)
    assert uy == pytest.approx(0.0, abs=1e-9)


def test_sharp_corner_exit_distance_and_invariance():
    t, x, y = _sharp_corner()
    cuts, info = detect_cuts(t, x, y, params=_params())
    assert info.n_duplicate_times == 0
    assert len(cuts) == 1
    cut = cuts[0]
    assert 60 <= cut.abs_angle_deg <= 120
    assert 1.5 <= cut.exit_distance_0p5 <= 3.5
    assert 0.7 <= cut.t0 <= 1.4

    xr, yr = rotate_xy(x + 10.0, y - 4.0, 37.0)
    cuts_r, _ = detect_cuts(t, xr, yr, params=_params())
    assert len(cuts_r) == 1
    assert cuts_r[0].abs_angle_deg == pytest.approx(cut.abs_angle_deg, abs=1.0)
    assert cuts_r[0].exit_distance_0p5 == pytest.approx(cut.exit_distance_0p5, abs=0.05)

    xf, yf = reflect_x(x, y)
    cuts_f, _ = detect_cuts(t, xf, yf, params=_params())
    assert len(cuts_f) == 1
    assert cuts_f[0].abs_angle_deg == pytest.approx(cut.abs_angle_deg, abs=1.0)
    assert cuts_f[0].exit_distance_0p5 == pytest.approx(cut.exit_distance_0p5, abs=0.05)
    assert cuts_f[0].angle_deg == pytest.approx(-cut.angle_deg, abs=1.0)


def test_angle_gate_rejects_shallow_and_turnback():
    t, x, y = _smooth_turn(angle=15.0, turn_s=0.4)
    cuts, _ = detect_cuts(t, x, y, params=_params())
    assert cuts == []
    t, x, y = _smooth_turn(angle=160.0, turn_s=0.4)
    cuts, _ = detect_cuts(t, x, y, params=_params())
    assert cuts == []
    t, x, y = _smooth_turn(angle=70.0, turn_s=0.3)
    cuts, _ = detect_cuts(t, x, y, params=_params())
    assert len(cuts) == 1
    assert 50 <= cuts[0].abs_angle_deg <= 95


def test_exit_window_does_not_cross_a_gap_or_a_short_series():
    t, x, y = _sharp_corner()
    # Truncate before a full half-second of exit exists.
    keep = t <= 1.3
    cuts, _ = detect_cuts(t[keep], x[keep], y[keep], params=_params())
    assert cuts == []
    # A long gap removes the exit samples. Interpolation must not bridge it.
    t2 = t.copy()
    t2[t > 1.05] += 1.0
    cuts, _ = detect_cuts(t2, x, y, params=_params())
    assert cuts == []


def test_low_speed_has_no_heading_cut():
    t = np.round(np.arange(0.0, 2.0, 0.1), 10)
    x = 0.2 * t
    y = 0.1 * t
    cuts, _ = detect_cuts(t, x, y, params=_params())
    assert cuts == []


def test_duplicate_timestamps_are_dropped_and_flagged():
    t, x, y = _sharp_corner()
    t = np.concatenate([t[:5], t[4:5], t[5:]])
    x = np.concatenate([x[:5], x[4:5], x[5:]])
    y = np.concatenate([y[:5], y[4:5], y[5:]])
    cuts, info = detect_cuts(t, x, y, params=_params())
    assert info.n_duplicate_times == 1
    assert len(cuts) == 1


def test_two_attempts_are_not_one_trajectory():
    t, x, y = _sharp_corner()
    cuts_a, _ = detect_cuts(t, x, y, params=_params())
    # A second, separate call cannot see the first attempt's frames.
    t_b = t + 50.0
    cuts_b, _ = detect_cuts(t_b, x + 30.0, y, params=_params())
    assert len(cuts_a) == 1 and len(cuts_b) == 1
    assert cuts_a[0].exit_distance_0p5 == pytest.approx(cuts_b[0].exit_distance_0p5, abs=0.05)


def test_nfl_window_stops_at_the_pass():
    t, x, y = _sharp_corner(t_corner=1.2, t_end=2.6)
    events = np.array([""] * len(t), dtype=object)
    events[t == 0.2] = "ball_snap"
    # Pass is only 0.2s after the corner, so the half-second exit does not fit.
    events[np.isclose(t, 1.4)] = "pass_forward"
    cuts, info = detect_cuts(t, x, y, events=events, params=_params(), require_pass_event=True)
    assert np.isfinite(info.pass_t)
    assert cuts == []
    events[np.isclose(t, 1.4)] = ""
    events[np.isclose(t, 2.2)] = "pass_forward"
    cuts, _ = detect_cuts(t, x, y, events=events, params=_params(), require_pass_event=True)
    assert len(cuts) == 1
    assert cuts[0].t0 + 0.5 <= 2.2 + 1e-6


def test_missing_pass_event_marks_nfl_cut_unavailable():
    t, x, y = _sharp_corner()
    events = np.array([""] * len(t), dtype=object)
    events[2] = "ball_snap"
    cuts, _ = detect_cuts(t, x, y, events=events, params=_params(), require_pass_event=True)
    assert cuts == []


def test_offense_direction_flip_is_a_reflection():
    t, x, y = _sharp_corner()
    xf, yf = to_offense_direction(x, y, "left", field_length=120.0)
    assert xf == pytest.approx(120.0 - x)
    cuts, _ = detect_cuts(t, x, y, params=_params())
    cuts_f, _ = detect_cuts(t, xf, yf, params=_params())
    assert len(cuts_f) == 1
    assert cuts_f[0].exit_distance_0p5 == pytest.approx(cuts[0].exit_distance_0p5, abs=0.05)
