"""Failure modes for the bounded detector, support, and demand penalty."""

import numpy as np
import pandas as pd

from bdb2027.config import load_config
from bdb2027.cuts import detect_cuts, detection_params
from bdb2027.cuts_v2 import detect_cuts_bounded
from bdb2027.demand_v2 import WeightedDemandModel, sklearn_alpha
from bdb2027.features import player_attempt_cut_weights
from bdb2027.support_v2 import apply_support, fit_support, supported_attempt_means


def _params():
    return detection_params(load_config())


def _path(future_heading, pass_time=1.6, contact_time=None):
    t = np.round(np.arange(0.0, 2.61, 0.1), 10)
    events = np.full(len(t), "", dtype=object)
    events[np.isclose(t, 0.2)] = "ball_snap"
    events[np.isclose(t, pass_time)] = "pass_forward"
    if contact_time is not None:
        events[np.isclose(t, contact_time)] = "first_contact"
    speed = np.where(t <= 1.0, 6.0, 5.0)
    heading = np.where(t <= 1.0, 0.0, 90.0)
    heading[t > pass_time] = future_heading
    x, y = np.zeros(len(t)), np.zeros(len(t))
    for i in range(1, len(t)):
        theta = np.radians(heading[i])
        x[i] = x[i - 1] + 0.1 * speed[i] * np.sin(theta)
        y[i] = y[i - 1] + 0.1 * speed[i] * np.cos(theta)
    return t, x, y, events


def test_historical_detector_changes_when_only_post_pass_heading_changes():
    params = _params()
    counts = []
    for heading in (90.0, 30.0):
        t, x, y, events = _path(heading)
        cuts, _ = detect_cuts(t, x, y, events=events, params=params, require_pass_event=True)
        counts.append(len(cuts))
    assert counts == [1, 0]


def test_bounded_detector_ignores_post_boundary_heading():
    params = _params()
    results = []
    for heading in (90.0, 30.0):
        t, x, y, events = _path(heading)
        cuts, _info, _reasons = detect_cuts_bounded(
            t, x, y, events=events, params=params, require_pass_event=True
        )
        results.append(cuts)
    assert len(results[0]) == len(results[1])
    if results[0]:
        assert results[0][0]["exit_distance_0p5"] == results[1][0]["exit_distance_0p5"]
        assert results[0][0]["latest_raw_timestamp"] <= 1.6 + 1e-9
        assert results[0][0]["t_out1"] <= 1.6 + 1e-9


def test_earlier_contact_is_the_boundary():
    params = _params()
    left_t, left_x, left_y, left_e = _path(90.0, pass_time=1.6, contact_time=1.2)
    right_t, right_x, right_y, right_e = _path(30.0, pass_time=1.6, contact_time=1.2)
    right_x = right_x.copy()
    right_y = right_y.copy()
    right_x[right_t > 1.2] += 5.0
    a, _, reasons_a = detect_cuts_bounded(left_t, left_x, left_y, events=left_e, params=params, require_pass_event=True)
    b, _, reasons_b = detect_cuts_bounded(right_t, right_x, right_y, events=right_e, params=params, require_pass_event=True)
    assert [row["t0"] for row in a] == [row["t0"] for row in b]
    assert reasons_a[0]["bound_event"] == "contact"
    assert reasons_a[0]["latest_raw_timestamp"] <= 1.2 + 1e-9
    assert reasons_b[0]["latest_raw_timestamp"] == reasons_a[0]["latest_raw_timestamp"]


def test_missing_pass_rejects_without_a_zero_measurement():
    params = _params()
    t, x, y, events = _path(90.0)
    events = events.copy()
    events[np.isclose(t, 1.6)] = ""
    cuts, _info, reasons = detect_cuts_bounded(t, x, y, events=events, params=params, require_pass_event=True)
    assert cuts == []
    assert reasons[0]["reason"] == "missing_pass_forward"


def test_outgoing_window_cannot_cross_the_boundary():
    params = _params()
    t = np.round(np.arange(0.0, 2.81, 0.1), 10)
    events = np.full(len(t), "", dtype=object)
    events[np.isclose(t, 0.1)] = "ball_snap"
    events[np.isclose(t, 1.6)] = "pass_forward"
    heading = np.zeros(len(t))
    for i, ti in enumerate(t):
        if ti < 0.8:
            heading[i] = 0.0
        elif ti <= 1.3:
            heading[i] = (ti - 0.8) / 0.5 * 70.0
        else:
            heading[i] = 70.0
    speed = np.full(len(t), 6.0)
    x, y = np.zeros(len(t)), np.zeros(len(t))
    for i in range(1, len(t)):
        theta = np.radians(heading[i])
        x[i] = x[i - 1] + 0.1 * speed[i] * np.sin(theta)
        y[i] = y[i - 1] + 0.1 * speed[i] * np.cos(theta)
    historical, _ = detect_cuts(t, x, y, events=events, params=params, require_pass_event=True)
    bounded, _, reasons = detect_cuts_bounded(t, x, y, events=events, params=params, require_pass_event=True)
    assert any(cut.t_out1 > 1.6 + 1e-9 for cut in historical)
    assert bounded == []
    assert reasons[0]["reason"] == "no_cut_inside_boundary"
    assert reasons[0]["latest_raw_timestamp"] <= 1.6 + 1e-9


def test_unsupported_cuts_are_omitted_not_zeroed():
    cfg = load_config()
    combine = pd.DataFrame(
        {
            "nfl_id": [1, 1, 2],
            "speed_in": [7.0, 7.0, 3.0],
            "abs_angle_deg": [60.0, 60.0, 40.0],
        }
    )
    nfl = pd.DataFrame({"nfl_id": [1] * 15, "speed_in": [7.0] * 15, "abs_angle_deg": [60.0] * 15})
    cells = fit_support(pd.concat([combine.iloc[:1]] * 15, ignore_index=True), nfl, cfg)
    scored = apply_support(
        pd.DataFrame(
            {
                "nfl_id": [1, 1, 2],
                "attempt_uid": ["a", "a", "b"],
                "speed_in": [7.0, 3.0, 3.0],
                "abs_angle_deg": [60.0, 40.0, 40.0],
                "residual": [1.0, 99.0, 5.0],
                "drill_family": ["SPEED_OUT", "SPEED_OUT", "SPEED_OUT"],
            }
        ),
        cells,
        cfg,
    )
    means = supported_attempt_means(scored)
    assert set(means["nfl_id"]) == {1}
    assert means["residual"].iloc[0] == 1.0
    assert 2 not in set(means["nfl_id"])


def test_evaluation_rows_do_not_change_frozen_support():
    cfg = load_config()
    train_c = pd.DataFrame({"nfl_id": [1] * 15, "speed_in": [7.0] * 15, "abs_angle_deg": [60.0] * 15})
    train_n = pd.DataFrame({"nfl_id": [2] * 15, "speed_in": [7.0] * 15, "abs_angle_deg": [60.0] * 15})
    frozen = fit_support(train_c, train_n, cfg)
    extra = pd.DataFrame({"nfl_id": [9] * 30, "speed_in": [3.0] * 30, "abs_angle_deg": [40.0] * 30})
    again = fit_support(train_c, train_n, cfg)
    contaminated = fit_support(pd.concat([train_c, extra], ignore_index=True), train_n, cfg)
    assert frozen == again
    assert any(row["supported"] for row in frozen)
    assert contaminated != frozen


def test_weight_and_penalty_rescale_together():
    frame = pd.DataFrame(
        [
            dict(nfl_id=p, attempt_uid=f"{p}-{j}", speed_in=s, abs_angle_deg=60.0, drill_family="SPEED_OUT", exit_distance_0p5=1.0 + 0.4 * s)
            for p in range(12)
            for j, s in enumerate([5.0, 6.0, 7.0, 8.0])
        ]
    )
    weights = player_attempt_cut_weights(frame)
    lam = 0.1
    first = WeightedDemandModel(alpha=lam).fit(frame, weights)
    second = WeightedDemandModel(alpha=lam).fit(frame, weights * 4.0)
    assert np.isclose(sklearn_alpha(lam, weights.sum()) * 4.0, sklearn_alpha(lam, (weights * 4).sum()))
    assert np.allclose(first.coef_, second.coef_, atol=1e-8)
    assert np.isclose(first.intercept_, second.intercept_, atol=1e-8)
