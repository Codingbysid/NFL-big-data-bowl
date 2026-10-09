"""Leakage, weights, eligibility, and cache checks that do not need the full tracking files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bdb2027.cohort import apply_funnel, assert_no_cohort_overlap
from bdb2027.config import load_config
from bdb2027.evaluation import equal_player_weights, summarize_frames
from bdb2027.io import HoldoutGuard, assert_outcomes_allowed, stage_cache_valid, write_stage_meta
from bdb2027.models import FORBIDDEN_FEATURES, DesignBuilder, fit_level, predict_level


def _plays():
    rows = [
        dict(nfl_position="WR", season_type="REG", season=2023, draft_year=2023, play_nullified_by_penalty="N",
             route_ran="OUT", pass_result="C", separation_at_pass_forward=3.0, nfl_id=1, game_id=1, play_id=1),
        dict(nfl_position="WR", season_type="PRE", season=2023, draft_year=2023, play_nullified_by_penalty="N",
             route_ran="OUT", pass_result="C", separation_at_pass_forward=3.0, nfl_id=1, game_id=2, play_id=1),
        dict(nfl_position="WR", season_type="POST", season=2023, draft_year=2023, play_nullified_by_penalty="N",
             route_ran="OUT", pass_result="C", separation_at_pass_forward=3.0, nfl_id=1, game_id=3, play_id=1),
        dict(nfl_position="WR", season_type="REG", season=2024, draft_year=2023, play_nullified_by_penalty="N",
             route_ran="OUT", pass_result="C", separation_at_pass_forward=4.0, nfl_id=1, game_id=4, play_id=1),
        dict(nfl_position="WR", season_type="REG", season=2023, draft_year=2023, play_nullified_by_penalty="Y",
             route_ran="OUT", pass_result="C", separation_at_pass_forward=3.0, nfl_id=1, game_id=5, play_id=1),
        dict(nfl_position="WR", season_type="REG", season=2023, draft_year=2023, play_nullified_by_penalty="N",
             route_ran="SCREEN", pass_result="C", separation_at_pass_forward=8.0, nfl_id=1, game_id=6, play_id=1),
        dict(nfl_position="WR", season_type="REG", season=2023, draft_year=2023, play_nullified_by_penalty="N",
             route_ran="OUT", pass_result="S", separation_at_pass_forward=np.nan, nfl_id=1, game_id=7, play_id=1),
        dict(nfl_position="WR", season_type="REG", season=2023, draft_year=2023, play_nullified_by_penalty="N",
             route_ran="OUT", pass_result="C", separation_at_pass_forward=np.nan, nfl_id=1, game_id=8, play_id=1),
        dict(nfl_position="TE", season_type="REG", season=2023, draft_year=2023, play_nullified_by_penalty="N",
             route_ran="OUT", pass_result="C", separation_at_pass_forward=3.0, nfl_id=2, game_id=9, play_id=1),
    ]
    frame = pd.DataFrame(rows)
    for col in ("down", "yards_to_go", "yardline_number", "cushion", "yardline_side", "possession_team",
                "offense_formation", "receiver_alignment", "team_coverage_man_zone", "in_motion_at_ball_snap"):
        frame[col] = np.nan
    frame["in_motion_at_ball_snap"] = False
    return frame


def test_eligibility_filters():
    cfg = load_config()
    out = apply_funnel(_plays(), cfg)
    kept = out.loc[out["eligible"]]
    assert list(kept["game_id"]) == [1]
    assert kept["separation_at_pass_forward"].isna().sum() == 0
    reasons = set(out.loc[~out["eligible"], "exclusion_reason"])
    assert {"regular_season", "rookie_season", "not_penalty", "not_excluded_family", "forward_pass", "has_separation", "wr_roster"} <= reasons


def test_no_cohort_overlap():
    assert_no_cohort_overlap([1, 2], [3])
    with pytest.raises(ValueError):
        assert_no_cohort_overlap([1, 2], [2, 3])


def test_holdout_guard():
    with pytest.raises(HoldoutGuard):
        assert_outcomes_allowed([2025], allow_outcomes=False)
    assert_outcomes_allowed([2023, 2024], allow_outcomes=False)
    assert_outcomes_allowed([2025], allow_outcomes=True)


def test_player_weights_sum_to_one():
    weights = equal_player_weights([1, 1, 2, 2, 2, 2])
    totals = pd.DataFrame({"id": [1, 1, 2, 2, 2, 2], "w": weights}).groupby("id")["w"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)


def test_missing_outcomes_are_not_zero_and_test_labels_do_not_change_predictions():
    cfg = load_config()
    rng = np.random.default_rng(0)
    rows = []
    for player in range(12):
        for route_i in range(8):
            rows.append(
                {
                    "nfl_id": player,
                    "route_ran": "OUT" if route_i % 2 == 0 else "IN",
                    "coverage_man_zone": "MAN_COVERAGE" if player % 2 == 0 else "ZONE_COVERAGE",
                    "down": "1",
                    "offense_formation": "SHOTGUN",
                    "receiver_alignment": "2x2",
                    "motion": "no",
                    "cushion": 5 + rng.normal(),
                    "yards_to_go": 10,
                    "yardline_number": 25,
                    "own_side": 1.0,
                    "forty": 4.4 + player * 0.01,
                    "ten_yd_split": 1.55,
                    "height": 72,
                    "weight": 190,
                    "age": 22,
                    "vertical": 35,
                    "broad_jump": 120,
                    "three_cone": np.nan if player < 3 else 6.9,
                    "short_shuttle": 4.2,
                    "trait": (player - 6) * 0.1,
                    "separation_at_pass_forward": 3 + 0.05 * player + rng.normal(0, 0.4),
                }
            )
    frame = pd.DataFrame(rows)
    # A missing label must be dropped, not replaced with zero.
    frame.loc[frame.index[:2], "separation_at_pass_forward"] = np.nan
    train = frame.iloc[:80].copy()
    test = frame.iloc[80:].copy()
    builder, model = fit_level(train, "M2", alpha=3.0, cfg=cfg)
    assert set(builder.columns_).isdisjoint(FORBIDDEN_FEATURES)
    assert not any(name.startswith("nfl_") for name in builder.columns_)
    assert "trait" in builder.columns_
    pred = predict_level(builder, model, test)
    test2 = test.copy()
    test2["separation_at_pass_forward"] = 0.0
    pred2 = predict_level(builder, model, test2)
    assert np.allclose(pred, pred2, equal_nan=True)
    y = test["separation_at_pass_forward"].to_numpy()
    assert np.isnan(y).sum() == 0 or summarize_frames(test["nfl_id"], np.nan_to_num(y, nan=0.0), np.nan_to_num(pred, nan=0.0))
    # The fitted target never contains a filled zero for the two blank training rows.
    assert train["separation_at_pass_forward"].isna().sum() == 2


def test_cache_invalidates_when_config_or_code_changes(tmp_path):
    write_stage_meta(tmp_path, sources={"players.csv": "abc"}, config_hash="cfg", code_hash="code")
    (tmp_path / "plays.parquet").write_text("x")
    assert stage_cache_valid(tmp_path, sources={"players.csv": "abc"}, config_hash="cfg", code_hash="code", require_files=["plays.parquet"])
    assert not stage_cache_valid(tmp_path, sources={"players.csv": "abc"}, config_hash="other", code_hash="code", require_files=["plays.parquet"])
    assert not stage_cache_valid(tmp_path, sources={"players.csv": "zzz"}, config_hash="cfg", code_hash="code", require_files=["plays.parquet"])
    assert not stage_cache_valid(tmp_path, sources={"players.csv": "abc"}, config_hash="cfg", code_hash="code", require_files=["missing.parquet"])
