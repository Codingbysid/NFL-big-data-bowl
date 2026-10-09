"""Tracking frames must not replicate a play outcome."""

import pandas as pd
import pytest

from bdb2027.io import assert_unique_keys, merge_frames_to_plays


def test_merge_keeps_one_row_per_frame_and_does_not_repeat_the_outcome_total():
    plays = pd.DataFrame(
        {
            "game_id": [1, 1],
            "play_id": [10, 11],
            "nfl_id": [5, 5],
            "separation_at_pass_forward": [3.0, 4.0],
        }
    )
    frames = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "play_id": [10, 10, 10, 11],
            "nfl_id": [5, 5, 5, 5],
            "x": [1, 2, 3, 4],
        }
    )
    merged = merge_frames_to_plays(frames, plays, ["game_id", "play_id", "nfl_id"])
    assert len(merged) == len(frames)
    collapsed = merged.groupby(["game_id", "play_id", "nfl_id"], as_index=False)["separation_at_pass_forward"].first()
    assert collapsed["separation_at_pass_forward"].sum() == pytest.approx(7.0)
    assert merged["separation_at_pass_forward"].sum() != pytest.approx(7.0)


def test_duplicate_play_keys_are_rejected():
    plays = pd.DataFrame({"game_id": [1, 1], "play_id": [10, 10], "nfl_id": [5, 5]})
    with pytest.raises(ValueError):
        assert_unique_keys(plays, ["game_id", "play_id", "nfl_id"], "plays")
