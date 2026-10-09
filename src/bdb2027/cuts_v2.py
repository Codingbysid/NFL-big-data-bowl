"""Information-bounded cut detection.

The historical detector in ``cuts.detect_cuts`` smooths the full trajectory
and only checks that the exit horizon ends before the pass. This version
removes every coordinate after the declared boundary before smoothing,
derivatives, window search, or exit projection. It does not replace the
archived detector.
"""

from __future__ import annotations

import numpy as np

from bdb2027.cuts import Cut, TrajectoryInfo, _event_time, detect_cuts

CONTACT_EVENTS = {
    "first_contact",
    "pass_arrived",
    "pass_outcome_caught",
    "tackle",
    "pass_outcome_touchdown",
}


def boundary_times(t: np.ndarray, events: np.ndarray | None) -> dict:
    snap_t = _event_time(t, events, {"ball_snap"})
    pass_t = _event_time(t, events, {"pass_forward"})
    contact_t = _event_time(t, events, CONTACT_EVENTS)
    candidates = [v for v in (pass_t, contact_t) if np.isfinite(v)]
    bound_t = float(min(candidates)) if candidates else float("nan")
    which = "none"
    if np.isfinite(bound_t):
        which = "contact" if np.isfinite(contact_t) and contact_t <= pass_t + 1e-9 else "pass_forward"
        if np.isfinite(contact_t) and (not np.isfinite(pass_t) or contact_t <= pass_t + 1e-9):
            which = "contact"
        elif np.isfinite(pass_t):
            which = "pass_forward"
    return {"snap_t": snap_t, "pass_t": pass_t, "contact_t": contact_t, "bound_t": bound_t, "bound_event": which}


def truncate_at_boundary(t, x, y, events, bound_t: float):
    """Drop frames strictly after the boundary.

    The last kept timestamp is the latest raw frame allowed into smoothing.
    A boundary that falls between samples does not keep the later sample.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(bound_t):
        kept = np.ones(len(t), dtype=bool)
    else:
        kept = t <= float(bound_t) + 1e-9
    ev = None if events is None else np.asarray(events)[kept]
    latest = float(t[kept][-1]) if kept.any() else float("nan")
    return t[kept], x[kept], y[kept], ev, latest


def detect_cuts_bounded(
    t,
    x,
    y,
    *,
    events=None,
    params: dict,
    require_pass_event: bool = False,
) -> tuple[list[dict], TrajectoryInfo | None, list[dict]]:
    """Detect cuts using only pre-boundary coordinates.

    Returns accepted measurements, trajectory info, and rejection records.
    An empty acceptance list is abstention, not a zero-ability measurement.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ev = None if events is None else np.asarray(events)
    times = boundary_times(t, ev)
    reasons: list[dict] = []
    if require_pass_event and not np.isfinite(times["pass_t"]):
        reasons.append({"reason": "missing_pass_forward", "latest_raw_timestamp": None, **times})
        return [], None, reasons
    if require_pass_event and not np.isfinite(times["snap_t"]):
        reasons.append({"reason": "missing_snap", "latest_raw_timestamp": None, **times})
        return [], None, reasons
    t_use, x_use, y_use, ev_use, latest = truncate_at_boundary(t, x, y, ev, times["bound_t"])
    if len(t_use) < 8:
        reasons.append({"reason": "insufficient_pre_boundary_frames", "latest_raw_timestamp": latest, **times})
        return [], None, reasons
    cuts, info = detect_cuts(t_use, x_use, y_use, events=ev_use, params=params, require_pass_event=require_pass_event)
    horizon = float(params["cuts"]["exit_horizon_s"])
    accepted: list[dict] = []
    for cut in cuts:
        record = _measurement(cut, latest, times)
        if np.isfinite(times["bound_t"]) and (
            cut.t_out1 > times["bound_t"] + 1e-6 or cut.t0 + horizon > times["bound_t"] + 1e-6
        ):
            reasons.append({**record, "reason": "window_crosses_boundary"})
            continue
        if record["latest_raw_timestamp"] > times["bound_t"] + 1e-6 if np.isfinite(times["bound_t"]) else False:
            reasons.append({**record, "reason": "smoothing_used_post_boundary_frame"})
            continue
        accepted.append(record)
    if not accepted and not reasons:
        reasons.append({"reason": "no_cut_inside_boundary", "latest_raw_timestamp": latest, **times})
    return accepted, info, reasons


def _measurement(cut: Cut, latest: float, times: dict) -> dict:
    row = cut.as_dict()
    row["latest_raw_timestamp"] = float(latest) if np.isfinite(latest) else None
    row.update(times)
    return row
