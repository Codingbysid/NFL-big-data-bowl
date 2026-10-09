"""Isolated-turn detection and half-second exit distance.

A cut is a change between a stable incoming heading and a stable outgoing
heading. The reference time t0 is the moment cumulative heading change
reaches half of that net angle. It is not the slowest frame.

exit_distance_0p5 is yards traveled along the outgoing direction in the
next 0.5 seconds. It is not separation from a defender.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from bdb2027.tracking import interp_at, kinematics, unit_from_heading, wrap_deg


@dataclass
class Cut:
    t0: float
    angle_deg: float
    abs_angle_deg: float
    speed_in: float
    speed_out: float
    speed_at_t0: float
    exit_distance_0p5: float
    speed_loss: float
    brake_time_s: float
    t_in0: float
    t_in1: float
    t_out0: float
    t_out1: float
    incoming_heading: float
    outgoing_heading: float
    t0_interpolated: bool
    exit_interpolated: bool
    heading_repair: bool
    segment_id: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrajectoryInfo:
    n_frames: int
    n_duplicate_times: int
    n_jumps: int
    n_heading_repairs: int
    duration_s: float
    snap_t: float
    pass_t: float
    bound_t: float

    def as_dict(self) -> dict:
        return asdict(self)


def _regions(mask: np.ndarray) -> list[tuple[int, int]]:
    if len(mask) == 0:
        return []
    flag = mask.astype(np.int8)
    delta = np.diff(flag, prepend=0, append=0)
    starts = np.flatnonzero(delta == 1)
    ends = np.flatnonzero(delta == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_regions(regions, t, rate, max_gap_s: float) -> list[tuple[int, int]]:
    if not regions:
        return []
    merged = [regions[0]]
    for start, end in regions[1:]:
        prev_s, prev_e = merged[-1]
        gap = t[start] - t[prev_e]
        prev_sign = np.sign(np.nanmean(rate[prev_s : prev_e + 1]))
        this_sign = np.sign(np.nanmean(rate[start : end + 1]))
        if gap <= max_gap_s and prev_sign != 0 and prev_sign == this_sign:
            merged[-1] = (prev_s, end)
        else:
            merged.append((start, end))
    return merged


def _stable_window(
    t, heading, speed, segment, anchor: int, direction: int, *,
    duration: float, search: float, max_range: float, min_speed: float,
):
    """Nearest stable heading window on one side of a turn.

    direction -1 searches backward (incoming). direction +1 searches forward.
    Returns (heading, mean_speed, i0, i1) or None.
    """
    n = len(t)
    if direction < 0:
        k = anchor - 1
        limit = t[anchor] - search
        while k > 0 and t[k] >= limit:
            if segment[k] != segment[anchor]:
                break
            j = int(np.searchsorted(t, t[k] - duration, side="left"))
            if j < k and segment[j] == segment[k] and (t[k] - t[j]) >= duration - 0.021 and (k - j) >= 3:
                sl = slice(j, k + 1)
                hd = heading[sl]
                if np.isfinite(hd).all() and (hd.max() - hd.min()) <= max_range and float(np.mean(speed[sl])) >= min_speed:
                    return float(np.mean(hd)), float(np.mean(speed[sl])), j, k
            k -= 1
        return None
    k = anchor + 1
    limit = t[anchor] + search
    while k < n - 1 and t[k] <= limit:
        if segment[k] != segment[anchor]:
            break
        j = int(np.searchsorted(t, t[k] + duration, side="right")) - 1
        if j > k and segment[j] == segment[k] and (t[j] - t[k]) >= duration - 0.021 and (j - k) >= 3:
            sl = slice(k, j + 1)
            hd = heading[sl]
            if np.isfinite(hd).all() and (hd.max() - hd.min()) <= max_range and float(np.mean(speed[sl])) >= min_speed:
                return float(np.mean(hd)), float(np.mean(speed[sl])), k, j
        k += 1
    return None


def _half_time(t, heading, i_from: int, i_to: int, incoming: float, angle: float):
    """Time when unwrapped heading has moved halfway from incoming to outgoing."""
    half = angle / 2.0
    prev_d = None
    prev_i = None
    for i in range(i_from, i_to + 1):
        if not np.isfinite(heading[i]):
            prev_d = None
            prev_i = None
            continue
        delta = float(heading[i] - incoming)
        if prev_d is not None and (prev_d - half) * (delta - half) <= 0 and abs(delta - prev_d) > 1e-8:
            weight = (half - prev_d) / (delta - prev_d)
            weight = min(1.0, max(0.0, weight))
            t0 = float(t[prev_i] + weight * (t[i] - t[prev_i]))
            return t0, weight > 1e-6
        prev_d = delta
        prev_i = i
    return None


def _event_time(t: np.ndarray, events: np.ndarray | None, names: set[str]) -> float:
    if events is None:
        return float("nan")
    hit = np.array([str(e) in names for e in events])
    if not hit.any():
        return float("nan")
    return float(np.min(t[hit]))


def detect_cuts(
    t,
    x,
    y,
    *,
    events=None,
    params: dict,
    require_pass_event: bool = False,
) -> tuple[list[Cut], TrajectoryInfo]:
    """Detect isolated cuts on one trajectory.

    When ``require_pass_event`` is true (NFL routes), a cut is kept only if
    snap and pass_forward tags exist and the half-second exit window ends
    before the pass and before any recorded catch or contact.
    """
    window = int(params["sg_window"])
    kin = kinematics(
        t, x, y,
        window=window,
        order=int(params.get("sg_order", 2)),
        gap_split_s=float(params["gap_split_s"]),
        speed_heading_min=float(params["speed_heading_min"]),
        max_plausible_speed=float(params["max_plausible_speed"]),
    )
    tt = kin.t
    ev = None if events is None else np.asarray(events)
    # kinematics may have dropped duplicate times; events must be aligned
    # by the caller on the same sorted unique timestamps. detect_cuts handles
    # duplicates itself, so events are sliced the same way.
    t_raw = np.asarray(t, dtype=float)
    order = np.argsort(t_raw, kind="mergesort")
    if ev is not None:
        ev = ev[order]
        _, unique_idx = np.unique(t_raw[order], return_index=True)
        unique_idx = np.sort(unique_idx)
        ev = ev[unique_idx]
        if len(ev) != len(tt):
            raise ValueError("Event array does not match the trajectory after time alignment")

    snap_t = _event_time(tt, ev, {"ball_snap"})
    pass_t = _event_time(tt, ev, {"pass_forward"})
    contact_t = _event_time(
        tt, ev,
        {"first_contact", "pass_arrived", "pass_outcome_caught", "tackle", "pass_outcome_touchdown"},
    )
    bound_candidates = [v for v in (pass_t, contact_t) if np.isfinite(v)]
    bound_t = float(min(bound_candidates)) if bound_candidates else float("nan")

    info = TrajectoryInfo(
        n_frames=int(len(tt)),
        n_duplicate_times=int(kin.n_duplicate_times),
        n_jumps=int(kin.n_jumps),
        n_heading_repairs=int(np.sum(kin.heading_repair)),
        duration_s=float(tt[-1] - tt[0]) if len(tt) else 0.0,
        snap_t=snap_t,
        pass_t=pass_t,
        bound_t=bound_t,
    )
    if len(tt) < 8:
        return [], info

    cuts_p = params["cuts"]
    rate = kin.heading_rate
    in_turn = np.isfinite(rate) & (np.abs(rate) >= float(cuts_p["turn_rate_min_deg_s"])) & (kin.speed >= 1.0)
    regions = _merge_regions(_regions(in_turn), tt, rate, float(cuts_p["merge_gap_s"]))
    found: list[Cut] = []
    gap_limit = float(params["gap_split_s"])

    for start, end in regions:
        incoming = _stable_window(
            tt, kin.heading_unwrapped, kin.speed, kin.segment_id, start, -1,
            duration=float(cuts_p["stable_window_s"]),
            search=float(cuts_p["stable_search_s"]),
            max_range=float(cuts_p["stable_heading_range_deg"]),
            min_speed=float(cuts_p["entry_speed_min"]),
        )
        outgoing = _stable_window(
            tt, kin.heading_unwrapped, kin.speed, kin.segment_id, end, +1,
            duration=float(cuts_p["stable_window_s"]),
            search=float(cuts_p["stable_search_s"]),
            max_range=float(cuts_p["stable_heading_range_deg"]),
            min_speed=float(cuts_p["outgoing_speed_min"]),
        )
        if incoming is None or outgoing is None:
            continue
        in_h, in_spd, in0, in1 = incoming
        out_h, out_spd, out0, out1 = outgoing
        if kin.segment_id[in0] != kin.segment_id[out1]:
            continue
        angle = float(wrap_deg(out_h - in_h))
        abs_angle = abs(angle)
        if not (float(cuts_p["angle_min_deg"]) <= abs_angle <= float(cuts_p["angle_max_deg"])):
            continue
        # Reject loops and double-moves whose total heading travel is much
        # larger than the net turn between the stable windows.
        span = kin.heading_unwrapped[in1 : out0 + 1]
        if np.isfinite(span).sum() >= 3:
            travel = float(np.nansum(np.abs(np.diff(span))))
            if travel > abs_angle + float(cuts_p["max_extra_heading_deg"]):
                continue
        timed = _half_time(tt, kin.heading_unwrapped, in1, out0, in_h, angle)
        if timed is None:
            continue
        t0, t0_interp = timed
        horizon = float(cuts_p["exit_horizon_s"])
        side = float(cuts_p["min_side_s"])
        seg = int(kin.segment_id[in0])
        seg_mask = kin.segment_id == seg
        seg_t0 = float(tt[seg_mask][0])
        seg_t1 = float(tt[seg_mask][-1])
        if t0 - seg_t0 < side or seg_t1 - t0 < side:
            continue
        if t0 - side < seg_t0 or t0 + horizon > seg_t1 + 1e-9:
            continue
        if require_pass_event:
            if not np.isfinite(snap_t) or not np.isfinite(pass_t):
                continue
            if t0 <= snap_t:
                continue
            if t0 + horizon > bound_t + 1e-9:
                continue
        p0x, p0_interp = interp_at(tt, kin.x_smooth, t0, gap_limit)
        p0y, _ = interp_at(tt, kin.y_smooth, t0, gap_limit)
        p1x, p1_interp = interp_at(tt, kin.x_smooth, t0 + horizon, gap_limit)
        p1y, _ = interp_at(tt, kin.y_smooth, t0 + horizon, gap_limit)
        if not all(np.isfinite(v) for v in (p0x, p0y, p1x, p1y)):
            continue
        ux, uy = unit_from_heading(out_h)
        exit_d = (p1x - p0x) * ux + (p1y - p0y) * uy
        speed_at, _ = interp_at(tt, kin.speed, t0, gap_limit)
        look = float(cuts_p["brake_lookback_s"])
        brake_mask = seg_mask & (tt >= t0 - look) & (tt <= t0) & np.isfinite(kin.speed)
        if brake_mask.any():
            b_idx = np.flatnonzero(brake_mask)[int(np.nanargmin(kin.speed[brake_mask]))]
            brake_time = float(tt[b_idx] - t0)
            min_speed = float(kin.speed[b_idx])
        else:
            brake_time = float("nan")
            min_speed = float("nan")
        turn_mask = (kin.segment_id == seg) & (tt >= tt[in1]) & (tt <= tt[out0]) & np.isfinite(kin.speed)
        min_turn = float(np.nanmin(kin.speed[turn_mask])) if turn_mask.any() else float("nan")
        speed_loss = float(in_spd - min_turn) if np.isfinite(min_turn) else float("nan")
        repair_mask = (tt >= tt[in0]) & (tt <= tt[out1]) & kin.heading_repair
        found.append(
            Cut(
                t0=float(t0),
                angle_deg=angle,
                abs_angle_deg=float(abs_angle),
                speed_in=float(in_spd),
                speed_out=float(out_spd),
                speed_at_t0=float(speed_at) if np.isfinite(speed_at) else float("nan"),
                exit_distance_0p5=float(exit_d),
                speed_loss=speed_loss,
                brake_time_s=brake_time,
                t_in0=float(tt[in0]),
                t_in1=float(tt[in1]),
                t_out0=float(tt[out0]),
                t_out1=float(tt[out1]),
                incoming_heading=float(in_h),
                outgoing_heading=float(out_h),
                t0_interpolated=bool(t0_interp or p0_interp),
                exit_interpolated=bool(p1_interp),
                heading_repair=bool(np.any(repair_mask)),
                segment_id=seg,
            )
        )

    found.sort(key=lambda c: c.t0)
    separated: list[Cut] = []
    min_sep = float(cuts_p["min_turn_separation_s"])
    for cut in found:
        if separated and cut.t0 - separated[-1].t0 < min_sep:
            if cut.abs_angle_deg > separated[-1].abs_angle_deg:
                separated[-1] = cut
            continue
        separated.append(cut)
    return separated, info


def detection_params(cfg: dict, sg_window: int | None = None) -> dict:
    """Flatten the yaml sections the detector reads."""
    return {
        "sg_window": int(sg_window if sg_window is not None else cfg["tracking"]["sg_window"]),
        "sg_order": int(cfg["tracking"]["sg_order"]),
        "gap_split_s": float(cfg["tracking"]["gap_split_s"]),
        "speed_heading_min": float(cfg["tracking"]["speed_heading_min"]),
        "max_plausible_speed": float(cfg["tracking"]["max_plausible_speed"]),
        "cuts": cfg["cuts"],
    }
