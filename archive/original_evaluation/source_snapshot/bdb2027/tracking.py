"""Trajectory smoothing, headings, and movement-frame geometry.

Positions are yards. Recorded ``dir`` is degrees, 0 along +y and increasing
clockwise, which matches ``atan2(vx, vy)``. There is no body-orientation
field in Combine tracking. ``dir`` is movement direction, not facing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter


def wrap_deg(angle) -> np.ndarray:
    """Wrap degrees to (-180, 180]."""
    out = (np.asarray(angle, dtype=float) + 180.0) % 360.0 - 180.0
    return out


def heading_deg(vx, vy) -> np.ndarray:
    """Movement heading in the NFL direction convention (0 = +y, clockwise)."""
    return np.degrees(np.arctan2(np.asarray(vx, dtype=float), np.asarray(vy, dtype=float)))


def unit_from_heading(heading: float) -> tuple[float, float]:
    """Unit vector (x, y) for a heading in degrees."""
    rad = np.radians(heading)
    return float(np.sin(rad)), float(np.cos(rad))


def rotate_xy(x, y, degrees: float):
    """Rotate coordinates counterclockwise about the origin."""
    rad = np.radians(degrees)
    c, s = np.cos(rad), np.sin(rad)
    xr = c * np.asarray(x, dtype=float) - s * np.asarray(y, dtype=float)
    yr = s * np.asarray(x, dtype=float) + c * np.asarray(y, dtype=float)
    return xr, yr


def reflect_x(x, y):
    """Reflect across the y axis. Headings flip sign; path lengths do not."""
    return -np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def to_offense_direction(x, y, play_direction: str, field_length: float = 120.0):
    """Flip a play so the offense moves toward increasing x.

    Local exit distance does not depend on this frame. It is used for
    plotting and for tests of reflection behavior.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if str(play_direction).lower() == "left":
        return field_length - x, y
    return x.copy(), y.copy()


def interp_at(t: np.ndarray, values: np.ndarray, t_query: float, max_gap_s: float) -> tuple[float, bool]:
    """Linear interpolation. Returns (value, interpolated).

    Interpolation across a gap larger than ``max_gap_s`` returns NaN.
    An exact sample is not flagged as interpolated.
    """
    t = np.asarray(t, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(t) == 0 or t_query < t[0] - 1e-9 or t_query > t[-1] + 1e-9:
        return float("nan"), False
    j = int(np.searchsorted(t, t_query))
    if j < len(t) and abs(t[j] - t_query) <= 1e-8:
        return float(values[j]), False
    if j == 0 or j >= len(t):
        return float("nan"), False
    if t[j] - t[j - 1] > max_gap_s:
        return float("nan"), False
    weight = (t_query - t[j - 1]) / (t[j] - t[j - 1])
    return float((1.0 - weight) * values[j - 1] + weight * values[j]), True


def drop_duplicate_times(t: np.ndarray, columns: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray], int]:
    """Keep the first row at each timestamp. Returns the number dropped."""
    t = np.asarray(t, dtype=float)
    if len(t) == 0:
        return t, columns, 0
    _, index = np.unique(t, return_index=True)
    index = np.sort(index)
    dropped = int(len(t) - len(index))
    return t[index], [np.asarray(col)[index] for col in columns], dropped


def _segment_bounds(t: np.ndarray, x: np.ndarray, y: np.ndarray, gap_split_s: float, max_speed: float) -> list[tuple[int, int]]:
    if len(t) == 0:
        return []
    break_list: list[int] = []
    if len(t) > 1:
        dt = np.diff(t)
        step = np.hypot(np.diff(x), np.diff(y))
        with np.errstate(divide="ignore", invalid="ignore"):
            too_fast = (dt > 0) & (step / np.maximum(dt, 1e-9) > max_speed)
        gap = dt > gap_split_s
        break_list = (np.where(gap | too_fast)[0] + 1).tolist()
    bounds = [0, *break_list, len(t)]
    return [(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:]) if hi > lo]


def unwrap_heading(heading_deg_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unwrap headings in degrees. Single-frame holes are bridged and flagged.

    Longer holes stay missing. No heading is invented across a long gap.
    """
    hd = np.asarray(heading_deg_arr, dtype=float)
    n = len(hd)
    out = np.full(n, np.nan)
    repair = np.zeros(n, dtype=bool)
    valid = np.isfinite(hd)
    bridged = valid.copy()
    for i in range(1, n - 1):
        if (not valid[i]) and valid[i - 1] and valid[i + 1]:
            bridged[i] = True
            repair[i] = True
    idx = np.flatnonzero(bridged)
    if len(idx) == 0:
        return out, repair
    splits = np.where(np.diff(idx) > 1)[0] + 1
    for run in np.split(idx, splits):
        vals = hd[run].astype(float).copy()
        for local, i in enumerate(run):
            if repair[i]:
                prev = hd[i - 1]
                nxt = hd[i + 1]
                vals[local] = prev + 0.5 * float(wrap_deg(nxt - prev))
        out[run] = np.rad2deg(np.unwrap(np.deg2rad(vals)))
    return out, repair


@dataclass
class Kinematics:
    t: np.ndarray
    x_smooth: np.ndarray
    y_smooth: np.ndarray
    speed: np.ndarray
    heading: np.ndarray
    heading_unwrapped: np.ndarray
    heading_rate: np.ndarray
    segment_id: np.ndarray
    heading_repair: np.ndarray
    n_duplicate_times: int
    n_jumps: int


def kinematics(
    t,
    x,
    y,
    *,
    window: int = 5,
    order: int = 2,
    gap_split_s: float = 0.15,
    speed_heading_min: float = 1.5,
    max_plausible_speed: float = 13.0,
) -> Kinematics:
    """Smooth one trajectory and estimate speed and heading from position.

    Savitzky-Golay is applied only inside contiguous, regularly sampled
    segments. Segments shorter than the window keep the raw coordinates.
    Derivatives use the actual timestamps and never cross a segment break.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order_idx = np.argsort(t, kind="mergesort")
    t, x, y = t[order_idx], x[order_idx], y[order_idx]
    t, (x, y), n_dup = drop_duplicate_times(t, [x, y])
    n = len(t)
    x_s = np.full(n, np.nan)
    y_s = np.full(n, np.nan)
    bounds = _segment_bounds(t, x, y, gap_split_s, max_plausible_speed)
    n_jumps = max(0, len(bounds) - 1)
    # Jumps were one reason for extra segments. Gaps are too. Count speed jumps
    # separately for the audit by re-checking raw steps.
    if n > 1:
        dt = np.diff(t)
        step = np.hypot(np.diff(x), np.diff(y))
        with np.errstate(divide="ignore", invalid="ignore"):
            n_jumps = int(np.sum((dt > 0) & (step / np.maximum(dt, 1e-9) > max_plausible_speed)))
    seg_id = np.full(n, -1, dtype=int)
    window = int(window)
    if window % 2 == 0:
        raise ValueError("Savitzky-Golay window must be odd")
    for s, (lo, hi) in enumerate(bounds):
        seg_id[lo:hi] = s
        length = hi - lo
        if length >= window and window >= order + 2:
            x_s[lo:hi] = savgol_filter(x[lo:hi], window, order, mode="interp")
            y_s[lo:hi] = savgol_filter(y[lo:hi], window, order, mode="interp")
        else:
            x_s[lo:hi] = x[lo:hi]
            y_s[lo:hi] = y[lo:hi]

    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    for lo, hi in bounds:
        if hi - lo < 3:
            continue
        vx[lo:hi] = np.gradient(x_s[lo:hi], t[lo:hi])
        vy[lo:hi] = np.gradient(y_s[lo:hi], t[lo:hi])
    speed = np.hypot(vx, vy)
    heading = np.full(n, np.nan)
    ok = np.isfinite(speed) & (speed >= speed_heading_min)
    heading[ok] = heading_deg(vx[ok], vy[ok])

    unwrapped = np.full(n, np.nan)
    repair = np.zeros(n, dtype=bool)
    rate = np.full(n, np.nan)
    for lo, hi in bounds:
        unwrapped[lo:hi], repair[lo:hi] = unwrap_heading(heading[lo:hi])
        uu = unwrapped[lo:hi]
        finite = np.isfinite(uu)
        idx = np.flatnonzero(finite)
        if len(idx) < 3:
            continue
        splits = np.where(np.diff(idx) > 1)[0] + 1
        for run in np.split(idx, splits):
            if len(run) < 3:
                continue
            rate[lo + run] = np.gradient(uu[run], t[lo + run])
    return Kinematics(
        t=t,
        x_smooth=x_s,
        y_smooth=y_s,
        speed=speed,
        heading=heading,
        heading_unwrapped=unwrapped,
        heading_rate=rate,
        segment_id=seg_id,
        heading_repair=repair,
        n_duplicate_times=n_dup,
        n_jumps=n_jumps,
    )
