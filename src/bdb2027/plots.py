"""Figures. Colors are distinguishable without relying on red-green contrast."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from bdb2027.config import load_config
from bdb2027.tracking import interp_at, kinematics, unit_from_heading

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
BLACK = "#000000"
GRAY = "#666666"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
    }
)


def save_figure(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _kinematics(t, x, y, window: int):
    cfg = load_config()["tracking"]
    return kinematics(
        t,
        x,
        y,
        window=window,
        order=int(cfg["sg_order"]),
        gap_split_s=float(cfg["gap_split_s"]),
        speed_heading_min=float(cfg["speed_heading_min"]),
        max_plausible_speed=float(cfg["max_plausible_speed"]),
    )


def plot_trajectory(t, x, y, path: Path, title: str, cut: dict | None = None, window: int = 5, bound_t: float | None = None) -> None:
    """Trajectory and speed trace. Titles must already be anonymized."""
    kin = _kinematics(t, x, y, window)
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.6), gridspec_kw={"height_ratios": [2.1, 1]})
    ax, ax2 = axes
    ax.plot(kin.x_smooth, kin.y_smooth, color=BLUE, lw=1.6, zorder=1)
    ax.scatter(kin.x_smooth[0], kin.y_smooth[0], color=BLACK, s=18, zorder=2, label="Start")
    if cut is not None and np.isfinite(cut.get("t0", np.nan)):
        gap = float(load_config()["tracking"]["gap_split_s"])
        x0, _ = interp_at(kin.t, kin.x_smooth, cut["t0"], gap)
        y0, _ = interp_at(kin.t, kin.y_smooth, cut["t0"], gap)
        ux, uy = unit_from_heading(cut["outgoing_heading"])
        reach = float(cut["exit_distance_0p5"])
        ax.scatter([x0], [y0], color=VERMILLION, s=28, zorder=3, label="Cut time")
        ax.annotate(
            "",
            xy=(x0 + ux * reach, y0 + uy * reach),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.8},
        )
        ax.plot([], [], color=GREEN, lw=1.8, label="Exit direction, 0.5 s")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (yards)")
    ax.set_ylabel("y (yards)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best", fontsize=9)
    rel = kin.t - kin.t[0]
    ax2.plot(rel, kin.speed, color=ORANGE, lw=1.5)
    if cut is not None and np.isfinite(cut.get("t0", np.nan)):
        t_rel = cut["t0"] - kin.t[0]
        ax2.axvline(t_rel, color=VERMILLION, lw=1.2, label="Cut time")
        ax2.axvspan(t_rel, t_rel + 0.5, color=GREEN, alpha=0.18, label="Exit window")
        ax2.legend(frameon=False, fontsize=9)
    if bound_t is not None and np.isfinite(bound_t):
        ax2.axvline(float(bound_t) - kin.t[0], color=BLACK, ls="--", lw=1.0, label="Information boundary")
        ax2.legend(frameon=False, fontsize=9)
    ax2.set_xlabel("Time from first frame (seconds)")
    ax2.set_ylabel("Speed (yards/second)")
    fig.tight_layout()
    save_figure(fig, path)


def plot_support(cells: pd.DataFrame, path: Path, icc: dict, title: str | None = None) -> None:
    speed = [c for c in cells["speed_bin"].unique()]
    angle = [c for c in cells["angle_bin"].unique()]
    grid = np.full((len(speed), len(angle)), np.nan)
    annot = [["" for _ in angle] for _ in speed]
    lookup = {(r.speed_bin, r.angle_bin): r for r in cells.itertuples()}
    for i, s in enumerate(speed):
        for j, a in enumerate(angle):
            row = lookup.get((s, a))
            if row is None:
                continue
            grid[i, j] = 1.0 if row.supported else 0.0
            annot[i][j] = f"{row.n_combine}/{row.n_nfl}"
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.imshow(grid, cmap="cividis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(angle)), [str(a) for a in angle], rotation=30, ha="right")
    ax.set_yticks(range(len(speed)), [str(s) for s in speed])
    for i in range(len(speed)):
        for j in range(len(angle)):
            if annot[i][j]:
                ink = "black" if grid[i, j] >= 0.5 else "white"
                ax.text(j, i, annot[i][j], ha="center", va="center", color=ink, fontsize=8)
    ax.set_xlabel("Absolute turn angle (degrees)")
    ax.set_ylabel("Incoming speed (yards/second)")
    icc_txt = "not estimated"
    if icc and np.isfinite(icc.get("icc", np.nan)):
        icc_txt = f"Cross-drill ICC {icc['icc']:.2f} (interval {icc.get('icc_p2_5', np.nan):.2f} to {icc.get('icc_p97_5', np.nan):.2f})"
    ax.set_title(title or f"2023 development support. Counts are Combine/NFL cuts.\n{icc_txt}")
    supported_color = plt.cm.cividis(1.0)
    unsupported_color = plt.cm.cividis(0.0)
    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=supported_color, markersize=10, label="Supported cell"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=unsupported_color, markersize=10, label="Not used for translation"),
    ]
    ax.legend(handles=legend, frameon=False, loc="upper right", fontsize=8)
    save_figure(fig, path)


def plot_translation(frame: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    part = frame.dropna(subset=["trait", "nfl_trait"])
    ax.scatter(part["trait"], part["nfl_trait"], color=BLUE, s=36, zorder=2)
    if len(part) >= 3 and part["trait"].std() > 1e-6:
        slope, intercept = np.polyfit(part["trait"], part["nfl_trait"], 1)
        xs = np.linspace(part["trait"].min(), part["trait"].max(), 40)
        ax.plot(xs, intercept + slope * xs, color=VERMILLION, lw=1.5, label=f"Slope {slope:.2f}")
        ax.legend(frameon=False)
    lims = [
        np.nanmin([part["trait"].min(), part["nfl_trait"].min()]),
        np.nanmax([part["trait"].max(), part["nfl_trait"].max()]),
    ]
    if np.isfinite(lims).all():
        pad = 0.15
        ax.plot([lims[0] - pad, lims[1] + pad], [lims[0] - pad, lims[1] + pad], color=GRAY, lw=0.8, ls="--")
    ax.set_xlabel("Combine cut-exit residual (yards)")
    ax.set_ylabel("NFL cut-exit residual (yards)")
    ax.set_title(title)
    save_figure(fig, path)


def plot_model_comparison(rows: pd.DataFrame, path: Path) -> None:
    """rows: model, mae, and optional lo, hi for the M2-M1 delta displayed as a note."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    order = [m for m in ["M0", "M1", "M2"] if m in set(rows["model"])]
    part = rows.set_index("model").loc[order]
    colors = {"M0": GRAY, "M1": BLUE, "M2": GREEN}
    xpos = np.arange(len(order))
    ax.bar(xpos, part["mae"], color=[colors[m] for m in order], width=0.72)
    ax.set_xticks(xpos, [{"M0": "Context", "M1": "Context +\nconventional", "M2": "Plus cut-exit\nresidual"}[m] for m in order])
    ax.set_ylabel("Player-mean absolute error (yards)")
    ax.set_title(rows["title"].iloc[0] if "title" in rows.columns else "Held-out separation error")
    save_figure(fig, path)


def plot_profile(players: pd.DataFrame, path: Path) -> None:
    """One row per player. Requires name, forty, trait, trait_loao_sd, pred_m1, pred_m2, observed."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.8))
    labels = players["label"].tolist()
    ypos = np.arange(len(players))
    axes[0].barh(ypos, players["trait"], color=BLUE, xerr=players["trait_loao_sd"], capsize=3)
    axes[0].axvline(0, color=GRAY, lw=0.8)
    axes[0].set_yticks(ypos, labels)
    axes[0].set_xlabel("Combine cut-exit residual (yards)")
    axes[0].set_title("Movement trait")
    width = 0.25
    axes[1].barh(ypos - width, players["pred_m1"], height=width, color=BLUE, label="Conventional")
    axes[1].barh(ypos, players["pred_m2"], height=width, color=GREEN, label="Plus tracking")
    axes[1].barh(ypos + width, players["observed"], height=width, color=ORANGE, label="Observed")
    axes[1].set_yticks(ypos, labels)
    axes[1].set_xlabel("Separation (yards)")
    axes[1].set_title("Same-route prediction")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, path)


def plot_failures(frame: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(frame["p"], frame["y"], color=BLUE, s=36)
    lo = np.nanmin([frame["p"].min(), frame["y"].min()])
    hi = np.nanmax([frame["p"].max(), frame["y"].max()])
    ax.plot([lo, hi], [lo, hi], color=GRAY, ls="--", lw=1)
    ax.set_xlabel("Predicted separation (yards)")
    ax.set_ylabel("Observed separation (yards)")
    ax.set_title(title)
    save_figure(fig, path)


def plot_montage(image_paths: list[Path], destination: Path, cols: int = 4) -> None:
    paths = [Path(p) for p in image_paths if Path(p).with_suffix(".png").exists() or Path(p).exists()]
    paths = [p if p.suffix == ".png" else p.with_suffix(".png") for p in paths]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return
    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    axes = np.atleast_1d(axes).ravel()
    for ax, path in zip(axes, paths):
        ax.imshow(plt.imread(path))
        ax.set_title(path.stem[:28], fontsize=7)
        ax.axis("off")
    for ax in axes[len(paths) :]:
        ax.axis("off")
    fig.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=80, bbox_inches="tight")
    plt.close(fig)
