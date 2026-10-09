"""Project paths, configuration, and reproducibility hashes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

HOLD_OUT_YEAR = 2025
RAW_FILES = [
    "players.csv",
    "combine_results.csv",
    "combine_tracking.csv",
    "player_career_successes.csv",
    "player_play.csv",
    "games.csv",
    "game_tracking_2023.csv",
    "game_tracking_2024.csv",
    "game_tracking_2025.csv",
]

# Outcome columns that must not be inspected for the held-out class
# until evaluate-heldout. pass_result is an eligibility flag, not a metric.
HOLDOUT_OUTCOME_COLUMNS = [
    "separation_at_pass_forward",
    "expected_yards_after_catch",
    "yards_gained",
    "rec_yards",
    "rush_yards",
    "yards_after_catch",
    "expected_points_added",
    "expected_points",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    path = Path(path) if path else project_root() / "config" / "research.yaml"
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration at {path} is not a mapping")
    return cfg


def config_hash(cfg: dict) -> str:
    blob = yaml.safe_dump(cfg, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def code_hash() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def drill_maps(cfg: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Return drill_name -> family and drill_name -> role."""
    family: dict[str, str] = {}
    role: dict[str, str] = {}
    for fam, names in cfg["drills"]["primary"].items():
        for name in names:
            family[name] = fam
            role[name] = "primary"
    for fam, names in cfg["drills"]["secondary_turnback"].items():
        for name in names:
            family[name] = fam
            role[name] = "secondary_turnback"
    for name in cfg["drills"]["excluded"]:
        family[name] = "EXCLUDED"
        role[name] = "excluded"
    return family, role


def drill_side(drill_name: str) -> str:
    if drill_name.endswith("_LEFT"):
        return "left"
    if drill_name.endswith("_RIGHT"):
        return "right"
    return "none"


class Project:
    def __init__(self, config_path: Path | None = None, data_root: Path | None = None):
        self.root = project_root()
        self.config_path = Path(config_path) if config_path else self.root / "config" / "research.yaml"
        self.config = load_config(self.config_path)
        self.data_root = self._resolve_data_root(data_root)
        self.reports = self.root / "reports"
        self.outputs = self.root / "outputs"
        self.interim = self.root / "data" / "interim"
        self.tmp = self.root / "data" / "tmp"
        self.figures = self.outputs / "figures"
        for path in (self.reports, self.outputs, self.interim, self.tmp, self.figures):
            path.mkdir(parents=True, exist_ok=True)

    def _resolve_data_root(self, explicit: Path | None) -> Path:
        candidates: list[Path] = []
        if explicit is not None:
            candidates.append(Path(explicit))
        configured = Path(self.config.get("data_root", "nfl-big-data-bowl-2027"))
        if not configured.is_absolute():
            configured = self.root / configured
        candidates.append(configured)
        candidates.append(self.root / "data" / "raw")
        candidates.append(Path("/kaggle/input/nfl-big-data-bowl-2027"))
        for candidate in candidates:
            if (candidate / "players.csv").exists():
                return candidate
        tried = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError(f"Could not find players.csv. Tried: {tried}")

    def raw(self, name: str) -> Path:
        path = self.data_root / name
        if not path.exists():
            raise FileNotFoundError(f"Missing competition file: {path}")
        return path
