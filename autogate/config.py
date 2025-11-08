"""Configuration loading utilities for Autogate."""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "io": {
        "train_fcs_dir": "./data/train",
        "train_gates_csv": "./data/train/gates.csv",
        "target_fcs_dir": "./data/target",
        "out_dir": "./out/gates",
        "labels_dir": None,
        "plots_dir": None,
        "eval_truth_csv": None,
    },
    "panel": {
        "channels": [],
        "transform": "asinh",
        "compensation": "auto",
        "ranges": {},
    },
    "imaging": {
        "bins": 256,
        "smooth_sigma": 0.0,
    },
    "registration": {
        "grid_spacing": 32,
        "levels": 3,
        "iterations": 200,
    },
    "selection": {
        "metric": "l2",
    },
    "logging": {
        "level": "INFO",
        "log_dir": "./out/logs",
    },
    "visualization": {
        "ungated_color": "#9e9e9e",
        "population_colors": {
            "default": {},
        },
    },
    "evaluation": {
        "mode": "auto",
        "label_column": "population",
        "sample_id_column": "sample_id",
        "event_index_column": "event_index",
        "include_ungated": True,
    },
}


def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively update mapping ``base`` with ``updates``."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path | str | None) -> Dict[str, Any]:
    """Load YAML configuration file, applying defaults."""
    config = copy.deepcopy(DEFAULT_CONFIG)

    if path is None:
        logging.getLogger(__name__).warning("No configuration file provided, using defaults.")
        return config

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Configuration file must define a mapping at the root level.")

    config = deep_update(config, loaded)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate that required configuration entries exist."""
    missing = []
    io_cfg = config.get("io", {})
    if not io_cfg.get("train_gates_csv"):
        missing.append("io.train_gates_csv")
    if not io_cfg.get("train_fcs_dir"):
        missing.append("io.train_fcs_dir")
    if not io_cfg.get("target_fcs_dir"):
        missing.append("io.target_fcs_dir")

    panel_cfg = config.get("panel", {})
    if not panel_cfg.get("channels"):
        missing.append("panel.channels")
    if not panel_cfg.get("ranges"):
        missing.append("panel.ranges")

    if missing:
        raise ValueError(f"Missing required configuration fields: {', '.join(missing)}")


def ensure_ranges(panel_cfg: Mapping[str, Any], channels: List[str]) -> None:
    """Ensure ranges are available for all channels."""
    ranges = panel_cfg.get("ranges", {})
    missing = [ch for ch in channels if ch not in ranges]
    if missing:
        raise ValueError(
            "Configuration missing ranges for channels: " + ", ".join(missing)
        )


def summarize_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Generate a summary suitable for logging."""
    summary = {
        "io": config.get("io", {}),
        "panel": {
            "channels": config.get("panel", {}).get("channels", []),
            "transform": config.get("panel", {}).get("transform"),
            "compensation": config.get("panel", {}).get("compensation"),
        },
        "imaging": config.get("imaging", {}),
        "registration": config.get("registration", {}),
        "selection": config.get("selection", {}),
    }
    return summary
