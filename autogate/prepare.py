"""Utilities to derive polygon gates and panel ranges from annotated CSV files."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import ConvexHull

LOGGER = logging.getLogger(__name__)


@dataclass
class GateSpec:
    """Description of a single polygon gate to be generated."""

    gate_id: str
    parent_id: str
    channel_x: str
    channel_y: str
    points: List[Tuple[float, float]]
    fcs_file: str
    population: str

    def to_record(self) -> Dict[str, str]:
        return {
            "gate_id": self.gate_id,
            "parent_id": self.parent_id,
            "type": "polygon",
            "channel_x": self.channel_x,
            "channel_y": self.channel_y,
            "points": json.dumps([(float(x), float(y)) for x, y in self.points]),
            "fcs_file": self.fcs_file,
            "population": self.population,
        }


def _load_spec(spec_path: Path | str) -> Dict[str, Any]:
    spec_path = Path(spec_path)
    with spec_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, Mapping):
        raise ValueError("Annotation spec must be a mapping at the root level")
    if "coordinate_systems" not in data:
        raise ValueError("Annotation spec missing 'coordinate_systems' section")
    return dict(data)


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], file_path: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Annotation CSV {file_path} missing required columns: {', '.join(missing)}"
        )


def _convex_hull(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("No points available to build polygon")
    if len(points) == 1:
        pt = points[0]
        return np.array([
            pt + [-1e-3, -1e-3],
            pt + [1e-3, -1e-3],
            pt + [1e-3, 1e-3],
            pt + [-1e-3, 1e-3],
        ])
    if len(points) == 2:
        p0, p1 = points
        delta = p1 - p0
        normal = np.array([-delta[1], delta[0]])
        if np.allclose(normal, 0):
            normal = np.array([0.0, 1.0])
        normal = normal / (np.linalg.norm(normal) + 1e-12) * 1e-3
        return np.array([p0 - normal, p0 + normal, p1 + normal, p1 - normal])

    hull = ConvexHull(points)
    return points[hull.vertices]


def _expand_polygon(polygon: np.ndarray, expand_frac: float = 0.05) -> np.ndarray:
    if expand_frac <= 0:
        return polygon
    centroid = polygon.mean(axis=0)
    ranges = polygon.max(axis=0) - polygon.min(axis=0)
    scale = 1.0 + expand_frac
    adjusted = centroid + (polygon - centroid) * scale
    # Prevent degenerate zero-width polygons by nudging using ranges
    adjusted[:, 0] += np.sign(adjusted[:, 0] - centroid[0]) * ranges[0] * 1e-3
    adjusted[:, 1] += np.sign(adjusted[:, 1] - centroid[1]) * ranges[1] * 1e-3
    return adjusted


def _default_gate_id(sample_id: str, population: str, channel_x: str, channel_y: str) -> str:
    slug_x = channel_x.replace(" ", "_")
    slug_y = channel_y.replace(" ", "_")
    return f"{sample_id}__{population}__{slug_x}__{slug_y}"


def _update_ranges(ranges: MutableMapping[str, List[float]], df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        lo = float(values.quantile(0.001))
        hi = float(values.quantile(0.999))
        if col not in ranges:
            ranges[col] = [lo, hi]
        else:
            ranges[col][0] = min(ranges[col][0], lo)
            ranges[col][1] = max(ranges[col][1], hi)


def build_gates_from_annotations(
    csv_root: Path | str,
    spec_path: Path | str,
    *,
    output_csv: Path | str,
    fcs_suffix: str = ".fcs",
    expand_frac: float = 0.05,
    min_points: int = 20,
) -> Dict[str, Any]:
    """Derive polygon gates and panel metadata from annotated CSV directories."""

    csv_root = Path(csv_root)
    if not csv_root.exists():
        raise FileNotFoundError(f"Annotation root not found: {csv_root}")

    spec = _load_spec(spec_path)
    coordinate_systems = spec["coordinate_systems"]
    if not isinstance(coordinate_systems, list):
        raise ValueError("coordinate_systems must be a list of mappings")

    label_column_default = spec.get("label_column", "cell_type")
    parent_id_default = spec.get("parent_id", "root")

    gates: List[GateSpec] = []
    ranges: Dict[str, List[float]] = {}
    populations_summary: Dict[str, int] = {}

    sample_dirs = [p for p in csv_root.iterdir() if p.is_dir()]
    if not sample_dirs:
        raise FileNotFoundError(f"No sample subdirectories found in {csv_root}")

    for sample_dir in sorted(sample_dirs):
        sample_id = sample_dir.name
        LOGGER.info("Processing annotations for sample %s", sample_id)
        for entry in coordinate_systems:
            file_name = entry.get("file")
            if not file_name:
                raise ValueError("Each coordinate system entry must define a 'file'")
            file_path = sample_dir / file_name
            if not file_path.exists():
                LOGGER.warning("Missing annotation CSV %s", file_path)
                continue

            channel_x = entry.get("channel_x")
            channel_y = entry.get("channel_y")
            if not channel_x or not channel_y:
                raise ValueError("coordinate_system entry must include channel_x and channel_y")

            label_column = entry.get("label_column", label_column_default)
            populations = entry.get("populations")
            if not populations:
                raise ValueError(
                    f"coordinate_system {file_name} must define populations to extract"
                )

            df = pd.read_csv(file_path)
            _ensure_columns(df, [channel_x, channel_y, label_column], file_path)
            _update_ranges(ranges, df, [channel_x, channel_y])

            for population in populations:
                mask = df[label_column].astype(str) == str(population)
                subset = df.loc[mask, [channel_x, channel_y]].dropna()
                count = len(subset)
                populations_summary[str(population)] = populations_summary.get(str(population), 0) + count
                if count == 0:
                    LOGGER.warning(
                        "Sample %s population %s has zero annotated points in %s",
                        sample_id,
                        population,
                        file_path.name,
                    )
                    continue
                if count < min_points:
                    LOGGER.warning(
                        "Population %s in %s has only %d points (<%d); polygon may be unstable",
                        population,
                        file_path.name,
                        count,
                        min_points,
                    )
                try:
                    polygon = _convex_hull(subset.to_numpy(dtype=float))
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to compute polygon for {population} in {file_path}"
                    ) from exc
                polygon = _expand_polygon(polygon, expand_frac)
                gate_id = entry.get("gate_id_prefix") or _default_gate_id(
                    sample_id, str(population), channel_x, channel_y
                )
                gate = GateSpec(
                    gate_id=f"{gate_id}",
                    parent_id=str(entry.get("parent_id", parent_id_default)),
                    channel_x=channel_x,
                    channel_y=channel_y,
                    points=[(float(x), float(y)) for x, y in polygon],
                    fcs_file=f"{sample_id}{fcs_suffix}",
                    population=str(population),
                )
                gates.append(gate)

    if not gates:
        raise ValueError("No gates generated from annotations; check spec and CSV files")

    records = [gate.to_record() for gate in gates]
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    LOGGER.info("Wrote %d gates to %s", len(records), output_path)

    channels = sorted(ranges.keys())
    summary = {
        "gates_csv": str(output_path),
        "channels": channels,
        "ranges": {ch: [float(lo), float(hi)] for ch, (lo, hi) in ranges.items()},
        "populations": populations_summary,
        "samples": [p.name for p in sorted(sample_dirs)],
    }
    return summary


def write_summary(summary: Mapping[str, Any], path: Path | str) -> None:
    """Persist summary dictionary as YAML."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(summary), fh, allow_unicode=True, sort_keys=True)
    LOGGER.info("Annotation summary written to %s", output)
