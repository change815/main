"""Gate CSV utilities and polygon transformations."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from .imaging import DensityImage
from .register import apply_transform_to_points

LOGGER = logging.getLogger(__name__)


@dataclass
class Gate:
    """Representation of a polygon gate."""

    gate_id: str
    parent_id: str
    gate_type: str
    channel_x: str
    channel_y: str
    points: List[Tuple[float, float]]
    fcs_file: Optional[str] = None

    def to_record(self) -> dict:
        pts = [(float(x), float(y)) for x, y in self.points]
        return {
            "gate_id": self.gate_id,
            "parent_id": self.parent_id,
            "type": self.gate_type,
            "channel_x": self.channel_x,
            "channel_y": self.channel_y,
            "points": json.dumps(pts),
            "fcs_file": self.fcs_file,
        }


def _parse_points(points_str: str) -> Optional[List[Tuple[float, float]]]:
    try:
        normalized = points_str.replace("(", "[").replace(")", "]")
        pts = json.loads(normalized)
        return [tuple(map(float, p)) for p in pts]
    except Exception:
        return None


def read_gates(path: Path | str) -> List[Gate]:
    """Read gates CSV into list of Gate objects."""
    path = Path(path)
    df = pd.read_csv(path)
    gates: List[Gate] = []
    for _, row in df.iterrows():
        pts = _parse_points(str(row["points"]))
        if pts is None:
            LOGGER.warning("Skipping gate %s due to invalid points format", row.get("gate_id"))
            continue
        gates.append(
            Gate(
                gate_id=str(row["gate_id"]),
                parent_id=str(row.get("parent_id", "")),
                gate_type=str(row.get("type", "polygon")),
                channel_x=str(row.get("channel_x")),
                channel_y=str(row.get("channel_y")),
                points=pts,
                fcs_file=str(row.get("fcs_file")) if not pd.isna(row.get("fcs_file")) else None,
            )
        )
    return gates


def write_gates(path: Path | str, gates: Iterable[Gate]) -> None:
    """Write gates to CSV file."""
    path = Path(path)
    records = [gate.to_record() for gate in gates]
    df = pd.DataFrame(records)
    if "fcs_file" in df.columns and df["fcs_file"].isnull().all():
        df = df.drop(columns=["fcs_file"])
    df.to_csv(path, index=False)


def transform_gate(
    gate: Gate,
    transform,
    source_density: DensityImage,
    target_density: DensityImage,
) -> Gate:
    """Apply transform to gate points via density image coordinate systems."""
    src_points = [source_density.data_to_index(x, y) for x, y in gate.points]
    transformed_idx = apply_transform_to_points(transform, src_points)
    transformed_points = [target_density.index_to_data(ix, iy) for ix, iy in transformed_idx]
    return Gate(
        gate_id=gate.gate_id,
        parent_id=gate.parent_id,
        gate_type=gate.gate_type,
        channel_x=gate.channel_x,
        channel_y=gate.channel_y,
        points=transformed_points,
        fcs_file=None,
    )


def points_in_polygon(points: np.ndarray, polygon: Sequence[Tuple[float, float]]):
    """Return boolean mask of points inside polygon."""
    path = MplPath(polygon)
    return path.contains_points(points)
