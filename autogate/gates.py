"""Gate CSV utilities and polygon transformations."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from .imaging import DensityImage
from .register import apply_transform_to_points

LOGGER = logging.getLogger(__name__)

ROOT_ALIASES = {"", "root", "ROOT", "Root", None}


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
    population: Optional[str] = None

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
            "population": self.population,
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
        population = None
        for col in ("population", "cell_type", "label"):
            if col in row and not pd.isna(row[col]):
                population = str(row[col])
                break
        gate_id = str(row["gate_id"])
        gates.append(
            Gate(
                gate_id=gate_id,
                parent_id=str(row.get("parent_id", "")),
                gate_type=str(row.get("type", "polygon")),
                channel_x=str(row.get("channel_x")),
                channel_y=str(row.get("channel_y")),
                points=pts,
                fcs_file=str(row.get("fcs_file")) if not pd.isna(row.get("fcs_file")) else None,
                population=population or gate_id,
            )
        )
    return gates


def write_gates(path: Path | str, gates: Iterable[Gate]) -> None:
    """Write gates to CSV file."""
    path = Path(path)
    records = [gate.to_record() for gate in gates]
    df = pd.DataFrame(records)
    drop_cols = []
    if "fcs_file" in df.columns and df["fcs_file"].isnull().all():
        drop_cols.append("fcs_file")
    if "population" in df.columns and df["population"].isnull().all():
        drop_cols.append("population")
    if drop_cols:
        df = df.drop(columns=drop_cols)
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
    transformed_points = [
        (float(np.round(x, 12)), float(np.round(y, 12))) for x, y in transformed_points
    ]
    return Gate(
        gate_id=gate.gate_id,
        parent_id=gate.parent_id,
        gate_type=gate.gate_type,
        channel_x=gate.channel_x,
        channel_y=gate.channel_y,
        points=transformed_points,
        fcs_file=None,
        population=gate.population,
    )


def points_in_polygon(points: np.ndarray, polygon: Sequence[Tuple[float, float]]):
    """Return boolean mask of points inside polygon."""
    path = MplPath(polygon)
    return path.contains_points(points)


def _normalize_parent_id(parent_id: Optional[str]) -> Optional[str]:
    if parent_id in ROOT_ALIASES:
        return None
    if parent_id is None:
        return None
    return str(parent_id)


def assign_populations(data: pd.DataFrame, gates: Sequence[Gate]) -> pd.DataFrame:
    """Assign each event in ``data`` to the deepest gate it belongs to.

    Parameters
    ----------
    data:
        Preprocessed events with columns covering all gate channels.
    gates:
        Sequence of gates (typically already transformed to target space).

    Returns
    -------
    pandas.DataFrame
        Table with event_index, population label, gate depth, and gate path.
    """

    if not len(gates):
        LOGGER.warning("No gates provided for population assignment")
        return pd.DataFrame(
            {
                "event_index": np.arange(len(data), dtype=int),
                "population": ["UNGATED"] * len(data),
                "depth": [-1] * len(data),
                "gate_path": [""] * len(data),
            }
        )

    gate_map: Dict[str, Gate] = {}
    for gate in gates:
        if gate.gate_id in gate_map:
            LOGGER.warning("Duplicate gate_id %s encountered; later gate overrides", gate.gate_id)
        gate_map[gate.gate_id] = gate

    def compute_depth(gate_id: str, stack: Optional[Tuple[str, ...]] = None) -> int:
        stack = stack or tuple()
        if gate_id in depth_cache:
            return depth_cache[gate_id]
        if gate_id in stack:
            cycle = " -> ".join(stack + (gate_id,))
            raise ValueError(f"Cycle detected in gate hierarchy: {cycle}")
        gate = gate_map[gate_id]
        parent_norm = _normalize_parent_id(gate.parent_id)
        if parent_norm is None or parent_norm not in gate_map:
            depth_cache[gate_id] = 0
        else:
            depth_cache[gate_id] = compute_depth(parent_norm, stack + (gate_id,)) + 1
        return depth_cache[gate_id]

    depth_cache: Dict[str, int] = {}
    for gid in gate_map:
        compute_depth(gid)

    gate_order = sorted(gates, key=lambda g: depth_cache.get(g.gate_id, 0))
    masks: Dict[str, np.ndarray] = {}

    for gate in gate_order:
        try:
            points = data[[gate.channel_x, gate.channel_y]].to_numpy()
        except KeyError:
            LOGGER.error(
                "Data missing channels required for gate %s (%s,%s); skipping",
                gate.gate_id,
                gate.channel_x,
                gate.channel_y,
            )
            continue
        mask = points_in_polygon(points, gate.points)
        parent_norm = _normalize_parent_id(gate.parent_id)
        if parent_norm and parent_norm in masks:
            mask = mask & masks[parent_norm]
        elif parent_norm and parent_norm not in gate_map:
            LOGGER.warning(
                "Gate %s references unknown parent %s; treating as root gate",
                gate.gate_id,
                gate.parent_id,
            )
        masks[gate.gate_id] = mask

    assignment_order = [gate_id for gate_id, gate in gate_map.items() if gate.population]
    if not assignment_order:
        LOGGER.warning("No populated gates detected; assigning using all gates")
        assignment_order = list(gate_map.keys())
    assignment_order = sorted(assignment_order, key=lambda g: depth_cache.get(g, 0))

    assigned_labels = np.array(["UNGATED"] * len(data), dtype=object)
    assigned_gate_ids = np.array([""] * len(data), dtype=object)
    assigned_depth = np.full(len(data), -1, dtype=int)
    assigned_path = np.array([""] * len(data), dtype=object)

    def build_path(gate_id: str) -> str:
        parts = [gate_id]
        current = gate_map[gate_id]
        while True:
            parent_norm = _normalize_parent_id(current.parent_id)
            if parent_norm is None or parent_norm not in gate_map:
                break
            parts.append(parent_norm)
            current = gate_map[parent_norm]
        return "/".join(reversed(parts))

    for gate_id in assignment_order:
        if gate_id not in masks:
            LOGGER.warning("Leaf gate %s has no computed mask; skipping", gate_id)
            continue
        mask = masks[gate_id]
        depth = depth_cache.get(gate_id, 0)
        update_mask = mask & (depth >= assigned_depth)
        if not np.any(update_mask):
            continue
        assigned_gate = gate_map[gate_id]
        assigned_labels[update_mask] = assigned_gate.population or gate_id
        assigned_gate_ids[update_mask] = gate_id
        assigned_depth[update_mask] = depth
        path_str = build_path(gate_id)
        assigned_path[update_mask] = path_str

    return pd.DataFrame(
        {
            "event_index": np.arange(len(data), dtype=int),
            "population": assigned_labels,
            "gate_id": assigned_gate_ids,
            "depth": assigned_depth,
            "gate_path": assigned_path,
        }
    )
