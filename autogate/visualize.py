"""Visualization helpers for population scatter plots."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
]


def _label_order(labels: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen = set()
    contains_ungated = False
    for label in labels:
        if label == "UNGATED":
            contains_ungated = True
            continue
        if label in seen:
            continue
        seen.add(label)
        unique.append(label)
    unique.sort()
    if contains_ungated:
        unique.append("UNGATED")
    return unique


def plot_population_scatter(
    data: pd.DataFrame,
    populations: pd.Series | np.ndarray,
    channel_x: str,
    channel_y: str,
    out_path: Path,
    sample_size: int | None = None,
    point_size: float = 4.0,
    alpha: float = 0.5,
    color_overrides: dict[str, str] | None = None,
    ungated_color: str = "#9e9e9e",
) -> None:
    """Render a scatter plot colored by population labels."""

    if channel_x not in data.columns or channel_y not in data.columns:
        raise KeyError(f"Missing channels {channel_x}/{channel_y} in data for visualization")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = data[[channel_x, channel_y]].copy()
    df["population"] = np.asarray(populations)

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=0)

    order = _label_order(df["population"].unique())
    if not order:
        order = ["UNGATED"]

    plt.figure(figsize=(6, 6))
    color_cycle = COLORS
    if len(order) > len(color_cycle):
        cmap = plt.get_cmap("tab20")
        color_cycle = [cmap(i) for i in np.linspace(0, 1, len(order))]

    overrides = color_overrides or {}
    color_map = {label: overrides.get(label, color_cycle[i % len(color_cycle)]) for i, label in enumerate(order)}
    if "UNGATED" in order:
        color_map["UNGATED"] = overrides.get("UNGATED", ungated_color)

    for label in order:
        subset = df[df["population"] == label]
        if subset.empty:
            continue
        plt.scatter(
            subset[channel_x],
            subset[channel_y],
            s=point_size,
            alpha=alpha if label != "UNGATED" else 0.2,
            label=label,
            color=color_map[label],
        )

    plt.xlabel(channel_x)
    plt.ylabel(channel_y)
    plt.title(f"Populations: {channel_x} vs {channel_y}")
    plt.legend(loc="best", fontsize="small", markerscale=3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

