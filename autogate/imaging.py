"""Point cloud to image conversion utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class DensityImage:
    """Container for density image and coordinate mapping."""

    image: np.ndarray
    bbox: Tuple[float, float, float, float]
    bins: int

    def data_to_index(self, x: float, y: float) -> Tuple[float, float]:
        """Map data coordinates to image index coordinates."""
        x_min, x_max, y_min, y_max = self.bbox
        ix = (x - x_min) / max(x_max - x_min, 1e-9) * (self.bins - 1)
        iy = (y - y_min) / max(y_max - y_min, 1e-9) * (self.bins - 1)
        return float(ix), float(iy)

    def index_to_data(self, ix: float, iy: float) -> Tuple[float, float]:
        """Map image index coordinates to data coordinates."""
        x_min, x_max, y_min, y_max = self.bbox
        x = ix / max(self.bins - 1, 1e-9) * (x_max - x_min) + x_min
        y = iy / max(self.bins - 1, 1e-9) * (y_max - y_min) + y_min
        return float(x), float(y)


def compute_density_image(
    data,
    channel_x: str,
    channel_y: str,
    bins: int = 256,
    smooth_sigma: float = 0.0,
    ranges: dict | None = None,
) -> DensityImage:
    """Convert channel pair point cloud into an 8-bit density image."""
    if channel_x not in data.columns or channel_y not in data.columns:
        raise KeyError(f"Channels {channel_x}/{channel_y} not available in data frame")

    x = data[channel_x].to_numpy()
    y = data[channel_y].to_numpy()

    if ranges and channel_x in ranges:
        x_min, x_max = ranges[channel_x]
    else:
        x_min, x_max = float(np.min(x)), float(np.max(x))
    if ranges and channel_y in ranges:
        y_min, y_max = ranges[channel_y]
    else:
        y_min, y_max = float(np.min(y)), float(np.max(y))

    if x_max == x_min:
        x_max += 1e-3
    if y_max == y_min:
        y_max += 1e-3

    hist, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[x_min, x_max], [y_min, y_max]],
    )
    hist = hist.astype(np.float32)
    if smooth_sigma and smooth_sigma > 0:
        hist = gaussian_filter(hist, sigma=smooth_sigma)

    if hist.max() > 0:
        hist = hist / hist.max()
    image = (hist * 255).astype(np.uint8)
    bbox = (x_min, x_max, y_min, y_max)
    return DensityImage(image=image, bbox=bbox, bins=bins)
