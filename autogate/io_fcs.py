"""FCS I/O, compensation, and transformation utilities."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional heavy dependency
    from fcsparser import parse
except Exception:  # pragma: no cover - fallback for environments without fcsparser
    parse = None  # type: ignore

LOGGER = logging.getLogger(__name__)


class FCSLoader:
    """Load and preprocess FCS-like data."""

    def __init__(
        self,
        channels: Iterable[str],
        transform: str = "asinh",
        compensation: str | Path | None = "auto",
        ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
        asinh_cofactor: float = 5.0,
    ) -> None:
        self.channels = list(channels)
        self.transform = transform
        self.compensation = compensation
        self.ranges = dict(ranges or {})
        self.asinh_cofactor = asinh_cofactor

    def read_fcs(self, path: Path | str) -> pd.DataFrame:
        """Read FCS/CSV/NPZ file into a DataFrame with configured preprocessing."""
        path = Path(path)
        LOGGER.info("Reading FCS data from %s", path)

        if path.suffix.lower() == ".fcs":
            if parse is None:
                raise ImportError("fcsparser is required to read .fcs files")
            meta, data = parse(str(path), reformat_meta=True)
            df = data[self.channels].copy()
            spill = self._extract_spillover(meta)
        elif path.suffix.lower() in {".csv"}:
            df = pd.read_csv(path)
            spill = None
        elif path.suffix.lower() in {".npz"}:
            arrays = np.load(path)
            df = pd.DataFrame({k: arrays[k] for k in arrays.files})
            spill = None
        elif path.suffix.lower() in {".npy"}:
            arr = np.load(path)
            if arr.ndim != 2 or arr.shape[1] != len(self.channels):
                raise ValueError(".npy array must be 2D with column count matching channels")
            df = pd.DataFrame(arr, columns=self.channels)
            spill = None
        else:
            raise ValueError(f"Unsupported data format: {path.suffix}")

        if not set(self.channels).issubset(df.columns):
            missing = set(self.channels) - set(df.columns)
            raise ValueError(f"Data file {path} missing required channels: {', '.join(missing)}")

        df = df[self.channels].astype(float)

        comp_matrix = self._load_compensation_matrix(spill)
        if comp_matrix is not None:
            LOGGER.info("Applying compensation matrix to %s", path.name)
            matrix = comp_matrix
            arr = df.to_numpy()
            compensated = arr @ matrix.T
            df = pd.DataFrame(compensated, columns=self.channels)
        else:
            LOGGER.info("No compensation applied to %s", path.name)

        df = self._apply_transform(df)
        df = self._clip_ranges(df)
        return df

    def _apply_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.transform == "asinh":
            LOGGER.debug("Applying asinh transform with cofactor %s", self.asinh_cofactor)
            return np.arcsinh(df / self.asinh_cofactor)
        if self.transform == "log10":
            LOGGER.debug("Applying log10 transform")
            return np.log10(np.clip(df, a_min=1e-3, a_max=None))
        LOGGER.debug("No transform applied (transform=%s)", self.transform)
        return df

    def _clip_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.ranges:
            return df
        clipped = df.copy()
        for ch, (lo, hi) in self.ranges.items():
            if ch not in clipped.columns:
                LOGGER.warning("Channel %s missing during clipping; skipping", ch)
                continue
            clipped[ch] = clipped[ch].clip(lo, hi)
        return clipped

    def _extract_spillover(self, meta: Mapping[str, str]) -> Optional[pd.DataFrame]:
        keys = ["SPILL", "SPILLOVER", "$SPILLOVER"]
        for key in keys:
            if key in meta:
                try:
                    value = meta[key]
                    if isinstance(value, str):
                        parts = value.strip().split(",")
                        size = int(parts[0])
                        channels = parts[1 : size + 1]
                        data = np.array(list(map(float, parts[size + 1 :])))
                        matrix = data.reshape(size, size)
                        return pd.DataFrame(matrix, index=channels, columns=channels)
                except Exception as exc:  # pragma: no cover - very unlikely
                    LOGGER.warning("Failed to parse spillover matrix: %s", exc)
        return None

    def _load_compensation_matrix(self, spill: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if self.compensation == "auto":
            if spill is None:
                return None
            missing = [ch for ch in self.channels if ch not in spill.index]
            if missing:
                LOGGER.warning("Spillover matrix missing channels: %s", ", ".join(missing))
                return None
            return spill.loc[self.channels, self.channels]

        if self.compensation is None:
            return None

        path = Path(self.compensation)
        if not path.exists():
            raise FileNotFoundError(f"Compensation matrix file not found: {path}")
        matrix = pd.read_csv(path, index_col=0)
        missing = [ch for ch in self.channels if ch not in matrix.index]
        if missing:
            raise ValueError(
                "Compensation matrix missing channels: " + ", ".join(missing)
            )
        return matrix.loc[self.channels, self.channels]


def parse_points(points_str: str) -> List[Tuple[float, float]]:
    """Parse points column string into list of tuples."""
    try:
        points = json.loads(points_str.replace("(", "[").replace(")", "]"))
        return [tuple(map(float, pt)) for pt in points]
    except Exception as exc:
        raise ValueError(f"Invalid points string: {points_str}") from exc
