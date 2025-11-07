"""Autogate package for elastic image registration based gating."""
from importlib.metadata import version, PackageNotFoundError

__all__ = ["__version__"]

try:
    __version__ = version("autogate")
except PackageNotFoundError:  # pragma: no cover - during local dev
    __version__ = "0.1.0"
