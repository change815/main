"""B-spline registration utilities."""
from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

LOGGER = logging.getLogger(__name__)


class RegistrationError(RuntimeError):
    """Raised when registration fails irrecoverably."""


def _bspline_mesh_size(image: sitk.Image, grid_spacing: int) -> List[int]:
    size = image.GetSize()
    spacing = image.GetSpacing()
    mesh_size = []
    for dim in range(image.GetDimension()):
        physical = (size[dim] - 1) * spacing[dim]
        mesh = max(1, int(round(physical / grid_spacing)))
        mesh_size.append(mesh)
    return mesh_size


def register_bspline(
    fixed: np.ndarray,
    moving: np.ndarray,
    grid_spacing: int = 32,
    levels: int = 3,
    iterations: int = 200,
) -> sitk.Transform:
    """Perform B-spline registration and return resulting transform."""
    fixed_image = sitk.GetImageFromArray(fixed.astype(np.float32))
    moving_image = sitk.GetImageFromArray(moving.astype(np.float32))

    transform = sitk.BSplineTransformInitializer(
        fixed_image,
        transformDomainMeshSize=_bspline_mesh_size(fixed_image, grid_spacing),
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=iterations)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetInitialTransform(transform, inPlace=False)

    shrink_factors = [2 ** i for i in range(levels)][::-1]
    smoothing_sigmas = [sigma for sigma in range(levels)][::-1]
    registration.SetShrinkFactorsPerLevel(shrink_factors)
    registration.SetSmoothingSigmasPerLevel(smoothing_sigmas)

    try:
        LOGGER.info(
            "Starting B-spline registration: levels=%s iterations=%s grid=%s",
            levels,
            iterations,
            grid_spacing,
        )
        final_transform = registration.Execute(fixed_image, moving_image)
        LOGGER.info("Registration completed with final metric %.6f", registration.GetMetricValue())
        return final_transform
    except Exception as exc:  # pragma: no cover - SITK exceptions hard to trigger deterministically
        raise RegistrationError(str(exc)) from exc


def apply_transform_to_points(
    transform: sitk.Transform,
    points: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Apply SimpleITK transform to a sequence of 2D points."""
    transformed = []
    for pt in points:
        x, y = transform.TransformPoint((float(pt[0]), float(pt[1])))
        transformed.append((float(x), float(y)))
    return transformed
