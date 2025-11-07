"""Training sample selection utilities."""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L2 distance between two images."""
    if a.shape != b.shape:
        raise ValueError("Images must have the same shape for L2 distance")
    diff = (a.astype(np.float32) - b.astype(np.float32)).ravel()
    return float(np.linalg.norm(diff))


def select_best(training_images: Dict[str, np.ndarray], target_image: np.ndarray) -> Tuple[str, float]:
    """Select the training image closest to the target using L2 distance."""
    if not training_images:
        raise ValueError("No training images provided for selection")

    best_id = None
    best_score = float("inf")

    for train_id, image in training_images.items():
        score = l2_distance(image, target_image)
        LOGGER.debug("L2 distance for %s: %.4f", train_id, score)
        if score < best_score:
            best_id = train_id
            best_score = score

    assert best_id is not None
    LOGGER.info("Selected training sample %s with L2 distance %.4f", best_id, best_score)
    return best_id, best_score
