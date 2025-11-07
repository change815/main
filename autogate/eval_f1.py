"""Precision/Recall/F1 evaluation utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .gates import Gate, points_in_polygon, read_gates
from .io_fcs import FCSLoader

LOGGER = logging.getLogger(__name__)


@dataclass
class GateMetrics:
    gate_id: str
    fcs_file: str
    precision: float
    recall: float
    f1: float
    predicted_count: int
    truth_count: int
    intersection_count: int
    warning: str | None = None


def _safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def evaluate_gate(
    data_points: np.ndarray,
    pred_gate: Gate,
    truth_gate: Gate,
) -> GateMetrics:
    pred_mask = points_in_polygon(data_points, pred_gate.points)
    truth_mask = points_in_polygon(data_points, truth_gate.points)

    predicted_count = int(pred_mask.sum())
    truth_count = int(truth_mask.sum())
    intersection = int((pred_mask & truth_mask).sum())

    precision = _safe_divide(intersection, predicted_count)
    recall = _safe_divide(intersection, truth_count)
    f1 = _safe_divide(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    warning = None
    if truth_count < 10 or predicted_count < 10:
        warning = "LOW_COUNTS"
        LOGGER.warning(
            "Low cell count for gate %s (pred=%s truth=%s); metrics may be unstable",
            pred_gate.gate_id,
            predicted_count,
            truth_count,
        )

    return GateMetrics(
        gate_id=pred_gate.gate_id,
        fcs_file="",
        precision=precision,
        recall=recall,
        f1=f1,
        predicted_count=predicted_count,
        truth_count=truth_count,
        intersection_count=intersection,
        warning=warning,
    )


def evaluate(
    config,
    predictions_dir: Path,
    truth_csv: Path,
) -> Dict[str, float]:
    loader = FCSLoader(
        channels=config["panel"]["channels"],
        transform=config["panel"].get("transform", "asinh"),
        compensation=config["panel"].get("compensation", "auto"),
        ranges=config["panel"].get("ranges", {}),
    )
    truth_gates = read_gates(truth_csv)
    truth_by_file: Dict[str, List[Gate]] = {}
    for gate in truth_gates:
        if gate.fcs_file is None:
            raise ValueError("Truth CSV must include fcs_file column for each gate")
        truth_by_file.setdefault(Path(gate.fcs_file).stem, []).append(gate)

    metrics: List[GateMetrics] = []
    micro_intersection = 0
    micro_pred = 0
    micro_truth = 0

    for file_stem, file_truth_gates in truth_by_file.items():
        prediction_path = predictions_dir / f"{file_stem}_gates.csv"
        if not prediction_path.exists():
            LOGGER.warning("Missing prediction file for %s", file_stem)
            continue
        pred_gates = read_gates(prediction_path)
        pred_by_id = {gate.gate_id: gate for gate in pred_gates}
        truth_by_id = {gate.gate_id: gate for gate in file_truth_gates}

        fcs_path = Path(config["io"]["target_fcs_dir"]) / f"{file_stem}.fcs"
        if not fcs_path.exists():
            # allow csv/npz fallback
            alternatives = list(fcs_path.parent.glob(f"{file_stem}.*"))
            if alternatives:
                fcs_path = alternatives[0]
        data = loader.read_fcs(fcs_path)

        for gate_id, truth_gate in truth_by_id.items():
            if gate_id not in pred_by_id:
                LOGGER.warning("Missing predicted gate %s for file %s", gate_id, file_stem)
                continue
            pred_gate = pred_by_id[gate_id]
            data_points = data[[pred_gate.channel_x, pred_gate.channel_y]].to_numpy()
            gm = evaluate_gate(data_points, pred_gate, truth_gate)
            gm.fcs_file = file_stem
            metrics.append(gm)
            micro_intersection += gm.intersection_count
            micro_pred += gm.predicted_count
            micro_truth += gm.truth_count

    if not metrics:
        raise ValueError("No evaluation metrics computed; ensure predictions and truth are aligned")

    macro_f1 = float(np.mean([m.f1 for m in metrics]))
    micro_precision = _safe_divide(micro_intersection, micro_pred)
    micro_recall = _safe_divide(micro_intersection, micro_truth)
    micro_f1 = _safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0

    records = [m.__dict__ for m in metrics]
    eval_path = predictions_dir / "evaluation.csv"
    pd.DataFrame(records).to_csv(eval_path, index=False)
    LOGGER.info("Evaluation written to %s", eval_path)
    LOGGER.info(
        "Macro F1: %.4f | Micro Precision: %.4f Recall: %.4f F1: %.4f",
        macro_f1,
        micro_precision,
        micro_recall,
        micro_f1,
    )

    return {
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "metrics": metrics,
    }
