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


@dataclass
class PopulationMetrics:
    sample_id: str
    population: str
    precision: float
    recall: float
    f1: float
    true_count: int
    pred_count: int
    tp: int
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


def _evaluate_gate_polygons(
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
        "mode": "gates",
    }


def _detect_label_column(df: pd.DataFrame, preferred: str | None) -> str:
    candidates = [preferred] if preferred else []
    candidates += ["population", "cell_type", "label"]
    for col in candidates:
        if col and col in df.columns:
            return col
    raise ValueError(
        "Truth label file must contain one of the columns: "
        + ", ".join([c for c in candidates if c]),
    )


def _normalize_truth_table(
    df: pd.DataFrame,
    *,
    sample_hint: str | None,
    label_column: str | None,
    sample_column: str | None,
    event_column: str | None,
) -> pd.DataFrame:
    df = df.copy()
    label_col = _detect_label_column(df, label_column)
    df.rename(columns={label_col: "population"}, inplace=True)

    if event_column and event_column in df.columns:
        df.rename(columns={event_column: "event_index"}, inplace=True)
    elif "event_index" not in df.columns:
        df["event_index"] = np.arange(len(df), dtype=int)

    if sample_column and sample_column in df.columns:
        df.rename(columns={sample_column: "sample_id"}, inplace=True)
    elif "sample_id" not in df.columns:
        if sample_hint is None:
            raise ValueError(
                "Truth labels require a sample identifier column or must be provided via filenames."
            )
        df["sample_id"] = sample_hint

    required = {"sample_id", "event_index", "population"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Truth labels missing required columns: {', '.join(sorted(missing))}")

    return df[list(required)].copy()


def _load_truth_labels(
    truth_path: Path,
    *,
    label_column: str | None,
    sample_column: str | None,
    event_column: str | None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if truth_path.is_dir():
        for csv_path in sorted(truth_path.glob("*.csv")):
            df = pd.read_csv(csv_path)
            frames.append(
                _normalize_truth_table(
                    df,
                    sample_hint=csv_path.stem,
                    label_column=label_column,
                    sample_column=sample_column,
                    event_column=event_column,
                )
            )
    else:
        df = pd.read_csv(truth_path)
        frames.append(
            _normalize_truth_table(
                df,
                sample_hint=None,
                label_column=label_column,
                sample_column=sample_column,
                event_column=event_column,
            )
        )

    truth = pd.concat(frames, ignore_index=True)
    truth["population"] = truth["population"].astype(str)
    return truth


def _evaluate_label_predictions(
    predictions_dir: Path,
    truth_labels: pd.DataFrame,
    *,
    include_ungated: bool = True,
) -> Dict[str, float]:
    prediction_files = sorted(predictions_dir.glob("*_labels.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No label CSV files found in {predictions_dir}")

    metrics: List[PopulationMetrics] = []
    macro_values: List[float] = []
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    for pred_file in prediction_files:
        sample_id = pred_file.stem.replace("_labels", "")
        pred = pd.read_csv(pred_file)
        if "population" not in pred.columns:
            raise ValueError(f"Predictions missing population column: {pred_file}")
        if "event_index" not in pred.columns:
            pred["event_index"] = np.arange(len(pred), dtype=int)

        truth_subset = truth_labels[truth_labels["sample_id"] == sample_id]
        if truth_subset.empty:
            LOGGER.warning("No truth annotations found for sample %s", sample_id)
            continue

        merged = pd.merge(
            pred[["event_index", "population"]],
            truth_subset,
            on="event_index",
            suffixes=("_pred", "_truth"),
        )
        if merged.empty:
            LOGGER.warning("No overlapping events for sample %s", sample_id)
            continue

        pred_labels = merged["population_pred"].astype(str)
        truth_labels_series = merged["population_truth"].astype(str)
        populations = sorted(set(truth_labels_series.unique()) | set(pred_labels.unique()))
        if not include_ungated:
            populations = [p for p in populations if p != "UNGATED"]

        for population in populations:
            pred_mask = pred_labels == population
            truth_mask = truth_labels_series == population
            tp = int(np.sum(pred_mask & truth_mask))
            fp = int(np.sum(pred_mask & ~truth_mask))
            fn = int(np.sum(~pred_mask & truth_mask))
            pred_count = int(pred_mask.sum())
            true_count = int(truth_mask.sum())
            precision = _safe_divide(tp, pred_count)
            recall = _safe_divide(tp, true_count)
            f1 = _safe_divide(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
            warning = None
            if true_count < 10 or pred_count < 10:
                warning = "LOW_COUNTS"
                LOGGER.warning(
                    "Low cell count for sample %s population %s (pred=%s truth=%s)",
                    sample_id,
                    population,
                    pred_count,
                    true_count,
                )

            metrics.append(
                PopulationMetrics(
                    sample_id=sample_id,
                    population=population,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    true_count=true_count,
                    pred_count=pred_count,
                    tp=tp,
                    warning=warning,
                )
            )
            macro_values.append(f1)
            micro_tp += tp
            micro_fp += int(np.sum(pred_mask & ~truth_mask))
            micro_fn += int(np.sum(~pred_mask & truth_mask))

    if not metrics:
        raise ValueError("No evaluation metrics computed for label predictions")

    macro_f1 = float(np.mean(macro_values)) if macro_values else 0.0
    micro_precision = _safe_divide(micro_tp, micro_tp + micro_fp)
    micro_recall = _safe_divide(micro_tp, micro_tp + micro_fn)
    micro_f1 = (
        _safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    records = [m.__dict__ for m in metrics]
    eval_path = predictions_dir / "evaluation.csv"
    pd.DataFrame(records).to_csv(eval_path, index=False)
    LOGGER.info("Label evaluation written to %s", eval_path)
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
        "mode": "labels",
    }


def evaluate(
    config,
    predictions_dir: Path,
    truth_path: Path,
    *,
    mode: str = "auto",
) -> Dict[str, float]:
    mode = mode.lower()
    if mode not in {"auto", "gates", "labels"}:
        raise ValueError("Evaluation mode must be one of auto/gates/labels")

    if mode == "auto":
        if truth_path.is_dir():
            mode = "labels"
        else:
            sample = pd.read_csv(truth_path, nrows=1)
            mode = "gates" if "points" in sample.columns else "labels"

    if mode == "gates":
        return _evaluate_gate_polygons(config, predictions_dir, truth_path)

    eval_cfg = config.get("evaluation", {})
    truth_labels = _load_truth_labels(
        truth_path,
        label_column=eval_cfg.get("label_column"),
        sample_column=eval_cfg.get("sample_id_column"),
        event_column=eval_cfg.get("event_index_column"),
    )
    include_ungated = bool(eval_cfg.get("include_ungated", True))
    return _evaluate_label_predictions(
        predictions_dir,
        truth_labels,
        include_ungated=include_ungated,
    )
