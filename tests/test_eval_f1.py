import pytest
from pathlib import Path

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from autogate.eval_f1 import evaluate


def test_evaluate_perfect_scores(tmp_path: Path):
    target_dir = tmp_path / "target"
    out_dir = tmp_path / "out"
    target_dir.mkdir()
    out_dir.mkdir()

    data = pd.DataFrame({"A": np.linspace(0.1, 0.9, 100), "B": np.linspace(0.1, 0.9, 100)})
    (target_dir / "sample.csv").write_text(data.to_csv(index=False))

    pred_gate_csv = out_dir / "sample_gates.csv"
    gate_points = "[(0,0),(1,0),(1,1),(0,1)]"
    pd.DataFrame(
        [
            {
                "gate_id": "g1",
                "parent_id": "root",
                "type": "polygon",
                "channel_x": "A",
                "channel_y": "B",
                "points": gate_points,
            }
        ]
    ).to_csv(pred_gate_csv, index=False)

    truth_csv = tmp_path / "truth.csv"
    pd.DataFrame(
        [
            {
                "gate_id": "g1",
                "parent_id": "root",
                "type": "polygon",
                "channel_x": "A",
                "channel_y": "B",
                "points": gate_points,
                "fcs_file": "sample.csv",
            }
        ]
    ).to_csv(truth_csv, index=False)

    config = {
        "io": {
            "target_fcs_dir": str(target_dir),
        },
        "panel": {
            "channels": ["A", "B"],
            "transform": "asinh",
            "compensation": None,
            "ranges": {"A": [0, 1], "B": [0, 1]},
        },
    }

    results = evaluate(config, out_dir, truth_csv, mode="gates")
    assert results["macro_f1"] == 1.0
    assert results["micro_f1"] == 1.0


def test_label_evaluation(tmp_path: Path):
    predictions_dir = tmp_path / "pred"
    predictions_dir.mkdir()
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()

    pred_df = pd.DataFrame(
        {
            "event_index": [0, 1, 2, 3],
            "population": ["C", "C", "UNGATED", "Mono"],
        }
    )
    pred_df.to_csv(predictions_dir / "sample_labels.csv", index=False)

    truth_df = pd.DataFrame(
        {
            "sample_id": ["sample"] * 4,
            "event_index": [0, 1, 2, 3],
            "cell_type": ["C", "Mono", "UNGATED", "Mono"],
        }
    )
    truth_df.to_csv(truth_dir / "labels.csv", index=False)

    config = {
        "evaluation": {
            "mode": "labels",
            "label_column": "cell_type",
            "sample_id_column": "sample_id",
            "event_index_column": "event_index",
        }
    }

    results = evaluate(config, predictions_dir, truth_dir, mode="labels")
    assert results["macro_f1"] > 0
    assert results["mode"] == "labels"
