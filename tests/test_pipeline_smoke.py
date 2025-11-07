import pytest
import pytest
from pathlib import Path

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("SimpleITK")

from autogate.pipeline import AutoGatePipeline


def _create_gaussian_cloud(path: Path, mean, cov, size=500):
    rng = np.random.default_rng(123)
    data = rng.multivariate_normal(mean, cov, size=size)
    df = pd.DataFrame(data, columns=["A", "B"])
    df.to_csv(path, index=False)


def test_pipeline_smoke(tmp_path: Path):
    train_dir = tmp_path / "train"
    target_dir = tmp_path / "target"
    out_dir = tmp_path / "out"
    train_dir.mkdir()
    target_dir.mkdir()
    out_dir.mkdir()

    _create_gaussian_cloud(train_dir / "train1.csv", mean=[0, 0], cov=[[0.2, 0], [0, 0.2]])
    _create_gaussian_cloud(train_dir / "train2.csv", mean=[1, 1], cov=[[0.2, 0], [0, 0.2]])
    _create_gaussian_cloud(target_dir / "target1.csv", mean=[0.1, 0.1], cov=[[0.3, 0], [0, 0.3]])

    gates_df = pd.DataFrame(
        [
            {
                "gate_id": "g1",
                "parent_id": "root",
                "type": "polygon",
                "channel_x": "A",
                "channel_y": "B",
                "points": "[(-1,-1),(1,-1),(1,1),(-1,1)]",
                "fcs_file": "train1.csv",
            },
            {
                "gate_id": "g2",
                "parent_id": "root",
                "type": "polygon",
                "channel_x": "A",
                "channel_y": "B",
                "points": "[(0,0),(2,0),(2,2),(0,2)]",
                "fcs_file": "train2.csv",
            },
        ]
    )
    gates_csv = tmp_path / "gates.csv"
    gates_df.to_csv(gates_csv, index=False)

    config = {
        "io": {
            "train_fcs_dir": str(train_dir),
            "train_gates_csv": str(gates_csv),
            "target_fcs_dir": str(target_dir),
            "out_dir": str(out_dir),
        },
        "panel": {
            "channels": ["A", "B"],
            "transform": "asinh",
            "compensation": None,
            "ranges": {"A": [-3, 3], "B": [-3, 3]},
        },
        "imaging": {
            "bins": 64,
            "smooth_sigma": 0.5,
        },
        "registration": {
            "grid_spacing": 16,
            "levels": 1,
            "iterations": 5,
        },
        "selection": {
            "metric": "l2",
        },
    }

    pipeline = AutoGatePipeline(config)
    summary = pipeline.run()
    assert summary["targets"], "Expect targets summary"
    output_files = list(out_dir.glob("*_gates.csv"))
    assert output_files, "Expected gate CSV outputs"
