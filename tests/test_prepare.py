from pathlib import Path

import numpy as np
import pandas as pd

from autogate.prepare import build_gates_from_annotations


def _write_annotation(root: Path, sample: str, file: str, points: np.ndarray, labels: list[str], columns: tuple[str, str]):
    sample_dir = root / sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(points, columns=columns)
    df["cell_type"] = labels
    df.to_csv(sample_dir / file, index=False)


def test_build_gates_from_annotations(tmp_path: Path):
    csv_root = tmp_path / "CSV" / "train"
    spec_path = tmp_path / "spec.yaml"
    csv_root.mkdir(parents=True)

    rng = np.random.default_rng(42)
    points_h = rng.normal(loc=[0.7, 0.8], scale=[0.02, 0.02], size=(30, 2))
    labels_h = ["H"] * len(points_h)
    points_bg = rng.normal(loc=[0.2, 0.2], scale=[0.05, 0.05], size=(30, 2))
    labels_bg = [""] * len(points_bg)
    combined = np.vstack([points_h, points_bg])
    labels = labels_h + labels_bg
    _write_annotation(
        csv_root,
        "001",
        "annotation_CD34 ECD-A_CD45 APC-A750-A.csv",
        combined,
        labels,
        ("CD45 APC-A750-A", "CD34 ECD-A"),
    )

    points_g = rng.normal(loc=[0.65, 0.4], scale=[0.03, 0.03], size=(35, 2))
    labels_g = ["G"] * len(points_g)
    _write_annotation(
        csv_root,
        "001",
        "annotation_CD117 PE-A_CD45 APC-A750-A.csv",
        points_g,
        labels_g,
        ("CD45 APC-A750-A", "CD117 PE-A"),
    )

    points_multi = rng.normal(loc=[0.55, 0.6], scale=[0.04, 0.04], size=(40, 2))
    labels_multi = ["C"] * len(points_multi)
    _write_annotation(
        csv_root,
        "001",
        "annotation_SSC-A_CD45 APC-A750-A.csv",
        points_multi,
        labels_multi,
        ("CD45 APC-A750-A", "SSC-A"),
    )

    spec_path.write_text(
        """
coordinate_systems:
  - file: "annotation_CD34 ECD-A_CD45 APC-A750-A.csv"
    channel_x: "CD45 APC-A750-A"
    channel_y: "CD34 ECD-A"
    populations: ["H"]
  - file: "annotation_CD117 PE-A_CD45 APC-A750-A.csv"
    channel_x: "CD45 APC-A750-A"
    channel_y: "CD117 PE-A"
    populations: ["G"]
  - file: "annotation_SSC-A_CD45 APC-A750-A.csv"
    channel_x: "CD45 APC-A750-A"
    channel_y: "SSC-A"
    populations: ["C"]
label_column: cell_type
parent_id: root
"""
    )

    output_csv = tmp_path / "gates.csv"
    summary = build_gates_from_annotations(
        csv_root=csv_root,
        spec_path=spec_path,
        output_csv=output_csv,
        fcs_suffix=".fcs",
        expand_frac=0.02,
        min_points=5,
    )

    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    assert not df.empty
    assert {"gate_id", "points", "population", "fcs_file"}.issubset(df.columns)
    assert summary["gates_csv"] == str(output_csv)
    assert "CD45 APC-A750-A" in summary["channels"]
    assert "H" in summary["populations"]
    # ranges should cover the generated data roughly between 0 and 1
    cd45_range = summary["ranges"]["CD45 APC-A750-A"]
    assert cd45_range[0] < 0.3
    assert cd45_range[1] > 0.6
