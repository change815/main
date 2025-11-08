import pytest
import pytest
from pathlib import Path

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
sitk = pytest.importorskip("SimpleITK")

from autogate.gates import Gate, assign_populations, read_gates, transform_gate
from autogate.imaging import DensityImage


def test_read_gates_skips_invalid(tmp_path: Path):
    csv_path = tmp_path / "gates.csv"
    df = pd.DataFrame(
        [
            {
                "gate_id": "g1",
                "parent_id": "root",
                "type": "polygon",
                "channel_x": "A",
                "channel_y": "B",
                "points": "[(0,0),(1,0),(1,1),(0,1)]",
            },
            {
                "gate_id": "g2",
                "parent_id": "root",
                "type": "polygon",
                "channel_x": "A",
                "channel_y": "B",
                "points": "invalid",
            },
        ]
    )
    df.to_csv(csv_path, index=False)
    gates = read_gates(csv_path)
    assert len(gates) == 1
    assert gates[0].gate_id == "g1"


def test_transform_gate_identity():
    gate = Gate(
        gate_id="g1",
        parent_id="root",
        gate_type="polygon",
        channel_x="A",
        channel_y="B",
        points=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
    )
    density = DensityImage(image=np.zeros((4, 4), dtype=np.uint8), bbox=(0, 1, 0, 1), bins=4)
    transform = sitk.Transform(2, sitk.sitkIdentity)
    transformed = transform_gate(gate, transform, density, density)
    assert transformed.points == gate.points


def test_assign_populations_hierarchy():
    df = pd.DataFrame(
        {
            "A": [0.1, 0.5, 0.8, 1.5],
            "B": [0.1, 0.5, 0.8, 1.5],
        }
    )
    gates = [
        Gate(
            gate_id="root_gate",
            parent_id="root",
            gate_type="polygon",
            channel_x="A",
            channel_y="B",
            points=[(0, 0), (1, 0), (1, 1), (0, 1)],
        ),
        Gate(
            gate_id="child_gate",
            parent_id="root_gate",
            gate_type="polygon",
            channel_x="A",
            channel_y="B",
            points=[(0.4, 0.4), (0.9, 0.4), (0.9, 0.9), (0.4, 0.9)],
        ),
    ]
    assignments = assign_populations(df, gates)
    assert list(assignments["population"]) == ["root_gate", "child_gate", "child_gate", "UNGATED"]
    assert assignments.loc[1, "depth"] > assignments.loc[0, "depth"]
