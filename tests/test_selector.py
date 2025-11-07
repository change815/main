import pytest

np = pytest.importorskip("numpy")

from autogate.selector import l2_distance, select_best


def test_l2_distance_and_selection():
    target = np.zeros((4, 4), dtype=np.uint8)
    training = {
        "t1": np.zeros((4, 4), dtype=np.uint8),
        "t2": np.ones((4, 4), dtype=np.uint8) * 10,
    }
    assert l2_distance(target, training["t1"]) == 0.0
    best_id, score = select_best(training, target)
    assert best_id == "t1"
    assert score == 0.0
