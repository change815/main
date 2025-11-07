import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from autogate.imaging import compute_density_image


def test_compute_density_image_basic():
    rng = np.random.default_rng(42)
    x = rng.normal(loc=2.0, scale=0.5, size=1000)
    y = rng.normal(loc=-1.0, scale=0.3, size=1000)
    df = pd.DataFrame({"A": x, "B": y})

    density = compute_density_image(df, "A", "B", bins=64, smooth_sigma=0.5)
    assert density.image.shape == (64, 64)
    assert density.image.dtype == np.uint8
    assert density.bbox[0] < density.bbox[1]
    assert density.bbox[2] < density.bbox[3]
