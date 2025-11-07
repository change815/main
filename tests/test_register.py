import pytest

np = pytest.importorskip("numpy")
sitk = pytest.importorskip("SimpleITK")

from autogate.register import register_bspline


def test_register_bspline_returns_transform():
    x = np.linspace(-1, 1, 32)
    xx, yy = np.meshgrid(x, x)
    img = np.exp(-(xx**2 + yy**2)) * 255
    fixed = img.astype(np.float32)
    moving = fixed.copy()

    transform = register_bspline(fixed, moving, grid_spacing=16, levels=1, iterations=5)
    assert isinstance(transform, sitk.Transform)
