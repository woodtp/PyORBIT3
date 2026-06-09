import numpy as np
import pytest
from orbit.core.spacecharge import Grid3D

pytestmark = pytest.mark.skipif(
    not hasattr(Grid3D, "to_numpy"),
    reason="PyORBIT compiled without numpy support (PyORBIT_EXPERIMENTAL_WITH_NUMPY not set)",
)


def test_to_numpy_shape():
    nX, nY, nZ = 4, 5, 6
    g = Grid3D(nX, nY, nZ)
    arr = g.to_numpy()
    assert arr.shape == (nZ, nX, nY)


def test_to_numpy_dtype():
    g = Grid3D(4, 5, 6)
    arr = g.to_numpy()
    assert arr.dtype == np.float64


def test_to_numpy_round_trip():
    nX, nY, nZ = 4, 5, 6
    g = Grid3D(nX, nY, nZ)
    g.setValue(1.5, 1, 2, 3)
    g.setValue(2.5, 3, 4, 2)
    arr = g.to_numpy()
    g2 = Grid3D(nX, nY, nZ)
    g2.from_numpy(arr)
    assert g2.getValueOnGrid(1, 2, 3) == 1.5
    assert g2.getValueOnGrid(3, 4, 2) == 2.5


def test_from_numpy_populates_grid():
    nX, nY, nZ = 3, 4, 5
    arr = np.arange(nZ * nX * nY, dtype=np.float64).reshape(nZ, nX, nY)
    g = Grid3D(nX, nY, nZ)
    g.from_numpy(arr)
    for iz in range(nZ):
        for ix in range(nX):
            for iy in range(nY):
                assert g.getValueOnGrid(ix, iy, iz) == arr[iz, ix, iy]


def test_from_numpy_wrong_ndim():
    g = Grid3D(4, 5, 6)
    arr = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="must be 3-dimensional"):
        g.from_numpy(arr)


def test_from_numpy_wrong_shape():
    g = Grid3D(4, 5, 6)
    arr = np.zeros((6, 4, 99), dtype=np.float64)
    with pytest.raises(ValueError, match="shape mismatch"):
        g.from_numpy(arr)


def test_to_numpy_returns_copy():
    nX, nY, nZ = 3, 4, 5
    g = Grid3D(nX, nY, nZ)
    g.setValue(42.0, 0, 0, 0)
    arr = g.to_numpy()
    arr[0, 0, 0] = 99.0
    assert g.getValueOnGrid(0, 0, 0) == 42.0


def test_from_numpy_non_contiguous():
    nX, nY, nZ = 3, 4, 5
    full = np.arange(2 * nZ * nX * nY, dtype=np.float64).reshape(2 * nZ, nX, nY)
    arr = full[::2]
    assert not arr.flags.c_contiguous
    g = Grid3D(nX, nY, nZ)
    g.from_numpy(arr)
    for iz in range(nZ):
        for ix in range(nX):
            for iy in range(nY):
                assert g.getValueOnGrid(ix, iy, iz) == arr[iz, ix, iy]
