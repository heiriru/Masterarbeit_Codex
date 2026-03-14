import numpy as np

from src.grid import Grid2D
from src.laplacian import laplacian_2d_periodic


def test_laplacian_of_sine_modes():
    grid = Grid2D(nx=128, ny=96, lx=2 * np.pi, ly=2 * np.pi)
    x, y = grid.mesh

    mx, my = 3, 2
    field = np.sin(mx * x) * np.cos(my * y)
    expected = -(mx**2 + my**2) * field

    numeric = laplacian_2d_periodic(field, grid.dx, grid.dy)
    max_error = np.max(np.abs(numeric - expected))

    assert max_error < 1e-2
