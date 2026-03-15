import numpy as np

from src.grid import Grid2D
from src.laplacian import laplacian_2d_dirichlet


def test_laplacian_of_sine_modes_dirichlet():
    grid = Grid2D(nx=128, ny=96, lx=2 * np.pi, ly=2 * np.pi)
    mx, my = 3, 2
    i = np.arange(grid.nx)
    j = np.arange(grid.ny)
    mode_x = np.sin(mx * np.pi * i / (grid.nx - 1))
    mode_y = np.sin(my * np.pi * j / (grid.ny - 1))
    field = np.outer(mode_x, mode_y)
    eigenvalue = (
        -4.0 * np.sin(mx * np.pi / (2.0 * (grid.nx - 1))) ** 2 / (grid.dx * grid.dx)
        -4.0 * np.sin(my * np.pi / (2.0 * (grid.ny - 1))) ** 2 / (grid.dy * grid.dy)
    )
    expected = eigenvalue * field

    numeric = laplacian_2d_dirichlet(field, grid.dx, grid.dy)
    max_error = np.max(np.abs(numeric[1:-1, 1:-1] - expected[1:-1, 1:-1]))

    assert max_error < 1e-10
