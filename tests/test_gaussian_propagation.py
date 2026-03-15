import numpy as np

from src.grid import Grid2D
from src.potential import free_potential
from src.simulation import SchrodingerSimulation2D, make_gaussian_packet


def test_free_gaussian_propagation_norm_stability():
    grid = Grid2D(nx=80, ny=80, lx=20.0, ly=20.0)
    x, y = grid.mesh
    potential = free_potential(x, y)

    sim = SchrodingerSimulation2D(grid=grid, potential=potential, dt=0.0005)
    psi0 = make_gaussian_packet(grid, sigma=1.0, x0=-2.0, y0=0.5, kx=1.0, ky=-0.5)
    norm0 = sim.norm(psi0)

    _, diag = sim.run(psi0=psi0, t_final=0.02, log_every=20)
    norm_final = diag.norms[-1]

    assert abs(norm0 - norm_final) < 5e-4
    assert max(abs(np.array(diag.norms) - norm0)) < 8e-4


def test_dirichlet_boundary_remains_zero():
    grid = Grid2D(nx=64, ny=64, lx=20.0, ly=20.0)
    x, y = grid.mesh
    potential = free_potential(x, y)

    sim = SchrodingerSimulation2D(grid=grid, potential=potential, dt=0.001)
    psi0 = make_gaussian_packet(grid, sigma=1.2, x0=0.0, y0=0.0, kx=0.0, ky=0.0)

    psi_final, _ = sim.run(psi0=psi0, t_final=0.05, log_every=10)

    assert np.allclose(psi_final[0, :], 0.0)
    assert np.allclose(psi_final[-1, :], 0.0)
    assert np.allclose(psi_final[:, 0], 0.0)
    assert np.allclose(psi_final[:, -1], 0.0)
