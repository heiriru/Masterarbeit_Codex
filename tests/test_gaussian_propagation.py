import numpy as np

from src.boundary import DirichletBoundary2D
from src.grid import Grid2D
from src.initial_conditions import make_gaussian_packet
from src.integrators import RK4Integrator, build_integrator
from src.models import LinearSchrodingerModel
from src.potential import free_potential
from src.simulation import SimulationRunner2D


def test_free_gaussian_propagation_particle_number_stability():
    grid = Grid2D(nx=80, ny=80, lx=20.0, ly=20.0)
    boundary = DirichletBoundary2D()
    x, y = grid.mesh
    potential = free_potential(x, y)

    model = LinearSchrodingerModel(grid=grid, potential=potential, boundary=boundary)
    sim = SimulationRunner2D(model=model, integrator=RK4Integrator(), dt=0.0005)
    psi0 = make_gaussian_packet(
        grid,
        boundary=boundary,
        sigma=1.0,
        x0=-2.0,
        y0=0.5,
        kx=1.0,
        ky=-0.5,
    )
    particle_number0 = model.particle_number(psi0)

    _, diag = sim.run(psi0=psi0, t_final=10, log_every=20)
    particle_number_final = diag.particle_numbers[-1]

    assert abs(particle_number0 - particle_number_final) < 5e-4
    assert max(abs(np.array(diag.particle_numbers) - particle_number0)) < 8e-4


def test_dirichlet_boundary_remains_zero():
    grid = Grid2D(nx=64, ny=64, lx=20.0, ly=20.0)
    boundary = DirichletBoundary2D()
    x, y = grid.mesh
    potential = free_potential(x, y)

    model = LinearSchrodingerModel(grid=grid, potential=potential, boundary=boundary)
    sim = SimulationRunner2D(model=model, integrator=RK4Integrator(), dt=0.001)
    psi0 = make_gaussian_packet(grid, boundary=boundary, sigma=1.2, x0=0.0, y0=0.0, kx=0.0, ky=0.0)

    psi_final, _ = sim.run(psi0=psi0, t_final=10, log_every=10)

    assert np.allclose(psi_final[0, :], 0.0)
    assert np.allclose(psi_final[-1, :], 0.0)
    assert np.allclose(psi_final[:, 0], 0.0)
    assert np.allclose(psi_final[:, -1], 0.0)


def test_rk4_integrator_matches_manual_single_step():
    grid = Grid2D(nx=32, ny=32, lx=8.0, ly=8.0)
    boundary = DirichletBoundary2D()
    potential = free_potential(*grid.mesh)
    model = LinearSchrodingerModel(grid=grid, potential=potential, boundary=boundary)
    integrator = RK4Integrator()
    psi0 = make_gaussian_packet(grid, boundary=boundary, sigma=0.8, x0=0.2, y0=-0.1)

    dt = 0.002
    stepped = integrator.step(model, psi0, t=0.0, dt=dt)

    k1 = model.rhs(psi0, 0.0)
    k2 = model.rhs(psi0 + 0.5 * dt * k1, 0.5 * dt)
    k3 = model.rhs(psi0 + 0.5 * dt * k2, 0.5 * dt)
    k4 = model.rhs(psi0 + dt * k3, dt)
    manual = model.apply_boundary(psi0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

    assert np.allclose(stepped, manual)


def test_factory_based_simulation_composition_smoke():
    grid = Grid2D(nx=24, ny=24, lx=6.0, ly=6.0)
    boundary = DirichletBoundary2D()
    potential = free_potential(*grid.mesh)
    model = LinearSchrodingerModel(grid=grid, potential=potential, boundary=boundary)
    integrator = build_integrator("rk4")
    psi0 = make_gaussian_packet(grid, boundary=boundary, sigma=0.7, target_particle_number=1.5)
    sim = SimulationRunner2D(model=model, integrator=integrator, dt=0.001)

    _, diag = sim.run(psi0=psi0, t_final=0.01, log_every=1, snapshot_every=1)

    assert diag.particle_numbers
    assert diag.snapshots
    assert abs(diag.particle_numbers[0] - 1.5) < 1e-6
