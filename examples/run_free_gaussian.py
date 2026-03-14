"""Run a free-particle Gaussian wave packet simulation."""

import logging

from config.simulation_parameters import SimulationParameters
from src.grid import Grid2D
from src.potential import free_potential
from src.simulation import SchrodingerSimulation2D, make_gaussian_packet


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    params = SimulationParameters()
    grid = Grid2D(params.nx, params.ny, params.lx, params.ly)
    x, y = grid.mesh

    potential = free_potential(x, y)
    psi0 = make_gaussian_packet(
        grid,
        sigma=params.sigma,
        x0=params.x0,
        y0=params.y0,
        kx=params.kx,
        ky=params.ky,
    )

    sim = SchrodingerSimulation2D(grid=grid, potential=potential, dt=params.dt)
    _, diag = sim.run(psi0=psi0, t_final=params.t_final, log_every=params.log_every)

    print(f"Final norm: {diag.norms[-1]:.8f}")
    print(f"Final energy: {diag.energies[-1]:.8f}")


if __name__ == "__main__":
    main()
