# 2D Schrodinger Equation Simulation (Python)

This project provides a modular scientific simulation for the time-dependent 2D Schrodinger equation:

\[
i\,\partial_t \psi(x,y,t) = \left(-\frac{1}{2}\nabla^2 + V(x,y)\right)\psi(x,y,t)
\]

The implementation uses:
- A uniform 2D grid
- Second-order finite differences for the Laplacian
- A layered model/integrator architecture
- Fourth-order Runge-Kutta (RK4) as the default time integrator
- Runtime logging and diagnostics (particle number, energy, and wave-packet center)
- Matplotlib-based visual summaries of the simulated state

## Project structure

```text
project_root/
|-- README.md
|-- requirements.txt
|-- config/
|   `-- simulation_parameters.py
|-- src/
|   |-- boundary.py
|   |-- diagnostics.py
|   |-- grid.py
|   |-- initial_conditions.py
|   |-- integrators.py
|   |-- models.py
|   |-- operators.py
|   |-- potential.py
|   |-- simulation.py
|   `-- visualization.py
|-- tests/
|   |-- test_gaussian_propagation.py
|   |-- test_laplacian.py
|   `-- test_visualization.py
`-- examples/
    `-- run_free_gaussian.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running a simulation

Run the free Gaussian example:

```bash
python examples/run_free_gaussian.py
```

You should see logged diagnostics in the terminal and generated plots inside `Results/`:

- `Results/free_gaussian_overview.png`: density and phase for selected times
- `Results/free_gaussian_overview_diagnostics.png`: particle-number drift, energy drift, and wave-packet center trajectory
- `Results/free_gaussian_overview_potential.png`: standalone plot of the potential landscape
- `Results/free_gaussian_density.gif`: animated density evolution
- `Results/free_gaussian_phase.gif`: animated phase evolution

## Running tests

Execute the full test suite:

```bash
pytest -q
```

### What the tests verify

- **Operator correctness:** the finite-difference Dirichlet Laplacian is checked against discrete sine eigenmodes.
- **Particle-number stability:** under free evolution, the RK4 integration approximately conserves the initial particle number.
- **Boundary preservation:** the Dirichlet boundary remains zero during time evolution.
- **Visualization smoke test:** diagnostics drifts are plotted in a way that avoids confusing axis offsets.

## Notes

- Units are dimensionless with \(\hbar = m = 1\).
- The Laplacian uses homogeneous Dirichlet boundary conditions, so \(\psi = 0\) on the boundary.
- The current physical model is a linear Schrodinger equation with Dirichlet boundaries.
- The codebase is prepared for later Gross-Pitaevskii extensions and a future Crank-Nicolson integrator.
- Additional potentials can be added in `src/potential.py`.
