# 2D Schrodinger Equation Simulation (Python)

This project provides a modular scientific simulation for the time-dependent 2D Schrodinger equation:

\[
i\,\partial_t \psi(x,y,t) = \left(-\frac{1}{2}\nabla^2 + V(x,y)\right)\psi(x,y,t)
\]

The implementation uses:
- A uniform 2D grid
- Second-order finite differences for the Laplacian
- Fourth-order Runge-Kutta (RK4) time integration
- Runtime logging and diagnostics (norm, energy, and wave-packet center)
- Matplotlib-based visual summaries of the simulated state

## Project structure

```text
project_root/
├── README.md
├── requirements.txt
├── config/
│   └── simulation_parameters.py
├── src/
│   ├── grid.py
│   ├── laplacian.py
│   ├── potential.py
│   ├── olver.py
│   └── simulation.py
├── tests/
│   ├── test_laplacian.py
│   └── test_gaussian_propagation.py
└── examples/
    └── run_free_gaussian.py
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
- `Results/free_gaussian_overview_diagnostics.png`: norm, energy, and wave-packet center trajectory
- `Results/free_gaussian_overview_potential.png`: standalone plot of the potential landscape
- `Results/free_gaussian_density.gif`: animated density evolution
- `Results/free_gaussian_phase.gif`: animated phase evolution

## Running tests

Execute the full test suite:

```bash
pytest -q
```

### What the tests verify

- **Laplacian correctness:** the finite-difference Laplacian is compared against an analytical result for trigonometric modes.
- **Gaussian propagation stability:** under free evolution, the RK4 integration approximately conserves total probability (wavefunction norm).

## Notes

- Units are dimensionless with \(\hbar = m = 1\).
- The Laplacian uses homogeneous Dirichlet boundary conditions, so \(\psi = 0\) on the boundary.
- Additional potentials can be added in `src/potential.py`.
