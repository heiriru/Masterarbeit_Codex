"""Grouped simulation parameters for Schrödinger-type runs."""

from dataclasses import dataclass, field


@dataclass
class GridParameters:
    nx: int = 128
    ny: int = 128
    lx: float = 10.0
    ly: float = 10.0


@dataclass
class InitialConditionParameters:
    kind: str = "gaussian"
    sigma: float = 1.0
    x0: float = 0.0
    y0: float = 0.0
    kx: float = 0.0
    ky: float = 0.0
    target_particle_number: float = 1.0


@dataclass
class ModelParameters:
    potential_type: str = "free"
    omega: float = 0.15
    boundary_type: str = "dirichlet"


@dataclass
class SolverParameters:
    dt: float = 0.001
    t_final: float = 10.0
    integrator: str = "rk4"
    log_every: int = 5
    snapshot_every: int | None = None


@dataclass
class OutputParameters:
    results_dir: str = "Results"
    save_png: bool = True
    save_gif: bool = True
    gif_fps: int = 12


@dataclass
class SimulationParameters:
    grid: GridParameters = field(default_factory=GridParameters)
    initial_condition: InitialConditionParameters = field(default_factory=InitialConditionParameters)
    model: ModelParameters = field(default_factory=ModelParameters)
    solver: SolverParameters = field(default_factory=SolverParameters)
    output: OutputParameters = field(default_factory=OutputParameters)
