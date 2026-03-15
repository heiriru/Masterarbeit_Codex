"""Visualization utilities for 2D Schrodinger simulations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from src.grid import Grid2D
from src.simulation import Diagnostics


def _extent(grid: Grid2D) -> list[float]:
    return [-grid.ly / 2, grid.ly / 2, -grid.lx / 2, grid.lx / 2]


def plot_simulation_summary(
    grid: Grid2D,
    potential: np.ndarray,
    diagnostics: Diagnostics,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Create a compact summary of state snapshots and simulation diagnostics."""

    if not diagnostics.snapshots:
        raise ValueError("No snapshots recorded. Run the simulation with snapshot_every set.")

    snapshot_indices = np.linspace(0, len(diagnostics.snapshots) - 1, num=min(3, len(diagnostics.snapshots)), dtype=int)
    selected_states = [diagnostics.snapshots[index] for index in snapshot_indices]
    selected_times = [diagnostics.snapshot_times[index] for index in snapshot_indices]

    density_max = max(float(np.max(np.abs(state) ** 2)) for state in selected_states)
    density_norm = Normalize(vmin=0.0, vmax=density_max if density_max > 0.0 else 1.0)
    extent = _extent(grid)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    fig.suptitle("2D Schrodinger Simulation Overview", fontsize=16)

    for column, (state, time) in enumerate(zip(selected_states, selected_times)):
        density = np.abs(state) ** 2
        phase = np.angle(state)

        density_ax = axes[0, column]
        density_im = density_ax.imshow(
            density.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="magma",
            norm=density_norm,
        )
        density_ax.set_title(f"Density |psi|^2 at t={time:.3f}")
        density_ax.set_xlabel("y")
        density_ax.set_ylabel("x")
        fig.colorbar(density_im, ax=density_ax, fraction=0.046, pad=0.04)

        phase_ax = axes[1, column]
        phase_im = phase_ax.imshow(
            phase.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )
        phase_ax.set_title(f"Phase arg(psi) at t={time:.3f}")
        phase_ax.set_xlabel("y")
        phase_ax.set_ylabel("x")
        fig.colorbar(phase_im, ax=phase_ax, fraction=0.046, pad=0.04)

    diagnostics_fig = plt.figure(figsize=(15, 4.5), constrained_layout=True)
    diag_axes = diagnostics_fig.subplots(1, 3)
    diagnostics_fig.suptitle("Simulation Diagnostics", fontsize=16)

    diag_axes[0].plot(diagnostics.times, diagnostics.norms, color="tab:blue", linewidth=2)
    diag_axes[0].set_title("Norm conservation")
    diag_axes[0].set_xlabel("time")
    diag_axes[0].set_ylabel("norm")
    diag_axes[0].grid(True, alpha=0.3)

    diag_axes[1].plot(diagnostics.times, diagnostics.energies, color="tab:green", linewidth=2)
    diag_axes[1].set_title("Energy evolution")
    diag_axes[1].set_xlabel("time")
    diag_axes[1].set_ylabel("energy")
    diag_axes[1].grid(True, alpha=0.3)

    diag_axes[2].plot(diagnostics.x_expectation, diagnostics.y_expectation, color="tab:red", linewidth=2)
    diag_axes[2].scatter(
        diagnostics.x_expectation[0],
        diagnostics.y_expectation[0],
        color="tab:orange",
        label="start",
        zorder=3,
    )
    diag_axes[2].scatter(
        diagnostics.x_expectation[-1],
        diagnostics.y_expectation[-1],
        color="tab:purple",
        label="end",
        zorder=3,
    )
    diag_axes[2].set_title("Wave-packet center trajectory")
    diag_axes[2].set_xlabel("<x>")
    diag_axes[2].set_ylabel("<y>")
    diag_axes[2].grid(True, alpha=0.3)
    diag_axes[2].legend()

    potential_fig, potential_ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    potential_fig.suptitle("Potential Landscape", fontsize=16)
    potential_im = potential_ax.imshow(
        potential.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="cividis",
    )
    potential_ax.set_xlabel("y")
    potential_ax.set_ylabel("x")
    potential_ax.set_title("Potential V(x, y)")
    potential_fig.colorbar(potential_im, ax=potential_ax, fraction=0.046, pad=0.04)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        diagnostics_fig.savefig(output_path.with_stem(f"{output_path.stem}_diagnostics"), dpi=160, bbox_inches="tight")
        potential_fig.savefig(output_path.with_stem(f"{output_path.stem}_potential"), dpi=160, bbox_inches="tight")

    return fig, diagnostics_fig, potential_fig


def create_density_gif(
    grid: Grid2D,
    diagnostics: Diagnostics,
    output_path: str | Path,
    fps: int = 12,
) -> Path:
    """Create an animated GIF of the density evolution."""

    return _create_state_gif(
        grid=grid,
        diagnostics=diagnostics,
        output_path=output_path,
        quantity="density",
        fps=fps,
    )


def create_phase_gif(
    grid: Grid2D,
    diagnostics: Diagnostics,
    output_path: str | Path,
    fps: int = 12,
) -> Path:
    """Create an animated GIF of the phase evolution."""

    return _create_state_gif(
        grid=grid,
        diagnostics=diagnostics,
        output_path=output_path,
        quantity="phase",
        fps=fps,
    )


def _create_state_gif(
    grid: Grid2D,
    diagnostics: Diagnostics,
    output_path: str | Path,
    quantity: str,
    fps: int,
) -> Path:
    if not diagnostics.snapshots:
        raise ValueError("No snapshots recorded. Run the simulation with snapshot_every set.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extent = _extent(grid)

    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    ax.set_xlabel("y")
    ax.set_ylabel("x")

    if quantity == "density":
        frames = [np.abs(state) ** 2 for state in diagnostics.snapshots]
        vmax = max(float(np.max(frame)) for frame in frames)
        image = ax.imshow(
            frames[0].T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="magma",
            norm=Normalize(vmin=0.0, vmax=vmax if vmax > 0.0 else 1.0),
        )
        ax.set_title(f"Density |psi|^2 at t={diagnostics.snapshot_times[0]:.3f}")
    elif quantity == "phase":
        frames = [np.angle(state) for state in diagnostics.snapshots]
        image = ax.imshow(
            frames[0].T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax.set_title(f"Phase arg(psi) at t={diagnostics.snapshot_times[0]:.3f}")
    else:
        raise ValueError(f"Unsupported quantity: {quantity}")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    def update(frame_index: int) -> tuple[plt.AxesImage]:
        image.set_data(frames[frame_index].T)
        time = diagnostics.snapshot_times[frame_index]
        if quantity == "density":
            ax.set_title(f"Density |psi|^2 at t={time:.3f}")
        else:
            ax.set_title(f"Phase arg(psi) at t={time:.3f}")
        return (image,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=max(1, int(round(1000 / fps))),
        blit=False,
    )
    ani.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return output_path
