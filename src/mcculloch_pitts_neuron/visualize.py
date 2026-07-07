"""Visualize McCulloch-Pitts logic gates using matplotlib.

Plots the decision regions for AND, OR, and NOT gates
as 2D grids showing where the neuron fires (1) or stays silent (0).
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.mcculloch_pitts_neuron.neuron import Neuron


def _build_and() -> Neuron:
    n = Neuron(threshold=2)
    n.add_excitatory(1)
    n.add_excitatory(1)
    return n


def _build_or() -> Neuron:
    n = Neuron(threshold=1)
    n.add_excitatory(1)
    n.add_excitatory(1)
    return n


def _build_not() -> Neuron:
    n = Neuron(threshold=1)
    n.add_excitatory(1)  # bias
    n.add_inhibitory()
    return n


def _gate_grid_2d(neuron: Neuron) -> np.ndarray:
    """Compute output for all 2-input combinations (00, 01, 10, 11)."""
    grid = np.zeros((2, 2), dtype=int)
    for x in range(2):
        for y in range(2):
            grid[x, y] = neuron.fire([x, y])
    return grid


def _gate_grid_not(neuron: Neuron) -> np.ndarray:
    """Compute NOT gate output: bias=1, input varies."""
    grid = np.zeros((1, 2), dtype=int)
    for x in range(2):
        grid[0, x] = neuron.fire([1, x])
    return grid


def plot_gates(save_path: str | None = None) -> None:
    """Plot AND, OR, NOT gates side by side.

    Args:
        save_path: If given, save the figure to this path instead of showing.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    cmap = mcolors.ListedColormap(["#2b2b2b", "#4ec9b0"])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # AND gate
    and_grid = _gate_grid_2d(_build_and())
    axes[0].imshow(and_grid, cmap=cmap, norm=norm, extent=[-0.5, 1.5, -0.5, 1.5])
    axes[0].set_title("AND gate (θ=2)", fontsize=13)
    axes[0].set_xlabel("Input x₂")
    axes[0].set_ylabel("Input x₁")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    for x in range(2):
        for y in range(2):
            axes[0].text(y, x, str(and_grid[x, y]),
                         ha="center", va="center", fontsize=16,
                         color="white" if and_grid[x, y] == 0 else "black")

    # OR gate
    or_grid = _gate_grid_2d(_build_or())
    axes[1].imshow(or_grid, cmap=cmap, norm=norm, extent=[-0.5, 1.5, -0.5, 1.5])
    axes[1].set_title("OR gate (θ=1)", fontsize=13)
    axes[1].set_xlabel("Input x₂")
    axes[1].set_ylabel("Input x₁")
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    for x in range(2):
        for y in range(2):
            axes[1].text(y, x, str(or_grid[x, y]),
                         ha="center", va="center", fontsize=16,
                         color="white" if or_grid[x, y] == 0 else "black")

    # NOT gate
    not_grid = _gate_grid_not(_build_not())
    axes[2].imshow(not_grid, cmap=cmap, norm=norm, extent=[-0.5, 1.5, -0.5, 0.5])
    axes[2].set_title("NOT gate (bias + inhibitory)", fontsize=13)
    axes[2].set_xlabel("Input x")
    axes[2].set_ylabel("Bias=1")
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0])
    for x in range(2):
        axes[2].text(x, 0, str(not_grid[0, x]),
                     ha="center", va="center", fontsize=16,
                     color="white" if not_grid[0, x] == 0 else "black")

    fig.suptitle("McCulloch-Pitts Neuron — Logic Gates (1943)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    plot_gates()
