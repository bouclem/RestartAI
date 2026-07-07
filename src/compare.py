"""Compare neuron variants by number of computable Boolean functions.

Metric: how many of the 16 two-input Boolean functions can a single
neuron compute?

For the MP neuron, the output is already binary — it directly computes
a Boolean function.

For the multi-threshold neuron, the output is graded (0..n). We
interpret "output == k" as firing (1) and all other levels as silent
(0). This lets the neuron compute non-monotonic functions like XOR
that a single MP neuron cannot.

The comparison shows each variant as a percentage of the base (MP
neuron = 100%), with the delta vs the parent variant.
"""

from itertools import combinations, permutations, product
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.mcculloch_pitts_neuron.neuron import Neuron
from src.multi_threshold_neuron.neuron import MultiThresholdNeuron
from src.adaptive_threshold_neuron.neuron import AdaptiveThresholdNeuron

# All 16 Boolean functions of 2 variables.
# Each function is a tuple of outputs for inputs (0,0), (0,1), (1,0), (1,1).
ALL_FUNCTIONS: Set[Tuple[int, ...]] = set(product([0, 1], repeat=4))
TOTAL_FUNCTIONS = len(ALL_FUNCTIONS)  # 16

INPUT_PAIRS = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Input roles: -1 = inhibitory, 0 = not connected, 1..3 = excitatory weight
MAX_WEIGHT = 3
ROLES = list(range(-1, MAX_WEIGHT + 1))
BIAS_OPTIONS = [0, 1]  # 0 = no bias, 1 = bias (always-on excitatory)


def _build_inputs(
    bias: int, x1_role: int, x2_role: int, x1: int, x2: int
) -> List[int]:
    """Build the input list for a given (x1, x2) combo."""
    exc: List[int] = []
    inh: List[int] = []

    if bias > 0:
        exc.append(1)
    if x1_role > 0:
        exc.append(x1)
    if x2_role > 0:
        exc.append(x2)
    if x1_role == -1:
        inh.append(x1)
    if x2_role == -1:
        inh.append(x2)

    return exc + inh


def enumerate_mp_functions() -> Set[Tuple[int, ...]]:
    """Brute-force all Boolean functions a single MP neuron can compute."""
    computable: Set[Tuple[int, ...]] = set()

    for bias in BIAS_OPTIONS:
        for x1_role in ROLES:
            for x2_role in ROLES:
                for theta in range(7):
                    try:
                        n = Neuron(threshold=theta)
                        if bias > 0:
                            n.add_excitatory(bias)
                        if x1_role > 0:
                            n.add_excitatory(x1_role)
                        if x2_role > 0:
                            n.add_excitatory(x2_role)
                        if x1_role == -1:
                            n.add_inhibitory()
                        if x2_role == -1:
                            n.add_inhibitory()

                        outputs = tuple(
                            n.fire(_build_inputs(bias, x1_role, x2_role, x1, x2))
                            for x1, x2 in INPUT_PAIRS
                        )
                        computable.add(outputs)
                    except (ValueError, IndexError):
                        continue

    return computable


def enumerate_mt_functions() -> Set[Tuple[int, ...]]:
    """Brute-force all Boolean functions a single multi-threshold neuron can compute.

    Uses "output == k" as the firing interpretation.
    """
    computable: Set[Tuple[int, ...]] = set()
    threshold_options = list(range(7))

    for bias in BIAS_OPTIONS:
        for x1_role in ROLES:
            for x2_role in ROLES:
                for num_t in range(1, 5):
                    for thresholds in combinations(threshold_options, num_t):
                        try:
                            n = MultiThresholdNeuron(list(thresholds))
                            if bias > 0:
                                n.add_excitatory(bias)
                            if x1_role > 0:
                                n.add_excitatory(x1_role)
                            if x2_role > 0:
                                n.add_excitatory(x2_role)
                            if x1_role == -1:
                                n.add_inhibitory()
                            if x2_role == -1:
                                n.add_inhibitory()

                            levels = [
                                n.fire(_build_inputs(bias, x1_role, x2_role, x1, x2))
                                for x1, x2 in INPUT_PAIRS
                            ]

                            # Each output level k → a Boolean function
                            for k in range(num_t + 1):
                                outputs = tuple(
                                    1 if level == k else 0 for level in levels
                                )
                                computable.add(outputs)
                        except (ValueError, IndexError):
                            continue

    return computable


def enumerate_at_functions() -> Set[Tuple[int, ...]]:
    """Brute-force all Boolean functions a single adaptive threshold neuron can compute.

    The adaptive neuron's threshold changes after each fire() call, so the
    order of inputs matters. We try all permutations of the 4 input pairs
    and all initial thresholds, collecting every function that can appear.
    """
    computable: Set[Tuple[int, ...]] = set()

    for bias in BIAS_OPTIONS:
        for x1_role in ROLES:
            for x2_role in ROLES:
                for init_theta in range(7):
                    for perm in permutations(INPUT_PAIRS):
                        try:
                            n = AdaptiveThresholdNeuron(threshold=init_theta)
                            if bias > 0:
                                n.add_excitatory(bias)
                            if x1_role > 0:
                                n.add_excitatory(x1_role)
                            if x2_role > 0:
                                n.add_excitatory(x2_role)
                            if x1_role == -1:
                                n.add_inhibitory()
                            if x2_role == -1:
                                n.add_inhibitory()

                            outputs = tuple(
                                n.fire(_build_inputs(bias, x1_role, x2_role, x1, x2))
                                for x1, x2 in perm
                            )
                            computable.add(outputs)
                        except (ValueError, IndexError):
                            continue

    return computable


def compare() -> Dict[str, dict]:
    """Run the comparison and return results."""
    mp_funcs = enumerate_mp_functions()
    mt_funcs = enumerate_mt_functions()
    at_funcs = enumerate_at_functions()

    mp_count = len(mp_funcs & ALL_FUNCTIONS)
    mt_count = len(mt_funcs & ALL_FUNCTIONS)
    at_count = len(at_funcs & ALL_FUNCTIONS)

    # MP is the base (100%)
    mp_pct = 100.0
    mt_pct = (mt_count / mp_count) * 100.0 if mp_count > 0 else 0.0
    mt_delta = mt_pct - mp_pct
    at_pct = (at_count / mp_count) * 100.0 if mp_count > 0 else 0.0
    at_delta = at_pct - mp_pct

    results = {
        "McCulloch-Pitts Neuron": {
            "count": mp_count,
            "total": TOTAL_FUNCTIONS,
            "pct": mp_pct,
            "delta": 0.0,
            "parent": None,
        },
        "Multi-Threshold Neuron": {
            "count": mt_count,
            "total": TOTAL_FUNCTIONS,
            "pct": mt_pct,
            "delta": mt_delta,
            "parent": "McCulloch-Pitts Neuron",
        },
        "Adaptive Threshold Neuron": {
            "count": at_count,
            "total": TOTAL_FUNCTIONS,
            "pct": at_pct,
            "delta": at_delta,
            "parent": "McCulloch-Pitts Neuron",
        },
    }

    return results


def print_results(results: Dict[str, dict]) -> None:
    """Print comparison results to console."""
    print(f"\n{'Neuron Variant':<30} {'Functions':>10} {'/16':>5} {'%':>8} {'Delta':>8} {'Parent':>30}")
    print("-" * 95)
    for name, r in results.items():
        delta_str = f"{r['delta']:+.1f}%" if r["parent"] else "(base)"
        parent_str = r["parent"] or "—"
        print(
            f"{name:<30} {r['count']:>10} {r['total']:>5} "
            f"{r['pct']:>7.1f}% {delta_str:>8} {parent_str:>30}"
        )
    print()


def plot_comparison(
    results: Dict[str, dict], save_path: str | None = None
) -> None:
    """Plot the comparison as a bar chart with delta annotations."""
    names = list(results.keys())
    counts = [results[n]["count"] for n in names]
    pcts = [results[n]["pct"] for n in names]
    deltas = [results[n]["delta"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#4ec9b0", "#569cd6", "#dcdcaa", "#ce9178"]
    bars = ax.bar(
        names, counts, color=colors[: len(names)], edgecolor="#2b2b2b", width=0.5
    )

    ax.set_ylabel("Computable Boolean Functions (out of 16)", fontsize=12)
    ax.set_title("Neuron Variant Comparison\n2-input Boolean Functions", fontsize=14, fontweight="bold")
    ax.set_ylim(0, TOTAL_FUNCTIONS + 2)
    ax.axhline(y=TOTAL_FUNCTIONS, color="#6a6a6a", linestyle="--", linewidth=0.8, label="All 16 functions")

    # Annotate each bar with count, percentage, and delta
    for i, (bar, count, pct, delta) in enumerate(zip(bars, counts, pcts, deltas)):
        height = bar.get_height()
        label = f"{count}/16\n{pct:.1f}%"
        if i > 0:
            label += f"\n({delta:+.1f}% vs parent)"
        else:
            label += "\n(base)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.3,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.legend(loc="upper left", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    results = compare()
    print_results(results)
    plot_comparison(results, save_path="src/comparison.png")
