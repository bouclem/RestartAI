"""Multi-threshold neuron.

A generalization of the McCulloch-Pitts neuron. Instead of a single
threshold, this neuron has multiple thresholds θ₁ ≤ θ₂ ≤ ... ≤ θₙ.
The output is the number of thresholds that the weighted sum meets
or exceeds, producing a graded response (0 to n) instead of binary.

Key advantage over MP neuron:
    A single multi-threshold neuron can compute non-monotonic functions
    like XOR, which is impossible with a single MP neuron (MP output is
    monotonic in each input).

Rules:
    1. Inputs are binary (0 or 1).
    2. Excitatory synapses have positive integer weights.
    3. Inhibitory synapses are an absolute veto (same as MP).
    4. Output = count of thresholds θᵢ where weighted_sum >= θᵢ.
    5. If any inhibitory input is active, output = 0.
    6. With a single threshold, this reduces exactly to the MP neuron.
"""

from typing import List


class MultiThresholdNeuron:
    """A multi-threshold neuron.

    Attributes:
        thresholds: Sorted list of firing thresholds (non-negative integers).
    """

    def __init__(self, thresholds: List[int]):
        if not thresholds:
            raise ValueError("Must provide at least one threshold")
        if any(t < 0 for t in thresholds):
            raise ValueError("Thresholds must be non-negative")
        self.thresholds = sorted(thresholds)
        self._excitatory_weights: List[int] = []
        self._inhibitory_count: int = 0

    def add_excitatory(self, weight: int = 1) -> None:
        """Add an excitatory synapse with a positive integer weight."""
        if weight <= 0:
            raise ValueError("Excitatory weight must be a positive integer")
        self._excitatory_weights.append(weight)

    def add_inhibitory(self) -> None:
        """Add an inhibitory synapse (absolute veto)."""
        self._inhibitory_count += 1

    @property
    def num_excitatory(self) -> int:
        return len(self._excitatory_weights)

    @property
    def num_inhibitory(self) -> int:
        return self._inhibitory_count

    @property
    def num_thresholds(self) -> int:
        return len(self.thresholds)

    def fire(self, inputs: List[int]) -> int:
        """Compute the neuron's output given binary inputs.

        Args:
            inputs: List of 0s and 1s. The first num_excitatory values
                    map to excitatory synapses, the rest to inhibitory.

        Returns:
            Number of thresholds met (0 to num_thresholds), or 0 if
            any inhibitory input is active.
        """
        total_inputs = self.num_excitatory + self.num_inhibitory
        if len(inputs) != total_inputs:
            raise ValueError(
                f"Expected {total_inputs} inputs "
                f"({self.num_excitatory} excitatory, "
                f"{self.num_inhibitory} inhibitory), got {len(inputs)}"
            )

        # Inhibitory veto — same as MP neuron.
        inhibitory_inputs = inputs[self.num_excitatory:]
        if any(v == 1 for v in inhibitory_inputs):
            return 0

        # Sum excitatory inputs.
        excitatory_inputs = inputs[: self.num_excitatory]
        total = sum(
            w * v for w, v in zip(self._excitatory_weights, excitatory_inputs)
        )

        # Count how many thresholds are met.
        return sum(1 for t in self.thresholds if total >= t)
