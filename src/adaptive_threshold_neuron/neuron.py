"""Adaptive threshold neuron.

A descendant of the McCulloch-Pitts neuron that can learn. The threshold
is not fixed — it adapts based on the neuron's own firing history.

Adaptation rule (homeostatic):
    - If the neuron fires, threshold increases by 1 (harder to fire next time).
    - If the neuron does not fire, threshold decreases by 1 (easier), min 0.

This is the first neuron in the project that can learn from experience.
It self-tunes to find an equilibrium firing rate.

Rules:
    1. Inputs are binary (0 or 1).
    2. Excitatory synapses have positive integer weights.
    3. Inhibitory synapses are an absolute veto (same as MP).
    4. Sum of excitatory inputs must reach or exceed current threshold to fire.
    5. After each fire() call, threshold adapts:
       - fired → threshold += 1
       - not fired → threshold = max(0, threshold - 1)
    6. With adaptation disabled, reduces exactly to the MP neuron.
"""

from typing import List


class AdaptiveThresholdNeuron:
    """A McCulloch-Pitts neuron with an adaptive threshold.

    Attributes:
        threshold: Current firing threshold (adapts after each call).
        initial_threshold: The starting threshold value.
    """

    def __init__(self, threshold: int):
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")
        self.threshold = threshold
        self.initial_threshold = threshold
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

    def fire(self, inputs: List[int]) -> int:
        """Compute output, then adapt threshold based on result.

        Args:
            inputs: List of 0s and 1s. First num_excitatory map to
                    excitatory synapses, the rest to inhibitory.

        Returns:
            1 if the neuron fires, 0 otherwise.
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
        vetoed = any(v == 1 for v in inhibitory_inputs)

        if vetoed:
            self._adapt(fired=False)
            return 0

        # Sum excitatory inputs.
        excitatory_inputs = inputs[: self.num_excitatory]
        total = sum(
            w * v for w, v in zip(self._excitatory_weights, excitatory_inputs)
        )

        fired = total >= self.threshold
        self._adapt(fired=fired)
        return 1 if fired else 0

    def _adapt(self, fired: bool) -> None:
        """Adjust threshold based on whether the neuron fired."""
        if fired:
            self.threshold += 1
        else:
            self.threshold = max(0, self.threshold - 1)

    def reset(self) -> None:
        """Reset threshold to its initial value."""
        self.threshold = self.initial_threshold
