"""McCulloch-Pitts neuron (1943).

Implements the binary threshold neuron described in:
    McCulloch, W.S. & Pitts, W. (1943). "A Logical Calculus of the Ideas
    Immanent in Nervous Activity." Bulletin of Mathematical Biophysics, 5, 115-133.

Rules from the paper:
    1. Neuron state is binary: firing (1) or silent (0).
    2. Time is discrete — output at t+1 depends on inputs at t.
    3. Excitatory synapses have positive integer weights.
    4. Sum of excitatory inputs must reach or exceed threshold θ to fire.
    5. Inhibitory synapses are an absolute veto: if ANY inhibitory input
       is active, the neuron cannot fire, regardless of excitation.
    6. Threshold is fixed and set by the neuron, not by the input.
    7. No learning — weights and connections are static.
"""

from typing import List


class Neuron:
    """A single McCulloch-Pitts neuron.

    Attributes:
        threshold: Fixed firing threshold (positive integer).
    """

    def __init__(self, threshold: int):
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")
        self.threshold = threshold
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
        """Compute the neuron's output given binary inputs.

        Args:
            inputs: List of 0s and 1s. The first num_excitatory values
                    map to excitatory synapses (in registration order),
                    the remaining values map to inhibitory synapses.

        Returns:
            1 if the neuron fires, 0 otherwise.

        Raises:
            ValueError: If the number of inputs doesn't match the number
                        of synapses.
        """
        total_inputs = self.num_excitatory + self.num_inhibitory
        if len(inputs) != total_inputs:
            raise ValueError(
                f"Expected {total_inputs} inputs "
                f"({self.num_excitatory} excitatory, "
                f"{self.num_inhibitory} inhibitory), got {len(inputs)}"
            )

        # Check inhibitory veto — any active inhibitory input blocks firing.
        inhibitory_inputs = inputs[self.num_excitatory:]
        if any(v == 1 for v in inhibitory_inputs):
            return 0

        # Sum excitatory inputs.
        excitatory_inputs = inputs[: self.num_excitatory]
        total = sum(
            w * v for w, v in zip(self._excitatory_weights, excitatory_inputs)
        )

        return 1 if total >= self.threshold else 0
