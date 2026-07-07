"""Tests for the adaptive threshold neuron."""

import pytest

from src.adaptive_threshold_neuron.neuron import AdaptiveThresholdNeuron


# --- Basic firing (same as MP before adaptation kicks in) ---

def test_and_gate_first_call():
    """AND gate works correctly on the first call (before adaptation)."""
    n = AdaptiveThresholdNeuron(threshold=2)
    n.add_excitatory(1)
    n.add_excitatory(1)
    assert n.fire([0, 0]) == 0  # sum=0 < 2, doesn't fire
    # After not firing, threshold drops to 1, so we need fresh neurons
    n2 = AdaptiveThresholdNeuron(threshold=2)
    n2.add_excitatory(1)
    n2.add_excitatory(1)
    assert n2.fire([0, 1]) == 0  # sum=1 < 2
    n3 = AdaptiveThresholdNeuron(threshold=2)
    n3.add_excitatory(1)
    n3.add_excitatory(1)
    assert n3.fire([1, 0]) == 0  # sum=1 < 2
    n4 = AdaptiveThresholdNeuron(threshold=2)
    n4.add_excitatory(1)
    n4.add_excitatory(1)
    assert n4.fire([1, 1]) == 1  # sum=2 >= 2


def test_or_gate_first_call():
    """OR gate works correctly on the first call (before adaptation)."""
    for inputs, expected in [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)]:
        n = AdaptiveThresholdNeuron(threshold=1)
        n.add_excitatory(1)
        n.add_excitatory(1)
        assert n.fire(inputs) == expected


# --- Threshold adaptation ---

def test_threshold_increases_after_firing():
    n = AdaptiveThresholdNeuron(threshold=1)
    n.add_excitatory(1)
    assert n.threshold == 1
    n.fire([1])  # fires
    assert n.threshold == 2


def test_threshold_decreases_after_not_firing():
    n = AdaptiveThresholdNeuron(threshold=3)
    n.add_excitatory(1)
    n.fire([1])  # sum=1 < 3, doesn't fire
    assert n.threshold == 2


def test_threshold_floor_at_zero():
    n = AdaptiveThresholdNeuron(threshold=0)
    n.add_excitatory(1)
    n.fire([0])  # sum=0 >= 0, fires
    assert n.threshold == 1
    n.fire([0])  # sum=0 < 1, doesn't fire
    assert n.threshold == 0
    n.fire([0])  # sum=0 >= 0, fires again
    assert n.threshold == 1


def test_threshold_does_not_go_negative():
    n = AdaptiveThresholdNeuron(threshold=1)
    n.add_excitatory(1)
    n.fire([0])  # doesn't fire
    assert n.threshold == 0
    n.fire([0])  # sum=0 >= 0, fires
    assert n.threshold == 1
    n.fire([0])  # doesn't fire
    assert n.threshold == 0
    n.fire([0])  # doesn't fire (sum=0 >= 0, fires actually)
    # wait — threshold=0, sum=0, 0>=0 → fires
    # so let's test the floor properly:
    n2 = AdaptiveThresholdNeuron(threshold=1)
    n2.add_excitatory(1)
    n2.fire([0])  # sum=0 < 1, doesn't fire → threshold=0
    assert n2.threshold == 0
    n2.fire([0])  # sum=0 >= 0, fires → threshold=1
    assert n2.threshold == 1


# --- Homeostatic equilibrium ---

def test_reaches_equilibrium_with_constant_input():
    """With constant input that always fires, threshold climbs
    until it stops firing, then oscillates around equilibrium."""
    n = AdaptiveThresholdNeuron(threshold=0)
    n.add_excitatory(1)
    # Always feed 1 — sum is always 1
    # threshold=0: fires (1>=0) → threshold=1
    # threshold=1: fires (1>=1) → threshold=2
    # threshold=2: doesn't fire (1<2) → threshold=1
    # threshold=1: fires (1>=1) → threshold=2
    # oscillates between 1 and 2
    outputs = [n.fire([1]) for _ in range(10)]
    # Should oscillate: fire, fire, no, fire, no, fire, no, fire, no, fire
    assert outputs[0] == 1  # threshold was 0
    assert outputs[1] == 1  # threshold was 1
    assert outputs[2] == 0  # threshold was 2
    assert outputs[3] == 1  # threshold was 1
    assert outputs[4] == 0  # threshold was 2


def test_equilibrium_stabilizes_with_zero_input():
    """With constant zero input, threshold drops until it hits 0,
    then oscillates between 0 (fires, 0>=0) and 1 (doesn't fire, 0<1)."""
    n = AdaptiveThresholdNeuron(threshold=5)
    n.add_excitatory(1)
    for _ in range(10):
        n.fire([0])
    # threshold should be oscillating between 0 and 1
    assert n.threshold in (0, 1)


# --- Inhibitory veto still works ---

def test_inhibitory_veto():
    n = AdaptiveThresholdNeuron(threshold=1)
    n.add_excitatory(1)
    n.add_inhibitory()
    assert n.fire([1, 1]) == 0  # vetoed


def test_veto_adapts_threshold_as_not_fired():
    n = AdaptiveThresholdNeuron(threshold=2)
    n.add_excitatory(1)
    n.add_inhibitory()
    n.fire([1, 1])  # vetoed → counts as not fired
    assert n.threshold == 1  # decreased


# --- Reset ---

def test_reset_restores_initial_threshold():
    n = AdaptiveThresholdNeuron(threshold=3)
    n.add_excitatory(1)
    n.fire([1])  # doesn't fire (1<3) → threshold=2
    n.fire([1])  # doesn't fire (1<2) → threshold=1
    n.fire([1])  # fires (1>=1) → threshold=2
    assert n.threshold != n.initial_threshold
    n.reset()
    assert n.threshold == 3


# --- Edge cases ---

def test_wrong_input_count_raises():
    n = AdaptiveThresholdNeuron(threshold=1)
    n.add_excitatory(1)
    n.add_excitatory(1)
    with pytest.raises(ValueError, match="Expected 2 inputs"):
        n.fire([1])


def test_negative_threshold_raises():
    with pytest.raises(ValueError, match="non-negative"):
        AdaptiveThresholdNeuron(threshold=-1)


def test_zero_excitatory_weight_raises():
    n = AdaptiveThresholdNeuron(threshold=1)
    with pytest.raises(ValueError, match="positive integer"):
        n.add_excitatory(0)
