"""Tests for the multi-threshold neuron."""

import pytest

from src.multi_threshold_neuron.neuron import MultiThresholdNeuron


# --- Reduces to MP neuron with single threshold ---

def test_single_threshold_matches_mp_and():
    n = MultiThresholdNeuron([2])
    n.add_excitatory(1)
    n.add_excitatory(1)
    assert n.fire([0, 0]) == 0
    assert n.fire([0, 1]) == 0
    assert n.fire([1, 0]) == 0
    assert n.fire([1, 1]) == 1


def test_single_threshold_matches_mp_or():
    n = MultiThresholdNeuron([1])
    n.add_excitatory(1)
    n.add_excitatory(1)
    assert n.fire([0, 0]) == 0
    assert n.fire([0, 1]) == 1
    assert n.fire([1, 0]) == 1
    assert n.fire([1, 1]) == 1


# --- Graded output with multiple thresholds ---

def test_two_thresholds_graded_output():
    n = MultiThresholdNeuron([1, 2])
    n.add_excitatory(1)
    n.add_excitatory(1)
    assert n.fire([0, 0]) == 0  # sum=0, no threshold met
    assert n.fire([0, 1]) == 1  # sum=1, meets θ₁=1 only
    assert n.fire([1, 0]) == 1  # sum=1, meets θ₁=1 only
    assert n.fire([1, 1]) == 2  # sum=2, meets both θ₁=1 and θ₂=2


def test_three_thresholds_graded_output():
    n = MultiThresholdNeuron([1, 2, 3])
    n.add_excitatory(1)
    n.add_excitatory(1)
    n.add_excitatory(1)
    assert n.fire([0, 0, 0]) == 0  # sum=0
    assert n.fire([1, 0, 0]) == 1  # sum=1, meets θ₁
    assert n.fire([1, 1, 0]) == 2  # sum=2, meets θ₁,θ₂
    assert n.fire([1, 1, 1]) == 3  # sum=3, meets all


# --- XOR with a single multi-threshold neuron ---

def test_xor_single_neuron():
    """XOR: fires when exactly one input is on.
    Use thresholds [1, 2] with weights [1, 1].
    Output 1 when sum=1 (meets θ₁ only), output 0 when sum=0 or sum=2.
    We interpret output==1 as firing, anything else as 0.
    """
    n = MultiThresholdNeuron([1, 2])
    n.add_excitatory(1)
    n.add_excitatory(1)
    # XOR truth table: output 1 only when exactly one input is 1
    assert n.fire([0, 0]) == 0  # sum=0 → 0 thresholds met → 0
    assert n.fire([0, 1]) == 1  # sum=1 → 1 threshold met → 1 (XOR=1)
    assert n.fire([1, 0]) == 1  # sum=1 → 1 threshold met → 1 (XOR=1)
    assert n.fire([1, 1]) == 2  # sum=2 → 2 thresholds met → 2 (XOR=0)


# --- Inhibitory veto ---

def test_inhibitory_veto_blocks_all_output():
    n = MultiThresholdNeuron([1, 2])
    n.add_excitatory(1)
    n.add_excitatory(1)
    n.add_inhibitory()
    assert n.fire([1, 1, 1]) == 0  # would be 2, but vetoed


def test_no_veto_when_inhibitory_silent():
    n = MultiThresholdNeuron([1, 2])
    n.add_excitatory(1)
    n.add_excitatory(1)
    n.add_inhibitory()
    assert n.fire([1, 1, 0]) == 2  # not vetoed


# --- Weighted inputs ---

def test_weighted_excitatory_inputs():
    n = MultiThresholdNeuron([2, 3, 5])
    n.add_excitatory(2)
    n.add_excitatory(1)
    assert n.fire([0, 0]) == 0  # sum=0
    assert n.fire([1, 0]) == 1  # sum=2, meets θ₁=2
    assert n.fire([1, 1]) == 2  # sum=3, meets θ₁=2, θ₂=3
    # sum=5 would need weight 2 on both or more inputs


# --- Thresholds are sorted internally ---

def test_unsorted_thresholds_get_sorted():
    n = MultiThresholdNeuron([3, 1, 2])
    assert n.thresholds == [1, 2, 3]


# --- Edge cases ---

def test_zero_threshold_always_met():
    n = MultiThresholdNeuron([0, 1])
    n.add_excitatory(1)
    assert n.fire([0]) == 1  # sum=0, meets θ₁=0 only
    assert n.fire([1]) == 2  # sum=1, meets both


def test_empty_thresholds_raises():
    with pytest.raises(ValueError, match="at least one threshold"):
        MultiThresholdNeuron([])


def test_negative_threshold_raises():
    with pytest.raises(ValueError, match="non-negative"):
        MultiThresholdNeuron([-1])


def test_wrong_input_count_raises():
    n = MultiThresholdNeuron([1])
    n.add_excitatory(1)
    n.add_excitatory(1)
    with pytest.raises(ValueError, match="Expected 2 inputs"):
        n.fire([1])


def test_zero_excitatory_weight_raises():
    n = MultiThresholdNeuron([1])
    with pytest.raises(ValueError, match="positive integer"):
        n.add_excitatory(0)
