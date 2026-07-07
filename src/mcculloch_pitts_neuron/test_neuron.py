"""Tests for the McCulloch-Pitts neuron."""

import pytest

from src.mcculloch_pitts_neuron.neuron import Neuron


# --- AND gate: threshold=2, two excitatory inputs (weight 1 each) ---

def test_and_gate_all_off():
    n = Neuron(threshold=2)
    n.add_excitatory(1)
    n.add_excitatory(1)
    assert n.fire([0, 0]) == 0


def test_and_gate_one_on():
    n = Neuron(threshold=2)
    n.add_excitatory(1)
    n.add_excitatory(1)
    assert n.fire([0, 1]) == 0
    assert n.fire([1, 0]) == 0


def test_and_gate_both_on():
    n = Neuron(threshold=2)
    n.add_excitatory(1)
    n.add_excitatory(1)
    assert n.fire([1, 1]) == 1


# --- OR gate: threshold=1, two excitatory inputs (weight 1 each) ---

@pytest.mark.parametrize("a,b,expected", [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 1),
])
def test_or_gate(a, b, expected):
    n = Neuron(threshold=1)
    n.add_excitatory(1)
    n.add_excitatory(1)
    assert n.fire([a, b]) == expected


# --- NOT gate: bias (always-on excitatory) + inhibitory input ---

def test_not_gate():
    n = Neuron(threshold=1)
    n.add_excitatory(1)  # bias — always fed 1
    n.add_inhibitory()   # the input x
    assert n.fire([1, 0]) == 1  # NOT 0 = 1
    assert n.fire([1, 1]) == 0  # NOT 1 = 0 (inhibitory veto)


# --- Inhibitory veto overrides excitation ---

def test_inhibitory_veto_blocks_firing():
    n = Neuron(threshold=1)
    n.add_excitatory(1)
    n.add_inhibitory()
    assert n.fire([1, 1]) == 0  # enough excitation but vetoed


def test_no_veto_when_inhibitory_silent():
    n = Neuron(threshold=1)
    n.add_excitatory(1)
    n.add_inhibitory()
    assert n.fire([1, 0]) == 1  # excitation present, no veto


def test_multiple_inhibitory_any_active_vetoes():
    n = Neuron(threshold=1)
    n.add_excitatory(1)
    n.add_inhibitory()
    n.add_inhibitory()
    assert n.fire([1, 0, 1]) == 0  # second inhibitory active
    assert n.fire([1, 1, 0]) == 0  # first inhibitory active
    assert n.fire([1, 0, 0]) == 1  # neither inhibitory active


# --- Threshold behavior ---

def test_threshold_not_reached():
    n = Neuron(threshold=3)
    n.add_excitatory(1)
    n.add_excitatory(1)
    n.add_excitatory(1)
    assert n.fire([1, 1, 0]) == 0  # sum=2 < 3


def test_threshold_exactly_met():
    n = Neuron(threshold=3)
    n.add_excitatory(1)
    n.add_excitatory(1)
    n.add_excitatory(1)
    assert n.fire([1, 1, 1]) == 1  # sum=3 >= 3


def test_weighted_excitatory_inputs():
    n = Neuron(threshold=3)
    n.add_excitatory(2)  # weight 2
    n.add_excitatory(1)  # weight 1
    assert n.fire([1, 0]) == 0  # sum=2 < 3
    assert n.fire([1, 1]) == 1  # sum=3 >= 3


# --- Edge cases ---

def test_zero_threshold_neuron_always_fires_without_inhibition():
    n = Neuron(threshold=0)
    assert n.fire([]) == 1


def test_zero_threshold_with_inhibitory_still_vetoed():
    n = Neuron(threshold=0)
    n.add_inhibitory()
    assert n.fire([1]) == 0
    assert n.fire([0]) == 1


def test_wrong_input_count_raises():
    n = Neuron(threshold=1)
    n.add_excitatory(1)
    n.add_excitatory(1)
    with pytest.raises(ValueError, match="Expected 2 inputs"):
        n.fire([1])


def test_negative_threshold_raises():
    with pytest.raises(ValueError, match="Threshold must be non-negative"):
        Neuron(threshold=-1)


def test_zero_excitatory_weight_raises():
    n = Neuron(threshold=1)
    with pytest.raises(ValueError, match="positive integer"):
        n.add_excitatory(0)


def test_negative_excitatory_weight_raises():
    n = Neuron(threshold=1)
    with pytest.raises(ValueError, match="positive integer"):
        n.add_excitatory(-1)
