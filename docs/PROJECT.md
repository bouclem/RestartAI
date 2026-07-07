# Project Overview

## What It Is

RestartAI is a project that restarts artificial intelligence research from the year 1943. We ignore everything that came after the McCulloch-Pitts neuron paper and build forward on our own.

## Architecture

### v1 — McCulloch-Pitts Neuron (`src/mcculloch_pitts_neuron/`)

The base unit: a single binary threshold neuron as described in the 1943 paper.

- **`neuron.py`** — `Neuron` class with excitatory/inhibitory synapses and fixed threshold
- **`test_neuron.py`** — 20 unit tests covering logic gates, veto, threshold, weighted inputs, edge cases
- **`visualize.py`** — Matplotlib visualization of AND, OR, NOT gate decision regions

Design follows the original paper:
- Inputs and outputs are binary (0 or 1)
- Excitatory synapses have positive integer weights; sum must reach threshold θ
- Inhibitory synapses are an absolute veto (any active → no firing)
- No learning — static weights and connections
- Discrete time — output at t+1 depends on inputs at t

### v2 — Multi-Threshold Neuron (`src/multi_threshold_neuron/`)

Generalization of the MP neuron with multiple thresholds and graded output.

- **`neuron.py`** — `MultiThresholdNeuron` class with multiple thresholds, graded output (0..n)
- **`test_neuron.py`** — 14 unit tests covering graded output, XOR, veto, threshold, edge cases

Key advantage: a single multi-threshold neuron can compute non-monotonic functions
like XOR, which is impossible with a single MP neuron.

### Comparison (`src/compare.py`)

Brute-force comparison of all neuron variants by number of computable 2-input
Boolean functions (out of 16). Generates a matplotlib bar chart with delta
annotations showing % improvement vs parent variant.

## Design

- Start from the simplest possible unit: a single binary threshold neuron
- Build, test, and document each step
- Let discoveries guide the direction — no predetermined roadmap
- Everything is documented as we go

## Direction

Open. The project evolves based on what we learn at each step.

## Stack

- Language: Python
- No external ML frameworks — everything built from scratch
