# Changelog

This is a history of what we did and discovered, not a changelog of version updates.

---

## Iteration 0 — Setup

- Project created
- Decided to restart AI research from scratch, as if post-1943 AI research never existed
- Starting point set: the McCulloch-Pitts neuron (1943 paper)
- Chose Python as the project language
- Set up documentation structure: README, PROJECT, CHANGELOG, LESSONS, METRICS

---

## Iteration 1 — The McCulloch-Pitts Neuron

- Implemented the McCulloch-Pitts neuron (1943 paper) in `src/v1/neuron.py`
- Binary threshold unit: fires (1) or silent (0)
- Excitatory synapses with positive integer weights — sum must reach threshold
- Inhibitory synapses with absolute veto — any active inhibitory input blocks firing
- Fixed threshold, no learning, discrete time (output at t+1 from inputs at t)
- Demonstrated logic gates: AND, OR, NOT (using bias + inhibitory)
- Wrote 20 tests covering gates, veto, threshold, weighted inputs, edge cases
- Added `requirements.txt` with numpy and matplotlib
- Added `visualize.py` — matplotlib plot of AND, OR, NOT gate decision regions
- All 20 tests pass

---

## Iteration 2 — Multi-Threshold Neuron & Comparison

- Renamed `src/v1/` to `src/mcculloch_pitts_neuron/`
- Added multi-threshold neuron in `src/multi_threshold_neuron/`
  - Multiple thresholds θ₁ ≤ θ₂ ≤ ... ≤ θₙ — output is graded (0 to n)
  - Counts how many thresholds the weighted sum meets
  - Can compute XOR with a single neuron (MP neuron cannot)
  - Inhibitory veto preserved (same as MP)
  - With a single threshold, reduces exactly to the MP neuron
- Added `src/compare.py` — brute-force comparison of all neuron variants
  - Metric: number of 2-input Boolean functions computable by a single neuron (out of 16)
  - MP neuron: 11/16 (base, 100%)
  - Multi-threshold neuron: 15/16 (+36.4% vs MP)
  - Matplotlib bar chart with delta annotations
- All 34 tests pass (20 MP + 14 multi-threshold)
