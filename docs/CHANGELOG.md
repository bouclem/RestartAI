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
