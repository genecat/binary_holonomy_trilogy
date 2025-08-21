# binary_holonomy_trilogy

Two tiny, laptop-friendly experiments to prototype ideas around spin-2 (TT-projected fluctuations) and stabilizer-based area-law estimates.

## Contents
- `src/sim_spin2.py` — Synthesizes a small metric fluctuation field, takes a spatial FFT, projects to **transverse–traceless (TT)**, and estimates the peak temporal frequency ω(k).  
- `src/entropy_cut.py` — Builds a toy CSS (stabilizer) model on a 3D cubic lattice and prints an **area-law-like** entanglement estimator via crossing ranks.

## Requirements
- Python **3.12+** (tested on 3.12.2)
- NumPy (bundled with most Python installs)
- (Optional for plots) `matplotlib` → `python3 -m pip install matplotlib`

## Quickstart

```bash
# from the repo root
python3 src/sim_spin2.py
python3 src/entropy_cut.py
