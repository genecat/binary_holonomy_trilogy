# binary_holonomy_trilogy

Two tiny, laptop-friendly experiments to prototype ideas around spin-2 (TT-projected fluctuations) and stabilizer-based area-law estimates.

## Contents
- `src/sim_spin2.py` — Synthesizes a small metric fluctuation field, takes a spatial FFT, projects to **transverse–traceless (TT)**, and estimates the peak temporal frequency ω(k).  
- `src/entropy_cut.py` — Builds a toy CSS (stabilizer) model on a 3D cubic lattice and prints an **area-law-like** entanglement estimator via crossing ranks.

## Requirements
- Python **3.12+** (you have 3.12.2 ✅)
- NumPy (bundled with your Python install in most cases)

## Quickstart

```bash
# from the repo root
python3 src/sim_spin2.py

