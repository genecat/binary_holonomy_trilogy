# binary_holonomy_trilogy

Two tiny, laptop-friendly experiments:
- `src/sim_spin2.py` — builds a small synthetic metric field, does a spatial FFT, applies a transverse–traceless (TT) projector, and estimates the peak temporal frequency ω(k).
- `src/entropy_cut.py` — toy CSS (stabilizer) model on a 3D lattice; prints an area-law–like entanglement estimator via crossing ranks.

## Requirements
- Python 3.12+
- NumPy
- (Optional for plots) matplotlib

### Install (optional)
```bash
python3 -m pip install -r requirements.txt

md
