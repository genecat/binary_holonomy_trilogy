# binary\_holonomy\_trilogy

Two tiny, laptop-friendly experiments:

* `src/sim_spin2.py` — builds a small synthetic metric field, does a spatial FFT, applies a transverse–traceless (TT) projector, and estimates the peak temporal frequency ω(k).
* `src/entropy_cut.py` — toy CSS (stabilizer) model on a 3D lattice; prints an area-law–like entanglement estimator via crossing ranks.

## Requirements

* Python 3.12+
* NumPy
* (Optional for plots) matplotlib

### Install (optional)

```bash
python3 -m pip install -r requirements.txt
```

## Quickstart

```bash
# from the repo root
python3 src/sim_spin2.py
python3 src/entropy_cut.py
```

## Optional CLI

```bash
# spin-2: change lattice/time or save a plot
python3 src/sim_spin2.py --L 16 --tsteps 64
python3 src/sim_spin2.py --plot

# entropy cut: choose cut side or save plots
python3 src/entropy_cut.py --L 6 --side x
python3 src/entropy_cut.py --L 6 --side x --plot
```

## Virtual environment (optional)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Makefile shortcuts

```bash
make install
make run-sim L=12 TSTEPS=48
make plot-sim L=12 TSTEPS=48
make run-entropy L=12 SIDE=x
make plot-entropy L=12 SIDE=x
make clean
```
