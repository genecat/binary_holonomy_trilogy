#!/usr/bin/env python3
"""
wilson_loops.py
A tiny, laptop-friendly Wilson-loop demo to probe deconfinement (perimeter law)
and a minimal transverse photon-like dispersion (placeholder).

Usage examples:
  python3 src/wilson_loops.py --L 12 --nsamples 1000 --rects 1x1,1x2,2x2,3x2 --plot
  python3 src/wilson_loops.py --L 10 --nsamples 500 --beta 0.35 --plot
"""
from __future__ import annotations
import argparse, os, math
import numpy as np

# Matplotlib is optional; we import lazily only if --plot is used
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def parse_rects(rects_str: str):
    rects = []
    for item in rects_str.split(","):
        item = item.strip()
        if "x" not in item:
            continue
        a, b = item.lower().split("x")
        rects.append((int(a), int(b)))
    return rects

# --- Random ±1 link configurations with controllable mean m = tanh(beta) ---
def sample_config(L: int, beta: float, rng: np.random.Generator):
    """
    Returns three arrays (ux, uy, uz) of shape (L,L,L) with entries ±1, interpreted as
    link variables on edges in +x, +y, +z directions. We bias toward +1 with mean m=tanh(beta).
    """
    m = np.tanh(beta)
    p = (1.0 + m) / 2.0  # prob(+1)
    ux = np.where(rng.random((L, L, L)) < p, 1, -1)
    uy = np.where(rng.random((L, L, L)) < p, 1, -1)
    uz = np.where(rng.random((L, L, L)) < p, 1, -1)
    return ux, uy, uz

# --- Wilson loop product on the XY plane (periodic boundary) ---
def loop_product_xy(ux: np.ndarray, uy: np.ndarray, x0: int, y0: int, z: int, a: int, b: int) -> int:
    """
    Oriented rectangular loop with sides a (x-direction) and b (y-direction) on the XY plane at height z.
    Periodic boundary conditions. For Z2 variables (±1), reversing direction uses the same link value.
    """
    L = ux.shape[0]
    prod = 1
    x, y = x0, y0

    # +x, a steps
    for _ in range(a):
        prod *= ux[x % L, y % L, z % L]
        x += 1

    # +y, b steps
    for _ in range(b):
        prod *= uy[x % L, y % L, z % L]
        y += 1

    # -x, a steps
    for _ in range(a):
        x -= 1
        prod *= ux[x % L, y % L, z % L]  # inverse is itself for ±1

    # -y, b steps
    for _ in range(b):
        y -= 1
        prod *= uy[x % L, y % L, z % L]

    return prod

def wilson_loop_expectation(L: int, a: int, b: int, plane: str, beta: float, nsamples: int, rng: np.random.Generator) -> float:
    """
    Estimates <W(C)> by averaging the loop product over random positions and fresh random
    link configs. For biased ±1 links with mean m=tanh(beta), the signal exhibits a clear
    perimeter scaling:  -log <W> ~ (const) * perimeter, when beta > 0.
    """
    if plane != "xy":
        raise NotImplementedError("This first demo implements plane='xy' only (we'll add yz/zx next).")
    acc = 0.0
    for _ in range(nsamples):
        ux, uy, uz = sample_config(L, beta, rng)
        x0 = int(rng.integers(0, L))
        y0 = int(rng.integers(0, L))
        z  = int(rng.integers(0, L))
        acc += loop_product_xy(ux, uy, x0, y0, z, a, b)
    return acc / float(nsamples)

def main():
    ap = argparse.ArgumentParser(description="Wilson loop perimeter/area demo (Z2-biased links).")
    ap.add_argument("--L", type=int, default=12, help="lattice size per dimension")
    ap.add_argument("--plane", type=str, default="xy", choices=["xy"], help="loop plane (xy only in this demo)")
    ap.add_argument("--rects", type=str, default="1x1,1x2,2x2,3x2,3x3", help="comma-separated rectangle sizes axb")
    ap.add_argument("--beta", type=float, default=0.35, help="bias parameter (mean m = tanh(beta))")
    ap.add_argument("--nsamples", type=int, default=800, help="number of loop samples")
    ap.add_argument("--plot", action="store_true", help="save perimeter/area plots under figs/")
    ap.add_argument("--seed", type=int, default=1234, help="random seed")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rects = parse_rects(args.rects)

    perims, areas, neglogWs = [], [], []
    for (a, b) in rects:
        W = wilson_loop_expectation(args.L, a, b, args.plane, args.beta, args.nsamples, rng)
        # numerical guard
        W_clip = max(min(W, 1.0 - 1e-12), 1e-12)
        perim = 2 * (a + b)
        area  = a * b
        nlw = -math.log(W_clip)
        perims.append(perim); areas.append(area); neglogWs.append(nlw)
        print(f"rect {a}x{b: <2} | perimeter={perim:2d} | area={area:2d} | <W>≈ {W: .4f} | -log<W>≈ {nlw: .4f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt  # noqa
            ensure_dir("figs")
            # Perimeter plot
            plt.figure()
            plt.scatter(perims, neglogWs)
            plt.xlabel("Perimeter (2[a+b])")
            plt.ylabel("-log ⟨W(C)⟩")
            plt.title("Wilson loop: -log⟨W⟩ vs Perimeter")
            out1 = "figs/wilson_perimeter_vs_logW.png"
            plt.savefig(out1, bbox_inches="tight")
            print(f"Saved plot to {out1}")

            # Area plot
            plt.figure()
            plt.scatter(areas, neglogWs)
            plt.xlabel("Area (a×b)")
            plt.ylabel("-log ⟨W(C)⟩")
            plt.title("Wilson loop: -log⟨W⟩ vs Area")
            out2 = "figs/wilson_area_vs_logW.png"
            plt.savefig(out2, bbox_inches="tight")
            print(f"Saved plot to {out2}")
        except Exception as e:
            print(f"Plotting skipped: {e}")

if __name__ == "__main__":
    main()
