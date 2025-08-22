#!/usr/bin/env python3
"""
gauge_mc.py
Z2 gauge Monte Carlo on a 3D periodic lattice with plaquette action.
- Link vars on +x,+y,+z edges: u_x, u_y, u_z ∈ {+1,-1}
- Action: S = -beta * sum_{plaquettes p} (product of 4 links around p)
- Metropolis flip of a single link uses local ΔS = 2*beta * sum(plaquettes touching that link)

Measures Wilson loops <W(C)> on rectangular loops in the XY plane and
plots -log <W> vs perimeter and vs area to look for perimeter-law behavior.

Examples:
  python3 src/gauge_mc.py --L 8 --beta 0.5 --therm 200 --sweeps 400 --measure-every 10 --rects 1x1,1x2,2x2,3x3 --plot
  python3 src/gauge_mc.py --L 10 --beta 0.35 --sweeps 300 --plot
"""
from __future__ import annotations
import argparse, os, math
import numpy as np

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def parse_rects(rects_str: str):
    rects = []
    for item in rects_str.split(","):
        item = item.strip()
        if not item or "x" not in item:
            continue
        a, b = item.lower().split("x")
        rects.append((int(a), int(b)))
    return rects

# --- Lattice helpers ---
def mod(i: int, L: int) -> int:
    return i % L

def random_links(L: int, rng: np.random.Generator):
    # hot start: i.i.d. ±1
    u = lambda: np.where(rng.random((L, L, L)) < 0.5, 1, -1)
    return u(), u(), u()

# --- Plaquette products (Z2): return ±1 ---
def plaq_xy(ux, uy, x, y, z, L):
    # edges: (x,y,z)->(x+1,y,z) [ux], (x+1,y,z)->(x+1,y+1,z) [uy at x+1],
    # (x+1,y+1,z)->(x,y+1,z) [ux at x reversed], (x,y+1,z)->(x,y,z) [uy at x,y]
    p = 1
    p *= ux[mod(x, L), mod(y, L), mod(z, L)]
    p *= uy[mod(x+1, L), mod(y, L), mod(z, L)]
    p *= ux[mod(x, L), mod(y+1, L), mod(z, L)]
    p *= uy[mod(x, L), mod(y, L), mod(z, L)]
    return p

def plaq_xz(ux, uz, x, y, z, L):
    p = 1
    p *= ux[mod(x, L), mod(y, L), mod(z, L)]
    p *= uz[mod(x+1, L), mod(y, L), mod(z, L)]
    p *= ux[mod(x, L), mod(y, L), mod(z+1, L)]
    p *= uz[mod(x, L), mod(y, L), mod(z, L)]
    return p

def plaq_yz(uy, uz, x, y, z, L):
    p = 1
    p *= uy[mod(x, L), mod(y, L), mod(z, L)]
    p *= uz[mod(x, L), mod(y+1, L), mod(z, L)]
    p *= uy[mod(x, L), mod(y, L), mod(z+1, L)]
    p *= uz[mod(x, L), mod(y, L), mod(z, L)]
    return p

# --- Local ΔS for flipping a single link ---
def delta_S_link_x(ux, uy, uz, x, y, z, beta, L):
    # x-link touches two XY plaquettes and two XZ plaquettes
    s = 0
    s += plaq_xy(ux, uy, x, y, z, L)
    s += plaq_xy(ux, uy, x, y-1, z, L)
    s += plaq_xz(ux, uz, x, y, z, L)
    s += plaq_xz(ux, uz, x, y, z-1, L)
    return 2.0 * beta * s

def delta_S_link_y(ux, uy, uz, x, y, z, beta, L):
    # y-link touches two XY and two YZ plaquettes
    s = 0
    s += plaq_xy(ux, uy, x, y, z, L)
    s += plaq_xy(ux, uy, x-1, y, z, L)
    s += plaq_yz(uy, uz, x, y, z, L)
    s += plaq_yz(uy, uz, x, y, z-1, L)
    return 2.0 * beta * s

def delta_S_link_z(ux, uy, uz, x, y, z, beta, L):
    # z-link touches two XZ and two YZ plaquettes
    s = 0
    s += plaq_xz(ux, uz, x, y, z, L)
    s += plaq_xz(ux, uz, x-1, y, z, L)
    s += plaq_yz(uy, uz, x, y, z, L)
    s += plaq_yz(uy, uz, x, y-1, z, L)
    return 2.0 * beta * s

def metropolis_sweep(ux, uy, uz, beta, rng, L):
    # iterate links in a random order per direction for mixing
    # X-links
    for x in rng.permutation(L):
        for y in rng.permutation(L):
            for z in rng.permutation(L):
                dS = delta_S_link_x(ux, uy, uz, x, y, z, beta, L)
                if dS <= 0 or rng.random() < np.exp(-dS):
                    ux[x, y, z] *= -1
    # Y-links
    for x in rng.permutation(L):
        for y in rng.permutation(L):
            for z in rng.permutation(L):
                dS = delta_S_link_y(ux, uy, uz, x, y, z, beta, L)
                if dS <= 0 or rng.random() < np.exp(-dS):
                    uy[x, y, z] *= -1
    # Z-links
    for x in rng.permutation(L):
        for y in rng.permutation(L):
            for z in rng.permutation(L):
                dS = delta_S_link_z(ux, uy, uz, x, y, z, beta, L)
                if dS <= 0 or rng.random() < np.exp(-dS):
                    uz[x, y, z] *= -1

# --- Wilson loop on XY plane (rect a×b starting at random (x0,y0,z)) ---
def loop_product_xy(ux, uy, x0, y0, z, a, b, L):
    prod = 1
    x, y = x0, y0
    for _ in range(a):
        prod *= ux[mod(x, L), mod(y, L), mod(z, L)]
        x += 1
    for _ in range(b):
        prod *= uy[mod(x, L), mod(y, L), mod(z, L)]
        y += 1
    for _ in range(a):
        x -= 1
        prod *= ux[mod(x, L), mod(y, L), mod(z, L)]
    for _ in range(b):
        y -= 1
        prod *= uy[mod(x, L), mod(y, L), mod(z, L)]
    return prod

def measure_wilson_xy(ux, uy, L, rects, nsamples_pos, rng):
    acc = np.zeros(len(rects), dtype=float)
    for n in range(nsamples_pos):
        x0 = int(rng.integers(0, L))
        y0 = int(rng.integers(0, L))
        z  = int(rng.integers(0, L))
        for i, (a, b) in enumerate(rects):
            acc[i] += loop_product_xy(ux, uy, x0, y0, z, a, b, L)
    return acc / float(nsamples_pos)

def run(L=8, beta=0.5, therm=200, sweeps=400, measure_every=10, rects_str="1x1,1x2,2x2,3x3",
        nsamples_pos=200, seed=1234, plot=False):
    rng = np.random.default_rng(seed)
    rects = parse_rects(rects_str)
    ux, uy, uz = random_links(L, rng)

    # Thermalization
    for _ in range(therm):
        metropolis_sweep(ux, uy, uz, beta, rng, L)

    # Measurements
    perims, areas = np.array([2*(a+b) for (a,b) in rects]), np.array([a*b for (a,b) in rects])
    accum = np.zeros(len(rects), dtype=float)
    nmeas = 0

    for sw in range(1, sweeps+1):
        metropolis_sweep(ux, uy, uz, beta, rng, L)
        if sw % measure_every == 0:
            W = measure_wilson_xy(ux, uy, L, rects, nsamples_pos, rng)
            accum += W
            nmeas += 1
            # progress line
            print(f"sweep {sw:4d}/{sweeps} | <W> (running avg): " +
                  ", ".join([f"{a}x{b}:{w/nmeas:+.3f}" for (a,b), w in zip(rects, accum)]))

    meanW = accum / max(nmeas, 1)
    # guard + convert to -log
    meanW = np.clip(meanW, 1e-12, 1-1e-12)
    neglogW = -np.log(meanW)

    # Print summary table
    print("\nSummary:")
    for (a,b), P, A, W, NL in zip(rects, perims, areas, meanW, neglogW):
        print(f"rect {a}x{b: <2} | perimeter={P:2d} | area={A:2d} | <W>≈ {W: .4f} | -log<W>≈ {NL: .4f}")

    if plot:
        try:
            import matplotlib.pyplot as plt
            ensure_dir("figs")
            # vs perimeter
            plt.figure()
            plt.scatter(perims, neglogW)
            # simple least-squares line
            m, c = np.polyfit(perims, neglogW, 1)
            xs = np.linspace(perims.min(), perims.max(), 100)
            plt.plot(xs, m*xs + c)
            plt.xlabel("Perimeter 2(a+b)")
            plt.ylabel("-log ⟨W(C)⟩")
            plt.title(f"Z2 gauge MC: -log⟨W⟩ vs perimeter (β={beta}, L={L})")
            out1 = "figs/mc_wilson_perimeter.png"
            plt.savefig(out1, bbox_inches="tight")
            print(f"Saved plot to {out1}")
            # vs area
            plt.figure()
            plt.scatter(areas, neglogW)
            m2, c2 = np.polyfit(areas, neglogW, 1)
            xs2 = np.linspace(areas.min(), areas.max(), 100)
            plt.plot(xs2, m2*xs2 + c2)
            plt.xlabel("Area a×b")
            plt.ylabel("-log ⟨W(C)⟩")
            plt.title(f"Z2 gauge MC: -log⟨W⟩ vs area (β={beta}, L={L})")
            out2 = "figs/mc_wilson_area.png"
            plt.savefig(out2, bbox_inches="tight")
            print(f"Saved plot to {out2}")
        except Exception as e:
            print(f"Plotting skipped: {e}")

def cli():
    ap = argparse.ArgumentParser(description="Z2 gauge MC with plaquette action; Wilson loops in XY plane.")
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--therm", type=int, default=200, help="thermalization sweeps")
    ap.add_argument("--sweeps", type=int, default=400, help="measurement sweeps after thermalization")
    ap.add_argument("--measure-every", type=int, default=10, dest="measure_every")
    ap.add_argument("--rects", type=str, default="1x1,1x2,2x2,3x3")
    ap.add_argument("--nsamples-pos", type=int, default=200, dest="nsamples_pos",
                    help="random loop starting positions per measurement")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    run(L=args.L, beta=args.beta, therm=args.therm, sweeps=args.sweeps,
        measure_every=args.measure_every, rects_str=args.rects,
        nsamples_pos=args.nsamples_pos, seed=args.seed, plot=args.plot)

if __name__ == "__main__":
    cli()
