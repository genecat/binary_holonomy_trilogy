# src/sim_spin2.py
# Minimal, self-contained demo:
# - builds TT projector
# - extracts a spatial Fourier mode h(k, t)
# - projects to TT
# - gets temporal peak frequency ω(k)
#
# Safe to run on a laptop (small L, small tsteps).

from __future__ import annotations
import numpy as np


def build_TT_projector(kvec: tuple[int, int, int]) -> np.ndarray:
    """
    Construct the transverse-traceless projector P_TT for a given k-vector
    in 3D. Returns a rank-4 tensor P_ij,kl.

    For k = 0 (no direction), we fall back to a pure traceless projector:
        P_TT,ij,kl = 0.5 * (δik δjl + δil δjk) - (1/3) δij δkl
    """
    d = np.eye(3)

    # Zero-k fallback: traceless part
    if kvec == (0, 0, 0):
        PTT = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        PTT[i, j, k, l] = 0.5 * (d[i, k] * d[j, l] + d[i, l] * d[j, k]) - (1.0 / 3.0) * d[i, j] * d[k, l]
        return PTT

    k = np.array(kvec, dtype=float)
    k = k / np.linalg.norm(k)
    P = d - np.outer(k, k)  # transverse projector in 3D

    # Standard TT projector:
    # P_TT,ij,kl = 0.5 * (P_ik P_jl + P_il P_jk) - 0.5 * (P_ij P_kl)
    PTT = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for l in range(3):
                for m in range(3):
                    PTT[i, j, l, m] = 0.5 * (P[i, l] * P[j, m] + P[i, m] * P[j, l] - P[i, j] * P[l, m])
    return PTT


def run_spin2_demo():
    """
    Pipeline:
      - synthesize small metric fluctuation h(t, x, y, z, i, j)
      - spatial FFT -> h_fft(t, kx, ky, kz, i, j)
      - pick a few lattice wave-vectors k
      - TT project each h_fft(:, k) in polarization indices
      - take the norm vs time and get peak ω from its temporal spectrum
    """
    # Small sizes to keep it light
    L = 12           # lattice length per spatial dimension
    tsteps = 48      # number of time steps
    rng = np.random.default_rng(1234)

    # Synthetic metric fluctuations (random noise + a mild oscillatory component)
    # Shape: (t, x, y, z, i, j) with i,j in {0..2}
    h = rng.standard_normal((tsteps, L, L, L, 3, 3)) * 0.2

    # Add a gentle oscillatory signal at a known lattice wave-vector to make peaks clearer
    injected_k = (1, 1, 0)  # lattice mode to inject
    kphase = (
        (2.0 * np.pi / L) * (injected_k[0] * np.arange(L)[:, None, None]
                             + injected_k[1] * np.arange(L)[None, :, None]
                             + injected_k[2] * np.arange(L)[None, None, :])
    )  # shape (L, L, L)
    # temporal frequency for the injected signal (arbitrary but < Nyquist)
    omega_inj = 0.35 * np.pi
    tgrid = np.arange(tsteps)

    # Put a small TT-like oscillation in (i,j)=(0,1) & (1,0)
    # (doesn't have to be perfect TT; projector will clean it)
    signal = 0.8 * np.sin(omega_inj * tgrid)[:, None, None, None] * np.cos(kphase)[None, :, :, :]
    h[..., 0, 1] += signal
    h[..., 1, 0] += signal

    # Spatial FFT over x,y,z only
    h_fft = np.fft.fftn(h, axes=(1, 2, 3))  # shape: (t, kx, ky, kz, i, j)

    # Choose a few lattice k-vectors to analyze (in index space, not physical units yet)
    k_list = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0)]

    results = []
    for k_idx in k_list:
        kx, ky, kz = k_idx
        # Take the complex 3x3 mode over time: (t, i, j)
        hk_t = h_fft[:, kx % L, ky % L, kz % L, :, :]

        # Build TT projector for this k
        PTT = build_TT_projector(k_idx)

        # Apply TT projection in polarization indices (i,j):
        # hk_tt[t, i, j] = PTT[i,j,k,l] * hk_t[t,k,l]
        hk_tt = np.einsum("ijkl, tkl -> tij", PTT, hk_t, optimize=True)

        # Reduce to a scalar time series (norm over i,j) so we can get a clean spectrum
        s_t = np.linalg.norm(hk_tt.reshape(tsteps, -1), axis=1)

        # Temporal spectrum and peak frequency (exclude DC at index 0)
        S_f = np.fft.rfft(s_t)
        freqs = np.fft.rfftfreq(tsteps, d=1.0)  # 'd' is the time step; using 1.0 arbitrary units
        if len(freqs) > 1:
            idx_peak = 1 + np.argmax(np.abs(S_f[1:]))
            omega = 2.0 * np.pi * freqs[idx_peak]  # convert cycles/step -> rad/step
        else:
            omega = 0.0

        # Physical lattice |k| in rad per lattice unit
        k_phys = (2.0 * np.pi / L) * np.linalg.norm(np.array(k_idx, dtype=float))

        vph = (omega / k_phys) if k_phys > 0 else 0.0
        results.append((np.linalg.norm(k_idx), k_phys, omega, vph))

    # Pretty print
    print("L =", L, "| tsteps =", tsteps)
    print("k (index-norm):", [f"{r[0]:.3f}" for r in results])
    print("|k| (phys rad/lu):", [f"{r[1]:.3f}" for r in results])
    print("ω(k) (rad/step):", [f"{r[2]:.3f}" for r in results])
    print("phase velocity ω/|k|:", [f"{r[3]:.3f}" for r in results])


if __name__ == "__main__":
    run_spin2_demo()
