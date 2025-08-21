from __future__ import annotations
import numpy as np
import argparse


def build_TT_projector(kvec: tuple[int, int, int]) -> np.ndarray:
    d = np.eye(3)

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
    P = d - np.outer(k, k)
    PTT = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for l in range(3):
                for m in range(3):
                    PTT[i, j, l, m] = 0.5 * (P[i, l] * P[j, m] + P[i, m] * P[j, l] - P[i, j] * P[l, m])
    return PTT


def run_spin2_demo(L: int = 12, tsteps: int = 48):
    rng = np.random.default_rng(1234)
    h = rng.standard_normal((tsteps, L, L, L, 3, 3)) * 0.2

    injected_k = (1, 1, 0)
    kphase = (
        (2.0 * np.pi / L) * (injected_k[0] * np.arange(L)[:, None, None]
                             + injected_k[1] * np.arange(L)[None, :, None]
                             + injected_k[2] * np.arange(L)[None, None, :])
    )
    omega_inj = 0.35 * np.pi
    tgrid = np.arange(tsteps)
    signal = 0.8 * np.sin(omega_inj * tgrid)[:, None, None, None] * np.cos(kphase)[None, :, :, :]
    h[..., 0, 1] += signal
    h[..., 1, 0] += signal

    h_fft = np.fft.fftn(h, axes=(1, 2, 3))
    k_list = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0)]

    results = []
    for k_idx in k_list:
        kx, ky, kz = k_idx
        hk_t = h_fft[:, kx % L, ky % L, kz % L, :, :]
        PTT = build_TT_projector(k_idx)
        hk_tt = np.einsum("ijkl, tkl -> tij", PTT, hk_t, optimize=True)
        s_t = np.linalg.norm(hk_tt.reshape(tsteps, -1), axis=1)

        S_f = np.fft.rfft(s_t)
        freqs = np.fft.rfftfreq(tsteps, d=1.0)
        if len(freqs) > 1:
            idx_peak = 1 + np.argmax(np.abs(S_f[1:]))
            omega = 2.0 * np.pi * freqs[idx_peak]
        else:
            omega = 0.0

        k_phys = (2.0 * np.pi / L) * np.linalg.norm(np.array(k_idx, dtype=float))
        vph = (omega / k_phys) if k_phys > 0 else 0.0
        results.append((np.linalg.norm(k_idx), k_phys, omega, vph))

    print("L =", L, "| tsteps =", tsteps)
    print("k (index-norm):", [f"{r[0]:.3f}" for r in results])
    print("|k| (phys rad/lu):", [f"{r[1]:.3f}" for r in results])
    print("ω(k) (rad/step):", [f"{r[2]:.3f}" for r in results])
    print("phase velocity ω/|k|:", [f"{r[3]:.3f}" for r in results])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spin-2 TT projection demo")
    parser.add_argument("--L", type=int, default=12, help="Lattice size")
    parser.add_argument("--tsteps", type=int, default=48, help="Number of time steps")
    args = parser.parse_args()
    run_spin2_demo(L=args.L, tsteps=args.tsteps)
