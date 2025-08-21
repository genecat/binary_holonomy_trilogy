# Research Roadmap (paper → working demos)

This repo tracks steps toward the trilogy (ϕ clock, A_{μν} info-geometry, ±1 holonomies) becoming a credible QG program.

## Stage 1 → 2: Emergent U(1) (next)
- [ ] Implement Wilson loops \(W(C)=\exp(i\sum_{e\in C}\theta_e)\) on rectangular loops.
- [ ] Measure \(\langle W(C)\rangle\) vs loop **perimeter** and **area**; target = **perimeter law** (deconfined U(1)).
- [ ] Photon sector: extract transverse 2-polarization dispersion; check \(\omega^2 \approx k^2\) at small |k|.
- [ ] Plot: perimeter vs \(-\log\langle W\rangle\); dispersion \(\omega(k)\) vs |k|.

## Stage 2 → 3: Spin-2 in the IR (TT sector)
- [ ] Increase L, tsteps; fit \(\omega(k)\) from TT correlators (not just peak-pick).
- [ ] Directional scans (kx, ky, kz) to test **isotropy** (no preferred frame).
- [ ] Check linear \(\omega \approx c|k|\) at small |k|; bound any Lorentz-violation.

## Stage 3: Constraints & symmetries
- [ ] Discrete Bianchi identities → continuum Bianchi.
- [ ] Begin reconstructing Dirac (hypersurface-deformation) algebra numerically in the IR.

## Stage 4: Exact stabilizer entropy (optional)
- [ ] Add binary symplectic reduction to get exact stabilizer \(S_A\); compare to crossing-rank estimator (area law).

## CI & reproducibility
- [ ] Add unit tests for Wilson loops & dispersion fits.
- [ ] Keep headless plots in CI; save figures under `figs/` with run metadata.

**Milestone tags**
- v0.2.0 — Wilson loops + photon dispersion (U(1) deconfinement)
- v0.3.0 — Improved TT dispersion & isotropy tests
- v0.4.0 — Exact stabilizer entropy; Bianchi checks
