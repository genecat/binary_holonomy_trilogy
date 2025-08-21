from __future__ import annotations
import numpy as np
import argparse

def build_lattice_links(L: int):
    links = []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for d in range(3):
                    links.append((x, y, z, d))
    idx_of = {lnk: i for i, lnk in enumerate(links)}
    return links, idx_of

def plaquette_links(x, y, z, plane, L):
    if plane == 'xy':
        return [
            (x, y, z, 0),
            ((x+1) % L, y, z, 1),
            (x, (y+1) % L, z, 0),
            (x, y, z, 1),
        ]
    if plane == 'yz':
        return [
            (x, y, z, 1),
            (x, y, (z+1) % L, 2),
            (x, (y+1) % L, z, 1),
            (x, y, z, 2),
        ]
    if plane == 'zx':
        return [
            (x, y, z, 2),
            (x, y, z, 0),
            ((x+1) % L, y, z, 2),
            (x, y, z, 0),
        ]
    raise ValueError("plane must be 'xy','yz','zx'")

def build_HZ(L: int, idx_of: dict) -> np.ndarray:
    rows = []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for plane in ('xy','yz','zx'):
                    row = np.zeros(len(idx_of), dtype=np.uint8)
                    for lnk in plaquette_links(x, y, z, plane, L):
                        row[idx_of[lnk]] ^= 1
                    rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, len(idx_of)), dtype=np.uint8)

def star_links(x, y, z, L):
    return [
        (x, y, z, 0),
        ((x-1) % L, y, z, 0),
        (x, y, z, 1),
        (x, (y-1) % L, z, 1),
        (x, y, z, 2),
        (x, y, (z-1) % L, 2),
    ]

def build_HX(L: int, idx_of: dict) -> np.ndarray:
    rows = []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                row = np.zeros(len(idx_of), dtype=np.uint8)
                for lnk in star_links(x, y, z, L):
                    row[idx_of[lnk]] ^= 1
                rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, len(idx_of)), dtype=np.uint8)

def gf2_rank(M: np.ndarray) -> int:
    A = M.copy().astype(np.uint8)
    m, n = A.shape
    r = 0
    for c in range(n):
        pivot = None
        for rr in range(r, m):
            if A[rr, c]:
                pivot = rr
                break
        if pivot is None:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        for rr in range(m):
            if rr != r and A[rr, c]:
                A[rr, :] ^= A[r, :]
        r += 1
        if r == m:
            break
    return r

def planar_cut_A_indices(L: int, links: list, side: str = "x", cut_at: int = 1) -> np.ndarray:
    mask = np.zeros(len(links), dtype=bool)
    for i, (x, y, z, d) in enumerate(links):
        if side == "x":
            if x <= cut_at:
                mask[i] = True
        elif side == "y":
            if y <= cut_at:
                mask[i] = True
        elif side == "z":
            if z <= cut_at:
                mask[i] = True
        else:
            raise ValueError("side must be 'x','y','z'")
    return mask

def entropy_css_crossing_estimate(HZ: np.ndarray, HX: np.ndarray, A_mask: np.ndarray) -> float:
    ln2 = np.log(2.0)
    HZ_A = HZ[:, A_mask]
    HX_A = HX[:, A_mask]
    rZ = gf2_rank(HZ_A)
    rX = gf2_rank(HX_A)
    return 0.5 * (rZ + rX) * ln2

def demo_entropy_curve(L: int = 6, side: str = "x"):
    links, idx_of = build_lattice_links(L)
    HZ = build_HZ(L, idx_of)
    HX = build_HX(L, idx_of)

    # simple proxy for "area" for x-cuts only
    def plane_area_x(cut):
        count = 0
        for (x, y, z, d) in links:
            if d == 0:
                if x == (cut % L):
                    count += 1
            else:
                if (x == ((cut + 1) % L)):
                    count += 1
        return count

    print(f"L={L}, n_links={len(links)}")
    for cut in range(1, L-1):
        A_mask = planar_cut_A_indices(L, links, side=side, cut_at=cut)
        S_cross = entropy_css_crossing_estimate(HZ, HX, A_mask)
        if side == "x":
            area = plane_area_x(cut)
            area_str = f"{area:4d}"
        else:
            area_str = " n/a"
        print(f"cut at {side}={cut:2d} | Area≈ {area_str} | S_cross≈ {S_cross:8.3f} nats")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entropy cut demo")
    parser.add_argument("--L", type=int, default=6, help="Lattice size")
    parser.add_argument("--side", type=str, default="x", choices=["x","y","z"], help="Cut orientation")
    args = parser.parse_args()
    demo_entropy_curve(L=args.L, side=args.side)
