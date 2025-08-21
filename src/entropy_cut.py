# src/entropy_cut.py
# Minimal CSS (stabilizer) lattice toy to show an "area-law"-like signal
# using a robust crossing-rank estimator for a planar bipartition.

from __future__ import annotations
import numpy as np

def build_lattice_links(L: int):
    """
    Build oriented links on a 3D LxLxL cubic lattice with periodic boundaries.
    Each link is identified by (x,y,z,dir) with dir in {0:x, 1:y, 2:z}.
    Return:
      - links: list of tuples (x,y,z,dir)
      - idx_of: dict mapping link tuple -> integer index
    """
    links = []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for d in range(3):
                    links.append((x, y, z, d))
    idx_of = {lnk: i for i, lnk in enumerate(links)}
    return links, idx_of

def plaquette_links(x, y, z, plane, L):
    """
    Return the 4 links forming a plaquette at (x,y,z) in plane ∈ {'xy','yz','zx'}.
    Links are oriented along +x or +y or +z directions on the torus.
    """
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
    raise ValueError("plane must be 'xy','yz', or 'zx'")

def build_HZ(L: int, idx_of: dict) -> np.ndarray:
    """
    Z-type plaquette checks on all xy, yz, zx faces.
    Each row is a binary indicator over links that the plaquette touches.
    """
    rows = []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for plane in ('xy','yz','zx'):
                    row = np.zeros(len(idx_of), dtype=np.uint8)
                    for lnk in plaquette_links(x, y, z, plane, L):
                        row[idx_of[lnk]] ^= 1  # mod 2
                    rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, len(idx_of)), dtype=np.uint8)

def star_links(x, y, z, L):
    """ six incident +direction links at a vertex """
    return [
        (x, y, z, 0),
        ((x-1) % L, y, z, 0),
        (x, y, z, 1),
        (x, (y-1) % L, z, 1),
        (x, y, z, 2),
        (x, y, (z-1) % L, 2),
    ]

def build_HX(L: int, idx_of: dict) -> np.ndarray:
    """
    X-type star (vertex) checks: one per lattice site, acts on 6 incident +direction links.
    """
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
    """
    Rank over GF(2) via Gaussian elimination.
    """
    A = M.copy().astype(np.uint8)
    m, n = A.shape
    r = 0
    for c in range(n):
        # find pivot
        pivot = None
        for rr in range(r, m):
            if A[rr, c]:
                pivot = rr
                break
        if pivot is None:
            continue
        # swap
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        # eliminate other rows
        for rr in range(m):
            if rr != r and A[rr, c]:
                A[rr, :] ^= A[r, :]
        r += 1
        if r == m:
            break
    return r

def planar_cut_A_indices(L: int, links: list, side: str = "x", cut_at: int = 1) -> np.ndarray:
    """
    Select link indices on one side of a planar cut (<= cut_at) along x|y|z.
    Returns a boolean mask over links.
    """
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
    """
    Robust area-law-like estimator:
      S_A ~ (rank(HZ_A) + rank(HX_A)) * (ln 2)/2
    where H*_A are the submatrices restricted to qubits in region A.
    """
    ln2 = np.log(2.0)
    HZ_A = HZ[:, A_mask]
    HX_A = HX[:, A_mask]
    rZ = gf2_rank(HZ_A)
    rX = gf2_rank(HX_A)
    return 0.5 * (rZ + rX) * ln2

def demo_entropy_curve(L: int = 6, side: str = "x"):
    """
    Build HZ, HX; sweep planar cuts; print area vs estimator.
    """
    links, idx_of = build_lattice_links(L)
    HZ = build_HZ(L, idx_of)
    HX = build_HX(L, idx_of)

    # area of an x = const plane in link units (rough proxy):
    # count links that cross the plane between x=cut and x=cut+1
    def plane_area_x(cut):
        count = 0
        for (x, y, z, d) in links:
            if d == 0:
                # x-link crossing the plane between cut and cut+1
                if x == (cut % L):
                    count += 1
            else:
                # y/z links lie in the plane at x=cut+1 (count them as boundary dof)
                if (x == ((cut + 1) % L)):
                    count += 1
        return count

    print(f"L={L}, n_links={len(links)}")
    for cut in range(1, L-1):
        A_mask = planar_cut_A_indices(L, links, side=side, cut_at=cut)
        S_cross = entropy_css_crossing_estimate(HZ, HX, A_mask)
        area = plane_area_x(cut) if side == "x" else None
        print(f"cut at {side}={cut:2d} | Area≈ {area:4d} | S_cross≈ {S_cross:8.3f} nats")


if __name__ == "__main__":
    demo_entropy_curve(L=6, side="x")
