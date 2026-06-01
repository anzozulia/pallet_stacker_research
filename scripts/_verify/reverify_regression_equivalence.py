"""
reverify_regression_equivalence.py — FOCUSED re-verification of the highest-risk
change this session: the constraint-aware BLOCK decoder (mode 4) per-bottom-box
support/load check (was block-aggregate). Confirms it STILL matches:

  (A) Cython == Numba bit-identical for mode 4 cstr blocks, driving the
      per-bottom-box rejection / 1x1 shrink path HARD via:
        - same-SKU uniform-shape boxes so k>1 / l>1 composite blocks form
        - tight support_ratio in {0.5,0.75,0.8,0.9,1.0} -> corner boxes of a
          multi-box bottom layer fail the per-box ratio -> shrink to 1x1
        - per-box rfs on/off, finite mlot, finite pallet cap, CoG envelope
        - z>0 stacking (small box height) so bottom-layer support is exercised
  (B) batch (decode_batch_blocks_njit_mode_cstr) == serial loop, bit-identical,
      for the SAME stress configs (shared _decode_cstr_blocks_loop hot path).

Plus a cross-check that this per-box change is OBSERVABLE: report how often the
block shrank (k*l after vs before would differ); we infer it from "z>0 multi-box
layers that ended up as 1x1 columns" via placement footprints. We don't need the
internal counter — bit-identity between the two independent ports under heavy
shrink traffic is the proof.

Run inside Docker (OMP_NUM_THREADS=1):
  docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
    -v "$(pwd)":/app -w /app pallet-packer:dev \
    python scripts/_verify/reverify_regression_equivalence.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pallet_packer._brkga_core import jit_decoders_cstr_cy as cy
from pallet_packer._brkga_core import jit_decoders_cstr as nb

_NO_LIMIT = 1e18


def uniform_block_dims(n_boxes, edge_xy, edge_z):
    """All boxes identical shape -> dense same-SKU composite blocks (k,l large).

    edge_z small so layers stack (z>0) and the per-bottom-box support check on
    the layer below is actually exercised.
    """
    dims = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    a, b, c = edge_xy, edge_xy, edge_z
    rots = [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]
    for i in range(n_boxes):
        for r in range(6):
            dims[i, r] = rots[r]
    n_rots = np.full(n_boxes, 6, dtype=np.int64)
    return n_rots, dims


def run_blocks(backend, order, n_rots, dims, sku, L, W, H, mp, n_skus, ca):
    n = order.shape[0]
    po = np.zeros((n, 6), dtype=np.int64)
    nbins = backend.decode_blocks_njit_mode_cstr(
        order, n_rots, dims, sku, L, W, H, mp, po, n_skus,
        ca["weights"], ca["mlot"], ca["rfs"],
        ca["pmw"], ca["support_ratio"], ca["require_centroid"],
        ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
        ca["cog_min_load_frac"], ca["cog_active"])
    return po, int(nbins)


def batch_blocks(backend, chroms, n_rots, dims, sku, L, W, H, mp, n_skus, ca):
    pop = chroms.shape[0]
    nb_boxes = n_rots.shape[0]
    po_b = np.zeros((pop, nb_boxes, 6), dtype=np.int64)
    nb_b = np.zeros(pop, dtype=np.int64)
    backend.decode_batch_blocks_njit_mode_cstr(
        chroms, n_rots, dims, sku, L, W, H, mp,
        ca["weights"], ca["mlot"], ca["rfs"],
        ca["pmw"], ca["support_ratio"], ca["require_centroid"],
        ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
        ca["cog_min_load_frac"], ca["cog_active"],
        po_b, nb_b, n_skus)
    return po_b, nb_b


def make_ca(rng, n_boxes, L, W, H, *, sr, tight_cog, finite_cap, finite_mlot, rfs_frac):
    pmw = (float(rng.uniform(0.3, 1.4)) * n_boxes * 3.0) if finite_cap else _NO_LIMIT
    mlot = (np.full(n_boxes, float(rng.uniform(3.0, 25.0)), dtype=np.float64)
            if finite_mlot else np.full(n_boxes, _NO_LIMIT, dtype=np.float64))
    if tight_cog:
        cx, cy = L / 2.0, W / 2.0
        hx = float(rng.uniform(0.05, 0.30)) * L
        hy = float(rng.uniform(0.05, 0.30)) * W
        cog = (cx - hx, cx + hx, cy - hy, cy + hy)
        cog_active = 1
    else:
        cog = (-_NO_LIMIT, _NO_LIMIT, -_NO_LIMIT, _NO_LIMIT)
        cog_active = int(rng.integers(0, 2))
    return {
        "weights": rng.uniform(0.8, 4.0, n_boxes).astype(np.float64),
        "mlot": mlot,
        "rfs": (rng.random(n_boxes) < rfs_frac).astype(np.int64),
        "pmw": float(pmw),
        "support_ratio": sr,
        "require_centroid": int(rng.integers(0, 2)),
        "cog_x_min": cog[0], "cog_x_max": cog[1],
        "cog_y_min": cog[2], "cog_y_max": cog[3],
        "cog_min_load_frac": float(rng.choice([0.0, 0.35, 1.0])),
        "cog_active": cog_active,
    }


def footprint_stats(po):
    """Return (n_placed, n_at_z_gt0) to confirm stacking/per-box paths fire."""
    placed = po[:, 5] == 1
    z_gt0 = placed & (po[:, 4] > 0)
    return int(placed.sum()), int(z_gt0.sum())


def main():
    rng = np.random.default_rng(31415926)

    SUPPORT_RATIOS = [0.5, 0.75, 0.8, 0.9, 1.0]  # all >0 -> per-box check binds
    PALLETS = [
        (1000, 800, 1200, 1),
        (1000, 800, 1200, 2),
        (600, 500, 900, 1),
        (400, 400, 800, 3),   # tiny xy -> few boxes per layer, sharp corners
        (1200, 1000, 1500, 2),
    ]
    # Edge sizes chosen so xy fits k,l in {2,3,4} and z stacks several layers.
    EDGE_XY = [120, 180, 250]
    EDGE_Z = [40, 60, 90]

    n_cy_nb = 0          # Cython-vs-Numba comparisons
    fail_cy_nb = 0
    n_batch = 0          # batch-vs-serial comparisons
    fail_batch = 0
    tot_placed = 0
    tot_z_gt0 = 0
    shrink_signal = 0    # instances with z>0 placements under tight sr (per-box path)
    rejections = 0       # instances with some box unplaced
    worst = None

    N = 500
    for it in range(N):
        L, W, H, mp = PALLETS[it % len(PALLETS)]
        n_boxes = int(rng.choice([16, 24, 36, 48, 60]))
        edge_xy = int(rng.choice(EDGE_XY))
        edge_z = int(rng.choice(EDGE_Z))
        n_rots, dims = uniform_block_dims(n_boxes, edge_xy, edge_z)

        # Few SKUs so same-SKU composite blocks (k>1,l>1) form heavily.
        n_skus = int(rng.choice([1, 1, 2, 3]))
        sku = rng.integers(0, n_skus, n_boxes).astype(np.int64)
        n_skus_eff = int(sku.max()) + 1

        sr = float(rng.choice(SUPPORT_RATIOS))
        ca = make_ca(rng, n_boxes, L, W, H, sr=sr,
                     tight_cog=(it % 4 == 0),
                     finite_cap=(it % 3 == 0),
                     finite_mlot=(it % 2 == 0),
                     rfs_frac=float(rng.choice([0.0, 0.25, 0.6, 1.0])))

        # ---- (A) single-chromosome Cython vs Numba, mode 4 ----
        order = np.argsort(rng.random(n_boxes)).astype(np.int64)
        po_cy, nb_cy = run_blocks(cy, order, n_rots, dims, sku, L, W, H, mp, n_skus_eff, ca)
        po_nb, nb_nb = run_blocks(nb, order, n_rots, dims, sku, L, W, H, mp, n_skus_eff, ca)
        n_cy_nb += 1
        npl, nz = footprint_stats(po_cy)
        tot_placed += npl
        tot_z_gt0 += nz
        if nz > 0:
            shrink_signal += 1
        if np.any(po_cy[:, 5] == 0):
            rejections += 1
        if not (np.array_equal(po_cy, po_nb) and nb_cy == nb_nb):
            fail_cy_nb += 1
            ndiff = int(np.sum(np.any(po_cy != po_nb, axis=1)))
            if worst is None or ndiff > worst[0]:
                worst = (ndiff, "cy_vs_nb", (L, W, H, mp, sr))
            if fail_cy_nb <= 3:
                rows = np.where(np.any(po_cy != po_nb, axis=1))[0]
                print(f"\n!!! CY!=NB mode4 @ {(L,W,H,mp,sr)} n_bins cy={nb_cy} nb={nb_nb}")
                print(f"    {len(rows)} differing rows; first 5:")
                for r in rows[:5]:
                    print(f"      row {r}: cy={po_cy[r].tolist()} nb={po_nb[r].tolist()}")
                print(f"    order={order.tolist()}")
                print(f"    sku={sku.tolist()} edge_xy={edge_xy} edge_z={edge_z}")
                print(f"    sr={sr} rfs={ca['rfs'].tolist()} mlot[0]={ca['mlot'][0]}"
                      f" pmw={ca['pmw']} cog_active={ca['cog_active']}")

        # ---- (B) batch vs serial loop (Cython), mode 4, same stress ----
        if it % 2 == 0:  # half the instances also batch-checked (keeps runtime sane)
            pop = int(rng.choice([1, 2, 8, 64]))
            chroms = rng.random((pop, 2 * n_boxes + 1)).astype(np.float64)
            po_b, nb_b = batch_blocks(cy, chroms, n_rots, dims, sku, L, W, H, mp, n_skus_eff, ca)
            po_s = np.zeros_like(po_b)
            nb_s = np.zeros_like(nb_b)
            for ci in range(pop):
                o = np.argsort(chroms[ci, :n_boxes]).astype(np.int64)
                nb_s[ci] = run_blocks(cy, o, n_rots, dims, sku, L, W, H, mp, n_skus_eff, ca)[1]
                # recompute placements via serial single-chrom call
                po_one, _ = run_blocks(cy, o, n_rots, dims, sku, L, W, H, mp, n_skus_eff, ca)
                po_s[ci] = po_one
            n_batch += 1
            if not (np.array_equal(po_b, po_s) and np.array_equal(nb_b, nb_s)):
                fail_batch += 1
                ndiff = sum(not np.array_equal(po_b[ci], po_s[ci]) for ci in range(pop))
                if fail_batch <= 3:
                    print(f"\n!!! BATCH!=SERIAL mode4 @ {(L,W,H,mp,sr)} pop={pop}"
                          f" {ndiff}/{pop} chroms differ; nbins_eq="
                          f"{np.array_equal(nb_b, nb_s)}")

    print("\n=== reverify regression-equivalence (mode-4 cstr blocks) ===")
    print(f"  Cython vs Numba (single-chrom): {n_cy_nb} comparisons, "
          f"{fail_cy_nb} mismatches")
    print(f"  batch vs serial (Cython):       {n_batch} comparisons, "
          f"{fail_batch} mismatches")
    print(f"  behaviour exercised: total_placed={tot_placed} "
          f"total_z>0_placements={tot_z_gt0} "
          f"instances_with_z>0(per-box path)={shrink_signal}/{n_cy_nb} "
          f"instances_with_rejections={rejections}/{n_cy_nb}")
    if worst is not None:
        print(f"  worst mismatch: {worst}")

    ok = (fail_cy_nb == 0 and fail_batch == 0)
    # Guard: if the per-box stacking path never fired, the probe proved nothing.
    if shrink_signal == 0:
        print("  WARNING: no z>0 placements -> per-box bottom-layer path NOT "
              "exercised; probe inconclusive")
        ok = False
    print(f"\n=== RESULT: {'PASS' if ok else 'FAIL'} "
          f"({n_cy_nb + n_batch} comparisons, "
          f"{fail_cy_nb + fail_batch} mismatches) ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
