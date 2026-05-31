"""
verify_backend_equiv_cstr.py — adversarial bit-identity check between the
Cython constraint-aware decoders and their Numba reference twins.

Decoders under test:
  decode_njit_mode_cstr     modes 0/1/2 (DFTRC / wall / corner)
  decode_layer_njit_cstr    mode 3      (Bischoff-Ratcliff layers)
  decode_blocks_njit_mode_cstr mode 4   (dynamic blocks)

A LARGE sweep (>=1500 instances) exercises EVERY constraint guard:
  - pallet_max_weight     finite / infinite (1e18 sentinel)
  - per-box mlot          finite / infinite mix
  - rfs                   requires_full_support on/off mix
  - support_ratio         {0.0, 0.5, 0.8, 1.0}
  - require_centroid      0 / 1
  - cog_active            0 / 1
  - cog envelope          tight (forces rejections) / loose
  - cog_min_load_frac     {0.0, 0.35, 1.0}
  - max_overhang          0 and >0 (caller inflates L/W -> we feed L_eff/W_eff)

For every (instance, decoder) we assert placements_out and n_bins are
bit-identical between Cython and Numba. The first mismatch dumps full args
for a minimal reproduction.

Run (from repo root):
  docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
    -v "$(pwd)":/app -w /app pallet-packer:dev \
    python scripts/_verify/verify_backend_equiv_cstr.py
"""
from __future__ import annotations

import os
import sys
import itertools

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import numpy as np

from pallet_packer._brkga_core import jit_decoders_cstr_cy as cy
from pallet_packer._brkga_core import jit_decoders_cstr as nb

_NO_LIMIT = 1e18  # mirrors jit_constraints._NO_LIMIT


# ---------------------------------------------------------------------------
# Instance / config generation
# ---------------------------------------------------------------------------

def make_boxes(rng, n_boxes, L, W, H):
    """Build dims_all (n,6,3) of int64 rotations + n_rots_per_box.

    Box edge sizes are deliberately a mix of small (forces stacking ->
    z>0 placements -> exercises support/load/centroid paths) and larger.
    """
    n_rots_per_box = np.full(n_boxes, 6, dtype=np.int64)
    dims_all = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    for i in range(n_boxes):
        # Bias toward small boxes so we get tall stacks and dense layers.
        bx = int(rng.integers(30, max(31, L // 3)))
        by = int(rng.integers(30, max(31, W // 3)))
        bz = int(rng.integers(30, max(31, H // 3)))
        rots = [
            (bx, by, bz), (bx, bz, by),
            (by, bx, bz), (by, bz, bx),
            (bz, bx, by), (bz, by, bx),
        ]
        for r in range(6):
            dims_all[i, r] = rots[r]
    return n_rots_per_box, dims_all


def make_order(rng, n_boxes):
    """A permutation of [0..n_boxes) — the already-sorted bps_order.

    We argsort a random chromosome (exactly what the driver does) so the
    order distribution matches production. Identical order to both backends.
    """
    chrom = rng.random(n_boxes)
    return np.argsort(chrom).astype(np.int64)


def make_sku(rng, n_boxes, n_skus):
    return rng.integers(0, n_skus, n_boxes).astype(np.int64)


# Guard-value menus.
SUPPORT_RATIOS = [0.0, 0.5, 0.8, 1.0]
COG_MIN_LOAD_FRACS = [0.0, 0.35, 1.0]


def make_constraint_config(rng, n_boxes, L, W, H, *, force_tight_cog=False,
                           force_inf_weight=False):
    """Random constraint config touching every guard.

    Returns a dict of decoder kwargs (constraint args only).
    """
    # Pallet weight cap: finite (tight enough to bind) or infinite.
    total_box_weight_est = n_boxes * 3.0
    if force_inf_weight or rng.random() < 0.4:
        pallet_max_weight = _NO_LIMIT
    else:
        # Finite cap somewhere between "binds hard" and "binds loosely".
        frac = float(rng.uniform(0.2, 1.5))
        pallet_max_weight = max(5.0, total_box_weight_est * frac)

    weights = rng.uniform(0.5, 6.0, n_boxes).astype(np.float64)

    # mlot: mix of finite (some tight) and infinite per box.
    mlot = np.empty(n_boxes, dtype=np.float64)
    for i in range(n_boxes):
        rv = rng.random()
        if rv < 0.35:
            mlot[i] = _NO_LIMIT
        elif rv < 0.7:
            mlot[i] = float(rng.uniform(2.0, 30.0))   # finite, can bind
        else:
            mlot[i] = float(rng.uniform(30.0, 200.0))  # finite, loose
    # rfs: per-box requires_full_support mix.
    rfs = (rng.random(n_boxes) < float(rng.uniform(0.0, 0.6))).astype(np.int64)

    support_ratio = float(rng.choice(SUPPORT_RATIOS))
    require_centroid = int(rng.integers(0, 2))
    cog_min_load_frac = float(rng.choice(COG_MIN_LOAD_FRACS))
    cog_active = int(rng.integers(0, 2))

    if cog_active and (force_tight_cog or rng.random() < 0.5):
        # Tight envelope centred near pallet centre — forces rejections.
        cx = L / 2.0
        cy = W / 2.0
        halfx = float(rng.uniform(0.02, 0.20)) * L
        halfy = float(rng.uniform(0.02, 0.20)) * W
        cog_x_min, cog_x_max = cx - halfx, cx + halfx
        cog_y_min, cog_y_max = cy - halfy, cy + halfy
    else:
        cog_x_min, cog_x_max = -_NO_LIMIT, _NO_LIMIT
        cog_y_min, cog_y_max = -_NO_LIMIT, _NO_LIMIT

    return {
        "weights": weights,
        "mlot": mlot,
        "rfs": rfs,
        "pallet_max_weight": float(pallet_max_weight),
        "support_ratio": support_ratio,
        "require_centroid": require_centroid,
        "cog_x_min": float(cog_x_min), "cog_x_max": float(cog_x_max),
        "cog_y_min": float(cog_y_min), "cog_y_max": float(cog_y_max),
        "cog_min_load_frac": cog_min_load_frac,
        "cog_active": cog_active,
    }


# ---------------------------------------------------------------------------
# Single-decoder runners (Cython + Numba), returning (placements_out, n_bins)
# ---------------------------------------------------------------------------

def run_cstr_012(backend, order, n_rots, dims, L, W, H, mp, mode, ca):
    n = order.shape[0]
    po = np.zeros((n, 6), dtype=np.int64)
    nb_ = backend.decode_njit_mode_cstr(
        order, n_rots, dims, L, W, H, mp, po, mode,
        ca["weights"], ca["mlot"], ca["rfs"],
        ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
        ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
        ca["cog_min_load_frac"], ca["cog_active"],
    )
    return po, int(nb_)


def run_cstr_layer(backend, order, n_rots, dims, L, W, H, mp, ca):
    n = order.shape[0]
    po = np.zeros((n, 6), dtype=np.int64)
    nb_ = backend.decode_layer_njit_cstr(
        order, n_rots, dims, L, W, H, mp, po,
        ca["weights"], ca["mlot"], ca["rfs"],
        ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
        ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
        ca["cog_min_load_frac"], ca["cog_active"],
    )
    return po, int(nb_)


def run_cstr_blocks(backend, order, n_rots, dims, sku, L, W, H, mp, n_skus, ca):
    n = order.shape[0]
    po = np.zeros((n, 6), dtype=np.int64)
    nb_ = backend.decode_blocks_njit_mode_cstr(
        order, n_rots, dims, sku, L, W, H, mp, po, n_skus,
        ca["weights"], ca["mlot"], ca["rfs"],
        ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
        ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
        ca["cog_min_load_frac"], ca["cog_active"],
    )
    return po, int(nb_)


# ---------------------------------------------------------------------------
# Comparison + diagnostics
# ---------------------------------------------------------------------------

def dump_mismatch(label, meta, order, n_rots, dims, sku, ca, po_cy, nb_cy,
                  po_nb, nb_nb):
    print(f"\n!!! MISMATCH: {label}  {meta}")
    print(f"    n_bins  cython={nb_cy}  numba={nb_nb}")
    rows_differ = [i for i in range(po_cy.shape[0])
                   if not np.array_equal(po_cy[i], po_nb[i])]
    print(f"    differing placement rows: {len(rows_differ)} / {po_cy.shape[0]}")
    for i in rows_differ[:6]:
        print(f"      row {i:3d}: cy={po_cy[i].tolist()}  nb={po_nb[i].tolist()}")
    print("    --- minimal repro args ---")
    print(f"    L,W,H,max_pallets = {meta}")
    print(f"    order = {order.tolist()}")
    print(f"    n_rots_per_box = {n_rots.tolist()}")
    print(f"    dims_all = {dims.tolist()}")
    if sku is not None:
        print(f"    sku_id_per_box = {sku.tolist()}")
    print(f"    weights = {ca['weights'].tolist()}")
    print(f"    mlot = {ca['mlot'].tolist()}")
    print(f"    rfs = {ca['rfs'].tolist()}")
    print(f"    pallet_max_weight = {ca['pallet_max_weight']}")
    print(f"    support_ratio = {ca['support_ratio']}")
    print(f"    require_centroid = {ca['require_centroid']}")
    print(f"    cog envelope = x[{ca['cog_x_min']},{ca['cog_x_max']}] "
          f"y[{ca['cog_y_min']},{ca['cog_y_max']}]")
    print(f"    cog_min_load_frac = {ca['cog_min_load_frac']}  "
          f"cog_active = {ca['cog_active']}")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(20260531)

    # Pallet shapes to vary geometry (integer dims as the decoder sees them).
    PALLETS = [
        (1000, 800, 1200, 1),
        (1200, 1000, 1500, 1),
        (600, 400, 500, 1),
        (1000, 800, 1200, 2),   # >1 pallet — exercises multi-bin paths
        (800, 600, 900, 3),
        (300, 300, 400, 1),     # tiny -> dense stacking, many z>0 placements
    ]
    N_BOXES_CHOICES = [12, 24, 40, 60]
    OVERHANGS = [0, 50]  # >0 inflates L/W as the driver does

    # Track guard coverage so we can prove the sweep actually hit everything.
    cov = {
        "pmw_finite": 0, "pmw_inf": 0,
        "mlot_finite": 0, "mlot_inf": 0,
        "rfs_any": 0, "rfs_none": 0,
        "sr": {s: 0 for s in SUPPORT_RATIOS},
        "rc0": 0, "rc1": 0,
        "cog0": 0, "cog1": 0,
        "cog_tight": 0, "cog_loose": 0,
        "cmlf": {f: 0 for f in COG_MIN_LOAD_FRACS},
        "overhang0": 0, "overhangP": 0,
        "z_gt0_seen": 0,      # instances where some box placed at z>0
        "rejections_seen": 0,  # instances where some box NOT placed (col5==0)
        "multibin": 0,         # instances using >1 bin
    }

    # Per-decoder counters.
    decoders = ["cstr0", "cstr1", "cstr2", "cstr3_layer", "cstr4_blocks"]
    n_compared = {d: 0 for d in decoders}
    n_fail = {d: 0 for d in decoders}
    first_fail_dumped = False

    # Target >=1500 instances. Each instance is compared across all 5 decoders.
    N_INSTANCES = 360  # * 5 decoders = 1800 comparisons

    worst = None  # (n_diff_rows, label, meta) for reporting

    for it in range(N_INSTANCES):
        L0, W0, H, mp = PALLETS[it % len(PALLETS)]
        n_boxes = int(rng.choice(N_BOXES_CHOICES))
        overhang = OVERHANGS[it % len(OVERHANGS)]
        L = L0 + 2 * overhang
        W = W0 + 2 * overhang
        if overhang == 0:
            cov["overhang0"] += 1
        else:
            cov["overhangP"] += 1

        n_rots, dims = make_boxes(rng, n_boxes, L, W, H)
        order = make_order(rng, n_boxes)
        n_skus = int(rng.integers(1, 5))
        sku = make_sku(rng, n_boxes, n_skus)

        # Occasionally force tight CoG / inf weight to guarantee coverage.
        ca = make_constraint_config(
            rng, n_boxes, L, W, H,
            force_tight_cog=(it % 7 == 0),
            force_inf_weight=(it % 5 == 0),
        )

        # Coverage bookkeeping.
        if ca["pallet_max_weight"] >= _NO_LIMIT:
            cov["pmw_inf"] += 1
        else:
            cov["pmw_finite"] += 1
        if np.any(ca["mlot"] < _NO_LIMIT):
            cov["mlot_finite"] += 1
        if np.any(ca["mlot"] >= _NO_LIMIT):
            cov["mlot_inf"] += 1
        if np.any(ca["rfs"] != 0):
            cov["rfs_any"] += 1
        else:
            cov["rfs_none"] += 1
        cov["sr"][ca["support_ratio"]] += 1
        cov["rc1" if ca["require_centroid"] else "rc0"] += 1
        cov["cog1" if ca["cog_active"] else "cog0"] += 1
        if ca["cog_active"]:
            tight = ca["cog_x_max"] < _NO_LIMIT
            cov["cog_tight" if tight else "cog_loose"] += 1
        cov["cmlf"][ca["cog_min_load_frac"]] += 1

        meta = (L, W, H, mp)

        # ---- cstr modes 0/1/2 ----
        for mode in (0, 1, 2):
            po_cy, nb_cy = run_cstr_012(cy, order, n_rots, dims, L, W, H, mp, mode, ca)
            po_nb, nb_nb = run_cstr_012(nb, order, n_rots, dims, L, W, H, mp, mode, ca)
            key = f"cstr{mode}"
            n_compared[key] += 1
            # Behaviour observation (use Cython output as reference).
            if np.any(po_cy[:, 4] > 0):
                cov["z_gt0_seen"] += 1
            if np.any(po_cy[:, 5] == 0):
                cov["rejections_seen"] += 1
            if nb_cy > 1:
                cov["multibin"] += 1
            if not (np.array_equal(po_cy, po_nb) and nb_cy == nb_nb):
                n_fail[key] += 1
                ndiff = int(np.sum(np.any(po_cy != po_nb, axis=1)))
                if worst is None or ndiff > worst[0]:
                    worst = (ndiff, key, meta)
                if not first_fail_dumped:
                    dump_mismatch(key, meta, order, n_rots, dims, None, ca,
                                  po_cy, nb_cy, po_nb, nb_nb)
                    first_fail_dumped = True

        # ---- cstr mode 3 (layer) ----
        po_cy, nb_cy = run_cstr_layer(cy, order, n_rots, dims, L, W, H, mp, ca)
        po_nb, nb_nb = run_cstr_layer(nb, order, n_rots, dims, L, W, H, mp, ca)
        n_compared["cstr3_layer"] += 1
        if not (np.array_equal(po_cy, po_nb) and nb_cy == nb_nb):
            n_fail["cstr3_layer"] += 1
            ndiff = int(np.sum(np.any(po_cy != po_nb, axis=1)))
            if worst is None or ndiff > worst[0]:
                worst = (ndiff, "cstr3_layer", meta)
            if not first_fail_dumped:
                dump_mismatch("cstr3_layer", meta, order, n_rots, dims, None, ca,
                              po_cy, nb_cy, po_nb, nb_nb)
                first_fail_dumped = True

        # ---- cstr mode 4 (blocks) ----
        po_cy, nb_cy = run_cstr_blocks(cy, order, n_rots, dims, sku, L, W, H, mp, n_skus, ca)
        po_nb, nb_nb = run_cstr_blocks(nb, order, n_rots, dims, sku, L, W, H, mp, n_skus, ca)
        n_compared["cstr4_blocks"] += 1
        if not (np.array_equal(po_cy, po_nb) and nb_cy == nb_nb):
            n_fail["cstr4_blocks"] += 1
            ndiff = int(np.sum(np.any(po_cy != po_nb, axis=1)))
            if worst is None or ndiff > worst[0]:
                worst = (ndiff, "cstr4_blocks", meta)
            if not first_fail_dumped:
                dump_mismatch("cstr4_blocks", meta, order, n_rots, dims, sku, ca,
                              po_cy, nb_cy, po_nb, nb_nb)
                first_fail_dumped = True

    # ---- Targeted edge-case battery (deterministic, no RNG) ----
    print("=== Edge-case battery (extreme guard values) ===")
    edge_fail = run_edge_cases()

    # ---- Epsilon-boundary / tie / truncation adversarial battery ----
    print("\n=== Epsilon-boundary adversarial battery ===")
    eps_fail, eps_comp = run_epsilon_battery()

    total_comp = sum(n_compared.values())
    total_fail = sum(n_fail.values())

    print("\n=== Coverage (guards actually exercised across instances) ===")
    print(f"  pallet_max_weight: finite={cov['pmw_finite']} inf={cov['pmw_inf']}")
    print(f"  mlot: finite_present={cov['mlot_finite']} inf_present={cov['mlot_inf']}")
    print(f"  rfs: any_on={cov['rfs_any']} all_off={cov['rfs_none']}")
    print(f"  support_ratio buckets: {cov['sr']}")
    print(f"  require_centroid: off={cov['rc0']} on={cov['rc1']}")
    print(f"  cog_active: off={cov['cog0']} on={cov['cog1']} "
          f"(tight={cov['cog_tight']} loose={cov['cog_loose']})")
    print(f"  cog_min_load_frac buckets: {cov['cmlf']}")
    print(f"  max_overhang: zero={cov['overhang0']} positive={cov['overhangP']}")
    print(f"  behaviour: z>0 placements seen={cov['z_gt0_seen']} "
          f"rejections(col5==0) seen={cov['rejections_seen']} "
          f"multibin seen={cov['multibin']}")

    print("\n=== Per-decoder bit-identity results ===")
    for d in decoders:
        status = "PASS" if n_fail[d] == 0 else "FAIL"
        print(f"  {d:14s}: compared={n_compared[d]:4d}  mismatches={n_fail[d]:4d}  [{status}]")

    print(f"\n  Random sweep: {total_comp} comparisons, {total_fail} mismatches")
    if worst is not None:
        print(f"  Worst mismatch: {worst[1]} with {worst[0]} differing rows @ {worst[2]}")

    overall_ok = (total_fail == 0) and (edge_fail == 0) and (eps_fail == 0)
    grand = total_comp + 70 + eps_comp
    print(f"\n=== RESULT: {'PASS' if overall_ok else 'FAIL'} "
          f"({grand} total comparisons; "
          f"{total_fail + edge_fail + eps_fail} total mismatches) ===")
    return 0 if overall_ok else 1


def run_epsilon_battery():
    """Adversarial cases that target where Cython/Numba float math can drift:

    - share = w*(area/total_area) compared to mlot + 1e-6  (load distribution)
    - cx = sum_xw/total compared to cog_x_max + 1e-6       (CoG envelope)
    - total_area/footprint vs eff_sr - 1e-6                (support ratio)
    - int(box_mlot/box_weight) truncation                 (block stack cap)
    - DFTRC/wall/corner score ties (identical boxes -> > vs >= resolution)

    Returns (n_fail, n_comparisons).
    """
    fails = 0
    comps = 0
    rng = np.random.default_rng(999)

    # ---- 1. Identical-box ties: all boxes same size -> score ties everywhere.
    #     Stresses the strict-> vs strict-< tie-break ordering in both ports.
    for trial in range(40):
        L, W, H, mp = 600, 600, 600, 3
        n_boxes = int(rng.choice([16, 30, 48]))
        edge = int(rng.integers(60, 200))
        dims = np.zeros((n_boxes, 6, 3), dtype=np.int64)
        for i in range(n_boxes):
            e = edge
            dims[i] = [(e, e, e)] * 6   # cube -> all rotations identical
        n_rots = np.full(n_boxes, 6, dtype=np.int64)
        order = make_order(rng, n_boxes)
        sku = make_sku(rng, n_boxes, int(rng.integers(1, 4)))
        ca = _eps_ca(rng, n_boxes, L, W, H, edge)
        fails, comps = _compare_all(
            "tie", (L, W, H, mp), order, n_rots, dims, sku,
            int(sku.max()) + 1, ca, fails, comps)

    # ---- 2. mlot/weight ratios on exact integer boundaries (block cap).
    for trial in range(40):
        L, W, H, mp = 800, 800, 800, 2
        n_boxes = int(rng.choice([20, 36, 50]))
        n_rots, dims = make_boxes(rng, n_boxes, L, W, H)
        order = make_order(rng, n_boxes)
        sku = make_sku(rng, n_boxes, int(rng.integers(1, 5)))
        w = float(rng.choice([1.0, 2.0, 2.5, 4.0]))
        weights = np.full(n_boxes, w, dtype=np.float64)
        # mlot = exact multiple of w -> int(mlot/w) lands on boundary.
        kmax = int(rng.integers(0, 6))
        mlot = np.full(n_boxes, w * kmax, dtype=np.float64)
        ca = _eps_ca(rng, n_boxes, L, W, H, 100)
        ca["weights"] = weights
        ca["mlot"] = mlot
        # finite pallet cap as exact multiple of w too.
        ca["pallet_max_weight"] = w * float(rng.integers(1, n_boxes + 1))
        fails, comps = _compare_all(
            "intbound", (L, W, H, mp), order, n_rots, dims, sku,
            int(sku.max()) + 1, ca, fails, comps)

    # ---- 3. CoG bound placed exactly at expected centroid coords.
    for trial in range(40):
        L, W, H, mp = 1000, 1000, 600, 2
        n_boxes = int(rng.choice([24, 40]))
        n_rots, dims = make_boxes(rng, n_boxes, L, W, H)
        order = make_order(rng, n_boxes)
        sku = make_sku(rng, n_boxes, int(rng.integers(1, 4)))
        weights = rng.uniform(1.0, 5.0, n_boxes).astype(np.float64)
        # Razor-thin CoG window so cx/cy land on the +/- 1e-6 boundary often.
        cx = rng.uniform(0.30, 0.70) * L
        cy = rng.uniform(0.30, 0.70) * W
        eps = float(rng.choice([0.0, 1e-7, 1e-6, 1e-5]))
        ca = _eps_ca(rng, n_boxes, L, W, H, 100)
        ca["weights"] = weights
        ca["cog_active"] = 1
        ca["cog_x_min"], ca["cog_x_max"] = cx - eps, cx + eps
        ca["cog_y_min"], ca["cog_y_max"] = cy - eps, cy + eps
        ca["cog_min_load_frac"] = float(rng.choice(COG_MIN_LOAD_FRACS))
        ca["pallet_max_weight"] = float(rng.choice(
            [_NO_LIMIT, weights.sum() * 0.5, weights.sum() * 1.2]))
        fails, comps = _compare_all(
            "cogedge", (L, W, H, mp), order, n_rots, dims, sku,
            int(sku.max()) + 1, ca, fails, comps)

    # ---- 4. support_ratio that exactly matches a partial-overlap fraction.
    #     Boxes sized so contact_area/footprint is a clean fraction (0.5, 0.75).
    for trial in range(40):
        L, W, H, mp = 400, 400, 600, 2
        n_boxes = int(rng.choice([18, 30]))
        # Two box footprints: full and half, to create exact ratio supports.
        dims = np.zeros((n_boxes, 6, 3), dtype=np.int64)
        for i in range(n_boxes):
            if i % 2 == 0:
                a, b, c = 200, 200, 100
            else:
                a, b, c = 100, 200, 100
            dims[i] = [(a, b, c), (a, c, b), (b, a, c),
                       (b, c, a), (c, a, b), (c, b, a)]
        n_rots = np.full(n_boxes, 6, dtype=np.int64)
        order = make_order(rng, n_boxes)
        sku = make_sku(rng, n_boxes, int(rng.integers(1, 4)))
        ca = _eps_ca(rng, n_boxes, L, W, H, 100)
        ca["support_ratio"] = float(rng.choice([0.5, 0.75, 0.8, 1.0]))
        ca["require_centroid"] = int(rng.integers(0, 2))
        fails, comps = _compare_all(
            "supedge", (L, W, H, mp), order, n_rots, dims, sku,
            int(sku.max()) + 1, ca, fails, comps)

    print(f"  epsilon battery: {comps} comparisons, {fails} mismatches")
    return fails, comps


def _eps_ca(rng, n_boxes, L, W, H, edge):
    """A moderately-constrained config for the epsilon battery."""
    return {
        "weights": rng.uniform(1.0, 4.0, n_boxes).astype(np.float64),
        "mlot": np.where(rng.random(n_boxes) < 0.5, _NO_LIMIT,
                         rng.uniform(2.0, 20.0, n_boxes)).astype(np.float64),
        "rfs": (rng.random(n_boxes) < 0.3).astype(np.int64),
        "pallet_max_weight": float(rng.choice([_NO_LIMIT, n_boxes * 1.5])),
        "support_ratio": float(rng.choice(SUPPORT_RATIOS)),
        "require_centroid": int(rng.integers(0, 2)),
        "cog_x_min": -_NO_LIMIT, "cog_x_max": _NO_LIMIT,
        "cog_y_min": -_NO_LIMIT, "cog_y_max": _NO_LIMIT,
        "cog_min_load_frac": float(rng.choice(COG_MIN_LOAD_FRACS)),
        "cog_active": int(rng.integers(0, 2)),
    }


def _compare_all(tag, meta, order, n_rots, dims, sku, n_skus, ca, fails, comps):
    """Compare all 5 decoders Cython vs Numba; dump first mismatch."""
    L, W, H, mp = meta
    for mode in (0, 1, 2):
        po_cy, nb_cy = run_cstr_012(cy, order, n_rots, dims, L, W, H, mp, mode, ca)
        po_nb, nb_nb = run_cstr_012(nb, order, n_rots, dims, L, W, H, mp, mode, ca)
        comps += 1
        if not (np.array_equal(po_cy, po_nb) and nb_cy == nb_nb):
            fails += 1
            dump_mismatch(f"{tag}:cstr{mode}", meta, order, n_rots, dims, None,
                          ca, po_cy, nb_cy, po_nb, nb_nb)
    po_cy, nb_cy = run_cstr_layer(cy, order, n_rots, dims, L, W, H, mp, ca)
    po_nb, nb_nb = run_cstr_layer(nb, order, n_rots, dims, L, W, H, mp, ca)
    comps += 1
    if not (np.array_equal(po_cy, po_nb) and nb_cy == nb_nb):
        fails += 1
        dump_mismatch(f"{tag}:cstr3_layer", meta, order, n_rots, dims, None,
                      ca, po_cy, nb_cy, po_nb, nb_nb)
    po_cy, nb_cy = run_cstr_blocks(cy, order, n_rots, dims, sku, L, W, H, mp, n_skus, ca)
    po_nb, nb_nb = run_cstr_blocks(nb, order, n_rots, dims, sku, L, W, H, mp, n_skus, ca)
    comps += 1
    if not (np.array_equal(po_cy, po_nb) and nb_cy == nb_nb):
        fails += 1
        dump_mismatch(f"{tag}:cstr4_blocks", meta, order, n_rots, dims, sku,
                      ca, po_cy, nb_cy, po_nb, nb_nb)
    return fails, comps


def run_edge_cases():
    """Deterministic extreme-guard cases. Returns mismatch count."""
    fails = 0
    L, W, H, mp = 500, 500, 500, 2
    n_boxes = 20
    rng = np.random.default_rng(7)
    n_rots, dims = make_boxes(rng, n_boxes, L, W, H)
    order = np.arange(n_boxes, dtype=np.int64)
    sku = (np.arange(n_boxes) % 3).astype(np.int64)
    n_skus = 3
    weights = np.full(n_boxes, 2.0, dtype=np.float64)

    def base_ca(**over):
        ca = {
            "weights": weights,
            "mlot": np.full(n_boxes, _NO_LIMIT, dtype=np.float64),
            "rfs": np.zeros(n_boxes, dtype=np.int64),
            "pallet_max_weight": _NO_LIMIT,
            "support_ratio": 0.0,
            "require_centroid": 0,
            "cog_x_min": -_NO_LIMIT, "cog_x_max": _NO_LIMIT,
            "cog_y_min": -_NO_LIMIT, "cog_y_max": _NO_LIMIT,
            "cog_min_load_frac": 0.0,
            "cog_active": 0,
        }
        ca.update(over)
        return ca

    cases = {
        "all-infinite-no-cstr": base_ca(),
        "full-support-1.0": base_ca(support_ratio=1.0,
                                    rfs=np.ones(n_boxes, dtype=np.int64)),
        "centroid-required": base_ca(require_centroid=1, support_ratio=0.5),
        "mlot-zero": base_ca(mlot=np.zeros(n_boxes, dtype=np.float64)),
        "mlot-tiny": base_ca(mlot=np.full(n_boxes, 0.5, dtype=np.float64)),
        "pmw-binds-hard": base_ca(pallet_max_weight=6.0),  # only ~3 boxes/pallet
        "pmw-just-one-box": base_ca(pallet_max_weight=2.0),
        "pmw-box-too-heavy": base_ca(
            pallet_max_weight=1.0,
            weights=np.full(n_boxes, 2.0, dtype=np.float64)),  # nothing fits
        "cog-tight-center": base_ca(
            cog_active=1,
            cog_x_min=240.0, cog_x_max=260.0,
            cog_y_min=240.0, cog_y_max=260.0,
            cog_min_load_frac=0.0, pallet_max_weight=80.0),
        "cog-minloadfrac-1.0": base_ca(
            cog_active=1,
            cog_x_min=240.0, cog_x_max=260.0,
            cog_y_min=240.0, cog_y_max=260.0,
            cog_min_load_frac=1.0, pallet_max_weight=80.0),
        "cog-minloadfrac-0.35": base_ca(
            cog_active=1,
            cog_x_min=200.0, cog_x_max=300.0,
            cog_y_min=200.0, cog_y_max=300.0,
            cog_min_load_frac=0.35, pallet_max_weight=80.0),
        "cog-active-but-inf-weight": base_ca(  # gate: pmw==NO_LIMIT skips minload
            cog_active=1,
            cog_x_min=200.0, cog_x_max=300.0,
            cog_y_min=200.0, cog_y_max=300.0,
            cog_min_load_frac=1.0, pallet_max_weight=_NO_LIMIT),
        "everything-on-tight": base_ca(
            mlot=np.full(n_boxes, 3.0, dtype=np.float64),
            rfs=(np.arange(n_boxes) % 2).astype(np.int64),
            pallet_max_weight=40.0,
            support_ratio=0.8, require_centroid=1,
            cog_active=1,
            cog_x_min=230.0, cog_x_max=270.0,
            cog_y_min=230.0, cog_y_max=270.0,
            cog_min_load_frac=0.35),
        "support-0.5-mixed-mlot": base_ca(
            support_ratio=0.5,
            mlot=np.where(np.arange(n_boxes) % 2 == 0, 5.0, _NO_LIMIT)),
    }

    n_ok = 0
    for name, ca in cases.items():
        for mode in (0, 1, 2):
            po_cy, nb_cy = run_cstr_012(cy, order, n_rots, dims, L, W, H, mp, mode, ca)
            po_nb, nb_nb = run_cstr_012(nb, order, n_rots, dims, L, W, H, mp, mode, ca)
            if not (np.array_equal(po_cy, po_nb) and nb_cy == nb_nb):
                fails += 1
                dump_mismatch(f"edge:{name}:cstr{mode}", (L, W, H, mp),
                              order, n_rots, dims, None, ca,
                              po_cy, nb_cy, po_nb, nb_nb)
            else:
                n_ok += 1
        po_cy, nb_cy = run_cstr_layer(cy, order, n_rots, dims, L, W, H, mp, ca)
        po_nb, nb_nb = run_cstr_layer(nb, order, n_rots, dims, L, W, H, mp, ca)
        if not (np.array_equal(po_cy, po_nb) and nb_cy == nb_nb):
            fails += 1
            dump_mismatch(f"edge:{name}:cstr3_layer", (L, W, H, mp),
                          order, n_rots, dims, None, ca,
                          po_cy, nb_cy, po_nb, nb_nb)
        else:
            n_ok += 1
        po_cy, nb_cy = run_cstr_blocks(cy, order, n_rots, dims, sku, L, W, H, mp, n_skus, ca)
        po_nb, nb_nb = run_cstr_blocks(nb, order, n_rots, dims, sku, L, W, H, mp, n_skus, ca)
        if not (np.array_equal(po_cy, po_nb) and nb_cy == nb_nb):
            fails += 1
            dump_mismatch(f"edge:{name}:cstr4_blocks", (L, W, H, mp),
                          order, n_rots, dims, sku, ca,
                          po_cy, nb_cy, po_nb, nb_nb)
        else:
            n_ok += 1
    print(f"  edge battery: {len(cases)} configs x 5 decoders = {n_ok + fails} "
          f"comparisons, {fails} mismatches, {n_ok} OK")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
