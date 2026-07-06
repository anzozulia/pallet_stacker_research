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

def run_cstr_012(backend, order, n_rots, dims, L, W, H, mp, mode, ca,
                 pl=0, pw=0, tr=0):
    n = order.shape[0]
    po = np.zeros((n, 6), dtype=np.int64)
    nb_ = backend.decode_njit_mode_cstr(
        order, n_rots, dims, L, W, H, mp, po, mode,
        ca["weights"], ca["mlot"], ca["rfs"],
        ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
        ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
        ca["cog_min_load_frac"], ca["cog_active"],
        pl, pw, tr,
    )
    return po, int(nb_)


def run_cstr_layer(backend, order, n_rots, dims, L, W, H, mp, ca,
                   pl=0, pw=0, tr=0):
    n = order.shape[0]
    po = np.zeros((n, 6), dtype=np.int64)
    nb_ = backend.decode_layer_njit_cstr(
        order, n_rots, dims, L, W, H, mp, po,
        ca["weights"], ca["mlot"], ca["rfs"],
        ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
        ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
        ca["cog_min_load_frac"], ca["cog_active"],
        pl, pw, tr,
    )
    return po, int(nb_)


def run_cstr_blocks(backend, order, n_rots, dims, sku, L, W, H, mp, n_skus, ca,
                    pl=0, pw=0, tr=0):
    n = order.shape[0]
    po = np.zeros((n, 6), dtype=np.int64)
    nb_ = backend.decode_blocks_njit_mode_cstr(
        order, n_rots, dims, sku, L, W, H, mp, po, n_skus,
        ca["weights"], ca["mlot"], ca["rfs"],
        ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
        ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
        ca["cog_min_load_frac"], ca["cog_active"],
        pl, pw, tr,
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
        "transitive_on": 0,
        "deck_rejections_seen": 0,
        "transitive_chain_seen": 0,
        # Round-3 (F20/F21/F25) coverage — the fixes are unconditional, so
        # "fired" means the deterministic case shows the POST-fix physical
        # outcome (the pre-fix code provably violated it; a pre-fix run of
        # this battery records the defect and FAILS the gate).
        "block_joint_rejections_seen": 0,
        "underfill_rejections_seen": 0,
        "epsilon_scale_seen": 0,
        # Round-6 (F30) coverage — floor centroid-over-deck (toppling).
        "floor_com_rejections_seen": 0,
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
        # Round-2: raw deck dims travel with overhang (F17 deck check) and
        # the transitive load flag alternates (F19).
        pl = L0 if overhang > 0 else 0
        pw = W0 if overhang > 0 else 0
        tr = int((it // 2) % 2)
        if overhang == 0:
            cov["overhang0"] += 1
        else:
            cov["overhangP"] += 1
        if tr:
            cov["transitive_on"] += 1

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
            po_cy, nb_cy = run_cstr_012(cy, order, n_rots, dims, L, W, H, mp, mode, ca, pl, pw, tr)
            po_nb, nb_nb = run_cstr_012(nb, order, n_rots, dims, L, W, H, mp, mode, ca, pl, pw, tr)
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
        po_cy, nb_cy = run_cstr_layer(cy, order, n_rots, dims, L, W, H, mp, ca, pl, pw, tr)
        po_nb, nb_nb = run_cstr_layer(nb, order, n_rots, dims, L, W, H, mp, ca, pl, pw, tr)
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
        po_cy, nb_cy = run_cstr_blocks(cy, order, n_rots, dims, sku, L, W, H, mp, n_skus, ca, pl, pw, tr)
        po_nb, nb_nb = run_cstr_blocks(nb, order, n_rots, dims, sku, L, W, H, mp, n_skus, ca, pl, pw, tr)
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

    # ---- Round-2 battery (F17 deck check / F19 transitive) ----
    print("=== Round-2 battery (deck contact + transitive load) ===")
    r2_fail = run_round2_battery(cov)

    # ---- Round-3 battery (F20 block siblings / F21 under-fill / F25) ----
    print("=== Round-3 battery (block aggregation + under-fill + eps) ===")
    r3_fail = run_round3_battery(cov)

    # ---- Round-6 battery (F30 floor centroid-over-deck / toppling) ----
    print("=== Round-6 battery (floor toppling rule) ===")
    r6_fail = run_round6_battery(cov)

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
    print(f"  transitive flag on: {cov['transitive_on']}")
    print(f"  ROUND-2 branch coverage: deck_rejections_seen="
          f"{cov['deck_rejections_seen']} transitive_chain_seen="
          f"{cov['transitive_chain_seen']}")
    print(f"  ROUND-3 branch coverage: block_joint_rejections_seen="
          f"{cov['block_joint_rejections_seen']} underfill_rejections_seen="
          f"{cov['underfill_rejections_seen']} epsilon_scale_seen="
          f"{cov['epsilon_scale_seen']}")
    print(f"  ROUND-6 branch coverage: floor_com_rejections_seen="
          f"{cov['floor_com_rejections_seen']}")
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

    overall_ok = (total_fail == 0) and (edge_fail == 0) and (eps_fail == 0) \
        and (r2_fail == 0) and (r3_fail == 0) and (r6_fail == 0) \
        and cov["deck_rejections_seen"] > 0 and cov["transitive_chain_seen"] > 0 \
        and cov["block_joint_rejections_seen"] > 0 \
        and cov["underfill_rejections_seen"] > 0 \
        and cov["epsilon_scale_seen"] > 0 \
        and cov["floor_com_rejections_seen"] > 0
    grand = total_comp + 70 + eps_comp
    if cov["deck_rejections_seen"] == 0 or cov["transitive_chain_seen"] == 0:
        print("\n  !! ROUND-2 COVERAGE HOLE: a new branch never fired")
    if (cov["block_joint_rejections_seen"] == 0
            or cov["underfill_rejections_seen"] == 0
            or cov["epsilon_scale_seen"] == 0):
        print("\n  !! ROUND-3 COVERAGE HOLE: a fix's expected outcome never "
              "observed (running against a pre-round-3 core?)")
    if cov["floor_com_rejections_seen"] == 0:
        print("\n  !! ROUND-6 COVERAGE HOLE: the floor-CoM fix's expected "
              "outcome never observed (running against a pre-round-6 core?)")
    print(f"\n=== RESULT: {'PASS' if overall_ok else 'FAIL'} "
          f"({grand} total comparisons; "
          f"{total_fail + edge_fail + eps_fail + r2_fail + r3_fail + r6_fail} "
          f"total mismatches) ===")
    return 0 if overall_ok else 1



def run_round2_battery(cov):
    """Deterministic F17/F19 cases: prove the new branches FIRE (behavior
    differs vs the legacy sentinel/flag-off run) and stay cy==nb
    bit-identical. Returns the mismatch count."""
    fails = 0

    def base_ca(n, mlot_val=_NO_LIMIT, w=1.0):
        return {
            "weights": np.full(n, w, dtype=np.float64),
            "mlot": np.full(n, mlot_val, dtype=np.float64),
            "rfs": np.zeros(n, dtype=np.int64),
            "pallet_max_weight": _NO_LIMIT,
            "support_ratio": 0.8, "require_centroid": 0,
            "cog_x_min": -_NO_LIMIT, "cog_x_max": _NO_LIMIT,
            "cog_y_min": -_NO_LIMIT, "cog_y_max": _NO_LIMIT,
            "cog_min_load_frac": 0.0, "cog_active": 0,
        }

    # ---- F17: narrow raw deck (300x300) + overhang to 600x600. Without the
    # deck check the floor spreads across the whole inflated area; with it,
    # off-deck floor spots are rejected.
    n = 6
    n_rots = np.full(n, 1, dtype=np.int64)
    dims = np.zeros((n, 6, 3), dtype=np.int64)
    dims[:, 0] = (200, 200, 100)
    order = np.arange(n, dtype=np.int64)
    ca = base_ca(n)
    for mode in (0, 1, 2):
        po_off, _ = run_cstr_012(cy, order, n_rots, dims, 600, 600, 400, 1,
                                 mode, ca, 0, 0, 0)
        po_c, nbc = run_cstr_012(cy, order, n_rots, dims, 600, 600, 400, 1,
                                 mode, ca, 300, 300, 0)
        po_n, nbn = run_cstr_012(nb, order, n_rots, dims, 600, 600, 400, 1,
                                 mode, ca, 300, 300, 0)
        if not (np.array_equal(po_c, po_n) and nbc == nbn):
            fails += 1
            print(f"  FAIL round2-deck cstr{mode}: cy != nb")
        if not np.array_equal(po_c, po_off):
            cov["deck_rejections_seen"] += 1
    po_off, _ = run_cstr_layer(cy, order, n_rots, dims, 600, 600, 400, 1,
                               ca, 0, 0, 0)
    po_c, nbc = run_cstr_layer(cy, order, n_rots, dims, 600, 600, 400, 1,
                               ca, 300, 300, 0)
    po_n, nbn = run_cstr_layer(nb, order, n_rots, dims, 600, 600, 400, 1,
                               ca, 300, 300, 0)
    if not (np.array_equal(po_c, po_n) and nbc == nbn):
        fails += 1
        print("  FAIL round2-deck layer: cy != nb")
    if not np.array_equal(po_c, po_off):
        cov["deck_rejections_seen"] += 1
    sku = np.zeros(n, dtype=np.int64)
    po_off, _ = run_cstr_blocks(cy, order, n_rots, dims, sku, 600, 600, 400,
                                1, 1, ca, 0, 0, 0)
    po_c, nbc = run_cstr_blocks(cy, order, n_rots, dims, sku, 600, 600, 400,
                                1, 1, ca, 300, 300, 0)
    po_n, nbn = run_cstr_blocks(nb, order, n_rots, dims, sku, 600, 600, 400,
                                1, 1, ca, 300, 300, 0)
    if not (np.array_equal(po_c, po_n) and nbc == nbn):
        fails += 1
        print("  FAIL round2-deck blocks: cy != nb")
    if not np.array_equal(po_c, po_off):
        cov["deck_rejections_seen"] += 1

    # ---- F19: pure column with tight finite mlot two levels down. Direct
    # model stacks all 8 (each link legal); the transitive check must cut
    # the column.
    n = 8
    n_rots = np.full(n, 1, dtype=np.int64)
    dims = np.zeros((n, 6, 3), dtype=np.int64)
    dims[:, 0] = (400, 400, 100)
    order = np.arange(n, dtype=np.int64)
    ca = base_ca(n, mlot_val=10.5, w=10.0)   # each link 10 <= 10.5
    for mode in (0, 1, 2):
        po_d, _ = run_cstr_012(cy, order, n_rots, dims, 400, 400, 1000, 1,
                               mode, ca, 0, 0, 0)
        po_c, nbc = run_cstr_012(cy, order, n_rots, dims, 400, 400, 1000, 1,
                                 mode, ca, 0, 0, 1)
        po_n, nbn = run_cstr_012(nb, order, n_rots, dims, 400, 400, 1000, 1,
                                 mode, ca, 0, 0, 1)
        if not (np.array_equal(po_c, po_n) and nbc == nbn):
            fails += 1
            print(f"  FAIL round2-transitive cstr{mode}: cy != nb")
        if not np.array_equal(po_c, po_d):
            cov["transitive_chain_seen"] += 1
    po_d, _ = run_cstr_layer(cy, order, n_rots, dims, 400, 400, 1000, 1,
                             ca, 0, 0, 0)
    po_c, nbc = run_cstr_layer(cy, order, n_rots, dims, 400, 400, 1000, 1,
                               ca, 0, 0, 1)
    po_n, nbn = run_cstr_layer(nb, order, n_rots, dims, 400, 400, 1000, 1,
                               ca, 0, 0, 1)
    if not (np.array_equal(po_c, po_n) and nbc == nbn):
        fails += 1
        print("  FAIL round2-transitive layer: cy != nb")
    if not np.array_equal(po_c, po_d):
        cov["transitive_chain_seen"] += 1
    sku = np.zeros(n, dtype=np.int64)
    po_d, _ = run_cstr_blocks(cy, order, n_rots, dims, sku, 400, 400, 1000,
                              1, 1, ca, 0, 0, 0)
    po_c, nbc = run_cstr_blocks(cy, order, n_rots, dims, sku, 400, 400, 1000,
                                1, 1, ca, 0, 0, 1)
    po_n, nbn = run_cstr_blocks(nb, order, n_rots, dims, sku, 400, 400, 1000,
                                1, 1, ca, 0, 0, 1)
    if not (np.array_equal(po_c, po_n) and nbc == nbn):
        fails += 1
        print("  FAIL round2-transitive blocks: cy != nb")
    if not np.array_equal(po_c, po_d):
        cov["transitive_chain_seen"] += 1
    print(f"  round2 battery: {fails} mismatches; "
          f"deck branch fired in {cov['deck_rejections_seen']} case(s), "
          f"transitive branch fired in {cov['transitive_chain_seen']} case(s)")
    return fails


def run_round3_battery(cov):
    """Deterministic F20/F21/F25 cases (hardening round 3).

    The round-3 fixes are UNCONDITIONAL (physics bugs, no flag), so unlike
    round 2 there is no flag-off run to diff against. Instead each case
    encodes the physically-correct POST-fix outcome that the pre-fix code
    provably violated (fuzz + live probes, docs/reports/37): the coverage
    counters only fire when the fixed behavior is observed, and every case
    still asserts cy == nb bit-identity. Returns the mismatch count.
    """
    fails = 0

    def base_ca(n, weights, mlot, sr=0.8):
        return {
            "weights": np.asarray(weights, dtype=np.float64),
            "mlot": np.asarray(mlot, dtype=np.float64),
            "rfs": np.zeros(n, dtype=np.int64),
            "pallet_max_weight": _NO_LIMIT,
            "support_ratio": sr, "require_centroid": 0,
            "cog_x_min": -_NO_LIMIT, "cog_x_max": _NO_LIMIT,
            "cog_y_min": -_NO_LIMIT, "cog_y_max": _NO_LIMIT,
            "cog_min_load_frac": 0.0, "cog_active": 0,
        }

    def single_rot(n, sizes):
        n_rots = np.full(n, 1, dtype=np.int64)
        dims = np.zeros((n, 6, 3), dtype=np.int64)
        for i, s in enumerate(sizes):
            dims[i, 0] = s
        return n_rots, dims

    # ---- F20: a 2x2x1 same-SKU block lands on a weak base. Each column's
    # weight (10) individually fits mlot=25; the aggregate (40) does not.
    # Pre-fix the block decoder checked all four columns against the
    # pre-block state and committed all four (base carried 40). Post-fix
    # at most 2 top boxes may rest on the base.
    n = 5
    n_rots, dims = single_rot(
        n, [(100, 100, 20)] + [(50, 50, 20)] * 4)
    order = np.arange(n, dtype=np.int64)
    sku = np.array([0, 1, 1, 1, 1], dtype=np.int64)
    ca = base_ca(n, [50.0] + [10.0] * 4, [25.0] + [_NO_LIMIT] * 4)
    for tr in (0, 1):
        po_c, nbc = run_cstr_blocks(cy, order, n_rots, dims, sku,
                                    100, 100, 200, 1, 2, ca, 0, 0, tr)
        po_n, nbn = run_cstr_blocks(nb, order, n_rots, dims, sku,
                                    100, 100, 200, 1, 2, ca, 0, 0, tr)
        if not (np.array_equal(po_c, po_n) and nbc == nbn):
            fails += 1
            print(f"  FAIL round3-block tr={tr}: cy != nb")
        placed_tops = int((po_c[1:, 5] == 1).sum())
        if po_c[0, 5] == 1 and placed_tops <= 2:
            cov["block_joint_rejections_seen"] += 1
        else:
            print(f"  round3-block tr={tr}: base placed={po_c[0, 5]} "
                  f"tops placed={placed_tops} (pre-fix behavior is 4)")

    # ---- F20 (m>1): 2x2x2 block, transitive relay per column = 20 <= 25,
    # aggregate 80. Post-fix at most 2 whole columns (4 boxes) can rest on
    # the base under the transitive model (2 cols x 20 = 40 > 25 already,
    # so really at most 1 column = 2 boxes).
    n = 9
    n_rots, dims = single_rot(
        n, [(100, 100, 20)] + [(50, 50, 20)] * 8)
    order = np.arange(n, dtype=np.int64)
    sku = np.array([0] + [1] * 8, dtype=np.int64)
    ca = base_ca(n, [50.0] + [10.0] * 8, [25.0] + [_NO_LIMIT] * 8)
    po_c, nbc = run_cstr_blocks(cy, order, n_rots, dims, sku,
                                100, 100, 200, 1, 2, ca, 0, 0, 1)
    po_n, nbn = run_cstr_blocks(nb, order, n_rots, dims, sku,
                                100, 100, 200, 1, 2, ca, 0, 0, 1)
    if not (np.array_equal(po_c, po_n) and nbc == nbn):
        fails += 1
        print("  FAIL round3-block-m2: cy != nb")
    placed_tops = int((po_c[1:, 5] == 1).sum())
    if po_c[0, 5] == 1 and placed_tops <= 2:
        cov["block_joint_rejections_seen"] += 1

    # ---- F21: under-fill. Pillar at the origin, a half-supported slab
    # cantilevers over the empty floor half (sr=0.5), then a FRAGILE box
    # (mlot=0) targets the floor spot under the cantilever — its top plane
    # meets the slab's bottom, so it would inherit half the slab's load.
    # Pre-fix every decoder placed it there (validate flags it); post-fix
    # that spot is rejected and the box lands elsewhere (or not at all).
    n = 3
    n_rots, dims = single_rot(
        n, [(200, 200, 100), (400, 200, 100), (200, 200, 100)])
    order = np.arange(n, dtype=np.int64)
    ca = base_ca(n, [1.0, 1.0, 5.0], [_NO_LIMIT, _NO_LIMIT, 0.0], sr=0.5)
    for tr in (0, 1):
        fired_any = False
        for mode in (0, 1, 2):
            po_c, nbc = run_cstr_012(cy, order, n_rots, dims, 400, 200, 300,
                                     1, mode, ca, 0, 0, tr)
            po_n, nbn = run_cstr_012(nb, order, n_rots, dims, 400, 200, 300,
                                     1, mode, ca, 0, 0, tr)
            if not (np.array_equal(po_c, po_n) and nbc == nbn):
                fails += 1
                print(f"  FAIL round3-underfill cstr{mode} tr={tr}: cy != nb")
            # Row 2 is the fragile box: post-fix it must NOT sit at z=0
            # while the slab (row 1) is placed above z=0 next to it.
            if po_c[1, 5] == 1 and po_c[1, 4] > 0:
                if po_c[2, 5] == 0 or po_c[2, 4] > 0:
                    fired_any = True
        po_c, nbc = run_cstr_layer(cy, order, n_rots, dims, 400, 200, 300,
                                   1, ca, 0, 0, tr)
        po_n, nbn = run_cstr_layer(nb, order, n_rots, dims, 400, 200, 300,
                                   1, ca, 0, 0, tr)
        if not (np.array_equal(po_c, po_n) and nbc == nbn):
            fails += 1
            print(f"  FAIL round3-underfill layer tr={tr}: cy != nb")
        sku3 = np.arange(n, dtype=np.int64)
        po_c, nbc = run_cstr_blocks(cy, order, n_rots, dims, sku3, 400, 200,
                                    300, 1, 3, ca, 0, 0, tr)
        po_n, nbn = run_cstr_blocks(nb, order, n_rots, dims, sku3, 400, 200,
                                    300, 1, 3, ca, 0, 0, tr)
        if not (np.array_equal(po_c, po_n) and nbc == nbn):
            fails += 1
            print(f"  FAIL round3-underfill blocks tr={tr}: cy != nb")
        if fired_any:
            cov["underfill_rejections_seen"] += 1
        else:
            print(f"  round3-underfill tr={tr}: fragile box still at z=0 "
                  f"under the cantilever (pre-fix behavior)")

    # ---- F25: scale-aware epsilon at 1e12. The carrier's mlot is 1e12;
    # a rider of 1e12+500 is within the relative tolerance
    # (max(1e-6, 1e-9*mlot) = 1000) and must be ACCEPTED post-fix (the
    # old absolute 1e-6 vanished below one ulp and rejected it); a rider
    # of 1e12+5000 is beyond the tolerance and must stay rejected.
    n = 2
    n_rots, dims = single_rot(n, [(400, 400, 100), (400, 400, 100)])
    order = np.arange(n, dtype=np.int64)
    for extra, expect_placed in ((500.0, 2), (5000.0, 1)):
        ca = base_ca(n, [1.0, 1e12 + extra], [1e12, _NO_LIMIT])
        po_c, nbc = run_cstr_012(cy, order, n_rots, dims, 400, 400, 1000,
                                 1, 0, ca, 0, 0, 0)
        po_n, nbn = run_cstr_012(nb, order, n_rots, dims, 400, 400, 1000,
                                 1, 0, ca, 0, 0, 0)
        if not (np.array_equal(po_c, po_n) and nbc == nbn):
            fails += 1
            print(f"  FAIL round3-eps extra={extra}: cy != nb")
        placed = int((po_c[:, 5] == 1).sum())
        if placed == expect_placed:
            if extra == 500.0:
                cov["epsilon_scale_seen"] += 1
        else:
            print(f"  round3-eps extra={extra}: placed={placed}, "
                  f"expected {expect_placed}")
            if extra == 5000.0:
                fails += 1     # over-tolerance accept would be a REAL bug

    print(f"  round3 battery: {fails} mismatches; block fix observed in "
          f"{cov['block_joint_rejections_seen']} case(s), under-fill fix in "
          f"{cov['underfill_rejections_seen']} case(s), scaled epsilon in "
          f"{cov['epsilon_scale_seen']} case(s)")
    return fails


def run_round6_battery(cov):
    """Deterministic F30 cases (hardening round 6): the floor
    centroid-over-deck (toppling) rule under overhang.

    Like round 3 the fix is unconditional physics (gated only on
    require_centroid), so each case encodes the physically-correct
    POST-fix outcome the pre-fix code provably violated (105/120 engine
    solves shipped a validate-clean toppling floor box at sr<0.5 —
    docs/reports/39), and every case asserts cy == nb bit-identity.

    Geometry (single-rotation, deck 400x400 inflated by overhang 400 to
    an 800x800 container; pl=pw=400 raw deck; sr=0.25):
      topple:   spacer 300-long at origin, then a 300x400 slab whose only
                floor spot is x=300 -> contact ratio 1/3 >= 0.25 but
                centroid 450 > 400: pre-fix placed toppling, post-fix
                must NOT sit at a toppling floor spot.
      boundary: spacer 200-long, slab 400-long at x=200 -> centroid
                EXACTLY 400 == deck edge: must stay ACCEPTED (pins the
                strict > comparison; a later >= "cleanup" fails here).
      one-past: spacer 201-long, slab 400-long at x=201 -> centroid
                400.5: one half-grid past the edge, must be rejected
                from the floor.
      rc=0:     the topple geometry with require_centroid=0 -> the
                pre-fix placement is legal again (gate respected).
    """
    fails = 0

    def ca_for(n, rc):
        return {
            "weights": np.full(n, 5.0, dtype=np.float64),
            "mlot": np.full(n, _NO_LIMIT, dtype=np.float64),
            "rfs": np.zeros(n, dtype=np.int64),
            "pallet_max_weight": _NO_LIMIT,
            "support_ratio": 0.25, "require_centroid": rc,
            "cog_x_min": -_NO_LIMIT, "cog_x_max": _NO_LIMIT,
            "cog_y_min": -_NO_LIMIT, "cog_y_max": _NO_LIMIT,
            "cog_min_load_frac": 0.0, "cog_active": 0,
        }

    def single_rot(sizes):
        n = len(sizes)
        n_rots = np.full(n, 1, dtype=np.int64)
        dims = np.zeros((n, 6, 3), dtype=np.int64)
        for i, s in enumerate(sizes):
            dims[i, 0] = s
        return n_rots, dims

    def decode_both(tag, sizes, rc):
        # mode 2 on purpose: it is the placement scoring that actually
        # picks the partially-overhanging floor spot (modes 0/1 skip it),
        # verified pre-fix — the toppling branch provably fires here.
        n = len(sizes)
        n_rots, dims = single_rot(sizes)
        order = np.arange(n, dtype=np.int64)
        ca = ca_for(n, rc)
        po_c, nbc = run_cstr_012(cy, order, n_rots, dims, 800, 800, 1200,
                                 1, 2, ca, 400, 400, 0)
        po_n, nbn = run_cstr_012(nb, order, n_rots, dims, 800, 800, 1200,
                                 1, 2, ca, 400, 400, 0)
        nonlocal fails
        if not (np.array_equal(po_c, po_n) and nbc == nbn):
            fails += 1
            print(f"  FAIL round6-{tag}: cy != nb")
        return po_c

    def floor_com_ok(po, dims):
        """True iff every placed floor row's centroid is over the deck."""
        for i in range(po.shape[0]):
            if po[i, 5] == 1 and po[i, 4] == 0:
                if (2 * po[i, 2] + dims[i, 0, 0] > 2 * min(
                        po[i, 2] + dims[i, 0, 0], 400)
                        and po[i, 2] + dims[i, 0, 0] > 400):
                    return False
                if (2 * po[i, 3] + dims[i, 0, 1] > 2 * min(
                        po[i, 3] + dims[i, 0, 1], 400)
                        and po[i, 3] + dims[i, 0, 1] > 400):
                    return False
        return True

    # topple: post-fix the slab may not occupy a toppling floor spot.
    sizes = [(300, 400, 200), (300, 400, 200)]
    po = decode_both("topple", sizes, rc=1)
    _, dims = single_rot(sizes)
    slab_on_floor_toppling = (po[1, 5] == 1 and po[1, 4] == 0
                              and po[1, 2] + 300 > 400
                              and 2 * po[1, 2] + 300 > 2 * 400)
    if floor_com_ok(po, dims) and not slab_on_floor_toppling:
        cov["floor_com_rejections_seen"] += 1
    else:
        print(f"  round6-topple: slab row={po[1].tolist()} "
              f"(pre-fix behavior: floor spot x=300, centroid 450)")

    # boundary: centroid exactly ON the deck edge stays accepted.
    po = decode_both("boundary", [(200, 400, 200), (400, 400, 200)], rc=1)
    if not (po[1, 5] == 1 and po[1, 4] == 0 and po[1, 2] == 200):
        fails += 1     # rejecting the boundary would be a REAL regression
        print(f"  FAIL round6-boundary: slab row={po[1].tolist()} "
              f"(centroid==edge must stay accepted)")

    # one-past: centroid a half-grid past the edge must leave the floor.
    po = decode_both("one-past", [(201, 400, 200), (400, 400, 200)], rc=1)
    if po[1, 5] == 1 and po[1, 4] == 0 and po[1, 2] == 201:
        fails += 1
        print("  FAIL round6-one-past: toppling floor spot accepted")
    else:
        cov["floor_com_rejections_seen"] += 1

    # rc=0: gate respected — the pre-fix placement is legal again.
    po = decode_both("rc0", [(300, 400, 200), (300, 400, 200)], rc=0)
    if not (po[1, 5] == 1 and po[1, 4] == 0 and po[1, 2] == 300):
        fails += 1
        print(f"  FAIL round6-rc0: slab row={po[1].tolist()} "
              f"(require_centroid=0 must keep the historical placement)")

    print(f"  round6 battery: {fails} mismatches; floor-CoM fix observed "
          f"in {cov['floor_com_rejections_seen']} case(s)")
    return fails


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
    pl = pw = tr = 0  # legacy defaults; round-2 args exercised by run_round2_battery
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
            po_cy, nb_cy = run_cstr_012(cy, order, n_rots, dims, L, W, H, mp, mode, ca, pl, pw, tr)
            po_nb, nb_nb = run_cstr_012(nb, order, n_rots, dims, L, W, H, mp, mode, ca, pl, pw, tr)
            if not (np.array_equal(po_cy, po_nb) and nb_cy == nb_nb):
                fails += 1
                dump_mismatch(f"edge:{name}:cstr{mode}", (L, W, H, mp),
                              order, n_rots, dims, None, ca,
                              po_cy, nb_cy, po_nb, nb_nb)
            else:
                n_ok += 1
        po_cy, nb_cy = run_cstr_layer(cy, order, n_rots, dims, L, W, H, mp, ca, pl, pw, tr)
        po_nb, nb_nb = run_cstr_layer(nb, order, n_rots, dims, L, W, H, mp, ca, pl, pw, tr)
        if not (np.array_equal(po_cy, po_nb) and nb_cy == nb_nb):
            fails += 1
            dump_mismatch(f"edge:{name}:cstr3_layer", (L, W, H, mp),
                          order, n_rots, dims, None, ca,
                          po_cy, nb_cy, po_nb, nb_nb)
        else:
            n_ok += 1
        po_cy, nb_cy = run_cstr_blocks(cy, order, n_rots, dims, sku, L, W, H, mp, n_skus, ca, pl, pw, tr)
        po_nb, nb_nb = run_cstr_blocks(nb, order, n_rots, dims, sku, L, W, H, mp, n_skus, ca, pl, pw, tr)
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
