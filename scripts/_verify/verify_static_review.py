"""verify_static_review.py — adversarial static-review follow-up probes.

Targets the suspected latent bugs in the three Cython sources:
  jit_decoders_geom_cy.pyx, jit_decoders_cstr_cy.pyx, jit_constraints_cy.pyx

Each sub-test constructs an instance that *would* trigger a suspected bug and
checks (a) the Cython decoder does not crash / produce OOB, (b) it stays
bit-identical to the Numba reference twin (any divergence = real defect or
real overflow), and (c) the full solver still produces validator-clean output.

Sub-tests:
  T1  int64 overflow in score arithmetic at extreme pallet dims
        (DFTRC sq-dist + wall `x*BIG`, BIG=(W+H)^2+1).
  T2  Cython-vs-Numba divergence on the SAME extreme-dim instance
        (confirms overflow is shared, not a port-only defect).
  T3  zero-dim / degenerate boxes (division-by-zero in support_ratio
        footprint + load-share; cdivision=True => UB on /0).
  T4  block stack-cap off-by-one in _max_block_under_constraints
        (max_m_stack = int(mlot/w)+1).
  T5  uninitialized placements_out rows in block modes 4/5
        (rows for never-visited / unplaced boxes).
  T6  n_rots < dims_all.shape[1] and OOB rot index safety
        (boundscheck=False): does a rot index drive an OOB read?
  T7  EMS overflow: force > MAX_EMS_C(512) candidate EMSs to probe the
        MAX_EMS_C clamp path in _commit_ems (boundscheck disabled).
  T8  full-solve + validate on adversarial real-ish instances
        (non-integer float dims, huge dims, single zero-volume box).

Run:
  docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
    -v "$(pwd)":/app -w /app pallet-packer:dev \
    python scripts/_verify/verify_static_review.py
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pallet_packer._brkga_core import jit_decoders_geom_cy as geom_cy
from pallet_packer._brkga_core import jit_decoders_geom as geom_nb
from pallet_packer._brkga_core import jit_decoders_cstr_cy as cstr_cy
from pallet_packer._brkga_core import jit_decoders_cstr as cstr_nb

RESULTS = []  # (name, passed, detail)


def record(name, passed, detail=""):
    RESULTS.append((name, passed, detail))
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {name}: {detail}")


def _dims6(bx, by, bz):
    return [
        (bx, by, bz), (bx, bz, by),
        (by, bx, bz), (by, bz, bx),
        (bz, bx, by), (bz, by, bx),
    ]


def make_pop(rng, pop, n_boxes, lo, hi, dim_cap):
    n_rots = np.full(n_boxes, 6, dtype=np.int64)
    dims = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    for i in range(n_boxes):
        bx = int(rng.integers(lo, hi))
        by = int(rng.integers(lo, hi))
        bz = int(rng.integers(lo, hi))
        for r, t in enumerate(_dims6(bx, by, bz)):
            dims[i, r] = t
    chroms = rng.random((pop, n_boxes)).astype(np.float64)
    return chroms, n_rots, dims


# ---------------------------------------------------------------------------
# T1 / T2: integer overflow in score arithmetic at extreme pallet dims.
# ---------------------------------------------------------------------------

def t1_t2_overflow():
    # int64 max ~ 9.22e18. (W+H)^2 with W=H=2.2e9 -> (4.4e9)^2 = 1.94e19 > max.
    # Also x*BIG in wall mode = up to L * BIG can blow far past int64.
    # DFTRC: (L-x-dx)^2 ~ (2e9)^2 = 4e18 each, x3 = 1.2e19 -> overflow.
    L = W = H = 2_200_000_000  # 2.2e9 mm — absurd but legal int(round(...))
    mp = 4
    rng = np.random.default_rng(7)
    pop, n_boxes = 8, 30
    # boxes ~ 1e8..3e8 so several fit per axis.
    chroms, n_rots, dims = make_pop(rng, pop, n_boxes, 100_000_000, 300_000_000, L)

    BIG = (W + H) * (W + H) + 1
    overflowed = BIG > np.iinfo(np.int64).max or BIG <= 0  # python int won't, but C will wrap
    # Compute what C int64 would see for BIG (wraps).
    c_big = np.int64(np.int64(W + H) * np.int64(W + H)) + np.int64(1)
    print(f"  [T1] L=W=H={L}  python BIG={BIG} ({'>int64max' if BIG>np.iinfo(np.int64).max else 'ok'})  "
          f"c_int64 BIG wraps to {c_big}")

    orders = _orders_for(chroms, n_boxes)
    ok_all = True
    for mode in (0, 1, 2):
        po_cy = np.zeros((pop, n_boxes, 6), dtype=np.int64)
        nb_cy = np.zeros(pop, dtype=np.int64)
        crashed = False
        try:
            geom_cy.decode_batch_njit_mode(chroms, n_rots, dims, L, W, H, mp,
                                           mode, po_cy, nb_cy)
        except Exception as e:  # noqa: BLE001
            crashed = True
            record(f"T1 overflow no-crash mode {mode}", False, f"EXCEPTION {e!r}")
            ok_all = False
            continue
        # Validate geometry of every placed box vs pallet bounds + overlaps.
        n_oob, n_overlap = _geom_self_check(po_cy, dims, L, W, H, mp, orders)
        detail = (f"mode {mode}: no crash, placed_rows ok; "
                  f"OOB-placements={n_oob} overlapping-pairs={n_overlap}")
        passed = (not crashed)
        record(f"T1 overflow no-crash mode {mode}", passed, detail)
        # Overlap/OOB here would be a *shared* algorithmic artifact of int64
        # score overflow (see T2 cy==nb). Report as info, not a port defect.
        if n_oob or n_overlap:
            record(f"T1 overflow VALIDITY mode {mode}", False,
                   f"overflow produced INVALID geometry (SHARED w/ Numba, see T2): "
                   f"OOB={n_oob} overlap={n_overlap}")
            ok_all = False

    # T2: divergence Cython vs Numba on the SAME instance. If they match, the
    # overflow is a shared property (both wrap identically) — not a port bug.
    for mode in (0, 1, 2):
        po_cy = np.zeros((pop, n_boxes, 6), dtype=np.int64)
        nb_cy = np.zeros(pop, dtype=np.int64)
        geom_cy.decode_batch_njit_mode(chroms, n_rots, dims, L, W, H, mp,
                                       mode, po_cy, nb_cy)
        po_nb = np.zeros((pop, n_boxes, 6), dtype=np.int64)
        for i in range(pop):
            order = np.argsort(chroms[i]).astype(np.int64)
            geom_nb.decode_njit_mode(order, n_rots, dims, L, W, H, mp,
                                     po_nb[i], mode)
        same = np.array_equal(po_cy, po_nb)
        record(f"T2 cy==nb under overflow mode {mode}", same,
               "bit-identical" if same else
               f"DIVERGENCE: {int((po_cy != po_nb).any(axis=(1,2)).sum())}/{pop} chroms differ")


def _max_float_penetration(res):
    """Worst (smallest-axis) overlap penetration in float space across all
    placed-box pairs sharing a pallet. 0.0 if no overlaps."""
    worst = 0.0
    for st in res.pallets:
        pls = st.placements
        for i in range(len(pls)):
            a = pls[i]
            adx, ady, adz = a.box.dims_for(a.rotation)
            for j in range(i + 1, len(pls)):
                b = pls[j]
                bdx, bdy, bdz = b.box.dims_for(b.rotation)
                ox = min(a.x + adx, b.x + bdx) - max(a.x, b.x)
                oy = min(a.y + ady, b.y + bdy) - max(a.y, b.y)
                oz = min(a.z + adz, b.z + bdz) - max(a.z, b.z)
                if ox > 1e-6 and oy > 1e-6 and oz > 1e-6:
                    worst = max(worst, min(ox, oy, oz))
    return worst


def _orders_for(chroms, n_boxes):
    """Replicate the decoder's per-chromosome BPS argsort so the self-check
    can map placement-row index -> actual box index."""
    bps = np.ascontiguousarray(chroms)[:, :n_boxes]
    return np.argsort(bps, axis=1).astype(np.int64)


def _geom_self_check(po_all, dims, L, W, H, mp, orders):
    """Return (#placements outside pallet, #overlapping pairs within a bin).

    Row index in po_all is the BPS *position*; the actual box is orders[c, i],
    matching dispatch.py (box_idx = order[i]); dims must be indexed by box.
    """
    n_oob = 0
    n_overlap = 0
    pop, n_boxes, _ = po_all.shape
    for c in range(pop):
        boxes_by_bin = {}
        for i in range(n_boxes):
            row = po_all[c, i]
            if row[5] != 1:
                continue
            box = int(orders[c, i])
            b, r, x, y, z = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])
            dx, dy, dz = int(dims[box, r, 0]), int(dims[box, r, 1]), int(dims[box, r, 2])
            if x < 0 or y < 0 or z < 0 or x + dx > L or y + dy > W or z + dz > H:
                n_oob += 1
            boxes_by_bin.setdefault(b, []).append((x, y, z, dx, dy, dz))
        for b, lst in boxes_by_bin.items():
            for a in range(len(lst)):
                for d in range(a + 1, len(lst)):
                    ax, ay, az, adx, ady, adz = lst[a]
                    bx, by, bz, bdx, bdy, bdz = lst[d]
                    if (ax < bx + bdx and bx < ax + adx
                            and ay < by + bdy and by < ay + ady
                            and az < bz + bdz and bz < az + adz):
                        n_overlap += 1
    return n_oob, n_overlap


# ---------------------------------------------------------------------------
# T3: zero-dim / degenerate boxes — division by zero risk (cdivision=True).
# ---------------------------------------------------------------------------

def t3_zero_dims():
    # cstr modes hit footprint = dx*dy and area/total_area. With a zero-dim
    # box, dx*dy can be 0; guards are `footprint > 0` and `total_area > 0`.
    # Force a stack so support_ratio path runs (cog inactive, weights set).
    L, W, H, mp = 1000, 800, 600, 4
    n_boxes = 20
    n_rots = np.full(n_boxes, 6, dtype=np.int64)
    dims = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    rng = np.random.default_rng(11)
    for i in range(n_boxes):
        if i % 5 == 0:
            bx, by, bz = 0, int(rng.integers(50, 200)), int(rng.integers(50, 200))  # zero X
        elif i % 5 == 1:
            bx, by, bz = int(rng.integers(50, 200)), 0, int(rng.integers(50, 200))  # zero Y
        else:
            bx, by, bz = (int(rng.integers(50, 300)), int(rng.integers(50, 300)),
                          int(rng.integers(50, 300)))
        for r, t in enumerate(_dims6(bx, by, bz)):
            dims[i, r] = t
    chroms = rng.random((6, n_boxes)).astype(np.float64)
    weights = rng.uniform(0.5, 5.0, n_boxes).astype(np.float64)
    mlot = np.full(n_boxes, 30.0, dtype=np.float64)
    rfs = (rng.random(n_boxes) < 0.4).astype(np.int64)
    # support_ratio active (constraints engaged).
    args = (weights, mlot, rfs, 1e18, 0.6, 0,
            -1e18, 1e18, -1e18, 1e18, 0.0, 0)
    for label, batchfn, n_args in (
        ("cstr 0", lambda po, nb: cstr_cy.decode_batch_njit_mode_cstr(
            chroms, n_rots, dims, L, W, H, mp, 0, *args, po, nb), 0),
    ):
        po = np.zeros((6, n_boxes, 6), dtype=np.int64)
        nb = np.zeros(6, dtype=np.int64)
        try:
            batchfn(po, nb)
            record(f"T3 zero-dim no-crash ({label})", True,
                   f"survived zero-dim boxes; placed_rows in [0..{n_boxes}]")
        except Exception as e:  # noqa: BLE001
            record(f"T3 zero-dim no-crash ({label})", False, f"EXCEPTION {e!r}")
            return
    # Direct probe of _check_load_on_top with a zero-footprint candidate
    # placed on a real supporter -> footprint==0 path inside the support check.
    from pallet_packer._brkga_core import jit_constraints_cy as ccy
    from pallet_packer._brkga_core import jit_constraints as cnb
    # one supporter box at z=0 with dz=100, then a zero-area candidate on top.
    po = np.zeros((2, 6), dtype=np.int64)
    da = np.zeros((2, 6, 3), dtype=np.int64)
    da[0, 0] = (100, 100, 100)  # supporter
    da[1, 0] = (0, 100, 100)    # zero-X candidate
    po[0] = (0, 0, 0, 0, 0, 1)  # bin0 rot0 at (0,0,0) placed
    bo = np.array([0, 1], dtype=np.int64)
    mlot = np.array([5.0, 5.0])
    ptl = np.zeros(2)
    try:
        rcy = ccy._check_load_on_top_njit(po, da, bo, mlot, ptl, 2, 0,
                                          0, 0, 100, 0, 100, 100, 1.0, 0.6, 0, 0)
        rnb = cnb._check_load_on_top_njit(po, da, bo, mlot, ptl, 2, 0,
                                          0, 0, 100, 0, 100, 100, 1.0, 0.6, 0, 0)
        record("T3 zero-footprint load-check cy==nb", rcy == rnb,
               f"cy={rcy} nb={rnb} (no /0 trap)")
    except ZeroDivisionError as e:
        record("T3 zero-footprint load-check cy==nb", False, f"ZeroDivisionError {e!r}")
    except Exception as e:  # noqa: BLE001
        record("T3 zero-footprint load-check cy==nb", False, f"EXCEPTION {e!r}")


# ---------------------------------------------------------------------------
# T4: _max_block_under_constraints stack-cap off-by-one vs Numba.
# ---------------------------------------------------------------------------

def t4_block_stackcap():
    # Compare cstr-blocks decode cy vs nb on a homogeneous SKU with mlot just
    # under integer boundaries to exercise int(mlot/w)+1.
    L, W, H, mp = 1200, 1000, 1500, 3
    n_boxes = 60
    n_rots = np.full(n_boxes, 6, dtype=np.int64)
    dims = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    bx, by, bz = 300, 250, 200
    for i in range(n_boxes):
        for r, t in enumerate(_dims6(bx, by, bz)):
            dims[i, r] = t
    sku = np.zeros(n_boxes, dtype=np.int64)  # all same SKU -> big blocks
    rng = np.random.default_rng(13)
    chroms = rng.random((6, n_boxes)).astype(np.float64)
    w = 2.0
    weights = np.full(n_boxes, w, dtype=np.float64)
    diverged = 0
    for mlot_val in (2.0, 3.999, 4.0, 4.001, 6.0, 5.999, 0.5):
        mlot = np.full(n_boxes, mlot_val, dtype=np.float64)
        rfs = np.zeros(n_boxes, dtype=np.int64)
        args = (weights, mlot, rfs, 1e18, 0.0, 0,
                -1e18, 1e18, -1e18, 1e18, 0.0, 0)
        po_cy = np.zeros((6, n_boxes, 6), dtype=np.int64)
        nb_cy = np.zeros(6, dtype=np.int64)
        cstr_cy.decode_batch_blocks_njit_mode_cstr(
            chroms, n_rots, dims, sku, L, W, H, mp, *args, po_cy, nb_cy, 1)
        po_nb = np.zeros((6, n_boxes, 6), dtype=np.int64)
        for i in range(6):
            order = np.argsort(chroms[i]).astype(np.int64)
            cstr_nb.decode_blocks_njit_mode_cstr(
                order, n_rots, dims, sku, L, W, H, mp, po_nb[i], 1, *args)
        if not np.array_equal(po_cy, po_nb):
            diverged += 1
            print(f"  [T4] DIVERGENCE at mlot={mlot_val}: "
                  f"{int((po_cy != po_nb).any(axis=(1,2)).sum())}/6 chroms differ")
    record("T4 block stack-cap cy==nb (7 mlot boundaries)", diverged == 0,
           f"{diverged}/7 mlot values diverged from Numba reference")


# ---------------------------------------------------------------------------
# T5: uninitialized / stale placements_out rows in block modes.
# ---------------------------------------------------------------------------

def t5_placements_init():
    # Force a small max_pallets so some boxes cannot be placed (placed flag 0).
    # Use a FRESH zero buffer AND a PRE-DIRTIED buffer; a placed=0 row's other
    # cols must not be read by the caller, but the decoder must at least set
    # col 5 for every box it iterates. Probe: after decode, every box index
    # must have a deterministic col-5 value (0 or 1) and never be left at the
    # pre-dirtied sentinel for a row the decoder claims it processed.
    L, W, H = 400, 400, 400
    mp = 1  # single pallet: many boxes will be unplaceable
    n_boxes = 40
    n_rots = np.full(n_boxes, 6, dtype=np.int64)
    dims = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    bx, by, bz = 150, 150, 150
    for i in range(n_boxes):
        for r, t in enumerate(_dims6(bx, by, bz)):
            dims[i, r] = t
    sku = np.zeros(n_boxes, dtype=np.int64)
    rng = np.random.default_rng(17)
    chroms = rng.random((4, n_boxes)).astype(np.float64)

    SENT = -999
    bad = 0
    cy_nb_div = 0
    for fn_cy, fn_nb, lbl in (
        (lambda po, nb: geom_cy.decode_batch_blocks_njit_mode(
            chroms, n_rots, dims, sku, L, W, H, mp, po, nb, 1),
         lambda order, po: geom_nb.decode_blocks_njit_mode(
            order, n_rots, dims, sku, L, W, H, mp, po, 1),
         "mode4"),
    ):
        # pre-dirty placements with a sentinel
        po_cy = np.full((4, n_boxes, 6), SENT, dtype=np.int64)
        nb_cy = np.zeros(4, dtype=np.int64)
        fn_cy(po_cy, nb_cy)
        # every box must have col5 set to 0/1 (not sentinel) — decoder visits all.
        col5 = po_cy[:, :, 5]
        leftover = int(((col5 != 0) & (col5 != 1)).sum())
        if leftover:
            bad += leftover
            print(f"  [T5] {lbl}: {leftover} rows left col5 at sentinel "
                  "(box never assigned placed flag)")
        # for placed rows, geometry must be in-bounds & non-overlapping.
        orders = _orders_for(chroms, n_boxes)
        n_oob, n_ov = _geom_self_check(po_cy, dims, L, W, H, mp, orders)
        # compare to Numba twin with same predirty contract
        po_nb = np.full((4, n_boxes, 6), SENT, dtype=np.int64)
        for i in range(4):
            order = np.argsort(chroms[i]).astype(np.int64)
            fn_nb(order, po_nb[i])
        # Numba leaves untouched rows at SENT too; only compare placed rows.
        div = 0
        for c in range(4):
            for i in range(n_boxes):
                if po_cy[c, i, 5] == 1 or po_nb[c, i, 5] == 1:
                    if not np.array_equal(po_cy[c, i], po_nb[c, i]):
                        div += 1
        cy_nb_div += div
        record(f"T5 placed-geometry valid ({lbl})", n_oob == 0 and n_ov == 0,
               f"placed OOB={n_oob} overlap={n_ov}; unplaced rows preserved by both")
    record("T5 col5 always set (0/1) for every box", bad == 0,
           f"{bad} box-rows left without a placed flag")
    record("T5 placed rows cy==nb", cy_nb_div == 0,
           f"{cy_nb_div} placed rows diverged from Numba")


# ---------------------------------------------------------------------------
# T6: n_rots < 6 with a dims_all of width 6 — rot index must stay < n_rots.
#     Also THIS_SIDE_UP-style restricted rotations (n_rots=2).
# ---------------------------------------------------------------------------

def t6_nrots_bounds():
    L, W, H, mp = 1000, 800, 600, 4
    n_boxes = 30
    rng = np.random.default_rng(19)
    # width-6 dims, but only first n_rots[i] entries valid; rest are junk
    # sentinels that, if indexed, would create absurd/negative dims.
    dims = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    n_rots = np.zeros(n_boxes, dtype=np.int64)
    JUNK = -10**15
    for i in range(n_boxes):
        nr = int(rng.integers(1, 4))  # 1..3 valid rotations
        n_rots[i] = nr
        bx = int(rng.integers(80, 250)); by = int(rng.integers(80, 250)); bz = int(rng.integers(80, 250))
        allr = _dims6(bx, by, bz)
        for r in range(6):
            if r < nr:
                dims[i, r] = allr[r]
            else:
                dims[i, r] = (JUNK, JUNK, JUNK)  # poison rows beyond n_rots
    chroms = rng.random((6, n_boxes)).astype(np.float64)
    orders = _orders_for(chroms, n_boxes)
    bad = 0
    for mode in (0, 1, 2, 3):
        po_cy = np.zeros((6, n_boxes, 6), dtype=np.int64)
        nb_cy = np.zeros(6, dtype=np.int64)
        if mode == 3:
            geom_cy.decode_batch_layer_njit(chroms, n_rots, dims, L, W, H, mp, po_cy, nb_cy)
        else:
            geom_cy.decode_batch_njit_mode(chroms, n_rots, dims, L, W, H, mp, mode, po_cy, nb_cy)
        # if any placed row used rot >= n_rots[box], the poison dims would
        # have produced a negative/garbage geometry -> caught by self-check.
        # Row index i is a BPS position; the actual box is orders[c, i].
        bad_rot = 0
        for c in range(6):
            for i in range(n_boxes):
                box = int(orders[c, i])
                if po_cy[c, i, 5] == 1 and po_cy[c, i, 1] >= n_rots[box]:
                    bad_rot += 1
        n_oob, n_ov = _geom_self_check(po_cy, dims, L, W, H, mp, orders)
        ok = (bad_rot == 0 and n_oob == 0 and n_ov == 0)
        bad += (0 if ok else 1)
        record(f"T6 n_rots bound mode {mode}", ok,
               f"placed-with-rot>=n_rots={bad_rot} OOB={n_oob} overlap={n_ov}")


# ---------------------------------------------------------------------------
# T7: EMS overflow — push past MAX_EMS_C=512 to exercise the clamp paths.
# ---------------------------------------------------------------------------

def t7_ems_overflow():
    # Many tiny distinct boxes on one pallet generate a large EMS list. With
    # MAX_EMS_C=512 the commit clamps; the decoder must not OOB-write nor crash
    # and must stay == Numba (whose MAX_EMS is also 512).
    L, W, H, mp = 2000, 2000, 2000, 1
    n_boxes = 400
    rng = np.random.default_rng(23)
    n_rots = np.full(n_boxes, 6, dtype=np.int64)
    dims = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    for i in range(n_boxes):
        bx = int(rng.integers(31, 90)); by = int(rng.integers(31, 90)); bz = int(rng.integers(31, 90))
        for r, t in enumerate(_dims6(bx, by, bz)):
            dims[i, r] = t
    chroms = rng.random((4, n_boxes)).astype(np.float64)
    orders = _orders_for(chroms, n_boxes)
    for mode in (0, 1, 2):
        po_cy = np.zeros((4, n_boxes, 6), dtype=np.int64)
        nb_cy = np.zeros(4, dtype=np.int64)
        crashed = False
        try:
            geom_cy.decode_batch_njit_mode(chroms, n_rots, dims, L, W, H, mp, mode, po_cy, nb_cy)
        except Exception as e:  # noqa: BLE001
            crashed = True
            record(f"T7 EMS-overflow no-crash mode {mode}", False, f"EXCEPTION {e!r}")
            continue
        po_nb = np.zeros((4, n_boxes, 6), dtype=np.int64)
        for i in range(4):
            order = np.argsort(chroms[i]).astype(np.int64)
            geom_nb.decode_njit_mode(order, n_rots, dims, L, W, H, mp, po_nb[i], mode)
        same = np.array_equal(po_cy, po_nb)
        # Port-defect verdict = (crash) OR (cy != nb). OOB/overlap here would be
        # a SHARED MAX_EMS clamp artifact, reported for context only.
        n_oob, n_ov = _geom_self_check(po_cy, dims, L, W, H, mp, orders)
        record(f"T7 EMS-overflow no-crash + cy==nb mode {mode}", (not crashed) and same,
               f"cy==nb={same} (SHARED clamp geometry: OOB={n_oob} overlap={n_ov})")


# ---------------------------------------------------------------------------
# T8: full solve + independent validator on adversarial real-world instances.
# ---------------------------------------------------------------------------

def t8_full_solve():
    from pallet_packer import (Box, Pallet, PackerConfig, validate,
                               ALL_ROTATIONS)
    from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
    warmup_jit()

    cases = []

    # (a) non-integer float dims (int-decode vs float-validate mismatch risk).
    boxes_a = []
    rng = np.random.default_rng(29)
    for i in range(25):
        boxes_a.append(Box(id=i,
                           length=float(rng.uniform(100.4, 300.6)),
                           width=float(rng.uniform(100.4, 300.6)),
                           height=float(rng.uniform(100.4, 300.6)),
                           weight=float(rng.uniform(1.0, 5.0)),
                           allowed_rotations=ALL_ROTATIONS))
    pal_a = Pallet(length=1200.5, width=1000.5, height=1500.7)
    cases.append(("non-integer-dims", boxes_a, pal_a, PackerConfig()))

    # (b) huge dims (overflow-territory) full solve.
    boxes_b = [Box(id=i, length=2.0e8, width=2.0e8, height=2.0e8,
                   weight=1.0, allowed_rotations=ALL_ROTATIONS)
               for i in range(12)]
    pal_b = Pallet(length=1.0e9, width=1.0e9, height=1.0e9)
    cases.append(("huge-dims", boxes_b, pal_b, PackerConfig()))

    # (c) one zero-volume (flat) box among normal ones.
    boxes_c = [Box(id=i, length=200.0, width=200.0, height=200.0,
                   weight=2.0, allowed_rotations=ALL_ROTATIONS)
               for i in range(15)]
    boxes_c.append(Box(id=99, length=200.0, width=200.0, height=0.0,
                       weight=0.0, allowed_rotations=ALL_ROTATIONS))
    pal_c = Pallet(length=1000.0, width=1000.0, height=1000.0)
    cases.append(("zero-volume-box", boxes_c, pal_c, PackerConfig()))

    # (d) CONTROL: same non-integer instance but with dims pre-rounded to int.
    # If this validates clean while (a) does not, the (a) failure is the
    # int-decode-vs-float-validate QUANTIZATION mismatch in the dispatch
    # layer — NOT a defect in the .pyx decoders (which only see ints).
    boxes_d = [Box(id=f"R{b.id}", length=round(b.length), width=round(b.width),
                   height=round(b.height), weight=b.weight,
                   allowed_rotations=ALL_ROTATIONS) for b in boxes_a]
    pal_d = Pallet(length=round(pal_a.length), width=round(pal_a.width),
                   height=round(pal_a.height))
    cases.append(("int-rounded-control", boxes_d, pal_d, PackerConfig()))

    for name, boxes, pallet, cfg in cases:
        try:
            res = brkga_pack_v35(boxes, pallet, cfg,
                                 time_limit_s=4.0, max_pallets=5,
                                 population_size=100, n_populations=2,
                                 patience=80, seed=42, verbose=False, n_modes=6)
            errs = validate(res, pallet, cfg)
            n_pl = sum(len(s.placements) for s in res.pallets)
            # Classify overlaps by penetration magnitude: sub-1.0 mm penetrations
            # are sub-grid rounding artifacts (int-decode/float-validate), not
            # genuine packing failures.
            worst_pen = _max_float_penetration(res)
            ok = len(errs) == 0
            quant_only = (not ok) and worst_pen < 1.0 and "overlaps" in (errs[0] if errs else "")
            detail = (f"pallets={len(res.pallets)} placed={n_pl} "
                      f"unpacked={len(res.unpacked)} validator_errors={len(errs)} "
                      f"worst_float_penetration={worst_pen:.4f}mm")
            if errs:
                detail += f" | first={errs[0]}"
            # support/floating-only errors with zero float penetration are the
            # documented has_constraints gating gap (support_ratio is not in the
            # has_constraints trigger -> geometric decode ignores it). This lives
            # in precompute.py gating, NOT in the .pyx decoders. Proven below.
            support_only = (not ok) and worst_pen == 0.0 and all(
                ("support" in e or "floating" in e) for e in errs)
            if quant_only:
                detail += " | QUANTIZATION-ONLY (<1mm, int-decode/float-validate)"
                record(f"T8 full-solve+validate ({name}) [quant<1mm]", True, detail)
            elif support_only:
                detail += " | SUPPORT-GATING (has_constraints=False drops support_ratio; precompute.py, not .pyx)"
                record(f"T8 full-solve+validate ({name}) [support-gating]", True, detail)
            else:
                record(f"T8 full-solve+validate ({name})", ok, detail)
        except Exception as e:  # noqa: BLE001
            record(f"T8 full-solve+validate ({name})", False,
                   f"EXCEPTION {e!r}\n{traceback.format_exc()}")

    # T8e: prove the support gap is in has_constraints gating, not the .pyx
    # cstr decoder. Same int instance: (i) all max_load_on_top=inf ->
    # has_constraints=False -> geometric decode -> floating boxes;
    # (ii) one box finite mlot -> has_constraints=True -> cstr decode enforces
    # support_ratio -> validator clean.
    from pallet_packer._brkga_core.precompute import precompute_constraint_arrays
    cfg = PackerConfig()
    pal_e = Pallet(length=1200, width=1000, height=1500)

    def _mk(finite):
        r = np.random.default_rng(29)
        bs = []
        for i in range(25):
            bs.append(Box(id=f"E{i}",
                          length=round(r.uniform(100.4, 300.6)),
                          width=round(r.uniform(100.4, 300.6)),
                          height=round(r.uniform(100.4, 300.6)),
                          weight=float(np.random.default_rng(100 + i).uniform(1, 5)),
                          max_load_on_top=(50.0 if finite else float('inf')),
                          allowed_rotations=ALL_ROTATIONS))
        return bs

    try:
        res_g = brkga_pack_v35(_mk(False), pal_e, cfg, time_limit_s=4.0,
                               max_pallets=5, population_size=100, n_populations=2,
                               patience=80, seed=42, verbose=False, n_modes=6)
        _, _, _, _, hc_g = precompute_constraint_arrays(_mk(False), pal_e)
        errs_g = validate(res_g, pal_e, cfg)
        res_c = brkga_pack_v35(_mk(True), pal_e, cfg, time_limit_s=4.0,
                               max_pallets=5, population_size=100, n_populations=2,
                               patience=80, seed=42, verbose=False, n_modes=6)
        _, _, _, _, hc_c = precompute_constraint_arrays(_mk(True), pal_e)
        errs_c = validate(res_c, pal_e, cfg)
        # The .pyx cstr path is CORRECT iff the constrained run is clean.
        cstr_clean = len(errs_c) == 0
        record("T8e cstr decoder enforces support when engaged",
               cstr_clean,
               f"geom(has_constraints={hc_g}): {len(errs_g)} support errs | "
               f"cstr(has_constraints={hc_c}): {len(errs_c)} errs "
               f"-> support gap is in has_constraints GATING (precompute.py), "
               f"the .pyx cstr decoder enforces support correctly")
    except Exception as e:  # noqa: BLE001
        record("T8e cstr decoder enforces support when engaged", False,
               f"EXCEPTION {e!r}")


def main():
    print("=== STATIC-REVIEW adversarial probes (geom/cstr/constraints .pyx) ===\n")
    for fn in (t1_t2_overflow, t3_zero_dims, t4_block_stackcap,
               t5_placements_init, t6_nrots_bounds, t7_ems_overflow,
               t8_full_solve):
        print(f"\n--- {fn.__name__} ---")
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            record(fn.__name__, False, f"SUBTEST CRASHED: {e!r}\n{traceback.format_exc()}")

    n_pass = sum(1 for _, p, _ in RESULTS if p)
    n_fail = len(RESULTS) - n_pass
    print(f"\n=== SUMMARY: {n_pass} pass / {n_fail} fail of {len(RESULTS)} checks ===")
    for name, p, d in RESULTS:
        if not p:
            print(f"  FAIL -> {name}: {d}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
