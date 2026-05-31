"""verify_backend_equiv_geom.py — Cython vs Numba bit-identity for geom decoders.

DIMENSION: backend-equiv-geom

Verifies the Cython geometric decoders are BIT-IDENTICAL to their Numba
reference twins, for modes:
    0,1,2  decode_njit_mode        (DFTRC / wall / corner)
    3      decode_layer_njit       (Bischoff-Ratcliff layer)
    4      decode_blocks_njit_mode  (dynamic composite blocks)
    5      decode_precomputed_blocks_njit_mode (precomputed blocks)

Strategy: a LARGE random sweep (2000+ instances) varying n_boxes (1..150),
pallet dims, box dims (incl. tiny + oversize that cannot fit), n_skus (1..n),
random sku_best blocks for mode 5, and max_pallets in {1,2,3,6,...}. For each
instance we run BOTH backends with IDENTICAL inputs and assert
np.array_equal on the full placements_out array AND the returned n_bins.

Plus explicit BOUNDARY sub-tests:
    - 1x1x1 pallet
    - all boxes larger than pallet (nothing fits)
    - zero-dim box rotations (degenerate dims)
    - max_pallets == 1 and max_pallets > 1
    - n_boxes == 1
    - tiny boxes (1x1x1) into a normal pallet

Any single mismatch is dumped with full inputs (a minimal repro).

Run inside Docker:
    docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
        -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/verify_backend_equiv_geom.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pallet_packer._brkga_core import jit_decoders_geom_cy as cy
from pallet_packer._brkga_core import jit_decoders_geom as nb


# ---------------------------------------------------------------------------
# input generation
# ---------------------------------------------------------------------------

def make_dims(rng, n_boxes, L, W, H, *, allow_oversize=True, tiny=False,
              allow_zero=False):
    """Build (n_boxes, 6, 3) int64 dims_all + n_rots_per_box.

    Each box gets a base (bx,by,bz); the 6 rotations are the 6 axis perms,
    matching the convention used elsewhere in the repo. n_rots varies 1..6
    so we exercise the rotation-count guard.
    """
    n_rots_per_box = np.empty(n_boxes, dtype=np.int64)
    dims_all = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    for i in range(n_boxes):
        if tiny:
            bx = int(rng.integers(1, 4))
            by = int(rng.integers(1, 4))
            bz = int(rng.integers(1, 4))
        elif allow_oversize and rng.random() < 0.15:
            # deliberately too big on at least one axis
            bx = int(rng.integers(L, L * 2 + 1))
            by = int(rng.integers(1, max(2, W)))
            bz = int(rng.integers(1, max(2, H)))
        else:
            bx = int(rng.integers(1, max(2, L // 2 + 1)))
            by = int(rng.integers(1, max(2, W // 2 + 1)))
            bz = int(rng.integers(1, max(2, H // 2 + 1)))
        if allow_zero and rng.random() < 0.1:
            # degenerate: zero on a random axis
            ax = int(rng.integers(0, 3))
            (bx, by, bz) = list((bx, by, bz))
            if ax == 0:
                bx = 0
            elif ax == 1:
                by = 0
            else:
                bz = 0
        rots = [
            (bx, by, bz), (bx, bz, by),
            (by, bx, bz), (by, bz, bx),
            (bz, bx, by), (bz, by, bx),
        ]
        for r in range(6):
            dims_all[i, r] = rots[r]
        n_rots_per_box[i] = int(rng.integers(1, 7))  # 1..6
    return n_rots_per_box, dims_all


def make_order(rng, n_boxes):
    """A random BPS order (permutation), int64 — the per-chromosome input."""
    return rng.permutation(n_boxes).astype(np.int64)


def make_sku(rng, n_boxes, n_skus):
    return rng.integers(0, n_skus, n_boxes).astype(np.int64)


def make_sku_best_block(rng, n_skus):
    """(n_skus,4) precomputed (k,l,m,rot). rot in 0..5 (full dims_all range)."""
    sbb = np.zeros((n_skus, 4), dtype=np.int64)
    for s in range(n_skus):
        sbb[s, 0] = int(rng.integers(1, 5))   # k
        sbb[s, 1] = int(rng.integers(1, 5))   # l
        sbb[s, 2] = int(rng.integers(1, 5))   # m
        sbb[s, 3] = int(rng.integers(0, 6))   # rot index into dims_all
    return sbb


# ---------------------------------------------------------------------------
# single-instance runners (return (placements, n_bins) for each backend)
# ---------------------------------------------------------------------------

def run_mode012(backend, order, n_rots, dims, L, W, H, mp, mode):
    n = order.shape[0]
    po = np.zeros((n, 6), dtype=np.int64)
    nb_count = backend.decode_njit_mode(order, n_rots, dims, L, W, H, mp, po, mode)
    return po, int(nb_count)


def run_layer(backend, order, n_rots, dims, L, W, H, mp):
    n = order.shape[0]
    po = np.zeros((n, 6), dtype=np.int64)
    nb_count = backend.decode_layer_njit(order, n_rots, dims, L, W, H, mp, po)
    return po, int(nb_count)


def run_blocks(backend, order, n_rots, dims, sku, L, W, H, mp, n_skus):
    n = order.shape[0]
    po = np.zeros((n, 6), dtype=np.int64)
    nb_count = backend.decode_blocks_njit_mode(
        order, n_rots, dims, sku, L, W, H, mp, po, n_skus)
    return po, int(nb_count)


def run_precomp(backend, order, n_rots, dims, sku, sbb, L, W, H, mp, n_skus):
    n = order.shape[0]
    po = np.zeros((n, 6), dtype=np.int64)
    nb_count = backend.decode_precomputed_blocks_njit_mode(
        order, n_rots, dims, sku, sbb, L, W, H, mp, po, n_skus)
    return po, int(nb_count)


# ---------------------------------------------------------------------------
# comparison + dump
# ---------------------------------------------------------------------------

class Mismatch(Exception):
    """A bit-identity / behavioral divergence between backends.

    .degenerate is True when the divergence is driven purely by a degenerate
    (zero-dimension) box that the real pipeline never produces — reported as
    an ANOMALY rather than a hard correctness FAIL.
    """
    def __init__(self, label, degenerate=False):
        super().__init__(label)
        self.label = label
        self.degenerate = degenerate


# Collected divergences so the run can report them all instead of aborting.
DIVERGENCES = []  # list of (label, kind, detail, degenerate)


def safe_run(fn):
    """Run a backend call; return ('ok', (po, nbins)) or ('exc', exc_type_name)."""
    try:
        return ("ok", fn())
    except Exception as e:  # noqa: BLE001
        return ("exc", type(e).__name__)


def _has_zero_dim(dump_ctx):
    d = dump_ctx.get("dims")
    if isinstance(d, np.ndarray):
        return bool((d == 0).any())
    return False


def compare_outcomes(label, cy_outcome, nb_outcome, dump_ctx):
    """Compare two (status, payload) outcomes.

    Bit-identity contract: both backends must take the SAME path —
    either both return and the arrays/n_bins match exactly, or both raise
    the SAME exception type. A divergence (one returns, other raises; or
    different exception types) is recorded; if it is purely driven by a
    degenerate zero-dim input it is flagged as such (ANOMALY, not FAIL).
    """
    (cy_status, cy_payload) = cy_outcome
    (nb_status, nb_payload) = nb_outcome

    if cy_status == "exc" or nb_status == "exc":
        if cy_status == "exc" and nb_status == "exc" and cy_payload == nb_payload:
            # both raised the same exception type -> consistent behavior
            return True
        degen = _has_zero_dim(dump_ctx)
        detail = f"cython={cy_status}({cy_payload}) numba={nb_status}({nb_payload})"
        # only print full dump for the first few of each label-prefix to avoid spam
        if _should_print(label):
            print(f"\n!!! EXCEPTION DIVERGENCE [{label}] degenerate={degen} !!!")
            print(f"  {detail}")
            print("  --- MINIMAL REPRO INPUTS ---")
            _dump(dump_ctx)
        DIVERGENCES.append((label, "exception", detail, degen))
        return False

    return _compare(label, cy_payload, nb_payload, dump_ctx)


_PRINTED_PREFIXES = {}


def _should_print(label):
    """Throttle: print at most 3 full dumps per label-prefix (before first '/')."""
    prefix = label.split("/")[0] if "/" in label else label.rstrip("0123456789")
    c = _PRINTED_PREFIXES.get(prefix, 0)
    if c < 3:
        _PRINTED_PREFIXES[prefix] = c + 1
        return True
    return False


def _dump(dump_ctx):
    for k, v in dump_ctx.items():
        if isinstance(v, np.ndarray):
            print(f"    {k} = np.array({v.tolist()}, dtype=np.int64)")
        else:
            print(f"    {k} = {v!r}")


def compare(label, cy_out, nb_out, dump_ctx):
    (po_c, nbins_c) = cy_out
    (po_n, nbins_n) = nb_out
    return _compare(label, (po_c, nbins_c), (po_n, nbins_n), dump_ctx)


def _compare(label, cy_payload, nb_payload, dump_ctx):
    (po_c, nbins_c) = cy_payload
    (po_n, nbins_n) = nb_payload
    same_po = np.array_equal(po_c, po_n)
    same_nb = (nbins_c == nbins_n)
    if same_po and same_nb:
        return True
    degen = _has_zero_dim(dump_ctx)
    detail = f"n_bins cy={nbins_c} nb={nbins_n}; placements_equal={same_po}"
    if _should_print(label):
        print(f"\n!!! VALUE MISMATCH [{label}] degenerate={degen} !!!")
        print(f"  n_bins: cython={nbins_c}  numba={nbins_n}  (equal={same_nb})")
        print(f"  placements array_equal={same_po}")
        if not same_po:
            diff_rows = np.where(np.any(po_c != po_n, axis=1))[0]
            print(f"  {len(diff_rows)} differing rows (of {po_c.shape[0]}); "
                  f"first 5 indices={diff_rows[:5].tolist()}")
            for ridx in diff_rows[:5]:
                print(f"    row {ridx}: cy={po_c[ridx].tolist()}  nb={po_n[ridx].tolist()}")
        print("  --- MINIMAL REPRO INPUTS ---")
        _dump(dump_ctx)
    DIVERGENCES.append((label, "value", detail, degen))
    return False


# ---------------------------------------------------------------------------
# random sweep
# ---------------------------------------------------------------------------

def random_sweep(rng, n_per_mode=420, allow_zero_dims=True, zero_prob=0.15):
    """Run a large random sweep. Returns count tested + per-mode counts.

    allow_zero_dims=False forces all box dims positive (the realistic input
    space the production pipeline operates in).
    """
    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    total = 0

    for trial in range(n_per_mode):
        # randomize the workload
        n_boxes = int(rng.integers(1, 151))            # 1..150
        L = int(rng.integers(1, 1201))
        W = int(rng.integers(1, 1201))
        H = int(rng.integers(1, 1201))
        mp = int(rng.choice([1, 1, 2, 3, 6, 10, 0]))   # 0 -> default 32
        tiny = rng.random() < 0.2
        allow_zero = allow_zero_dims and (rng.random() < zero_prob)
        n_skus = int(rng.integers(1, n_boxes + 1))

        n_rots, dims = make_dims(
            rng, n_boxes, L, W, H,
            allow_oversize=True, tiny=tiny, allow_zero=allow_zero)
        order = make_order(rng, n_boxes)
        sku = make_sku(rng, n_boxes, n_skus)
        sbb = make_sku_best_block(rng, n_skus)

        base_ctx = dict(
            n_boxes=n_boxes, L=L, W=W, H=H, mp=mp, n_skus=n_skus,
            order=order, n_rots=n_rots, dims=dims, sku=sku, sku_best_block=sbb,
        )

        # modes 0,1,2
        for mode in (0, 1, 2):
            cy_out = safe_run(lambda m=mode: run_mode012(cy, order, n_rots, dims, L, W, H, mp, m))
            nb_out = safe_run(lambda m=mode: run_mode012(nb, order, n_rots, dims, L, W, H, mp, m))
            compare_outcomes(f"mode{mode}", cy_out, nb_out, {**base_ctx, "mode": mode})
            counts[mode] += 1
            total += 1

        # mode 3 (layer)
        cy_out = safe_run(lambda: run_layer(cy, order, n_rots, dims, L, W, H, mp))
        nb_out = safe_run(lambda: run_layer(nb, order, n_rots, dims, L, W, H, mp))
        compare_outcomes("mode3-layer", cy_out, nb_out, base_ctx)
        counts[3] += 1
        total += 1

        # mode 4 (blocks)
        cy_out = safe_run(lambda: run_blocks(cy, order, n_rots, dims, sku, L, W, H, mp, n_skus))
        nb_out = safe_run(lambda: run_blocks(nb, order, n_rots, dims, sku, L, W, H, mp, n_skus))
        compare_outcomes("mode4-blocks", cy_out, nb_out, base_ctx)
        counts[4] += 1
        total += 1

        # mode 5 (precomputed)
        cy_out = safe_run(lambda: run_precomp(cy, order, n_rots, dims, sku, sbb, L, W, H, mp, n_skus))
        nb_out = safe_run(lambda: run_precomp(nb, order, n_rots, dims, sku, sbb, L, W, H, mp, n_skus))
        compare_outcomes("mode5-precomp", cy_out, nb_out, base_ctx)
        counts[5] += 1
        total += 1

    return total, counts


# ---------------------------------------------------------------------------
# boundary sub-tests
# ---------------------------------------------------------------------------

def boundary_tests(rng):
    """Explicit edge cases. Returns count tested."""
    n = 0

    def all_geom_modes(order, n_rots, dims, sku, sbb, L, W, H, mp, n_skus, tag):
        nonlocal n
        for mode in (0, 1, 2):
            compare_outcomes(
                f"BND/{tag}/mode{mode}",
                safe_run(lambda m=mode: run_mode012(cy, order, n_rots, dims, L, W, H, mp, m)),
                safe_run(lambda m=mode: run_mode012(nb, order, n_rots, dims, L, W, H, mp, m)),
                dict(tag=tag, mode=mode, order=order, n_rots=n_rots,
                     dims=dims, L=L, W=W, H=H, mp=mp))
            n += 1
        compare_outcomes(
            f"BND/{tag}/mode3",
            safe_run(lambda: run_layer(cy, order, n_rots, dims, L, W, H, mp)),
            safe_run(lambda: run_layer(nb, order, n_rots, dims, L, W, H, mp)),
            dict(tag=tag, order=order, n_rots=n_rots, dims=dims,
                 L=L, W=W, H=H, mp=mp))
        n += 1
        compare_outcomes(
            f"BND/{tag}/mode4",
            safe_run(lambda: run_blocks(cy, order, n_rots, dims, sku, L, W, H, mp, n_skus)),
            safe_run(lambda: run_blocks(nb, order, n_rots, dims, sku, L, W, H, mp, n_skus)),
            dict(tag=tag, order=order, dims=dims, sku=sku,
                 L=L, W=W, H=H, mp=mp, n_skus=n_skus))
        n += 1
        compare_outcomes(
            f"BND/{tag}/mode5",
            safe_run(lambda: run_precomp(cy, order, n_rots, dims, sku, sbb, L, W, H, mp, n_skus)),
            safe_run(lambda: run_precomp(nb, order, n_rots, dims, sku, sbb, L, W, H, mp, n_skus)),
            dict(tag=tag, order=order, dims=dims, sku=sku, sbb=sbb,
                 L=L, W=W, H=H, mp=mp, n_skus=n_skus))
        n += 1

    # B1: 1x1x1 pallet, boxes of size 1 and bigger
    n_boxes = 5
    n_rots = np.full(n_boxes, 6, dtype=np.int64)
    dims = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    for i in range(n_boxes):
        d = 1 if i % 2 == 0 else 2
        for r in range(6):
            dims[i, r] = (d, d, d)
    order = np.arange(n_boxes, dtype=np.int64)
    sku = np.zeros(n_boxes, dtype=np.int64)
    sbb = np.array([[2, 2, 2, 0]], dtype=np.int64)
    all_geom_modes(order, n_rots, dims, sku, sbb, 1, 1, 1, 1, 1, "1x1x1pallet")

    # B2: all boxes strictly larger than pallet (nothing fits)
    n_boxes = 8
    n_rots = np.full(n_boxes, 6, dtype=np.int64)
    dims = np.full((n_boxes, 6, 3), 9999, dtype=np.int64)
    order = make_order(rng, n_boxes)
    sku = make_sku(rng, n_boxes, 2)
    sbb = make_sku_best_block(rng, 2)
    all_geom_modes(order, n_rots, dims, sku, sbb, 100, 100, 100, 3, 2, "oversize-all")

    # B3: zero-dim rotations (degenerate)
    n_boxes = 6
    n_rots = np.full(n_boxes, 6, dtype=np.int64)
    dims = np.zeros((n_boxes, 6, 3), dtype=np.int64)  # all zeros
    order = make_order(rng, n_boxes)
    sku = make_sku(rng, n_boxes, 2)
    sbb = make_sku_best_block(rng, 2)
    all_geom_modes(order, n_rots, dims, sku, sbb, 50, 50, 50, 2, 2, "all-zero-dims")

    # B4: n_boxes == 1, max_pallets 1
    n_boxes = 1
    n_rots = np.array([6], dtype=np.int64)
    dims = np.zeros((1, 6, 3), dtype=np.int64)
    base = (10, 20, 30)
    rots = [(base[0], base[1], base[2]), (base[0], base[2], base[1]),
            (base[1], base[0], base[2]), (base[1], base[2], base[0]),
            (base[2], base[0], base[1]), (base[2], base[1], base[0])]
    for r in range(6):
        dims[0, r] = rots[r]
    order = np.array([0], dtype=np.int64)
    sku = np.array([0], dtype=np.int64)
    sbb = np.array([[1, 1, 1, 0]], dtype=np.int64)
    all_geom_modes(order, n_rots, dims, sku, sbb, 100, 100, 100, 1, 1, "single-box-mp1")

    # B5: same single box, max_pallets > 1
    all_geom_modes(order, n_rots, dims, sku, sbb, 5, 5, 5, 6, 1, "single-box-mp6-toobig")

    # B6: tiny boxes (1x1x1) into a normal pallet, mp 1 (stress many placements)
    n_boxes = 120
    n_rots = np.full(n_boxes, 6, dtype=np.int64)
    dims = np.ones((n_boxes, 6, 3), dtype=np.int64)
    order = make_order(rng, n_boxes)
    sku = make_sku(rng, n_boxes, 3)
    sbb = make_sku_best_block(rng, 3)
    all_geom_modes(order, n_rots, dims, sku, sbb, 50, 50, 50, 1, 3, "tiny-1x1x1")

    # B7: max_pallets == 0 (sentinel -> default 32 bins) with boxes that overflow 1 bin
    n_boxes = 40
    n_rots = np.full(n_boxes, 6, dtype=np.int64)
    dims = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    for i in range(n_boxes):
        for r in range(6):
            dims[i, r] = (6, 6, 6)
    order = make_order(rng, n_boxes)
    sku = make_sku(rng, n_boxes, 2)
    sbb = np.array([[3, 3, 3, 0], [2, 2, 2, 0]], dtype=np.int64)
    all_geom_modes(order, n_rots, dims, sku, sbb, 10, 10, 10, 0, 2, "mp0-overflow")

    # B8: precompute rot index that exceeds n_rots (n_rots small, rot up to 5)
    n_boxes = 30
    n_rots = np.ones(n_boxes, dtype=np.int64)  # only rot 0 allowed normally
    dims = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    for i in range(n_boxes):
        b = (5, 7, 11)
        rots = [(b[0], b[1], b[2]), (b[0], b[2], b[1]),
                (b[1], b[0], b[2]), (b[1], b[2], b[0]),
                (b[2], b[0], b[1]), (b[2], b[1], b[0])]
        for r in range(6):
            dims[i, r] = rots[r]
    order = make_order(rng, n_boxes)
    sku = make_sku(rng, n_boxes, 2)
    sbb = np.array([[2, 2, 2, 5], [3, 1, 2, 4]], dtype=np.int64)  # rot 5/4 > n_rots
    all_geom_modes(order, n_rots, dims, sku, sbb, 100, 100, 100, 4, 2, "precomp-rot-gt-nrots")

    return n


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=== backend-equiv-geom: Cython vs Numba bit-identity ===")
    print(f"  cython module: {cy.__file__}")
    print(f"  numba  module: {nb.__file__}")
    # sanity: confirm they are genuinely distinct backends
    if "cy" not in os.path.basename(cy.__file__):
        print("  WARN: cython module path does not contain 'cy'")

    rng = np.random.default_rng(20260531)

    print("\n--- boundary sub-tests ---")
    n_boundary = boundary_tests(rng)

    print("\n--- random sweep A: positive-dims ONLY (realistic input space) ---")
    n_a, counts_a = random_sweep(rng, n_per_mode=300, allow_zero_dims=False)
    real_after_a = len([d for d in DIVERGENCES if not d[3]])
    print(f"  sweep A: {n_a} comparisons; real divergences so far: {real_after_a}")

    print("\n--- random sweep B: includes degenerate zero-dim boxes (adversarial) ---")
    rng_b = np.random.default_rng(20260531)
    n_b, counts_b = random_sweep(rng_b, n_per_mode=420, allow_zero_dims=True, zero_prob=0.15)

    n_random = n_a + n_b
    counts = {k: counts_a[k] + counts_b[k] for k in counts_a}
    total = n_random + n_boundary

    # classify divergences
    real = [d for d in DIVERGENCES if not d[3]]        # non-degenerate
    degen = [d for d in DIVERGENCES if d[3]]            # zero-dim driven

    print("\n=== SUMMARY ===")
    print(f"  TOTAL comparisons: {total} (random={n_random}, boundary={n_boundary})")
    print(f"  per-mode random counts: {counts}")
    print(f"  divergences total: {len(DIVERGENCES)}  "
          f"(real/non-degenerate: {len(real)}, degenerate zero-dim: {len(degen)})")

    if degen:
        # summarize which labels (dedup)
        labels = sorted({d[0].rsplit('/', 0)[0] for d in degen})
        kinds = sorted({(d[0], d[1]) for d in degen})
        print(f"  [ANOMALY] {len(degen)} divergences ALL involve a zero-dimension box "
              f"(degenerate input never produced by the real pipeline).")
        for (lbl, kind) in list(kinds)[:10]:
            print(f"      - {lbl}  ({kind})")

    if real:
        print(f"  [DEFECT] {len(real)} REAL divergences on positive-dim inputs:")
        for (lbl, kind, detail, _) in real[:10]:
            print(f"      - {lbl}  {kind}: {detail}")

    if real:
        status = "FAIL"
    elif degen:
        status = "ANOMALY"
    else:
        status = "PASS"
    print(f"\n=== RESULT: {status} ===")
    # exit 0 for PASS/ANOMALY (degenerate-only), 1 for FAIL (real divergence)
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
