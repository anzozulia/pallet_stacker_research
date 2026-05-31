"""verify_edge_geom.py — adversarial geometric EDGE-CASE probe.

Drives a battery of degenerate geometric inputs through BOTH
  (a) decode_chromosome  (the per-chromosome decoder path), and
  (b) brkga_pack_v35     (the full solve / driver path),
then asserts for each:
  * no crash / exception
  * validate() returns ZERO errors
  * every placement is in-bounds, non-negative, integer-faithful
  * the "should be impossible to pack" cases leave items in .unpacked
    (NOT silently dropped, NOT illegally placed)
  * placed + unpacked == all boxes (conservation: no box vanishes)

Edge cases (per mandate):
  1. empty box list (n=0)            -> empty PackResult, no crash
  2. single box that fits
  3. single box LARGER than pallet in EVERY dim -> unpacked, no illegal place
  4. box exactly == pallet dims      -> packs, fills 100%
  5. two boxes
  6. 200 identical SKU
  7. box that fits ONLY in one rotation (others would overflow)
  8. pallet smaller than every box   -> all unpacked

Run inside Docker (OMP_NUM_THREADS=1 for thread-invariant correctness):
    docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
        -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/verify_edge_geom.py
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pallet_packer import (
    Box, Pallet, PackerConfig, Rotation, ALL_ROTATIONS, NO_ROTATION, validate,
)
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer._brkga_core.dispatch import decode_chromosome
from pallet_packer._brkga_core.precompute import (
    precompute_box_dims_and_sku, precompute_constraint_arrays, _NO_LIMIT,
)
from benchmarks.br import geometric_only_config


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
FAILS = []          # list of (case, message)
def fail(case, msg):
    FAILS.append((case, msg))
    print(f"    [FAIL] {case}: {msg}")

def ok(case, msg):
    print(f"    [ pass] {case}: {msg}")


def check_result_sane(case, result, pallet, config, all_boxes,
                       expect_all_unpacked=False, expect_n_placed=None,
                       label=""):
    """Run every invariant on a PackResult. Returns True if all pass."""
    tag = f"{case}{('/' + label) if label else ''}"
    good = True

    # (a) validator must be clean
    try:
        errs = validate(result, pallet, config)
    except Exception as e:
        fail(tag, f"validate() raised {type(e).__name__}: {e}")
        return False
    if errs:
        good = False
        fail(tag, f"validator returned {len(errs)} error(s); first: {errs[0]}")

    # (b) conservation: placed + unpacked == all boxes (by identity)
    placed_ids = [id(p.box) for st in result.pallets for p in st.placements]
    unpacked_ids = [id(b) for b in result.unpacked]
    n_placed = len(placed_ids)
    n_unpacked = len(unpacked_ids)
    if n_placed + n_unpacked != len(all_boxes):
        good = False
        fail(tag, f"conservation broken: placed {n_placed} + unpacked "
                  f"{n_unpacked} != total {len(all_boxes)}")
    # No duplicate placement of the same box object
    if len(set(placed_ids)) != len(placed_ids):
        good = False
        fail(tag, f"a box object was placed more than once "
                  f"({len(placed_ids)} placements, {len(set(placed_ids))} unique)")
    # No box both placed AND unpacked
    overlap = set(placed_ids) & set(unpacked_ids)
    if overlap:
        good = False
        fail(tag, f"{len(overlap)} box(es) both placed and unpacked")

    # (c) geometry sanity directly on placements (independent of validator
    #     support semantics): negative coords / out-of-bounds.
    for st in result.pallets:
        for p in st.placements:
            if p.x < -1e-6 or p.y < -1e-6 or p.z < -1e-6:
                good = False
                fail(tag, f"negative coord: {p.box.id} at ({p.x},{p.y},{p.z})")
            if (p.x2 > pallet.length + 1e-6 or
                    p.y2 > pallet.width + 1e-6 or
                    p.z2 > pallet.height + 1e-6):
                good = False
                fail(tag, f"out-of-bounds: {p.box.id} extent "
                          f"({p.x2},{p.y2},{p.z2}) vs pallet "
                          f"({pallet.length},{pallet.width},{pallet.height})")
            # rotation must be in the box's allowed set
            if p.rotation not in p.box.allowed_rotations:
                good = False
                fail(tag, f"{p.box.id} placed with disallowed rotation "
                          f"{p.rotation.name}")

    # (d) expectation checks
    if expect_all_unpacked and n_placed != 0:
        good = False
        fail(tag, f"expected ALL unpacked, but {n_placed} got placed "
                  f"(illegal: cannot fit)")
    if expect_n_placed is not None and n_placed != expect_n_placed:
        good = False
        fail(tag, f"expected {expect_n_placed} placed, got {n_placed}")

    if good:
        ok(tag, f"placed={n_placed} unpacked={n_unpacked} validator=clean")
    return good


def run_full(boxes, pallet, config, max_pallets=1, t=2.0, seed=42):
    """Full solve via the driver."""
    return brkga_pack_v35(
        boxes, pallet, config,
        time_limit_s=t, max_pallets=max_pallets,
        population_size=60, n_populations=2, patience=40,
        local_search_budget_s=0.5, seed=seed, verbose=False, n_modes=6,
    )


def run_decode(boxes, pallet, config, max_pallets=1, mode=0, rng_seed=1,
               extreme_chrom=None):
    """Direct per-chromosome decode. Builds the precompute arrays the way the
    driver does and passes the geometric/constraint path explicitly."""
    n = len(boxes)
    n_rots_arr, dims_all, sku = precompute_box_dims_and_sku(boxes)
    weights, mlot, rfs, pmw, has_c = precompute_constraint_arrays(boxes, pallet)
    sr = float(config.support_ratio) if has_c else 0.0
    chrom_size = 2 * n + 1
    if extreme_chrom is not None:
        chrom = extreme_chrom
    else:
        chrom = np.random.default_rng(rng_seed).random(chrom_size)
    return decode_chromosome(
        chrom, boxes, pallet, config, n_rots_arr, dims_all, mode,
        max_pallets=max_pallets,
        sku_id_per_box=sku,
        weights=weights, mlot=mlot, rfs=rfs,
        pallet_max_weight=pmw, has_constraints=has_c, support_ratio=sr,
    )


def decode_all_modes(case, boxes, pallet, config, all_boxes,
                     expect_all_unpacked=False, max_pallets=1):
    """Run the per-chromosome decoder across every mode 0..5 and a couple of
    adversarial chromosomes; sanity-check each result."""
    good = True
    for mode in range(6):
        for rs in (1, 7):
            try:
                r = run_decode(boxes, pallet, config, max_pallets=max_pallets,
                               mode=mode, rng_seed=rs)
            except Exception as e:
                good = False
                fail(f"{case}/decode m{mode} s{rs}",
                     f"raised {type(e).__name__}: {e}\n{traceback.format_exc()}")
                continue
            good &= check_result_sane(
                case, r, pallet, config, all_boxes,
                expect_all_unpacked=expect_all_unpacked,
                label=f"decode m{mode} s{rs}")
    # adversarial chromosomes: all-zeros, all-ones, reversed
    n = len(boxes)
    if n > 0:
        for name, chrom in [
            ("zeros", np.zeros(2 * n + 1)),
            ("ones", np.ones(2 * n + 1)),
            ("descending", np.linspace(1.0, 0.0, 2 * n + 1)),
        ]:
            try:
                r = run_decode(boxes, pallet, config, max_pallets=max_pallets,
                               mode=4, extreme_chrom=chrom)
            except Exception as e:
                good = False
                fail(f"{case}/decode adv-{name}",
                     f"raised {type(e).__name__}: {e}")
                continue
            good &= check_result_sane(
                case, r, pallet, config, all_boxes,
                expect_all_unpacked=expect_all_unpacked,
                label=f"decode adv-{name}")
    return good


# ===========================================================================
# Edge cases
# ===========================================================================
def case_1_empty():
    print("\n=== Case 1: empty box list (n=0) ===")
    pallet = Pallet(length=1000, width=800, height=600)
    cfg = geometric_only_config(max_pallets=1)
    good = True
    # full solve
    try:
        r = run_full([], pallet, cfg)
        if r.pallets or r.unpacked:
            good = False
            fail("1/full", f"non-empty result: pallets={len(r.pallets)} "
                           f"unpacked={len(r.unpacked)}")
        else:
            ok("1/full", "empty PackResult, no crash")
    except Exception as e:
        good = False
        fail("1/full", f"raised {type(e).__name__}: {e}")
    # decode path
    try:
        n_rots, dims, sku = precompute_box_dims_and_sku([])
        r = decode_chromosome(np.array([]), [], pallet, cfg, n_rots, dims, 0,
                              max_pallets=1, sku_id_per_box=sku)
        if r.pallets or r.unpacked:
            good = False
            fail("1/decode", "non-empty result on empty input")
        else:
            ok("1/decode", "empty PackResult, no crash")
    except Exception as e:
        good = False
        fail("1/decode", f"raised {type(e).__name__}: {e}")
    return good


def case_2_single_fits():
    print("\n=== Case 2: single box that fits ===")
    pallet = Pallet(length=1000, width=800, height=600)
    cfg = geometric_only_config(max_pallets=1)
    boxes = [Box(id="B0", length=300, width=200, height=150)]
    good = True
    r = run_full(boxes, pallet, cfg)
    good &= check_result_sane("2", r, pallet, cfg, boxes,
                              expect_n_placed=1, label="full")
    good &= decode_all_modes("2", boxes, pallet, cfg, boxes)
    return good


def case_3_single_too_big_every_dim():
    print("\n=== Case 3: single box LARGER than pallet in EVERY dim ===")
    pallet = Pallet(length=1000, width=800, height=600)
    cfg = geometric_only_config(max_pallets=1)
    # bigger in all 3 dims AND every rotation overflows (since min dim 1200 > max pallet dim 1000)
    boxes = [Box(id="HUGE", length=1200, width=1100, height=1050)]
    good = True
    r = run_full(boxes, pallet, cfg)
    good &= check_result_sane("3", r, pallet, cfg, boxes,
                              expect_all_unpacked=True, label="full")
    good &= decode_all_modes("3", boxes, pallet, cfg, boxes,
                             expect_all_unpacked=True)
    return good


def case_4_exact_fit():
    print("\n=== Case 4: box exactly == pallet dims ===")
    pallet = Pallet(length=1000, width=800, height=600)
    cfg = geometric_only_config(max_pallets=1)
    boxes = [Box(id="EXACT", length=1000, width=800, height=600,
                 allowed_rotations=NO_ROTATION)]
    good = True
    r = run_full(boxes, pallet, cfg)
    good &= check_result_sane("4", r, pallet, cfg, boxes,
                              expect_n_placed=1, label="full")
    # should fill 100%
    if r.pallets and r.pallets[0].placements:
        util = r.total_volume_utilisation
        if abs(util - 1.0) < 1e-9:
            ok("4/full", f"utilisation == 100% ({util*100:.4f}%)")
        else:
            fail("4/full", f"exact-fit box but utilisation {util*100:.4f}% != 100%")
            good = False
    good &= decode_all_modes("4", boxes, pallet, cfg, boxes)
    return good


def case_5_two_boxes():
    print("\n=== Case 5: two boxes ===")
    pallet = Pallet(length=1000, width=800, height=600)
    cfg = geometric_only_config(max_pallets=1)
    boxes = [
        Box(id="A", length=500, width=400, height=300),
        Box(id="B", length=400, width=400, height=300),
    ]
    good = True
    r = run_full(boxes, pallet, cfg)
    good &= check_result_sane("5", r, pallet, cfg, boxes,
                              expect_n_placed=2, label="full")
    good &= decode_all_modes("5", boxes, pallet, cfg, boxes)
    return good


def case_6_identical_200():
    print("\n=== Case 6: 200 identical SKU ===")
    pallet = Pallet(length=1200, width=1000, height=1200)
    cfg = geometric_only_config(max_pallets=1)
    # 200x150x100 -> fits 6*6*12 = 432 nominally; pallet holds many, some leftover.
    boxes = [Box(id=f"S{i:03d}", length=200, width=150, height=100)
             for i in range(200)]
    good = True
    r = run_full(boxes, pallet, cfg, t=3.0)
    good &= check_result_sane("6", r, pallet, cfg, boxes, label="full")
    n_placed = sum(len(st.placements) for st in r.pallets)
    ok("6/full", f"placed {n_placed}/200 identical units (cap-bounded)")
    # decode: just modes (200 boxes x 6 modes x 2 seeds is fine)
    good &= decode_all_modes("6", boxes, pallet, cfg, boxes)
    return good


# ---------------------------------------------------------------------------
# Case 6 root-cause: minimal deterministic floating-box reproduction.
# DEFECT: on the pure-geometric path (has_constraints=False), the decoder does
# NOT enforce support_ratio (support_ratio_value=0.0 in the driver), so it
# produces placements where a box rests on only partial support. The
# independent validator — which uses the SAME geometric_only_config whose
# docstring promises "no overhang of one box over another" (support_ratio=1.0)
# — correctly flags these as floating. The engine thus violates its own
# config contract on weightless workloads, including the headline BR set.
# ---------------------------------------------------------------------------
def case_6b_floating_repro_minimal():
    print("\n=== Case 6b: MINIMAL deterministic floating-box repro (n=5) ===")
    pallet = Pallet(length=400, width=300, height=400)
    cfg = geometric_only_config(max_pallets=1)
    boxes = [Box(id=f"B{i}", length=200, width=150, height=100) for i in range(5)]
    n = 5
    n_rots_arr, dims_all, sku = precompute_box_dims_and_sku(boxes)
    weights, mlot, rfs, pmw, has_c = precompute_constraint_arrays(boxes, pallet)
    print(f"    has_constraints={has_c} -> support enforced in decode? "
          f"{has_c}  (config.support_ratio={cfg.support_ratio})")
    chrom = np.random.default_rng(0).random(2 * n + 1)
    r = decode_chromosome(chrom, boxes, pallet, cfg, n_rots_arr, dims_all, 0,
                          max_pallets=1, sku_id_per_box=sku,
                          weights=weights, mlot=mlot, rfs=rfs,
                          pallet_max_weight=pmw, has_constraints=has_c,
                          support_ratio=0.0)
    errs = validate(r, pallet, cfg)
    # This is the KNOWN DEFECT; we assert it reproduces so the finding is
    # captured, but it does NOT contribute to the pass/fail of case 6 above
    # (which already records the FAIL).
    if errs:
        print(f"    [DEFECT-REPRO] n=5 seed=0 mode=0 -> {len(errs)} validator "
              f"error(s): {errs[0]}")
        for st in r.pallets:
            for p in st.placements:
                print(f"        {p.box.id} {p.rotation.name} "
                      f"pos=({p.x},{p.y},{p.z}) dims=({p.dx},{p.dy},{p.dz})")
        # report as informational repro, not a separate failure
        return True
    print("    [unexpected] minimal repro did NOT reproduce (env changed?)")
    return False


def case_7_only_one_rotation():
    print("\n=== Case 7: box that fits ONLY in one rotation ===")
    # Pallet is a tall thin slot. Box is long+thin: it only fits if its long
    # axis is vertical. dims (900,300,300); pallet (350,350,1000).
    #   LWH dx=900 > 350 -> overflow
    #   only HWL/WHL (L vertical, dz=900<=1000, dx/dy=300<=350) fit.
    pallet = Pallet(length=350, width=350, height=1000)
    cfg = geometric_only_config(max_pallets=1)
    boxes = [Box(id="ROD", length=900, width=300, height=300,
                 allowed_rotations=list(ALL_ROTATIONS))]
    good = True
    r = run_full(boxes, pallet, cfg)
    good &= check_result_sane("7", r, pallet, cfg, boxes,
                              expect_n_placed=1, label="full")
    if r.pallets and r.pallets[0].placements:
        rot = r.pallets[0].placements[0].rotation
        # vertical extent must be the 900 axis
        dz = r.pallets[0].placements[0].dz
        if abs(dz - 900) < 1e-6:
            ok("7/full", f"chose a fitting rotation ({rot.name}, dz={dz})")
        else:
            fail("7/full", f"placed with dz={dz}, expected 900 (only-fit axis)")
            good = False
    good &= decode_all_modes("7", boxes, pallet, cfg, boxes)

    # 7b: restrict allowed_rotations to ONLY the non-fitting ones -> must be unpacked
    print("--- Case 7b: only-disallowed rotations (must stay unpacked) ---")
    boxes_b = [Box(id="ROD2", length=900, width=300, height=300,
                   allowed_rotations=[Rotation.LWH, Rotation.WLH])]  # H vertical=300, dx=900>350
    r2 = run_full(boxes_b, pallet, cfg)
    good &= check_result_sane("7b", r2, pallet, cfg, boxes_b,
                              expect_all_unpacked=True, label="full")
    good &= decode_all_modes("7b", boxes_b, pallet, cfg, boxes_b,
                             expect_all_unpacked=True)
    return good


def case_8_pallet_smaller_than_every_box():
    print("\n=== Case 8: pallet smaller than every box ===")
    pallet = Pallet(length=100, width=100, height=100)
    cfg = geometric_only_config(max_pallets=1)
    boxes = [
        Box(id="X0", length=200, width=200, height=200),
        Box(id="X1", length=150, width=300, height=120),
        Box(id="X2", length=101, width=101, height=101),
    ]
    good = True
    r = run_full(boxes, pallet, cfg)
    good &= check_result_sane("8", r, pallet, cfg, boxes,
                              expect_all_unpacked=True, label="full")
    good &= decode_all_modes("8", boxes, pallet, cfg, boxes,
                             expect_all_unpacked=True)
    return good


# ===========================================================================
# Bonus adversarial: degenerate dims (zero / 1mm pallet, multi-pallet overflow)
# ===========================================================================
def case_9_degenerate_dims():
    print("\n=== Case 9 (bonus): degenerate / pathological dims ===")
    cfg = geometric_only_config(max_pallets=1)
    good = True

    # 9a: 1mm pallet — nothing fits.
    pallet = Pallet(length=1, width=1, height=1)
    boxes = [Box(id="N0", length=10, width=10, height=10)]
    try:
        r = run_full(boxes, pallet, cfg)
        good &= check_result_sane("9a", r, pallet, cfg, boxes,
                                  expect_all_unpacked=True, label="full(1mm pallet)")
    except Exception as e:
        good = False
        fail("9a/full", f"raised {type(e).__name__}: {e}")

    # 9b: zero-height pallet (cap == 0). Should not divide-by-zero; nothing fits.
    pallet0 = Pallet(length=1000, width=800, height=0)
    try:
        r = run_full(boxes, pallet0, cfg)
        good &= check_result_sane("9b", r, pallet0, cfg, boxes,
                                  expect_all_unpacked=True, label="full(0-height pallet)")
    except Exception as e:
        good = False
        fail("9b/full", f"raised {type(e).__name__}: {e}")
    try:
        r = run_decode(boxes, pallet0, cfg)
        good &= check_result_sane("9b", r, pallet0, cfg, boxes,
                                  expect_all_unpacked=True, label="decode(0-height pallet)")
    except Exception as e:
        good = False
        fail("9b/decode", f"raised {type(e).__name__}: {e}")

    # 9c: multi-pallet overflow — 5 exact-fit boxes, max_pallets=3.
    # Only 3 can be placed; 2 must be unpacked (NOT dropped, NOT 4+ pallets).
    pallet_e = Pallet(length=500, width=500, height=500)
    cfg_mp = geometric_only_config(max_pallets=3)
    boxes_e = [Box(id=f"E{i}", length=500, width=500, height=500,
                   allowed_rotations=NO_ROTATION) for i in range(5)]
    try:
        r = run_full(boxes_e, pallet_e, cfg_mp, max_pallets=3)
        good &= check_result_sane("9c", r, pallet_e, cfg_mp, boxes_e, label="full(mp=3)")
        n_placed = sum(len(st.placements) for st in r.pallets)
        if len(r.pallets) > 3:
            good = False
            fail("9c/full", f"opened {len(r.pallets)} pallets, cap was 3")
        if n_placed > 3:
            good = False
            fail("9c/full", f"placed {n_placed} exact-fit boxes into 3 single-box pallets")
        else:
            ok("9c/full", f"placed {n_placed} into {len(r.pallets)} pallet(s), "
                          f"{len(r.unpacked)} unpacked (cap respected)")
    except Exception as e:
        good = False
        fail("9c/full", f"raised {type(e).__name__}: {e}")
    return good


def main():
    print("=== EDGE-GEOM adversarial probe ===")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','default')}")
    warmup_jit()
    # confirm Cython backend live
    from pallet_packer._brkga_core import dispatch
    print(f"backend decode_njit_mode={dispatch.decode_njit_mode.__module__} "
          f"batch_available={dispatch._BATCH_AVAILABLE}")

    results = {}
    for fn in (case_1_empty, case_2_single_fits,
               case_3_single_too_big_every_dim, case_4_exact_fit,
               case_5_two_boxes, case_6_identical_200,
               case_6b_floating_repro_minimal,
               case_7_only_one_rotation,
               case_8_pallet_smaller_than_every_box,
               case_9_degenerate_dims):
        try:
            results[fn.__name__] = fn()
        except Exception as e:
            results[fn.__name__] = False
            fail(fn.__name__, f"UNCAUGHT {type(e).__name__}: {e}\n"
                              f"{traceback.format_exc()}")

    print("\n" + "=" * 60)
    print("=== SUMMARY ===")
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    n_fail = sum(1 for v in results.values() if not v)
    print(f"\nTotal sub-failures recorded: {len(FAILS)}")
    if FAILS:
        print("Distinct failing checks:")
        for case, msg in FAILS[:30]:
            print(f"  - {case}: {msg.splitlines()[0]}")
    overall = "PASS" if n_fail == 0 else "FAIL"
    print(f"\n=== OVERALL: {overall} ({len(results)-n_fail}/{len(results)} cases clean) ===")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
