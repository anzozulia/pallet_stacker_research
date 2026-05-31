"""verify_edge_degenerate.py — adversarial probe for DEGENERATE / MALFORMED inputs.

Dimension: edge-degenerate. Public-API surface (brkga_pack_v35 + decode_chromosome)
fed garbage:
  - zero-dimension box (length=0)
  - negative-dimension box
  - NaN / inf box dimension
  - zero-height pallet
  - negative pallet dims
  - box with empty allowed_rotations
  - box with weight = NaN / inf
  - pallet max_weight = 0
  - duplicate box ids
  - non-integer (sub-mm) dims  -> int-decode vs float-validate divergence

For each case we document EXACTLY what happens. Classification:
  CRASH    : raised an exception (acceptable-but-noted for garbage input)
  HANG     : exceeded a wall-clock guard (BAD)
  CLEAN    : returned a result AND validate() is clean AND result is
             physically sane (best outcome)
  SILENT-WRONG (CRITICAL): validate() returns [] (clean) but the geometry is
             physically impossible (overlap / out-of-bounds / negative coords
             under the ORIGINAL float dims, or a box that cannot exist).

Each sub-test is wrapped in a wall-clock guard (subprocess-free: we rely on
short time_limit_s + a hard alarm) so a single hang can't kill the whole probe.

Run inside Docker:
    docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
        -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/verify_edge_degenerate.py
"""
from __future__ import annotations

import math
import os
import signal
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pallet_packer import (Box, Pallet, PackerConfig, Rotation, ALL_ROTATIONS,
                           validate)
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer._brkga_core.dispatch import decode_chromosome
from pallet_packer._brkga_core.precompute import (
    precompute_box_dims_and_sku, precompute_constraint_arrays)

EPS = 1e-6
TIME = 3.0          # short solve budget per case
HARD_GUARD_S = 25   # SIGALRM guard around any single case (>> TIME -> hang)

# ---------------------------------------------------------------------------
# Guard machinery
# ---------------------------------------------------------------------------
class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)

RESULTS = []  # (case, classification, detail)


def record(case, cls, detail):
    RESULTS.append((case, cls, detail))
    print(f"  [{cls:>13}] {case}: {detail}", flush=True)


# ---------------------------------------------------------------------------
# Physical-sanity re-check (INDEPENDENT of validate()).
# Uses the ORIGINAL float Box dims as the Placement.dims property does.
# Returns list of physical violations. Empty = physically sane.
# ---------------------------------------------------------------------------
def physical_violations(result, pallet, allow_overhang_ov=0.0):
    viol = []
    for st in result.pallets:
        pls = st.placements
        # bounds
        for p in pls:
            if (p.x < -EPS or p.y < -EPS or p.z < -EPS or
                    p.x2 > pallet.length + allow_overhang_ov + EPS or
                    p.y2 > pallet.width + allow_overhang_ov + EPS or
                    p.z2 > pallet.height + EPS):
                viol.append(
                    f"OOB {p.box.id} @({p.x:.3f},{p.y:.3f},{p.z:.3f})"
                    f"+({p.dx:.3f},{p.dy:.3f},{p.dz:.3f}) "
                    f"pallet=({pallet.length},{pallet.width},{pallet.height})")
            # degenerate / non-finite extents
            if not (math.isfinite(p.dx) and math.isfinite(p.dy) and math.isfinite(p.dz)):
                viol.append(f"NONFINITE-DIM {p.box.id} dims=({p.dx},{p.dy},{p.dz})")
            if p.dx <= 0 or p.dy <= 0 or p.dz <= 0:
                viol.append(f"NONPOS-DIM {p.box.id} dims=({p.dx},{p.dy},{p.dz})")
        # pairwise overlap under float dims
        for i, a in enumerate(pls):
            for b in pls[i + 1:]:
                if a.overlaps(b):
                    viol.append(f"OVERLAP {a.box.id}<->{b.box.id}")
    return viol


def n_placed(result):
    return sum(len(st.placements) for st in result.pallets)


# ---------------------------------------------------------------------------
# Generic runner for a brkga_pack_v35 case
# ---------------------------------------------------------------------------
def run_solve_case(case, boxes, pallet, config=None, max_pallets=1,
                   expect_overhang_ov=0.0):
    cfg = config or PackerConfig(max_pallets=max_pallets)
    signal.alarm(HARD_GUARD_S)
    try:
        r = brkga_pack_v35(
            boxes, pallet, cfg,
            time_limit_s=TIME, max_pallets=max_pallets,
            population_size=80, n_populations=2, patience=50,
            local_search_budget_s=1.0, seed=42, verbose=False, n_modes=6,
        )
    except _Timeout:
        record(case, "HANG", f">{HARD_GUARD_S}s with time_limit_s={TIME}")
        return None
    except Exception as e:  # noqa: BLE001
        record(case, "CRASH", f"{type(e).__name__}: {e}")
        return None
    finally:
        signal.alarm(0)

    # got a result — validate + physical sanity
    try:
        errs = validate(r, pallet, cfg)
    except Exception as e:  # noqa: BLE001
        record(case, "CRASH", f"validate() raised {type(e).__name__}: {e}")
        return r
    phys = physical_violations(r, pallet, expect_overhang_ov)
    placed = n_placed(r)
    summary = (f"placed={placed} unpacked={len(r.unpacked)} "
               f"pallets={len(r.pallets)} validator_errs={len(errs)} "
               f"phys_viol={len(phys)}")
    if not errs and phys:
        record(case, "SILENT-WRONG", summary + " | " + "; ".join(phys[:3]))
    elif errs and not phys:
        record(case, "CLEAN", summary + " (validator flagged; physically sane)")
    elif errs and phys:
        record(case, "CLEAN", summary + " (both flagged)")
    else:
        record(case, "CLEAN", summary)
    return r


# ---------------------------------------------------------------------------
# Standard "good" filler box so degenerate box sits among real ones.
# ---------------------------------------------------------------------------
def good_box(i, L=30, W=30, H=30):
    return Box(id=f"g{i}", length=L, width=W, height=H, weight=1.0)


def std_pallet():
    return Pallet(length=120, width=100, height=100)


# ===========================================================================
# CASES
# ===========================================================================
def case_zero_dim():
    boxes = [good_box(0), good_box(1),
             Box(id="zero", length=0.0, width=30, height=30, weight=1.0)]
    run_solve_case("zero-dim-box (length=0)", boxes, std_pallet())


def case_all_zero_dim():
    boxes = [Box(id="z", length=0.0, width=0.0, height=0.0, weight=1.0)]
    run_solve_case("all-zero-dim-box", boxes, std_pallet())


def case_negative_dim():
    boxes = [good_box(0),
             Box(id="neg", length=-30.0, width=30, height=30, weight=1.0)]
    run_solve_case("negative-dim-box (length=-30)", boxes, std_pallet())


def case_nan_dim():
    boxes = [good_box(0),
             Box(id="nan", length=float("nan"), width=30, height=30, weight=1.0)]
    run_solve_case("NaN-dim-box", boxes, std_pallet())


def case_inf_dim():
    boxes = [good_box(0),
             Box(id="inf", length=float("inf"), width=30, height=30, weight=1.0)]
    run_solve_case("inf-dim-box", boxes, std_pallet())


def case_zero_height_pallet():
    boxes = [good_box(0), good_box(1)]
    run_solve_case("zero-height-pallet", boxes,
                   Pallet(length=120, width=100, height=0.0))


def case_zero_all_pallet():
    boxes = [good_box(0)]
    run_solve_case("all-zero-pallet", boxes, Pallet(length=0, width=0, height=0))


def case_negative_pallet():
    boxes = [good_box(0), good_box(1)]
    run_solve_case("negative-pallet-dims", boxes,
                   Pallet(length=-120, width=100, height=100))


def case_empty_rotations():
    boxes = [good_box(0),
             Box(id="norot", length=30, width=30, height=30, weight=1.0,
                 allowed_rotations=[])]
    run_solve_case("empty-allowed_rotations", boxes, std_pallet())


def case_nan_weight():
    boxes = [good_box(0),
             Box(id="wnan", length=30, width=30, height=30,
                 weight=float("nan"))]
    # add a weight constraint so the weight path is exercised
    run_solve_case("NaN-weight-box (pallet max_weight set)", boxes,
                   Pallet(length=120, width=100, height=100, max_weight=1000.0))


def case_inf_weight():
    boxes = [good_box(0),
             Box(id="winf", length=30, width=30, height=30,
                 weight=float("inf"))]
    run_solve_case("inf-weight-box (pallet max_weight set)", boxes,
                   Pallet(length=120, width=100, height=100, max_weight=1000.0))


def case_pallet_maxweight_zero():
    boxes = [good_box(0, 30, 30, 30), good_box(1, 30, 30, 30)]
    # boxes weigh 1kg each; pallet allows 0 -> nothing should pack (or all unpacked)
    run_solve_case("pallet-max_weight=0", boxes,
                   Pallet(length=120, width=100, height=100, max_weight=0.0))


def case_duplicate_ids():
    boxes = [Box(id="dup", length=30, width=30, height=30, weight=1.0)
             for _ in range(6)]
    run_solve_case("duplicate-box-ids (6x 'dup')", boxes, std_pallet())


def case_noninteger_dims():
    """Sub-mm dims -> int(round()) decode vs float-dim validate divergence.

    Pick dims whose rounding shifts the footprint so boxes that decode as
    non-overlapping on the int grid actually overlap (or go OOB) under the
    true float dims. 0.4 rounds DOWN -> int box smaller than real -> real
    boxes may overlap. We make many same-SKU boxes tile tightly so any
    rounding gap is amplified.
    """
    # length 10.4 rounds to 10. If 12 are placed edge-to-edge on int grid
    # at x=0,10,20,... the real boxes (10.4 wide) overlap by 0.4 each.
    boxes = [Box(id=f"f{i}", length=10.4, width=10.4, height=10.4, weight=0.1)
             for i in range(40)]
    run_solve_case("non-integer-dims (10.4 rounds to 10)", boxes,
                   Pallet(length=120, width=100, height=100))


def case_noninteger_up():
    """0.6 rounds UP -> int box BIGGER than real. Int-grid packing may push
    boxes past the pallet edge on the int grid but they're flagged OOB; the
    real risk is the reverse: int says it fits, real coords drift. We test
    that validator catches whatever int-decode produces."""
    boxes = [Box(id=f"u{i}", length=15.6, width=15.6, height=15.6, weight=0.1)
             for i in range(30)]
    run_solve_case("non-integer-dims-up (15.6 rounds to 16)", boxes,
                   Pallet(length=120, width=100, height=100))


def case_tiny_subunit():
    """Dims < 0.5 round to 0 -> zero-volume int box. What does decode do?"""
    boxes = [good_box(0),
             Box(id="tiny", length=0.4, width=0.4, height=0.4, weight=0.1)]
    run_solve_case("sub-0.5 dims round to 0 (0.4mm box)", boxes, std_pallet())


# ---------------------------------------------------------------------------
# decode_chromosome direct (per-chromosome path) on a couple of nasty cases —
# bypasses the driver's n==0 guard and exercises the decoder directly.
# ---------------------------------------------------------------------------
def case_decode_direct_nan():
    boxes = [good_box(0),
             Box(id="nan", length=float("nan"), width=30, height=30, weight=1.0)]
    pallet = std_pallet()
    cfg = PackerConfig(max_pallets=1)
    signal.alarm(HARD_GUARD_S)
    try:
        # precompute itself does int(round(NaN)) -> may raise here; that is the
        # SAME crash the solve path hits, so catch it as CRASH not PROBE-ERROR.
        n_rots_arr, dims_all, sku = precompute_box_dims_and_sku(boxes)
        chrom = np.array([0.3, 0.7], dtype=np.float64)
        r = decode_chromosome(chrom, boxes, pallet, cfg, n_rots_arr, dims_all,
                              mode=0, max_pallets=1)
    except _Timeout:
        record("decode_chromosome direct NaN-dim", "HANG", f">{HARD_GUARD_S}s")
        return
    except Exception as e:  # noqa: BLE001
        record("decode_chromosome direct NaN-dim", "CRASH",
               f"{type(e).__name__}: {e}")
        return
    finally:
        signal.alarm(0)
    errs = validate(r, pallet, cfg)
    phys = physical_violations(r, pallet)
    s = f"placed={n_placed(r)} validator_errs={len(errs)} phys_viol={len(phys)}"
    if not errs and phys:
        record("decode_chromosome direct NaN-dim", "SILENT-WRONG",
               s + " | " + "; ".join(phys[:2]))
    else:
        record("decode_chromosome direct NaN-dim", "CLEAN", s)


def case_precompute_constraint_nan():
    """Does the constraint precompute choke on NaN weight / NaN max_weight?"""
    boxes = [Box(id="w", length=30, width=30, height=30, weight=float("nan"))]
    pallet = Pallet(length=120, width=100, height=100, max_weight=float("nan"))
    try:
        out = precompute_constraint_arrays(boxes, pallet)
        has_cstr = out[4]
        pmw = out[3]
        # NaN max_weight: math.isinf(nan) is False -> pmw=float(nan), has_cstr=True
        detail = f"has_constraints={has_cstr} pallet_max_weight={pmw} weights={out[0]}"
        # Flag the danger: NaN pmw makes weight-budget comparisons (w > pmw) all
        # False -> weight limit silently disabled.
        if isinstance(pmw, float) and math.isnan(pmw):
            record("precompute NaN max_weight", "SILENT-WRONG",
                   detail + " | NaN pmw -> all (w>pmw) comparisons False -> "
                   "weight limit silently DISABLED")
        else:
            record("precompute NaN max_weight", "CLEAN", detail)
    except Exception as e:  # noqa: BLE001
        record("precompute NaN max_weight", "CRASH", f"{type(e).__name__}: {e}")


# ===========================================================================
def main():
    print("=== warmup ===", flush=True)
    try:
        warmup_jit()
    except Exception as e:  # noqa: BLE001
        print(f"warmup failed: {e}")

    print("\n=== Degenerate BOX dims ===", flush=True)
    for fn in (case_zero_dim, case_all_zero_dim, case_negative_dim,
               case_nan_dim, case_inf_dim, case_tiny_subunit):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            record(fn.__name__, "PROBE-ERROR", f"{type(e).__name__}: {e}\n"
                   + traceback.format_exc())

    print("\n=== Degenerate PALLET ===", flush=True)
    for fn in (case_zero_height_pallet, case_zero_all_pallet,
               case_negative_pallet):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            record(fn.__name__, "PROBE-ERROR", f"{type(e).__name__}: {e}")

    print("\n=== Degenerate ROTATIONS / WEIGHT / IDS ===", flush=True)
    for fn in (case_empty_rotations, case_nan_weight, case_inf_weight,
               case_pallet_maxweight_zero, case_duplicate_ids):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            record(fn.__name__, "PROBE-ERROR", f"{type(e).__name__}: {e}")

    print("\n=== Non-integer dims (int-decode vs float-validate) ===", flush=True)
    for fn in (case_noninteger_dims, case_noninteger_up):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            record(fn.__name__, "PROBE-ERROR", f"{type(e).__name__}: {e}")

    print("\n=== Direct decode_chromosome + precompute ===", flush=True)
    for fn in (case_decode_direct_nan, case_precompute_constraint_nan):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            record(fn.__name__, "PROBE-ERROR", f"{type(e).__name__}: {e}")

    # -------- summary --------
    print("\n=== SUMMARY ===", flush=True)
    counts = {}
    for _, cls, _ in RESULTS:
        counts[cls] = counts.get(cls, 0) + 1
    for cls in ("SILENT-WRONG", "HANG", "PROBE-ERROR", "CRASH", "CLEAN"):
        if cls in counts:
            print(f"  {cls:>13}: {counts[cls]}")
    sw = [r for r in RESULTS if r[1] == "SILENT-WRONG"]
    hg = [r for r in RESULTS if r[1] == "HANG"]
    print()
    if sw:
        print(f"CRITICAL: {len(sw)} SILENT-WRONG case(s):")
        for case, _, detail in sw:
            print(f"    - {case}: {detail}")
    if hg:
        print(f"HANG: {len(hg)} case(s) exceeded guard:")
        for case, _, detail in hg:
            print(f"    - {case}: {detail}")
    overall = "FAIL" if (sw or hg) else "PASS"
    print(f"\n=== OVERALL: {overall} "
          f"(SILENT-WRONG={len(sw)} HANG={len(hg)}) ===")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
