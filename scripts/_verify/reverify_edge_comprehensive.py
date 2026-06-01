"""reverify_edge_comprehensive.py — NEW edge interactions from THIS session's fixes.

Re-verification (branch block-edge-decoder-constraints). Existing verify_edge_geom.py
and verify_edge_degenerate.py cover the geometric/degenerate baseline. THIS probe drills
the edge interactions INTRODUCED by the session fixes (group co-location, D2-from-config,
input gate, max_pallets propagation):

  (a) empty box list, max_pallets>1            -> empty result, no crash, no group dispatch trap
  (b) single box (grouped & ungrouped), mp>1   -> _pack_with_groups handles n=1, placed=1
  (c) group LARGER than one pallet, mp=3        -> group NOT split; overflow unpacked
  (d) ALL boxes one group, mp=5                 -> all on ONE pallet or overflow; no split
  (e) weightless single box + DEFAULT config    -> D2 active (sr=0.8, centroid) -> valid, placed
  (f) max_pallets=1 with groups                 -> normal path (no _pack_with_groups), group trivially co-located
  (g) input gate: degenerate REJECTED, valid edges PASS

Invariants per result: validate()==[], conservation (placed+unpacked==n by identity),
no box placed twice, in-bounds, no GROUP spans >1 pallet.

Run inside Docker (OMP_NUM_THREADS=1):
    docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
        -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/reverify_edge_comprehensive.py
"""
from __future__ import annotations

import math
import os
import sys
import traceback
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pallet_packer import (
    Box, Pallet, PackerConfig, ALL_ROTATIONS, THIS_SIDE_UP, NO_ROTATION, validate,
    validate_packing_input, check_packing_input, PackingInputError,
)
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit

FAILS = []
PASSES = []


def fail(case, msg):
    FAILS.append((case, msg))
    print(f"  [FAIL] {case}: {msg}", flush=True)


def ok(case, msg):
    PASSES.append(case)
    print(f"  [pass] {case}: {msg}", flush=True)


def group_spans(result):
    """Return {group: [pallet_indices]} for any group spanning >1 pallet."""
    gp = defaultdict(set)
    for pi, st in enumerate(result.pallets):
        for p in st.placements:
            g = getattr(p.box, "group", None)
            if g is not None:
                gp[g].add(pi)
    return {g: sorted(s) for g, s in gp.items() if len(s) > 1}


def check_sane(case, result, pallet, config, all_boxes, expect_placed=None,
               expect_all_unpacked=False, expect_no_split=True):
    good = True
    try:
        errs = validate(result, pallet, config)
    except Exception as e:
        fail(case, f"validate() raised {type(e).__name__}: {e}")
        return False
    if errs:
        good = False
        fail(case, f"validator returned {len(errs)} error(s); first: {errs[0]}")

    placed_ids = [id(p.box) for st in result.pallets for p in st.placements]
    unpacked_ids = [id(b) for b in result.unpacked]
    n_placed = len(placed_ids)
    n_unpacked = len(unpacked_ids)
    if n_placed + n_unpacked != len(all_boxes):
        good = False
        fail(case, f"conservation: placed {n_placed} + unpacked {n_unpacked} "
                   f"!= total {len(all_boxes)}")
    if len(set(placed_ids)) != len(placed_ids):
        good = False
        fail(case, f"box placed more than once ({len(placed_ids)} vs "
                   f"{len(set(placed_ids))} unique)")
    overlap = set(placed_ids) & set(unpacked_ids)
    if overlap:
        good = False
        fail(case, f"{len(overlap)} box(es) both placed and unpacked")

    for st in result.pallets:
        for p in st.placements:
            if p.x < -1e-6 or p.y < -1e-6 or p.z < -1e-6:
                good = False
                fail(case, f"negative coord {p.box.id} ({p.x},{p.y},{p.z})")
            if (p.x2 > pallet.length + 1e-6 or p.y2 > pallet.width + 1e-6
                    or p.z2 > pallet.height + 1e-6):
                good = False
                fail(case, f"OOB {p.box.id} ({p.x2},{p.y2},{p.z2}) vs pallet "
                           f"({pallet.length},{pallet.width},{pallet.height})")

    if expect_no_split:
        spans = group_spans(result)
        if spans:
            good = False
            fail(case, f"GROUP SPLIT across pallets: {spans}")

    if expect_all_unpacked and n_placed != 0:
        good = False
        fail(case, f"expected all unpacked, {n_placed} placed")
    if expect_placed is not None and n_placed != expect_placed:
        good = False
        fail(case, f"expected {expect_placed} placed, got {n_placed}")

    if good:
        ok(case, f"placed={n_placed} unpacked={n_unpacked} "
                 f"pallets={len(result.pallets)} validator=clean")
    return good


def solve(boxes, pallet, config, max_pallets=1, t=4.0, seed=42):
    return brkga_pack_v35(
        boxes, pallet, config, time_limit_s=t, max_pallets=max_pallets,
        population_size=120, n_populations=2, patience=80,
        local_search_budget_s=0.5, seed=seed, verbose=False, n_modes=6)


# (a) empty box list with max_pallets>1 (group dispatch must NOT trap on empty)
def case_a_empty_mp():
    print("\n=== (a) empty box list, max_pallets=3 ===")
    pallet = Pallet(length=1000, width=800, height=600)
    cfg = PackerConfig(support_ratio=0.0, require_centroid_supported=False)
    try:
        r = solve([], pallet, cfg, max_pallets=3)
        if r.pallets or r.unpacked:
            fail("a", f"non-empty result pallets={len(r.pallets)} unpacked={len(r.unpacked)}")
        else:
            ok("a", "empty PackResult, no crash, no group-dispatch trap")
    except Exception as e:
        fail("a", f"raised {type(e).__name__}: {e}\n{traceback.format_exc()}")


# (b) single box grouped & ungrouped with max_pallets>1
def case_b_single_box_mp():
    print("\n=== (b) single box (grouped & ungrouped), max_pallets=3 ===")
    pallet = Pallet(length=1000, width=800, height=600, max_weight=500)
    cfg = PackerConfig(support_ratio=0.0, require_centroid_supported=False)
    # ungrouped single box, mp>1 -> normal path (no group dispatch)
    b_ung = [Box(id="U0", length=300, width=200, height=150, weight=5.0)]
    try:
        r = solve(b_ung, pallet, cfg, max_pallets=3)
        check_sane("b/ungrouped", r, pallet, cfg, b_ung, expect_placed=1)
    except Exception as e:
        fail("b/ungrouped", f"raised {type(e).__name__}: {e}\n{traceback.format_exc()}")
    # grouped single box, mp>1 -> _pack_with_groups must handle n=1
    b_grp = [Box(id="G0", length=300, width=200, height=150, weight=5.0, group="solo")]
    try:
        r = solve(b_grp, pallet, cfg, max_pallets=3)
        check_sane("b/grouped(n=1)", r, pallet, cfg, b_grp, expect_placed=1)
    except Exception as e:
        fail("b/grouped(n=1)", f"raised {type(e).__name__}: {e}\n{traceback.format_exc()}")


# (c) a group whose total volume EXCEEDS one pallet, max_pallets=3 -> no split
def case_c_group_bigger_than_pallet():
    print("\n=== (c) group volume > one pallet, max_pallets=3 (NO split) ===")
    # pallet vol = 1000*800*600 = 4.8e8. One box = 500*400*300 = 6e7.
    # 12 boxes = 7.2e8 > pallet vol. All in ONE group "BIG".
    pallet = Pallet(length=1000, width=800, height=600, max_weight=10000)
    cfg = PackerConfig(support_ratio=0.0, require_centroid_supported=False)
    boxes = [Box(id=f"BIG-{i:02d}", length=500, width=400, height=300, weight=5.0,
                 group="BIG", allowed_rotations=ALL_ROTATIONS) for i in range(12)]
    pallet_vol = pallet.length * pallet.width * pallet.height
    grp_vol = sum(b.volume for b in boxes)
    print(f"    group vol {grp_vol:.0f} vs pallet vol {pallet_vol:.0f} "
          f"(ratio {grp_vol/pallet_vol:.2f}x)")
    try:
        r = solve(boxes, pallet, cfg, max_pallets=3, t=6.0)
        # must NOT split: all placed boxes on the SAME single pallet; overflow unpacked
        good = check_sane("c", r, pallet, cfg, boxes)
        # extra: the group that overflows a pallet should occupy exactly ONE pallet
        n_pallets_with_placements = sum(1 for st in r.pallets if st.placements)
        if n_pallets_with_placements > 1:
            fail("c", f"group spread over {n_pallets_with_placements} pallets "
                      f"(must be 1 — overflow goes to unpacked, not another pallet)")
        elif good:
            n_placed = sum(len(st.placements) for st in r.pallets)
            ok("c/single-pallet", f"oversize group on 1 pallet, "
                                  f"{n_placed} placed, {len(r.unpacked)} overflow->unpacked")
    except Exception as e:
        fail("c", f"raised {type(e).__name__}: {e}\n{traceback.format_exc()}")


# (d) ALL boxes one group, max_pallets=5 -> all on one pallet or overflow, no split
def case_d_all_one_group_mp5():
    print("\n=== (d) all boxes one group, max_pallets=5 (NO split) ===")
    pallet = Pallet(length=1200, width=1000, height=1200, max_weight=5000)
    cfg = PackerConfig(support_ratio=0.0, require_centroid_supported=False)
    boxes = [Box(id=f"ONE-{i:02d}", length=300, width=250, height=200, weight=3.0,
                 group="ALL", allowed_rotations=ALL_ROTATIONS) for i in range(40)]
    try:
        r = solve(boxes, pallet, cfg, max_pallets=5, t=6.0)
        good = check_sane("d", r, pallet, cfg, boxes)
        n_pallets_with_placements = sum(1 for st in r.pallets if st.placements)
        if n_pallets_with_placements > 1:
            fail("d", f"single group spread over {n_pallets_with_placements} pallets")
        elif good:
            n_placed = sum(len(st.placements) for st in r.pallets)
            ok("d/single-pallet", f"all-one-group on 1 pallet, {n_placed} placed, "
                                  f"{len(r.unpacked)} unpacked")
    except Exception as e:
        fail("d", f"raised {type(e).__name__}: {e}\n{traceback.format_exc()}")


# (e) weightless single box + DEFAULT config (sr=0.8, centroid, D2 active) -> valid
def case_e_weightless_default_cfg():
    print("\n=== (e) weightless single box + DEFAULT config (D2 active) ===")
    pallet = Pallet(length=1000, width=800, height=600)
    cfg = PackerConfig()  # support_ratio=0.8, require_centroid_supported=True
    print(f"    config support_ratio={cfg.support_ratio} "
          f"require_centroid_supported={cfg.require_centroid_supported}")
    box = [Box(id="W0", length=300, width=200, height=150)]  # weight defaults 0
    try:
        r = solve(box, pallet, cfg, max_pallets=1)
        check_sane("e", r, pallet, cfg, box, expect_placed=1)
    except Exception as e:
        fail("e", f"raised {type(e).__name__}: {e}\n{traceback.format_exc()}")
    # also weightless single box with mp>1 default cfg (stability path + normal dispatch)
    try:
        r = solve(box, pallet, cfg, max_pallets=3)
        check_sane("e/mp3", r, pallet, cfg, box, expect_placed=1)
    except Exception as e:
        fail("e/mp3", f"raised {type(e).__name__}: {e}\n{traceback.format_exc()}")


# (f) max_pallets=1 WITH groups -> normal path, group trivially co-located
def case_f_mp1_with_groups():
    print("\n=== (f) max_pallets=1 with groups (normal path, no _pack_with_groups) ===")
    pallet = Pallet(length=1200, width=1000, height=1200, max_weight=5000)
    cfg = PackerConfig(support_ratio=0.0, require_centroid_supported=False)
    boxes = []
    for g in ("GA", "GB"):
        for k in range(10):
            boxes.append(Box(id=f"{g}-{k}", length=300, width=250, height=200,
                             weight=2.0, group=g, allowed_rotations=ALL_ROTATIONS))
    try:
        r = solve(boxes, pallet, cfg, max_pallets=1, t=5.0)
        # only one pallet possible -> trivially co-located
        good = check_sane("f", r, pallet, cfg, boxes)
        if good and len(r.pallets) <= 1:
            ok("f/one-pallet", f"groups on single pallet, {len(r.pallets)} pallet(s)")
        elif len(r.pallets) > 1:
            fail("f", f"max_pallets=1 but opened {len(r.pallets)} pallets")
    except Exception as e:
        fail("f", f"raised {type(e).__name__}: {e}\n{traceback.format_exc()}")


# (g) input gate: degenerate REJECTED, valid edges PASS
def case_g_input_gate():
    print("\n=== (g) input gate edge matrix ===")
    P = Pallet(length=1000, width=800, height=600, max_weight=500)

    def R(label, boxes, pallet, want_reject, **kw):
        try:
            probs = validate_packing_input(boxes, pallet, **kw)
        except Exception as e:
            fail(f"g/{label}", f"validate_packing_input raised {type(e).__name__}: {e}")
            return
        rejected = len(probs) > 0
        if rejected == want_reject:
            ok(f"g/{label}", ("rejected: " + str(probs[:1])) if rejected else "passed (clean)")
        else:
            fail(f"g/{label}", f"want_reject={want_reject} but rejected={rejected} "
                               f"problems={probs[:2]}")

    def B(**kw):
        d = dict(id="x", length=100, width=100, height=100, weight=1.0,
                 allowed_rotations=ALL_ROTATIONS)
        d.update(kw)
        return Box(**d)

    # --- degenerate -> MUST be rejected ---
    R("non-integer-dim", [B(length=100.49)], P, True)
    R("negative-dim", [B(length=-30)], P, True)
    R("zero-dim", [B(length=0)], P, True)
    R("all-zero-dim", [B(length=0, width=0, height=0)], P, True)
    R("NaN-dim", [B(length=float("nan"))], P, True)
    R("inf-dim", [B(length=float("inf"))], P, True)
    R("NaN-weight-cap", [B()], Pallet(length=1000, width=800, height=600,
                                      max_weight=float("nan")), True)
    R("zero-weight-cap", [B()], Pallet(length=1000, width=800, height=600,
                                       max_weight=0), True)
    R("over-500-boxes", [B(id=f"b{i}") for i in range(501)], P, True)
    R("empty-list", [], P, True)
    R("negative-weight", [B(weight=-1)], P, True)
    R("NaN-mlot", [B(max_load_on_top=float("nan"))], P, True)
    R("empty-rotations", [B(allowed_rotations=[])], P, True)
    R("duplicate-ids", [B(id="dup"), B(id="dup")], P, True)
    R("non-integer-pallet", [B()], Pallet(length=1000.5, width=800, height=600), True)

    # --- valid edge cases -> MUST pass ---
    R("single-box", [B()], P, False)
    R("exact-fit-box", [B(length=1000, width=800, height=600)], P, False)
    # oversize-unpackable is geometrically fine input -> gate must NOT reject
    R("oversize-unpackable", [B(length=2000, width=2000, height=2000)], P, False)
    R("fractional-weight", [B(weight=2.5)], P, False)
    R("integer-valued-float-dims", [B(length=100.0, width=100.0, height=100.0)], P, False)
    R("fragile-mlot-0", [B(max_load_on_top=0.0)], P, False)
    R("inf-mlot", [B(max_load_on_top=float("inf"))], P, False)
    R("inf-weight-cap", [B()], Pallet(length=1000, width=800, height=600,
                                      max_weight=float("inf")), False)
    R("500-boxes-exact", [B(id=f"b{i}") for i in range(500)], P, False)
    R("over-500-cap-disabled", [B(id=f"b{i}") for i in range(501)], P, False,
      max_boxes=None)

    # check_packing_input raises on degenerate, returns None on valid
    try:
        check_packing_input([B(length=-5)], P)
        fail("g/check-raises", "check_packing_input did NOT raise on negative dim")
    except PackingInputError as e:
        ok("g/check-raises", f"raised PackingInputError ({len(e.problems)} problem(s))")
    try:
        out = check_packing_input([B()], P)
        if out is None:
            ok("g/check-clean", "returned None on valid input")
        else:
            fail("g/check-clean", f"expected None, got {out!r}")
    except Exception as e:
        fail("g/check-clean", f"raised on valid input: {type(e).__name__}: {e}")


def main():
    print("=== reverify_edge_comprehensive ===")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','default')}", flush=True)
    warmup_jit()
    for fn in (case_a_empty_mp, case_b_single_box_mp,
               case_c_group_bigger_than_pallet, case_d_all_one_group_mp5,
               case_e_weightless_default_cfg, case_f_mp1_with_groups,
               case_g_input_gate):
        try:
            fn()
        except Exception as e:
            fail(fn.__name__, f"UNCAUGHT {type(e).__name__}: {e}\n{traceback.format_exc()}")

    print("\n" + "=" * 60)
    print(f"=== SUMMARY: {len(PASSES)} passed, {len(FAILS)} failed ===")
    if FAILS:
        for case, msg in FAILS:
            print(f"  FAIL {case}: {msg.splitlines()[0]}")
    print(f"=== OVERALL: {'PASS' if not FAILS else 'FAIL'} ===")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
