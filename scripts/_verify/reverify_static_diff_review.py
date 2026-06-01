"""reverify_static_diff_review.py — adversarial re-verification of the
session's code changes (branch block-edge-decoder-constraints, vs cead45d).

Each sub-test encodes a concrete static-review hypothesis about ONE change and
tries to break it. status PASS only when the hypothesis is refuted (no defect).

  S1  Per-box block support check — cy==nb bit-equivalence on the EXACT class
      of instances the fix targets: a wide bottom layer whose corner box sits
      on a partial supporter (aggregate average >= support_ratio but a corner
      box < support_ratio). Both backends must agree AND the shrink-to-(1,1)
      fallback must match.
  S2  SKU key fix — (a) fragile (mlot=0) + sturdy (mlot=inf) SAME-SIZE boxes
      must get DIFFERENT sku ids (no crush merge); (b) a purely geometric /
      weightless workload must keep a BYTE-IDENTICAL sku partition vs the old
      key (length,width,height,weight,rot_key).
  S3  _pack_with_groups — conservation (placed+unpacked == input), group
      co-location (every group on exactly one pallet), max_pallets cap honored,
      pallet ids unique, and the max_pallets==1 base case does NOT recurse.
  S4  use_cstr_path — a STABILITY-ONLY workload (support_ratio>0, all weights
      0, mlot=inf -> has_constraints False) must (a) route the cstr path so the
      validator sees no floating boxes, and (b) NOT auto-enable v2_seed.
  S5  input_validation — _is_cap NaN/-inf rejection, integer-float acceptance,
      max_load_on_top operator-precedence (line 157), negative/zero dims,
      box-count cap, duplicate ids, +inf caps.

Run:
  docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
    -v "$(pwd)":/app -w /app pallet-packer:dev \
    python scripts/_verify/reverify_static_diff_review.py
"""
from __future__ import annotations

import os
import sys
import math
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

RESULTS = []  # (name, passed, detail)


def record(name, passed, detail=""):
    RESULTS.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def _dims6(bx, by, bz):
    return [(bx, by, bz), (bx, bz, by), (by, bx, bz),
            (by, bz, bx), (bz, bx, by), (bz, by, bx)]


# ---------------------------------------------------------------------------
# S1: per-box block support check cy==nb on partial-support corner instances.
# ---------------------------------------------------------------------------

def s1_per_box_block_equiv():
    from pallet_packer._brkga_core import jit_decoders_cstr_cy as cstr_cy
    from pallet_packer._brkga_core import jit_decoders_cstr as cstr_nb

    # Homogeneous SKU so the block decoder builds wide bottom layers (k*l>1).
    # Stack one big SKU layer, then a second SKU layer that the per-box support
    # test must scrutinize box-by-box. Sweep support_ratio across the boundary
    # where aggregate-vs-per-box verdicts diverge.
    L, W, H, mp = 1200, 1000, 1500, 2
    n_boxes = 80
    n_rots = np.full(n_boxes, 6, dtype=np.int64)
    dims = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    bx, by, bz = 300, 250, 200
    for i in range(n_boxes):
        for r, t in enumerate(_dims6(bx, by, bz)):
            dims[i, r] = t
    # two SKUs, alternating, identical size -> stacks that exercise per-box.
    sku = (np.arange(n_boxes) % 2).astype(np.int64)
    n_skus = 2  # MUST match max(sku)+1 or the batch skur_all array is OOB.
    rng = np.random.default_rng(101)
    chroms = rng.random((8, n_boxes)).astype(np.float64)
    weights = np.full(n_boxes, 2.0, dtype=np.float64)
    mlot = np.full(n_boxes, 100.0, dtype=np.float64)

    diverged = 0
    total = 0
    for sr in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1.0):
        for rfs_on in (0, 1):
            rfs = np.full(n_boxes, rfs_on, dtype=np.int64)
            args = (weights, mlot, rfs, 1e18, sr, 0,
                    -1e18, 1e18, -1e18, 1e18, 0.0, 0)
            po_cy = np.zeros((8, n_boxes, 6), dtype=np.int64)
            nb_cy = np.zeros(8, dtype=np.int64)
            cstr_cy.decode_batch_blocks_njit_mode_cstr(
                chroms, n_rots, dims, sku, L, W, H, mp, *args, po_cy, nb_cy, n_skus)
            po_nb = np.zeros((8, n_boxes, 6), dtype=np.int64)
            for i in range(8):
                order = np.argsort(chroms[i]).astype(np.int64)
                cstr_nb.decode_blocks_njit_mode_cstr(
                    order, n_rots, dims, sku, L, W, H, mp, po_nb[i], n_skus, *args)
            total += 1
            if not np.array_equal(po_cy, po_nb):
                diverged += 1
                d = int((po_cy != po_nb).any(axis=(1, 2)).sum())
                print(f"  [S1] DIVERGENCE sr={sr} rfs={rfs_on}: {d}/8 chroms differ")
    record("S1 per-box block cy==nb (18 sr/rfs combos)", diverged == 0,
           f"{diverged}/{total} combos diverged")

    # Also exercise centroid-required path (require_centroid=1) which the
    # per-box loop also funnels through _check_load_on_top.
    diverged2 = 0
    for sr in (0.5, 0.8, 1.0):
        args = (weights, mlot, np.zeros(n_boxes, np.int64), 1e18, sr, 1,
                -1e18, 1e18, -1e18, 1e18, 0.0, 0)
        po_cy = np.zeros((8, n_boxes, 6), dtype=np.int64)
        nb_cy = np.zeros(8, dtype=np.int64)
        cstr_cy.decode_batch_blocks_njit_mode_cstr(
            chroms, n_rots, dims, sku, L, W, H, mp, *args, po_cy, nb_cy, n_skus)
        po_nb = np.zeros((8, n_boxes, 6), dtype=np.int64)
        for i in range(8):
            order = np.argsort(chroms[i]).astype(np.int64)
            cstr_nb.decode_blocks_njit_mode_cstr(
                order, n_rots, dims, sku, L, W, H, mp, po_nb[i], n_skus, *args)
        if not np.array_equal(po_cy, po_nb):
            diverged2 += 1
    record("S1b per-box block + centroid cy==nb", diverged2 == 0,
           f"{diverged2}/3 centroid sr combos diverged")


# ---------------------------------------------------------------------------
# S2: SKU key fix — no fragile/sturdy merge + geometric partition unchanged.
# ---------------------------------------------------------------------------

def s2_sku_key():
    from pallet_packer import Box, ALL_ROTATIONS
    from pallet_packer._brkga_core.precompute import precompute_box_dims_and_sku

    # (a) fragile + sturdy, identical geometry+weight -> must be DIFFERENT skus.
    fragile = Box(id="frag", length=100, width=100, height=100, weight=5.0,
                  max_load_on_top=0.0, allowed_rotations=ALL_ROTATIONS)
    sturdy = Box(id="sturdy", length=100, width=100, height=100, weight=5.0,
                 max_load_on_top=float('inf'), allowed_rotations=ALL_ROTATIONS)
    rfs_box = Box(id="rfs", length=100, width=100, height=100, weight=5.0,
                  max_load_on_top=float('inf'), requires_full_support=True,
                  allowed_rotations=ALL_ROTATIONS)
    out = precompute_box_dims_and_sku([fragile, sturdy, rfs_box])
    sku_ids = out[2]  # return is (n_rots_arr, dims_all, sku_id_per_box)
    distinct = len(set(int(x) for x in sku_ids))
    record("S2a fragile/sturdy/rfs same-size -> distinct skus",
           distinct == 3, f"sku_ids={list(int(x) for x in sku_ids)} distinct={distinct}/3")

    # (b) geometric/weightless workload — partition must match the OLD key
    # (length,width,height,weight,rot_key). Build a mixed-size weightless set;
    # recompute the old partition by hand and compare equivalence classes.
    rng = np.random.default_rng(202)
    gboxes = []
    sizes = [(100, 100, 100), (100, 200, 100), (150, 150, 150), (100, 100, 100),
             (200, 100, 100), (150, 150, 150), (100, 200, 100), (300, 100, 100)]
    for i, (l, w, h) in enumerate(sizes):
        gboxes.append(Box(id=i, length=l, width=w, height=h, weight=0.0,
                          allowed_rotations=ALL_ROTATIONS))
    out_g = precompute_box_dims_and_sku(gboxes)
    new_ids = [int(x) for x in out_g[2]]

    def _old_key(b):
        rot = tuple(sorted(r.name for r in b.allowed_rotations))
        return (round(b.length, 6), round(b.width, 6), round(b.height, 6),
                round(b.weight, 6), rot)
    old_map = {}
    old_ids = []
    for b in gboxes:
        k = _old_key(b)
        if k not in old_map:
            old_map[k] = len(old_map)
        old_ids.append(old_map[k])
    # equivalence-class equality: same partition (ignore id labelling).
    def _partition(ids):
        groups = {}
        for idx, v in enumerate(ids):
            groups.setdefault(v, set()).add(idx)
        return frozenset(frozenset(s) for s in groups.values())
    same_partition = _partition(new_ids) == _partition(old_ids)
    record("S2b geometric SKU partition unchanged vs old key",
           same_partition,
           f"new={new_ids} old={old_ids} same_partition={same_partition}")


# ---------------------------------------------------------------------------
# S3: _pack_with_groups — conservation, co-location, cap, ids, base case.
# ---------------------------------------------------------------------------

def s3_pack_with_groups():
    from pallet_packer import Box, Pallet, PackerConfig, validate, ALL_ROTATIONS
    from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
    warmup_jit()

    pallet = Pallet(length=1200, width=1000, height=1200)
    cfg = PackerConfig()

    # Mixed groups + singletons. Group A (3 boxes), group B (2 boxes),
    # plus 10 ungrouped singletons. Enough volume to force multiple pallets.
    boxes = []
    bid = 0
    for g, n in (("A", 3), ("B", 2), ("C", 4)):
        for _ in range(n):
            boxes.append(Box(id=bid, length=400, width=400, height=400,
                             weight=3.0, group=g, allowed_rotations=ALL_ROTATIONS))
            bid += 1
    for _ in range(10):
        boxes.append(Box(id=bid, length=400, width=400, height=400,
                         weight=3.0, allowed_rotations=ALL_ROTATIONS))
        bid += 1
    n_in = len(boxes)

    res = brkga_pack_v35(boxes, pallet, cfg, time_limit_s=4.0, max_pallets=6,
                         population_size=80, n_populations=2, patience=60,
                         seed=7, n_modes=6)
    n_placed = sum(len(s.placements) for s in res.pallets)
    n_unpacked = len(res.unpacked)
    # conservation: every input box id appears exactly once across placed+unpacked
    placed_ids = [pl.box.id for s in res.pallets for pl in s.placements]
    unpacked_ids = [b.id for b in res.unpacked]
    all_ids = placed_ids + unpacked_ids
    conserved = (sorted(all_ids) == sorted(b.id for b in boxes)
                 and len(all_ids) == n_in)
    record("S3a group-pack conservation (no dup/loss)", conserved,
           f"in={n_in} placed={n_placed} unpacked={n_unpacked} "
           f"unique_out={len(set(all_ids))}")

    # co-location: each group's placed boxes must share exactly one pallet.
    id_to_group = {b.id: getattr(b, "group", None) for b in boxes}
    group_pallets = {}
    for s in res.pallets:
        for pl in s.placements:
            g = id_to_group[pl.box.id]
            if g is not None:
                group_pallets.setdefault(g, set()).add(s.pallet_id)
    split = {g: pids for g, pids in group_pallets.items() if len(pids) > 1}
    record("S3b group co-location (each group <=1 pallet)", not split,
           f"group->pallets={ {g: sorted(p) for g, p in group_pallets.items()} } split={split}")

    # max_pallets cap honored.
    record("S3c max_pallets cap honored", len(res.pallets) <= 6,
           f"pallets_used={len(res.pallets)} cap=6")

    # pallet ids unique.
    pids = [s.pallet_id for s in res.pallets]
    record("S3d pallet ids unique", len(pids) == len(set(pids)),
           f"ids={pids}")

    # validator clean.
    errs = validate(res, pallet, cfg)
    record("S3e group-pack validates clean", len(errs) == 0,
           f"validator_errors={len(errs)}" + (f" first={errs[0]}" if errs else ""))

    # base case: max_pallets==1 with a group must NOT go through _pack_with_groups
    # (it returns a normal single-pallet solve). Prove no infinite recursion +
    # a valid result.
    res1 = brkga_pack_v35(boxes, pallet, cfg, time_limit_s=3.0, max_pallets=1,
                          population_size=80, n_populations=2, patience=60,
                          seed=7, n_modes=6)
    errs1 = validate(res1, pallet, cfg)
    record("S3f max_pallets==1 base case (no recursion, valid)",
           len(res1.pallets) <= 1 and len(errs1) == 0,
           f"pallets={len(res1.pallets)} errs={len(errs1)}")

    # adversarial: a single group BIGGER than a pallet -> overflow to unpacked,
    # never split across pallets.
    big_group = [Box(id=1000 + i, length=700, width=700, height=700,
                     weight=2.0, group="BIG", allowed_rotations=ALL_ROTATIONS)
                 for i in range(8)]
    resb = brkga_pack_v35(big_group, pallet, cfg, time_limit_s=3.0, max_pallets=4,
                          population_size=60, n_populations=2, patience=40,
                          seed=3, n_modes=6)
    gp = {}
    for s in resb.pallets:
        for pl in s.placements:
            gp.setdefault("BIG", set()).add(s.pallet_id)
    big_split = len(gp.get("BIG", set())) > 1
    placed_b = sum(len(s.placements) for s in resb.pallets)
    record("S3g oversized group never splits across pallets", not big_split,
           f"BIG on pallets={sorted(gp.get('BIG', set()))} placed={placed_b} "
           f"unpacked={len(resb.unpacked)}")


# ---------------------------------------------------------------------------
# S4: use_cstr_path — stability-only workload enforced + no v2-seed auto-enable.
# ---------------------------------------------------------------------------

def s4_use_cstr_path():
    from pallet_packer import Box, Pallet, PackerConfig, validate, ALL_ROTATIONS
    from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
    from pallet_packer._brkga_core.precompute import precompute_constraint_arrays
    warmup_jit()

    pallet = Pallet(length=1200, width=1000, height=1500)
    # support_ratio > 0 but ALL boxes weightless + mlot=inf -> has_constraints
    # must be False, yet support must still be enforced (D2 fix).
    cfg = PackerConfig(support_ratio=0.8, require_centroid_supported=True)
    rng = np.random.default_rng(303)
    boxes = []
    for i in range(30):
        boxes.append(Box(id=i,
                         length=int(rng.integers(150, 350)),
                         width=int(rng.integers(150, 350)),
                         height=int(rng.integers(150, 350)),
                         weight=0.0,
                         max_load_on_top=float('inf'),
                         allowed_rotations=ALL_ROTATIONS))
    _, _, _, _, hc = precompute_constraint_arrays(boxes, pallet)
    res = brkga_pack_v35(boxes, pallet, cfg, time_limit_s=5.0, max_pallets=5,
                         population_size=100, n_populations=2, patience=80,
                         seed=11, n_modes=6)
    errs = validate(res, pallet, cfg)
    support_errs = [e for e in errs if "support" in e or "floating" in e
                    or "centroid" in e]
    record("S4a stability-only weightless workload validates (no floating)",
           len(errs) == 0,
           f"has_constraints={hc} (expected False) validator_errors={len(errs)} "
           f"support/floating={len(support_errs)}"
           + (f" first={errs[0]}" if errs else ""))

    # S4b: prove use_v2_seed stays keyed on has_constraints, NOT use_cstr_path.
    # Patch PalletPacker to count instantiations; with use_v2_seed=None on a
    # weightless stability-only workload, v2 seed must NOT auto-enable.
    import pallet_packer.packer as packer_mod
    orig_init = packer_mod.PalletPacker.__init__
    calls = {"n": 0}

    def counting_init(self, *a, **k):
        calls["n"] += 1
        return orig_init(self, *a, **k)
    packer_mod.PalletPacker.__init__ = counting_init
    try:
        calls["n"] = 0
        brkga_pack_v35(boxes, pallet, cfg, time_limit_s=2.0, max_pallets=1,
                       population_size=60, n_populations=2, patience=40,
                       seed=11, n_modes=6, use_v2_seed=None)
        weightless_v2 = calls["n"]
        # control: same shape but with real weight cap / finite mlot ->
        # has_constraints True -> v2 seed SHOULD auto-enable.
        wboxes = [Box(id=i, length=b.length, width=b.width, height=b.height,
                      weight=5.0, max_load_on_top=50.0,
                      allowed_rotations=ALL_ROTATIONS)
                  for i, b in enumerate(boxes)]
        calls["n"] = 0
        brkga_pack_v35(wboxes, pallet, cfg, time_limit_s=2.0, max_pallets=1,
                       population_size=60, n_populations=2, patience=40,
                       seed=11, n_modes=6, use_v2_seed=None)
        weighted_v2 = calls["n"]
    finally:
        packer_mod.PalletPacker.__init__ = orig_init
    record("S4b v2-seed NOT auto-enabled by stability-only (keyed on has_constraints)",
           weightless_v2 == 0 and weighted_v2 > 0,
           f"PalletPacker instantiations: weightless={weightless_v2} (expect 0) "
           f"weighted={weighted_v2} (expect >0)")


# ---------------------------------------------------------------------------
# S5: input_validation edge logic.
# ---------------------------------------------------------------------------

def s5_input_validation():
    from pallet_packer import (Box, Pallet, ALL_ROTATIONS,
                               validate_packing_input, check_packing_input,
                               PackingInputError)
    from pallet_packer.input_validation import _is_cap

    # _is_cap unit checks.
    cap_cases = {
        float('inf'): True, float('-inf'): False, float('nan'): False,
        0.0: False, -1.0: False, 1.0: True, None: False, 1: True,
    }
    cap_ok = all(_is_cap(v) is exp for v, exp in cap_cases.items())
    record("S5a _is_cap (inf/-inf/nan/0/neg/pos/None)", cap_ok,
           f"{ {repr(v): _is_cap(v) for v in cap_cases} }")

    good_pallet = Pallet(length=1200, width=1000, height=1500)

    def _good_box(i=0, **over):
        kw = dict(id=i, length=100, width=100, height=100, weight=1.0,
                  allowed_rotations=ALL_ROTATIONS)
        kw.update(over)
        return Box(**kw)

    # integer-valued float dims accepted.
    p = validate_packing_input(
        [Box(id=0, length=100.0, width=100.0, height=100.0, weight=1.0,
             allowed_rotations=ALL_ROTATIONS)], good_pallet)
    record("S5b integer-valued float dims accepted", p == [], f"problems={p}")

    # fractional dim rejected.
    p = validate_packing_input([_good_box(length=100.5)], good_pallet)
    record("S5c fractional dim rejected", any("length" in x for x in p),
           f"problems={p}")

    # negative / zero dims rejected.
    p_neg = validate_packing_input([_good_box(width=-5)], good_pallet)
    p_zero = validate_packing_input([_good_box(height=0)], good_pallet)
    record("S5d negative+zero dims rejected",
           bool(p_neg) and bool(p_zero),
           f"neg={bool(p_neg)} zero={bool(p_zero)}")

    # NaN pallet.max_weight rejected (the headline D7 defect).
    p = validate_packing_input([_good_box()],
                               Pallet(length=1200, width=1000, height=1500,
                                      max_weight=float('nan')))
    record("S5e NaN pallet.max_weight rejected",
           any("max_weight" in x for x in p), f"problems={p}")

    # max_load_on_top: 0 (fragile) OK, +inf OK, -1 rejected, NaN rejected.
    p_frag = validate_packing_input([_good_box(max_load_on_top=0.0)], good_pallet)
    p_inf = validate_packing_input([_good_box(max_load_on_top=float('inf'))], good_pallet)
    p_neg_m = validate_packing_input([_good_box(max_load_on_top=-1.0)], good_pallet)
    p_nan_m = validate_packing_input([_good_box(max_load_on_top=float('nan'))], good_pallet)
    p_ninf_m = validate_packing_input([_good_box(max_load_on_top=float('-inf'))], good_pallet)
    mlot_ok = (p_frag == [] and p_inf == [] and bool(p_neg_m)
               and bool(p_nan_m) and bool(p_ninf_m))
    record("S5f max_load_on_top precedence (0/+inf ok; -1/nan/-inf rejected)",
           mlot_ok,
           f"frag={p_frag==[]} inf={p_inf==[]} neg={bool(p_neg_m)} "
           f"nan={bool(p_nan_m)} ninf={bool(p_ninf_m)}")

    # duplicate ids rejected.
    p = validate_packing_input([_good_box(i=5), _good_box(i=5)], good_pallet)
    record("S5g duplicate ids rejected", any("duplicate" in x for x in p),
           f"problems={p}")

    # missing id rejected.
    p = validate_packing_input([_good_box(id=None)], good_pallet)
    record("S5h missing id rejected", any("id is required" in x for x in p),
           f"problems={p}")

    # box-count cap enforced + disabled with None.
    many = [_good_box(i=i) for i in range(501)]
    p_cap = validate_packing_input(many, good_pallet)
    p_nocap = validate_packing_input(many, good_pallet, max_boxes=None)
    record("S5i box-count cap (501>500 rejected; None disables)",
           any("too many" in x for x in p_cap) and p_nocap == [],
           f"capped={any('too many' in x for x in p_cap)} nocap_clean={p_nocap==[]}")

    # empty rotations rejected; empty box list rejected.
    p_rot = validate_packing_input([_good_box(allowed_rotations=[])], good_pallet)
    p_empty = validate_packing_input([], good_pallet)
    record("S5j empty rotations + empty box list rejected",
           bool(p_rot) and any("at least one box" in x for x in p_empty),
           f"rot={bool(p_rot)} empty={bool(p_empty)}")

    # check_packing_input raises with .problems populated.
    raised = False
    probs = None
    try:
        check_packing_input([_good_box(length=-1)], good_pallet)
    except PackingInputError as e:
        raised = True
        probs = e.problems
    record("S5k check_packing_input raises PackingInputError",
           raised and probs and len(probs) >= 1,
           f"raised={raised} n_problems={len(probs) if probs else 0}")

    # a fully valid request passes.
    p = validate_packing_input([_good_box(i=i) for i in range(5)], good_pallet)
    record("S5l valid request -> empty problems", p == [], f"problems={p}")


def main():
    print("=== RE-VERIFY static-diff-review (session changes vs cead45d) ===")
    for fn in (s1_per_box_block_equiv, s2_sku_key, s3_pack_with_groups,
               s4_use_cstr_path, s5_input_validation):
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
