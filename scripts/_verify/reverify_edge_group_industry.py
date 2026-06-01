"""reverify_edge_group_industry.py — adversarial group-split + 10 industry cases.

Two adversarial fronts that the session fixes touch:

  1. CAP-BINDING group split: force the rare _pack_with_groups branch where
     len(assigned) == max_pallets and a bundle must drop onto the emptiest pallet.
     A group must NEVER split even when capacity binds. Many groups, each <1 pallet,
     more groups than pallets.

  2. ALL 10 INDUSTRY CASES: run each with its own config + a generous multi-pallet
     cap. Assert validator-clean, conservation, and NO group spans >1 pallet
     (IND6 is grouped; the rest are ungrouped -> trivially no split). This is the
     regression net for D2-from-config + max_pallets propagation + group dispatch
     on real workloads.

Run inside Docker (OMP_NUM_THREADS=1):
    docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
        -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/reverify_edge_group_industry.py
"""
from __future__ import annotations

import os
import sys
import traceback
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pallet_packer import (Box, Pallet, PackerConfig, ALL_ROTATIONS, THIS_SIDE_UP,
                           validate)
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from benchmarks.industry import cases as industry_cases

FAILS = []
PASSES = []


def fail(c, m):
    FAILS.append((c, m)); print(f"  [FAIL] {c}: {m}", flush=True)


def ok(c, m):
    PASSES.append(c); print(f"  [pass] {c}: {m}", flush=True)


def group_spans(result):
    gp = defaultdict(set)
    for pi, st in enumerate(result.pallets):
        for p in st.placements:
            g = getattr(p.box, "group", None)
            if g is not None:
                gp[g].add(pi)
    return {g: sorted(s) for g, s in gp.items() if len(s) > 1}


def check(case, result, pallet, config, n_input, expect_no_split=True):
    good = True
    try:
        errs = validate(result, pallet, config)
    except Exception as e:
        fail(case, f"validate raised {type(e).__name__}: {e}"); return False
    if errs:
        good = False; fail(case, f"{len(errs)} validator err(s); first: {errs[0]}")
    placed_ids = [id(p.box) for st in result.pallets for p in st.placements]
    n_placed = len(placed_ids)
    n_unpacked = len(result.unpacked)
    if n_placed + n_unpacked != n_input:
        good = False
        fail(case, f"conservation: {n_placed}+{n_unpacked} != {n_input}")
    if len(set(placed_ids)) != n_placed:
        good = False; fail(case, "a box placed more than once")
    for st in result.pallets:
        for p in st.placements:
            if (p.x < -1e-6 or p.y < -1e-6 or p.z < -1e-6 or
                    p.x2 > pallet.length + 1e-6 or p.y2 > pallet.width + 1e-6 or
                    p.z2 > pallet.height + 1e-6):
                good = False; fail(case, f"OOB/neg {p.box.id}")
    if expect_no_split:
        spans = group_spans(result)
        if spans:
            good = False; fail(case, f"GROUP SPLIT: {spans}")
    if good:
        ok(case, f"placed={n_placed} unpacked={n_unpacked} pallets={len(result.pallets)} clean")
    return good


def case_cap_binding_groups():
    print("\n=== CAP-BINDING group split (more groups than pallets) ===")
    # 6 groups of 6 boxes, each group ~ small; pallet large enough for ~2-3 groups.
    # max_pallets=2 forces the cap-binding branch (6 bundles -> only 2 pallets).
    pallet = Pallet(length=1200, width=1000, height=1200, max_weight=10000)
    cfg = PackerConfig(support_ratio=0.0, require_centroid_supported=False)
    boxes = []
    for gi in range(6):
        for k in range(6):
            boxes.append(Box(id=f"G{gi}-{k}", length=300, width=250, height=200,
                             weight=2.0, group=f"G{gi}",
                             allowed_rotations=ALL_ROTATIONS))
    for cap in (1, 2, 3):
        try:
            r = brkga_pack_v35(boxes, pallet, cfg, time_limit_s=8, max_pallets=cap,
                               population_size=200, n_populations=3, patience=120,
                               seed=42, n_modes=6, verbose=False)
            check(f"cap-bind/mp={cap}", r, pallet, cfg, len(boxes))
        except Exception as e:
            fail(f"cap-bind/mp={cap}",
                 f"raised {type(e).__name__}: {e}\n{traceback.format_exc()}")


def case_mixed_grouped_ungrouped_overflow():
    print("\n=== MIXED grouped + ungrouped with overflow (no split) ===")
    pallet = Pallet(length=1000, width=800, height=600, max_weight=10000)
    cfg = PackerConfig(support_ratio=0.0, require_centroid_supported=False)
    boxes = []
    # 2 groups + many ungrouped singletons -> total volume >> capacity
    for gi in range(2):
        for k in range(8):
            boxes.append(Box(id=f"GRP{gi}-{k}", length=400, width=300, height=250,
                             weight=3.0, group=f"GRP{gi}",
                             allowed_rotations=ALL_ROTATIONS))
    for k in range(30):
        boxes.append(Box(id=f"S{k}", length=300, width=250, height=200, weight=2.0,
                         allowed_rotations=ALL_ROTATIONS))
    try:
        r = brkga_pack_v35(boxes, pallet, cfg, time_limit_s=8, max_pallets=3,
                           population_size=200, n_populations=3, patience=120,
                           seed=42, n_modes=6, verbose=False)
        check("mixed-overflow", r, pallet, cfg, len(boxes))
    except Exception as e:
        fail("mixed-overflow", f"raised {type(e).__name__}: {e}\n{traceback.format_exc()}")


def case_industry_all():
    print("\n=== ALL 10 industry cases (multi-pallet cap, validator+conservation+no-split) ===")
    for c in industry_cases():
        n = len(c.boxes)
        is_grouped = any(getattr(b, "group", None) is not None for b in c.boxes)
        try:
            r = brkga_pack_v35(c.boxes, c.pallet, c.config, time_limit_s=10,
                               max_pallets=10, population_size=200, n_populations=3,
                               patience=120, seed=42, n_modes=6, verbose=False)
            tag = c.name.split()[0]
            good = check(f"{tag} (grouped={is_grouped})", r, c.pallet, c.config, n)
            n_placed = sum(len(st.placements) for st in r.pallets)
            print(f"        {tag}: n={n} placed={n_placed} "
                  f"unpacked={len(r.unpacked)} pallets={len(r.pallets)}")
        except Exception as e:
            fail(c.name.split()[0], f"raised {type(e).__name__}: {e}\n"
                                    f"{traceback.format_exc()}")


def main():
    print("=== reverify_edge_group_industry ===")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','default')}", flush=True)
    warmup_jit()
    for fn in (case_cap_binding_groups, case_mixed_grouped_ungrouped_overflow,
               case_industry_all):
        try:
            fn()
        except Exception as e:
            fail(fn.__name__, f"UNCAUGHT {type(e).__name__}: {e}\n{traceback.format_exc()}")
    print("\n" + "=" * 60)
    print(f"=== SUMMARY: {len(PASSES)} passed, {len(FAILS)} failed ===")
    for c, m in FAILS:
        print(f"  FAIL {c}: {m.splitlines()[0]}")
    print(f"=== OVERALL: {'PASS' if not FAILS else 'FAIL'} ===")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
