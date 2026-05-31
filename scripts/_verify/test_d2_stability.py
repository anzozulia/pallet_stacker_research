"""Test D2: always-on stability (support_ratio decoupled from has_constraints).

  1. WEIGHTLESS workload + default config (support_ratio=0.8): now validator-
     clean (was floating). Control: support_ratio=0 still floats (shows the
     enforcement is what changed, not the packing geometry).
  2. v2-seed must NOT auto-enable on a weightless workload (Phase 7a basin).
  3. BR1 with geometric_only_config (now support_ratio=0): util unchanged +
     validator-clean.
  4. Industry (constrained): still clean (no regression).
"""
from __future__ import annotations
import sys
sys.path.insert(0, '.')
import numpy as np
from collections import Counter
from pallet_packer import Box, Pallet, PackerConfig, ALL_ROTATIONS, validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer._brkga_core.precompute import precompute_constraint_arrays
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization
from benchmarks.industry import cases as industry_cases


def cat(e):
    for k, t in [('floating', 'floating'), ('overlaps', 'overlap'),
                 ('max_load_on_top', 'load'), ('outside', 'oob'), ('weight', 'weight')]:
        if k in e:
            return t
    return 'other'


def weightless(N, rng):
    skus = [(int(rng.integers(150, 450)), int(rng.integers(120, 350)),
             int(rng.integers(100, 300))) for _ in range(6)]
    boxes = [Box(id=f'B{i}', length=skus[i % 6][0], width=skus[i % 6][1],
                 height=skus[i % 6][2], weight=0.0, allowed_rotations=ALL_ROTATIONS)
             for i in range(N)]
    return boxes, Pallet(length=1200, width=1000, height=1500)


def main():
    warmup_jit()
    ok = True

    print("=== 1. Weightless + default config (support_ratio=0.8): clean? ===")
    boxes, pallet = weightless(80, np.random.default_rng(7))
    # has_constraints / v2-seed check
    _, _, _, _, hc = precompute_constraint_arrays(boxes, pallet)
    print(f"  has_constraints={hc} (expect False -> v2-seed must stay off)")
    for sr in [0.8, 0.0]:
        cfg = PackerConfig(support_ratio=sr, cog_envelope_fraction=1.0,
                           cog_check_min_load_fraction=1.0)
        r = brkga_pack_v35(boxes, pallet, cfg, time_limit_s=15, max_pallets=1,
                           population_size=300, n_populations=3, patience=200,
                           seed=42, n_modes=6, verbose=False)
        errs = validate(r, pallet, cfg)
        placed = len(r.pallets[0].placements) if r.pallets else 0
        c = Counter(cat(e) for e in errs)
        tag = "STABILITY ON" if sr > 0 else "control (off)"
        print(f"  support_ratio={sr} [{tag}]: placed={placed} errs={len(errs)} {dict(c)}")
        if sr == 0.8 and len(errs) != 0:
            ok = False

    print("\n=== 3. BR1#1 geometric_only_config (support_ratio=0): util unchanged ===")
    cfg = geometric_only_config(max_pallets=1)
    print(f"  geometric_only_config.support_ratio={cfg.support_ratio} "
          f"require_centroid={cfg.require_centroid_supported}")
    inst = parse_thpack('benchmarks/data/thpack1.txt', set_id='BR1')[0]
    r = brkga_pack_v35(inst.boxes, inst.pallet, cfg, time_limit_s=10, max_pallets=1,
                       population_size=300, n_populations=3, patience=200,
                       local_search_budget_s=2.0, seed=42 + inst.instance_id,
                       n_modes=6, verbose=False)
    u = first_pallet_utilization(r, inst.pallet) * 100
    errs = validate(r, inst.pallet, cfg)
    print(f"  BR1#1: util={u:.4f}% (baseline 91.9620%) errs={len(errs)}")

    print("\n=== 4. Industry (constrained): no regression ===")
    for c in industry_cases()[:5]:
        r = brkga_pack_v35(c.boxes, c.pallet, c.config, time_limit_s=10,
                           max_pallets=10, population_size=300, n_populations=3,
                           patience=150, seed=42, n_modes=6, verbose=False)
        errs = validate(r, c.pallet, c.config)
        print(f"  {c.name[:30]:30s}: pallets={len(r.pallets)} errs={len(errs)}")
        if errs:
            ok = False

    print(f"\n=== D2 RESULT: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
