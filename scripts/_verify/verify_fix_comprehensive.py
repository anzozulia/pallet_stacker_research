"""Comprehensive regression check for the block-decoder constraint fix
(Part A: mlot/rfs in SKU key; Part B: per-box support/load in block decode).

  1. Constrained ladder (realistic + adversarial) -> expect 0 errors at the cap.
  2. All 10 industry cases -> expect 0 errors (no regression) + density.
  3. Geometric BR converged -> expect util unchanged (cstr path untouched; the
     residual geometric 'floating' is the SEPARATE D2 weightless-stability issue,
     not this fix — flagged, not counted against this fix).
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, '.')
import numpy as np
from collections import Counter
from pallet_packer import Box, Pallet, PackerConfig, THIS_SIDE_UP, validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization
from benchmarks.industry import cases as industry_cases


def cat(e):
    for k, t in [('floating', 'floating'), ('overlaps', 'overlap'),
                 ('max_load_on_top', 'load_bear'), ('outside', 'oob'),
                 ('weight', 'weight'), ('rotation', 'rot')]:
        if k in e:
            return t
    return 'other'


def realistic(N, rng):
    skus = []
    for s in range(8):
        l, w, h = int(rng.integers(150, 450)), int(rng.integers(120, 350)), int(rng.integers(100, 300))
        skus.append((l, w, h, float(rng.integers(2, 15)), 0.0 if s < 2 else 40.0))
    boxes = [Box(id=f'B{i}', length=skus[i % 8][0], width=skus[i % 8][1],
                 height=skus[i % 8][2], weight=skus[i % 8][3],
                 allowed_rotations=THIS_SIDE_UP, max_load_on_top=skus[i % 8][4])
             for i in range(N)]
    return boxes, Pallet(length=1200, width=1000, height=1500, max_weight=900)


def adversarial(N, rng):
    skus = [(int(rng.integers(150, 450)), int(rng.integers(120, 350)),
             int(rng.integers(100, 300)), float(rng.integers(2, 15))) for _ in range(8)]
    boxes = [Box(id=f'B{i}', length=skus[i % 8][0], width=skus[i % 8][1],
                 height=skus[i % 8][2], weight=skus[i % 8][3],
                 allowed_rotations=THIS_SIDE_UP,
                 max_load_on_top=0.0 if i % 9 == 0 else float(rng.integers(15, 60)))
             for i in range(N)]
    return boxes, Pallet(length=1200, width=1000, height=1500, max_weight=900)


def main():
    warmup_jit()
    total_err = 0

    print("=== 1. Constrained ladder (target: 0 errs) ===")
    for name, gen, Ns in [("realistic", realistic, [200, 350, 500, 750]),
                          ("adversarial", adversarial, [350, 500])]:
        for N in Ns:
            boxes, pallet = gen(N, np.random.default_rng(42))
            cfg = PackerConfig()
            r = brkga_pack_v35(boxes, pallet, cfg, time_limit_s=25, max_pallets=40,
                               population_size=300, n_populations=3, patience=200,
                               seed=42, n_modes=6, verbose=False)
            errs = validate(r, pallet, cfg)
            total_err += len(errs)
            print(f"  {name:11s} N={N:>4}: pallets={len(r.pallets):>3} "
                  f"unpkd={len(r.unpacked):>3} errs={len(errs):>3} "
                  f"{dict(Counter(cat(e) for e in errs))}", flush=True)

    print("\n=== 2. Industry 10 cases (target: 0 errs, no regression) ===")
    for c in industry_cases():
        r = brkga_pack_v35(c.boxes, c.pallet, c.config, time_limit_s=12,
                           max_pallets=10, population_size=300, n_populations=3,
                           patience=150, seed=42, n_modes=6, verbose=False)
        errs = validate(r, c.pallet, c.config)
        total_err += len(errs)
        print(f"  {c.name[:34]:34s}: pallets={len(r.pallets):>2} "
              f"unpkd={len(r.unpacked):>3} errs={len(errs):>3} "
              f"{dict(Counter(cat(e) for e in errs))}", flush=True)

    print("\n=== 3. Geometric BR converged (regression guard; D2 floating is separate) ===")
    cfg = geometric_only_config(max_pallets=1)
    for fn, sid in [("thpack1", "BR1"), ("thpack3", "BR3")]:
        inst = parse_thpack(f"benchmarks/data/{fn}.txt", set_id=sid)[0]
        r = brkga_pack_v35(inst.boxes, inst.pallet, cfg, time_limit_s=600,
                           max_pallets=1, population_size=300, n_populations=3,
                           patience=15, local_search_budget_s=0.0,
                           use_lns=False, use_v2_hybrid_polish=False,
                           seed=42 + inst.instance_id, n_modes=6, verbose=False)
        u = first_pallet_utilization(r, inst.pallet) * 100
        print(f"  {sid}#{inst.instance_id} converged: util={u:.4f}%", flush=True)

    print(f"\n=== CONSTRAINED+INDUSTRY errors: {total_err} (target 0) ===")
    return 0 if total_err == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
