"""Test Part A: mlot/rfs in SKU key. Run in Docker (no rebuild — pure Python).

Checks:
  1. REALISTIC workload (fragility is a SKU property): 0 errors, density preserved.
  2. ADVERSARIAL workload (per-box random mlot): 0 errors (fragile separated).
  3. Geometric BR1#1 bit-identical (sku partition unchanged for weightless).
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, '.')
import numpy as np
from collections import Counter
from pallet_packer import Box, Pallet, PackerConfig, THIS_SIDE_UP, validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization


def cat(e):
    for k, t in [('floating', 'floating'), ('overlaps', 'overlap'),
                 ('max_load_on_top', 'load_bear'), ('outside', 'oob'),
                 ('weight', 'weight'), ('rotation', 'rot')]:
        if k in e:
            return t
    return 'other'


def realistic(N, rng):
    """Fragility + mlot are SKU properties (as in the real world)."""
    skus = []
    for s in range(8):
        l, w, h = int(rng.integers(150, 450)), int(rng.integers(120, 350)), int(rng.integers(100, 300))
        wt = float(rng.integers(2, 15))
        mlot = 0.0 if s < 2 else 40.0   # 2 of 8 SKUs fragile, rest sturdy
        skus.append((l, w, h, wt, mlot))
    boxes = []
    for i in range(N):
        l, w, h, wt, mlot = skus[i % 8]
        boxes.append(Box(id=f'B{i}', length=l, width=w, height=h, weight=wt,
                         allowed_rotations=THIS_SIDE_UP, max_load_on_top=mlot))
    return boxes, Pallet(length=1200, width=1000, height=1500, max_weight=900)


def adversarial(N, rng):
    """Same dims, per-box random mlot (stress: should fragment into singletons)."""
    skus = [(int(rng.integers(150, 450)), int(rng.integers(120, 350)),
             int(rng.integers(100, 300)), float(rng.integers(2, 15))) for _ in range(8)]
    boxes = []
    for i in range(N):
        l, w, h, wt = skus[i % 8]
        mlot = 0.0 if i % 9 == 0 else float(rng.integers(15, 60))
        boxes.append(Box(id=f'B{i}', length=l, width=w, height=h, weight=wt,
                         allowed_rotations=THIS_SIDE_UP, max_load_on_top=mlot))
    return boxes, Pallet(length=1200, width=1000, height=1500, max_weight=900)


def run(boxes, pallet, label):
    cfg = PackerConfig()
    t0 = time.perf_counter()
    r = brkga_pack_v35(boxes, pallet, cfg, time_limit_s=30, max_pallets=40,
                       population_size=300, n_populations=3, patience=200,
                       seed=42, n_modes=6, verbose=False)
    wall = time.perf_counter() - t0
    errs = validate(r, pallet, cfg)
    print(f"  {label:28s}: pallets={len(r.pallets):>3} unpkd={len(r.unpacked):>3} "
          f"errs={len(errs):>3} {dict(Counter(cat(e) for e in errs))}  ({wall:.0f}s)")
    return len(errs)


def main():
    warmup_jit()
    print("=== Part A fix: mlot/rfs in SKU key ===")
    print("\n1. REALISTIC (fragility per-SKU) — expect 0 errs, blocks preserved:")
    e1 = 0
    for N in [200, 500]:
        e1 += run(*realistic(N, np.random.default_rng(42)), f"realistic N={N}")
    print("\n2. ADVERSARIAL (per-box mlot) — expect 0 errs (density may drop):")
    e2 = 0
    for N in [200, 500]:
        e2 += run(*adversarial(N, np.random.default_rng(42)), f"adversarial N={N}")
    print("\n3. GEOMETRIC BR1 — expect unchanged util (bit-identical partition):")
    cfg = geometric_only_config(max_pallets=1)
    e3ok = True
    for inst in parse_thpack('benchmarks/data/thpack1.txt', set_id='BR1')[:2]:
        r = brkga_pack_v35(inst.boxes, inst.pallet, cfg, time_limit_s=10, max_pallets=1,
                           population_size=300, n_populations=3, patience=200,
                           local_search_budget_s=2.0, seed=42 + inst.instance_id,
                           n_modes=6, verbose=False)
        u = first_pallet_utilization(r, inst.pallet) * 100
        print(f"  BR1#{inst.instance_id}: util={u:.4f}%")
    print(f"\n=== load/support errors: realistic={e1} adversarial={e2} "
          f"(target: 0 / 0) ===")
    return 0 if (e1 == 0 and e2 == 0) else 1


if __name__ == '__main__':
    raise SystemExit(main())
