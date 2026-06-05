"""Floor-first block-shape fix verification (runs inside pallet-packer:dev).

Checks two things at once:
  1. FLATNESS — the user's bug: few identical boxes on a large/tall pallet must
     spread across the floor, not build a corner tower. Tested via the exact
     production cstr path (support+centroid+overhang) AND geometric-only, plus a
     small sweep.
  2. BR DENSITY NEUTRALITY — the headline metric must not regress. Runs the
     block-building (v2) BR path on thpack1/3/5/7.

Run pre-fix (baseline) and post-fix, compare.
"""
import sys, time
import numpy as np
from pallet_packer import Box, Pallet, PackerConfig, ALL_ROTATIONS
from pallet_packer.brkga_v3_5 import brkga_pack_v35

TAG = sys.argv[1] if len(sys.argv) > 1 else "RUN"


def _cstr_cfg(overhang):
    # Mirrors the service adapter._config exactly.
    return PackerConfig(
        support_ratio=0.8, require_centroid_supported=True,
        enforce_load_bearing=True, cog_envelope_fraction=1.0,
        cog_check_min_load_fraction=1.0, allow_pallet_overhang=overhang)


def _geom_cfg():
    return PackerConfig(
        support_ratio=0.0, require_centroid_supported=False,
        allow_pallet_overhang=False, heavy_on_bottom=False,
        cog_envelope_fraction=1.0, cog_check_min_load_fraction=1.0,
        enforce_load_bearing=False)


def _verdict(pl):
    if not pl or not pl.placements:
        return "EMPTY", {}
    zs = sorted({round(p.z) for p in pl.placements})
    xs = sorted({round(p.x) for p in pl.placements})
    top_z = max(round(p.z) for p in pl.placements)
    n_layers = len(zs)
    v = "FLAT" if top_z <= 260 else ("2-LAYER" if top_z <= 510 else
         ("3-LAYER" if top_z <= 760 else "TOWER"))
    return v, dict(items=len(pl.placements), top_z=top_z, distinct_z=zs,
                   distinct_x=xs, n_layers=n_layers)


def _run(boxes, pallet, cfg, seed, t, max_pallets):
    res = brkga_pack_v35(boxes, pallet, cfg, time_limit_s=t,
                         max_pallets=max_pallets, seed=seed,
                         population_size=300, n_populations=3, patience=150,
                         n_modes=6, validate_input=True, verbose=False)
    pl = res.pallets[0] if res.pallets else None
    v, info = _verdict(pl)
    info["unpacked"] = len(res.unpacked)
    info["n_pallets"] = len(res.pallets)
    return v, info


def flatness():
    print(f"\n===== FLATNESS [{TAG}] =====")
    # 1) Exact user bug case via the production cstr path.
    boxes = [Box(id=f"box-{i}", length=400, width=300, height=250, weight=6.4,
                 allowed_rotations=list(ALL_ROTATIONS)) for i in range(10)]
    pallet = Pallet(length=1200, width=800, height=1800, max_weight=1000,
                    max_overhang=40)
    t0 = time.time()
    v, info = _run(boxes, pallet, _cstr_cfg(True), seed=7, t=12, max_pallets=2)
    print(f"  BUG cstr  10x(400x300x250) 1200x800x1800 sr=0.8: {v}  {info}  ({time.time()-t0:.1f}s)")
    # 2) Same case geometric-only (Test C regime).
    pallet_g = Pallet(length=1200, width=800, height=1800)
    v, info = _run(boxes, pallet_g, _geom_cfg(), seed=7, t=12, max_pallets=1)
    print(f"  BUG geom  10x(400x300x250) 1200x800x1800        : {v}  {info}")
    # 3) Test B: 8 boxes that exactly tile one 4x2 layer.
    boxes8 = boxes[:8]
    v, info = _run(boxes8, pallet_g, _geom_cfg(), seed=7, t=12, max_pallets=1)
    print(f"  TESTB     8x(400x300x250) 1200x800x1800 (geom)  : {v}  {info}")
    # 4) Small sweep.
    for (n, L, W, H, sd) in [(4, 2000, 2000, 2000, 1), (12, 1200, 800, 1800, 3),
                             (6, 1500, 1200, 2000, 5)]:
        bs = [Box(id=f"b{i}", length=400, width=300, height=250, weight=0.0,
                  allowed_rotations=list(ALL_ROTATIONS)) for i in range(n)]
        v, info = _run(bs, Pallet(length=L, width=W, height=H), _geom_cfg(),
                       seed=sd, t=8, max_pallets=1)
        print(f"  sweep n={n} {L}x{W}x{H} seed={sd}: {v}  top_z={info['top_z']} layers={info['n_layers']} unp={info['unpacked']}")


def br():
    print(f"\n===== BR DENSITY (v2 block-building) [{TAG}] =====")
    from benchmarks.br import run_set
    sample = 8
    for s in ("thpack1", "thpack3", "thpack5", "thpack7"):
        t0 = time.time()
        summ, _ = run_set(f"benchmarks/data/{s}.txt", sample=sample, use_v2=True)
        print(f"  {summ.set_id}: v2 {summ.util_mean:5.2f}% ± {summ.util_stdev:4.2f}  "
              f"(n={sample}, {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    flatness()
    br()
    print(f"\n[{TAG}] DONE")
