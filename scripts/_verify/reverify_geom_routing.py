"""Prove the GEOMETRIC path is genuinely taken under geometric_only_config,
and that the SKU partition is unchanged by the precompute mlot/rfs key change.

This is the mechanism check behind the regression result: the geometric BR
decode path is byte-identical to commit cead45d (geom decoders untouched), so
the ONLY way geometric BR could change is (a) routing flips to the constraint
path, or (b) the SKU partition shifts. We falsify both here, in Docker.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pallet_packer._brkga_core.precompute import (
    precompute_box_dims_and_sku,
    precompute_constraint_arrays,
    precompute_cog_envelope,
)
from benchmarks.br import parse_thpack, geometric_only_config


def main():
    ok = True
    cfg = geometric_only_config(max_pallets=1)
    print("geometric_only_config:")
    print(f"  support_ratio              = {cfg.support_ratio}")
    print(f"  require_centroid_supported = {cfg.require_centroid_supported}")
    print(f"  allow_pallet_overhang      = {cfg.allow_pallet_overhang}")
    print(f"  enforce_load_bearing       = {cfg.enforce_load_bearing}")
    print(f"  cog_envelope_fraction      = {cfg.cog_envelope_fraction}")
    print(f"  cog_check_min_load_fraction= {cfg.cog_check_min_load_fraction}")

    for path, sid, iid in [("benchmarks/data/thpack1.txt", "BR1", 1),
                           ("benchmarks/data/thpack3.txt", "BR3", 1),
                           ("benchmarks/data/thpack5.txt", "BR5", 1),
                           ("benchmarks/data/thpack7.txt", "BR7", 1)]:
        inst = next(i for i in parse_thpack(path, set_id=sid)
                    if i.instance_id == iid)
        boxes, pallet = inst.boxes, inst.pallet

        # 1) Box defaults must be geometric (mlot=inf, rfs=False, weight=0).
        mlots = {getattr(b, "max_load_on_top", float("inf")) for b in boxes}
        rfss = {bool(getattr(b, "requires_full_support", False)) for b in boxes}
        wts = {round(float(getattr(b, "weight", 0.0)), 6) for b in boxes}

        # 2) SKU partition: emulate OLD key (no mlot/rfs) vs NEW key.
        import math
        def part_new(bx):
            n_rots_arr, dims_all, sku_id_per_box = precompute_box_dims_and_sku(bx)
            return tuple(int(x) for x in sku_id_per_box.tolist())
        def part_old(bx):
            sku_to_id, out = {}, []
            for b in bx:
                rot_key = tuple(sorted(r.name for r in b.allowed_rotations))
                key = (round(b.length, 6), round(b.width, 6),
                       round(b.height, 6), round(b.weight, 6), rot_key)
                sku_to_id.setdefault(key, len(sku_to_id))
                out.append(sku_to_id[key])
            return tuple(out)
        p_old, p_new = part_old(boxes), part_new(boxes)
        n_sku_old = len(set(p_old))
        n_sku_new = len(set(p_new))
        same_partition = (p_old == p_new)

        # 3) Routing flags exactly as driver.py computes them.
        weights_arr, mlot_arr, rfs_arr, pmw, has_constraints = \
            precompute_constraint_arrays(boxes, pallet)
        (cog_xmin, cog_xmax, cog_ymin, cog_ymax,
         cog_minload, cog_active) = precompute_cog_envelope(pallet, cfg)
        stability_active = (float(cfg.support_ratio) > 0.0
                            or bool(cfg.require_centroid_supported))
        overhang_active = (bool(cfg.allow_pallet_overhang)
                           and float(pallet.max_overhang) > 0.0)
        use_cstr_path = bool(has_constraints or stability_active
                             or cog_active or overhang_active)

        geom = (not use_cstr_path)
        print(f"\n  {sid}#{iid}  N={len(boxes)}")
        print(f"    box mlot set={mlots}  rfs set={rfss}  weight set={wts}")
        print(f"    SKU partition: old #SKU={n_sku_old} new #SKU={n_sku_new} "
              f"IDENTICAL={same_partition}")
        print(f"    has_constraints={has_constraints} stability={stability_active} "
              f"cog_active={cog_active} overhang={overhang_active}")
        print(f"    => use_cstr_path={use_cstr_path}  GEOMETRIC_PATH={geom}")

        if not same_partition:
            print("    FAIL: SKU partition changed for geometric workload")
            ok = False
        if not geom:
            print("    FAIL: geometric workload routed through CONSTRAINT path")
            ok = False

    print(f"\nSUMMARY: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
