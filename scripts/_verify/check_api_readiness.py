"""API-readiness check: worst-case loads at the cap through the FULL pipeline
the service will use — input gate -> solve(validate_input=True) -> validate() ->
to_json() serialization. Confirms the algorithm side is build-ready.
"""
from __future__ import annotations
import sys, time, json
sys.path.insert(0, '.')
import numpy as np
from pallet_packer import (Box, Pallet, PackerConfig, THIS_SIDE_UP, ALL_ROTATIONS,
                           validate, check_packing_input, PackingInputError, to_json)
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit


def worst_case(N, rng, grouped):
    """Mixed SKUs, ~15% fragile (mlot=0), per-SKU weights, weight-capped pallet,
    integer dims. grouped=True attaches LTL-style groups (triggers the wrapper)."""
    skus = []
    for s in range(12):
        skus.append((int(rng.integers(120, 420)), int(rng.integers(100, 340)),
                     int(rng.integers(90, 280)), float(rng.integers(2, 18)),
                     0.0 if s < 2 else float(rng.integers(20, 60))))
    boxes = []
    for i in range(N):
        l, w, h, wt, mlot = skus[i % 12]
        g = f"CUST{i % 15}" if grouped else None   # 15 customer groups
        boxes.append(Box(id=f"B{i:04d}", length=l, width=w, height=h, weight=wt,
                         allowed_rotations=THIS_SIDE_UP, max_load_on_top=mlot, group=g))
    return boxes, Pallet(length=1200, width=1000, height=1500, max_weight=900)


def run(label, boxes, pallet, budget):
    cfg = PackerConfig()  # default: support_ratio=0.8 (stability enforced)
    # 1. input gate
    try:
        check_packing_input(boxes, pallet, max_boxes=500)
        gate = "PASS"
    except PackingInputError as e:
        gate = f"REJECTED: {e.problems[:1]}"
    # 2. solve (gate also runs inside via validate_input=True)
    t0 = time.perf_counter()
    r = brkga_pack_v35(boxes, pallet, cfg, time_limit_s=budget, max_pallets=30,
                       population_size=300, n_populations=3, patience=150,
                       seed=42, n_modes=6, validate_input=True, verbose=False)
    wall = time.perf_counter() - t0
    # 3. validate
    errs = validate(r, pallet, cfg)
    placed = sum(len(p.placements) for p in r.pallets)
    # 4. serialize
    t1 = time.perf_counter()
    js = to_json(r, pallet)
    blob = json.dumps(js)
    ser = time.perf_counter() - t1
    parsed = json.loads(blob)  # round-trip
    keys_ok = all(k in parsed for k in ("input_summary", "pallets", "unpacked_items"))
    item_ok = (len(parsed["pallets"]) == 0 or
               all(k in parsed["pallets"][0]["items"][0]
                   for k in ("item_id", "position", "dimensions", "orientation"))
               if parsed["pallets"] and parsed["pallets"][0]["items"] else True)
    print(f"\n[{label}] N={len(boxes)} budget={budget}s")
    print(f"  gate={gate}")
    print(f"  solve: wall={wall:.1f}s ({'<=budget' if wall<=budget*1.15 else 'OVERRUN +'+str(int(wall/budget*100-100))+'%'}) "
          f"pallets={len(r.pallets)} placed={placed} unpacked={len(r.unpacked)} validator_errs={len(errs)}")
    print(f"  serialize: {ser*1000:.0f}ms  json_bytes={len(blob):,}  roundtrip_ok={keys_ok and item_ok}")
    print(f"  conservation: {placed + len(r.unpacked) == len(boxes)}")
    ok = (len(errs) == 0 and keys_ok and item_ok and placed + len(r.unpacked) == len(boxes))
    return ok


def main():
    warmup_jit()
    rng = np.random.default_rng(42)
    print("=== API readiness: worst-case loads at the 500-box cap ===")
    a = run("A non-grouped 500 (common path)", *worst_case(500, rng, grouped=False), budget=90)
    b = run("B grouped 500 (LTL wrapper path)", *worst_case(500, np.random.default_rng(7), grouped=True), budget=90)
    print(f"\n=== READY: {'YES' if (a and b) else 'NO — see above'} ===")
    return 0 if (a and b) else 1


if __name__ == '__main__':
    raise SystemExit(main())
