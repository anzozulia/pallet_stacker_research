"""recheck_edge-geom.py — INDEPENDENT counter-probe for the claimed
'support_ratio silently not enforced on geometric workloads' defect.

Re-derives from first principles. Does NOT trust verify_edge_geom.py.

Tests:
  A. Confirm config + precompute structural facts:
       - geometric_only_config.support_ratio == 1.0
       - precompute_constraint_arrays(BR1#1) -> has_constraints == False
       - driver therefore passes support_ratio=0.0 to decoder
  B. Minimal decode repro (n=5): build it independently, then MANUALLY
     recompute the support ratio of the offending box from the placement
     coordinates, and confirm the validator's number matches my hand-calc.
     i.e. the floating is REAL geometry, not a validator EPS quirk.
  C. Headline-benchmark check: run the ACTUAL driver (brkga_pack_v35) on
     real BR1 instances with the project's own geometric_only_config, then
     validate(). Count how many instances are validator-DIRTY. This is the
     load-bearing claim ('headline BR results are NOT validator-clean').
  D. Counterfactual: if I FORCE support_ratio enforcement in the decode path
     (pass has_constraints=True with a finite weight so the cstr/geom path
     enforces support=1.0), does the SAME chromosome produce a validator-clean
     packing? This isolates 'decoder can honor support' from 'driver chooses
     not to'.
  E. Is validator support semantics sane? Build a KNOWN-good fully-supported
     stack by hand and confirm validate() passes it (no false positives), and
     build a KNOWN floating box and confirm it flags it. Guards against the
     'probe / validator is wrong' hypothesis.

Run:
  docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
    -v "$(pwd)":/app -w /app pallet-packer:dev \
    python scripts/_verify/recheck_edge-geom.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from pallet_packer import (
    Box, Pallet, PackerConfig, Rotation, NO_ROTATION, validate,
)
from pallet_packer.packer import PackResult, PalletState
from pallet_packer.models import Placement
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer._brkga_core.dispatch import decode_chromosome
from pallet_packer._brkga_core.precompute import (
    precompute_box_dims_and_sku, precompute_constraint_arrays,
)
from benchmarks.br import geometric_only_config, parse_thpack, first_pallet_utilization


def manual_support_ratio(target, placements):
    """Independently recompute support ratio for `target` placement from
    coordinates only (mirror of validator section 4, written fresh)."""
    if target.z <= 1e-9:
        return 1.0, "on floor"
    footprint = target.dx * target.dy
    if footprint <= 0:
        return 1.0, "zero footprint"
    supported = 0.0
    contributors = []
    for q in placements:
        if q is target:
            continue
        # supporter top must coincide with target bottom
        if abs(target.z - q.z2) > 1e-6:
            continue
        ox = max(0.0, min(target.x2, q.x2) - max(target.x, q.x))
        oy = max(0.0, min(target.y2, q.y2) - max(target.y, q.y))
        if ox * oy > 0:
            supported += ox * oy
            contributors.append((q.box.id, ox * oy))
    return supported / footprint, contributors


def main():
    print("=== RECHECK edge-geom: support_ratio enforcement ===")
    warmup_jit()
    EPS = 1e-6

    # ---- A. structural facts -------------------------------------------
    print("\n[A] Structural facts")
    cfg = geometric_only_config(max_pallets=1)
    print(f"  geometric_only_config.support_ratio = {cfg.support_ratio}")
    assert cfg.support_ratio == 1.0, "config support_ratio not 1.0"

    insts = parse_thpack("benchmarks/data/thpack1.txt", set_id="BR1")
    br1_1 = insts[0]
    w, mlot, rfs, pmw, has_c = precompute_constraint_arrays(br1_1.boxes, br1_1.pallet)
    print(f"  BR1#1: has_constraints={has_c}  pallet.max_weight={getattr(br1_1.pallet,'max_weight',None)}")
    print(f"  -> driver support_ratio_value = "
          f"{float(cfg.support_ratio) if has_c else 0.0} (config={cfg.support_ratio})")
    structural_ok = (cfg.support_ratio == 1.0 and has_c is False)
    print(f"  [{'CONFIRM' if structural_ok else 'REFUTE'}] config wants 1.0 but driver passes 0.0 on BR")

    # ---- E. validator self-consistency (guard against probe/validator bug) ---
    print("\n[E] Validator support semantics sanity (no false pos/neg)")
    pal = Pallet(length=400, width=300, height=400)
    bA = Box(id="A", length=200, width=150, height=100)
    bB = Box(id="B", length=200, width=150, height=100)
    # Fully supported: B directly on top of A, same footprint
    st_good = PalletState(pallet=pal, pallet_id="P0", config=cfg)
    st_good.placements = [
        Placement(box=bA, rotation=Rotation.LWH, x=0, y=0, z=0),
        Placement(box=bB, rotation=Rotation.LWH, x=0, y=0, z=100),
    ]
    good = PackResult(pallets=[st_good], unpacked=[])
    e_good = validate(good, pal, cfg)
    sr_good, _ = manual_support_ratio(good.pallets[0].placements[1], good.pallets[0].placements)
    print(f"  fully-supported stack: validator_errors={len(e_good)} manual_sr={sr_good:.3f}")
    # Floating: B shifted so only half rests on A
    st_float = PalletState(pallet=pal, pallet_id="P0", config=cfg)
    st_float.placements = [
        Placement(box=bA, rotation=Rotation.LWH, x=0, y=0, z=0),
        Placement(box=bB, rotation=Rotation.LWH, x=100, y=0, z=100),  # 50% overhang in x
    ]
    floaty = PackResult(pallets=[st_float], unpacked=[])
    e_float = validate(floaty, pal, cfg)
    sr_float, contrib = manual_support_ratio(floaty.pallets[0].placements[1], floaty.pallets[0].placements)
    print(f"  half-overhang stack:   validator_errors={len(e_float)} manual_sr={sr_float:.3f} contrib={contrib}")
    validator_sane = (len(e_good) == 0 and len(e_float) >= 1 and abs(sr_float - 0.5) < 1e-6)
    print(f"  [{'OK' if validator_sane else 'BAD'}] validator: clean on supported, flags floating, "
          f"sr matches hand-calc")

    # ---- B. minimal decode repro, independently rebuilt -----------------
    print("\n[B] Minimal decode repro (n=5, seed=0, mode=0)")
    pal_b = Pallet(length=400, width=300, height=400)
    boxes_b = [Box(id=f"B{i}", length=200, width=150, height=100) for i in range(5)]
    nr, da, sku = precompute_box_dims_and_sku(boxes_b)
    w2, mlot2, rfs2, pmw2, hc2 = precompute_constraint_arrays(boxes_b, pal_b)
    chrom = np.random.default_rng(0).random(2 * 5 + 1)
    rB = decode_chromosome(chrom, boxes_b, pal_b, cfg, nr, da, 0,
                           max_pallets=1, sku_id_per_box=sku,
                           weights=w2, mlot=mlot2, rfs=rfs2,
                           pallet_max_weight=pmw2, has_constraints=hc2,
                           support_ratio=0.0)  # mirrors driver on geometric
    eB = validate(rB, pal_b, cfg)
    placements_b = [p for st in rB.pallets for p in st.placements]
    print(f"  placed={len(placements_b)} validator_errors={len(eB)}")
    worst = None
    for p in placements_b:
        sr, contrib = manual_support_ratio(p, placements_b)
        if sr < cfg.support_ratio - EPS:
            print(f"    FLOAT {p.box.id} pos=({p.x},{p.y},{p.z}) dims=({p.dx},{p.dy},{p.dz}) "
                  f"manual_sr={sr:.4f} supporters={contrib}")
            if worst is None or sr < worst:
                worst = sr
    repro_b = (len(eB) >= 1 and worst is not None)
    print(f"  [{'REPRO' if repro_b else 'no-repro'}] minimal decode floats; "
          f"worst hand-calc sr={worst}")

    # ---- D. counterfactual: can the decoder HONOR support if asked? -----
    print("\n[D] Counterfactual: force support enforcement (has_constraints=True)")
    # Give boxes a finite weight + pallet a finite cap so has_constraints flips True.
    boxes_d = [Box(id=f"B{i}", length=200, width=150, height=100, weight=1.0) for i in range(5)]
    pal_d = Pallet(length=400, width=300, height=400, max_weight=1000.0)
    nrd, dad, skud = precompute_box_dims_and_sku(boxes_d)
    wd, mlotd, rfsd, pmwd, hcd = precompute_constraint_arrays(boxes_d, pal_d)
    print(f"  has_constraints now = {hcd} -> driver would pass support_ratio={cfg.support_ratio}")
    rD = decode_chromosome(chrom, boxes_d, pal_d, cfg, nrd, dad, 0,
                           max_pallets=1, sku_id_per_box=skud,
                           weights=wd, mlot=mlotd, rfs=rfsd,
                           pallet_max_weight=pmwd, has_constraints=hcd,
                           support_ratio=float(cfg.support_ratio))  # 1.0
    # validate with a weight-aware cfg copy that still has support_ratio=1.0
    eD = validate(rD, pal_d, cfg)
    nplaced_d = sum(len(st.placements) for st in rD.pallets)
    print(f"  same chrom, support enforced -> placed={nplaced_d} validator_errors={len(eD)}")
    counterfactual_ok = (len(eD) == 0)
    print(f"  [{'OK' if counterfactual_ok else 'STILL-DIRTY'}] enforcing support yields clean pack")

    # ---- C. THE HEADLINE CLAIM: full driver on real BR1 ----------------
    print("\n[C] Headline: brkga_pack_v35 on real BR1 (n=10) with geometric_only_config")
    dirty = 0
    total_errs = 0
    worst_inst = (None, 0)
    floating_only = True   # are ALL errors floating/support? (vs overlap/bounds)
    for inst in insts[:10]:
        r = brkga_pack_v35(
            inst.boxes, inst.pallet, cfg,
            time_limit_s=3.0, max_pallets=1,
            population_size=120, n_populations=2,
            patience=60, local_search_budget_s=1.0,
            seed=42 + inst.instance_id, verbose=False, n_modes=6,
        )
        errs = validate(r, inst.pallet, cfg)
        util = first_pallet_utilization(r, inst.pallet)
        for e in errs:
            if "floating" not in e and "support" not in e:
                floating_only = False
        if errs:
            dirty += 1
            total_errs += len(errs)
            if len(errs) > worst_inst[1]:
                worst_inst = (inst.instance_id, len(errs))
        print(f"  BR1#{inst.instance_id:>2} N={len(inst.boxes):>4d} util={util*100:5.2f}% "
              f"errs={len(errs):>2d}{'  <-- example: '+errs[0] if errs else ''}")
    print(f"\n  BR1 summary: {dirty}/10 instances validator-DIRTY, "
          f"{total_errs} total errors, worst=BR1#{worst_inst[0]} ({worst_inst[1]})")
    print(f"  all errors are floating/support-type? {floating_only}")
    headline_dirty = (dirty >= 1)
    print(f"  [{'CONFIRM' if headline_dirty else 'REFUTE'}] headline BR1 produces "
          f"validator-rejected (floating) placements under shipped geometric_only_config")

    # ---- VERDICT SUMMARY ------------------------------------------------
    print("\n" + "=" * 64)
    print("=== RECHECK VERDICT INPUTS ===")
    print(f"  A structural (config 1.0, driver 0.0 on BR): {structural_ok}")
    print(f"  E validator self-consistent (no false pos/neg): {validator_sane}")
    print(f"  B minimal decode floats per hand-calc:          {repro_b}")
    print(f"  D enforcing support -> clean (decoder CAN honor): {counterfactual_ok}")
    print(f"  C HEADLINE driver BR1 dirty count:              {dirty}/10")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
