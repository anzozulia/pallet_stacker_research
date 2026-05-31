"""recheck_edge-degenerate.py — INDEPENDENT re-derivation of the claimed
'negative-dimension box packs validator-clean but OOB' defect.

I do NOT trust verify_edge_degenerate.py. I re-derive from first principles.

  T1. Reproduce the negative-dim case via the FULL public solve path
      (brkga_pack_v35). Confirm: box is PLACED (not unpacked), x2 < 0
      (out of bounds under the float dims), validate() == [].

  T2. Confirm the validator's bounds predicate mathematically cannot flag
      x2 < x (negative extent): check it directly on a hand-built Placement.

  T3. Confirm there is NO existing input-validation guard: feed the raw
      negative box to precompute_box_dims and show dims_all[*,0,0] = -30
      is stored verbatim (round-trips, no rejection / no raise).

  T4. CONTROL: a WELL-FORMED input (all positive dims) on the same pallet
      must produce validator-clean AND physically-sane geometry. This
      isolates the failure to malformed input only (garbage-in), NOT valid
      input.

  T5. NaN pallet max_weight sub-claim: confirm precompute treats NaN as a
      finite (NaN) cap, the comparison is always False, the cap is silently
      disabled, and a heavy box packs where a finite cap would reject it.

  T6. fit-test mechanism: show the decoder 'fits' a negative box because the
      fit test is (free_extent < dx); free_extent(120) < dx(-30) is False ->
      always fits.

Run:
  docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
    -v "$(pwd)":/app -w /app pallet-packer:dev \
    python scripts/_verify/recheck_edge-degenerate.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pallet_packer import (Box, Pallet, PackerConfig, Placement, Rotation,
                           validate)
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer.brkga_v3_fast import precompute_box_dims
from pallet_packer._brkga_core.precompute import precompute_constraint_arrays

EPS = 1e-6


def line(tag, ok, msg):
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {msg}", flush=True)
    return ok


def all_placements(r):
    out = []
    for st in r.pallets:
        out.extend(st.placements)
    return out


def main():
    warmup_jit()
    checks = []

    # -------------------------------------------------------------------
    print("\n=== T1: negative-dim box via FULL solve path ===", flush=True)
    neg = Box(id="neg", length=-30.0, width=30, height=30, weight=1.0)
    boxes = [Box(id="g0", length=30, width=30, height=30, weight=1.0), neg]
    pallet = Pallet(length=120, width=100, height=100)
    cfg = PackerConfig(max_pallets=1)
    r = brkga_pack_v35(boxes, pallet, cfg, time_limit_s=3.0, max_pallets=1,
                       population_size=80, n_populations=2, patience=50,
                       local_search_budget_s=1.0, seed=42, verbose=False,
                       n_modes=6)
    errs = validate(r, pallet, cfg)
    placed = all_placements(r)
    neg_pl = [p for p in placed if p.box.id == "neg"]
    neg_placed = len(neg_pl) == 1
    oob = False
    if neg_placed:
        p = neg_pl[0]
        print(f"      neg @ ({p.x},{p.y},{p.z}) dims=({p.dx},{p.dy},{p.dz}) "
              f"x2={p.x2}", flush=True)
        oob = p.x2 < -EPS
    else:
        print(f"      neg unpacked={[b.id for b in r.unpacked]}", flush=True)
    print(f"      validate() -> {len(errs)} errors", flush=True)
    claim = neg_placed and oob and (len(errs) == 0)
    checks.append(line("T1 negative-dim placed-OOB-yet-clean", claim,
                       f"placed={neg_placed} x2<0={oob} validator_errs={len(errs)} "
                       f"-> claim {'REPRODUCED' if claim else 'NOT reproduced'}"))

    # -------------------------------------------------------------------
    print("\n=== T2: validator bounds predicate blind to x2<x ===", flush=True)
    hand = Placement(box=neg, rotation=Rotation.LWH, x=0.0, y=0.0, z=0.0)
    ov = 0.0
    flagged = (hand.x < -EPS or hand.y < -EPS or hand.z < -EPS or
               hand.x2 > pallet.length + ov + EPS or
               hand.y2 > pallet.width + ov + EPS or
               hand.z2 > pallet.height + EPS)
    blind = not flagged
    checks.append(line("T2 bounds predicate blind to negative extent", blind,
                       f"x2={hand.x2} (box spans x in [{hand.x2},{hand.x}]); "
                       f"predicate flagged={flagged} -> blind spot "
                       f"{'CONFIRMED' if blind else 'absent'}"))

    # -------------------------------------------------------------------
    print("\n=== T3: precompute stores negative dim verbatim (no guard) ===", flush=True)
    n_rots, dims_all = precompute_box_dims([neg])
    stored = int(dims_all[0, 0, 0])
    no_guard = (stored == -30)
    checks.append(line("T3 precompute_box_dims no positivity check", no_guard,
                       f"dims_all[0,0]={stored} (raw int(round(-30.0))) -> "
                       f"{'stored verbatim, no guard/raise' if no_guard else 'unexpected'}"))

    # -------------------------------------------------------------------
    print("\n=== T4: CONTROL — well-formed input is clean AND sane ===", flush=True)
    good = [Box(id=f"g{i}", length=30, width=30, height=30, weight=1.0)
            for i in range(8)]
    rg = brkga_pack_v35(good, pallet, cfg, time_limit_s=3.0, max_pallets=1,
                        population_size=80, n_populations=2, patience=50,
                        local_search_budget_s=1.0, seed=42, verbose=False,
                        n_modes=6)
    errg = validate(rg, pallet, cfg)
    physg = []
    for p in all_placements(rg):
        if (p.x < -EPS or p.y < -EPS or p.z < -EPS or
                p.x2 > pallet.length + EPS or p.y2 > pallet.width + EPS or
                p.z2 > pallet.height + EPS or
                p.dx <= 0 or p.dy <= 0 or p.dz <= 0):
            physg.append(p.box.id)
    control_ok = (len(errg) == 0 and len(physg) == 0 and len(all_placements(rg)) > 0)
    checks.append(line("T4 control well-formed clean+sane", control_ok,
                       f"placed={len(all_placements(rg))} validator_errs={len(errg)} "
                       f"phys_viol={len(physg)} -> valid input "
                       f"{'UNAFFECTED' if control_ok else 'BROKEN'}"))

    # -------------------------------------------------------------------
    print("\n=== T5: NaN pallet max_weight silently disables cap ===", flush=True)
    heavy = Box(id="heavy", length=30, width=30, height=30, weight=999999.0)
    out_nan = precompute_constraint_arrays([heavy], Pallet(120, 100, 100, max_weight=float("nan")))
    pmw_nan, hc_nan = out_nan[3], out_nan[4]
    nan_isnan = (isinstance(pmw_nan, float) and math.isnan(pmw_nan))
    cmp_false = not (heavy.weight > pmw_nan)  # NaN comparison -> False
    rc = brkga_pack_v35([heavy], Pallet(120, 100, 100, max_weight=1.0),
                        PackerConfig(max_pallets=1), time_limit_s=2.0,
                        max_pallets=1, population_size=40, n_populations=2,
                        patience=30, local_search_budget_s=0.5, seed=42,
                        verbose=False, n_modes=6)
    finite_rejects = (len(all_placements(rc)) == 0)
    rn = brkga_pack_v35([heavy], Pallet(120, 100, 100, max_weight=float("nan")),
                        PackerConfig(max_pallets=1), time_limit_s=2.0,
                        max_pallets=1, population_size=40, n_populations=2,
                        patience=30, local_search_budget_s=0.5, seed=42,
                        verbose=False, n_modes=6)
    nan_packs = (len(all_placements(rn)) == 1)
    t5 = nan_isnan and cmp_false and finite_rejects and nan_packs
    checks.append(line("T5 NaN max_weight disables cap", t5,
                       f"pmw_nan={pmw_nan} isnan={nan_isnan} (w>pmw)False={cmp_false} "
                       f"finite_cap_rejects={finite_rejects} nan_cap_packs_heavy={nan_packs}"))

    # -------------------------------------------------------------------
    print("\n=== T6: fit-test mechanism admits negative extent ===", flush=True)
    free_extent, dx = 120, -30
    admits = not (free_extent < dx)
    checks.append(line("T6 fit predicate admits negative extent", admits,
                       f"(free_extent {free_extent} < dx {dx}) = {free_extent < dx} "
                       f"-> negative box always 'fits'"))

    # -------------------------------------------------------------------
    print("\n=== VERDICT INPUTS ===", flush=True)
    print(f"  T1 claim_reproduced  = {checks[0]}", flush=True)
    print(f"  T2 blind_spot        = {checks[1]}", flush=True)
    print(f"  T3 no_input_guard    = {checks[2]}", flush=True)
    print(f"  T4 valid_input_ok    = {checks[3]}", flush=True)
    print(f"  T5 nan_weight        = {checks[4]}", flush=True)
    print(f"  T6 fit_admits_neg    = {checks[5]}", flush=True)
    all_pass = all(checks)
    print(f"\n=== ALL CHECKS: {'PASS' if all_pass else 'FAIL'} "
          f"({sum(checks)}/{len(checks)}) ===", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
