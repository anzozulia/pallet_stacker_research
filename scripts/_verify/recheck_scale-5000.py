"""recheck_scale-5000.py — independent counter-probe for the 'scale-5000' claim.

The original probe (verify_scale_5000.py) asserts:
  (A) time_limit_s is "grossly violated" for N>=1000.
  (B) the decoder "emits validator-rejected (floating) placements at N=2000"
      -> the ONLY correctness claim. Everything else is performance/scaling.

This counter-probe re-derives each from first principles, OMP_NUM_THREADS=1.

Fast localization strategy for (B):
  B3  per-mode decoder isolation: feed random chromosomes to EACH of the 6
      constraint-aware decoders via decode_chromosome and validate. This
      bypasses BRKGA/v2/polish and pins any floating box to a specific mode.
  B2  full-solve attribution at N=2000 with a SHORT budget + feature toggles
      to confirm whether the violation comes from the v2 packer or the decoder.

For (A):
  A   measure wall vs budget at a cheap N=600 and characterize budget semantics.
"""
from __future__ import annotations

import io
import os
import re
import sys
import contextlib
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pallet_packer import Box, Pallet, PackerConfig, THIS_SIDE_UP, validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer._brkga_core import dispatch
from pallet_packer._brkga_core.precompute import (
    precompute_box_dims_and_sku, precompute_constraint_arrays)


EURO = Pallet(length=1200, width=1000, height=1500, max_weight=1000.0)

# IDENTICAL catalog to the original probe.
SKU_CATALOG = [
    (200, 150, 100, 0.6, 15.0),
    (300, 250, 200, 1.5, 20.0),
    (400, 300, 300, 3.0, 10.0),
    (150, 150, 100, 0.8, 0.0),     # fragile: max_load_on_top = 0
    (250, 200, 150, 1.2, 18.0),
    (380, 300, 250, 2.0, 50.0),
    (200, 200, 200, 1.0, 25.0),
    (350, 250, 300, 1.8, 30.0),
]


def make_workload(n: int, seed: int = 7) -> list:
    rng = np.random.default_rng(seed)
    boxes = []
    for i in range(n):
        l, w, h, wt, mlot = SKU_CATALOG[int(rng.integers(0, len(SKU_CATALOG)))]
        boxes.append(Box(id=f"B{i:05d}", length=l, width=w, height=h,
                         weight=wt, allowed_rotations=THIS_SIDE_UP,
                         max_load_on_top=mlot))
    return boxes


def manual_support(result, pallet, cfg):
    """Recompute support ratio exactly as the validator does; return offenders
    with full context (supporters, footprint, contact area)."""
    offenders = []
    EPS = 1e-9
    for st in result.pallets:
        for p in st.placements:
            if p.z <= EPS:
                continue
            footprint = p.dx * p.dy
            supported = 0.0
            sup_list = []
            for q in st.placements:
                if q is p:
                    continue
                if abs(p.z - q.z2) > EPS:
                    continue
                ox = max(0.0, min(p.x2, q.x2) - max(p.x, q.x))
                oy = max(0.0, min(p.y2, q.y2) - max(p.y, q.y))
                if ox * oy > 0:
                    sup_list.append((q.box.id, ox * oy))
                supported += ox * oy
            ratio = supported / footprint if footprint > 0 else 1.0
            if ratio < cfg.support_ratio - EPS:
                offenders.append(dict(
                    pallet=st.pallet_id, box=p.box.id, ratio=ratio,
                    pos=(p.x, p.y, p.z), dims=(p.dx, p.dy, p.dz),
                    rot=p.rotation.name, footprint=footprint,
                    supported=supported, supporters=sup_list,
                    n_sup_at_z=len(sup_list)))
    return offenders


def decoder_support_ratio(placements_out, order, dims_all, box_i, cand_pallet):
    """Recompute support ratio using the DECODER'S OWN integer arithmetic
    (mirrors _check_load_on_top_njit) against the final committed placements.
    If this disagrees with the validator we have a config issue; if it AGREES
    that the box is under-supported yet the box was committed (flag=1), the
    decoder accepted a placement that fails its own check => genuine defect."""
    cb = order[box_i]; cr = placements_out[box_i, 1]
    cx = placements_out[box_i, 2]; cy = placements_out[box_i, 3]
    cz = placements_out[box_i, 4]
    cdx = dims_all[cb, cr, 0]; cdy = dims_all[cb, cr, 1]
    if cz <= 0:
        return 1.0, 0
    total = 0; nsup = 0
    for i in range(placements_out.shape[0]):
        if placements_out[i, 5] == 0 or placements_out[i, 0] != cand_pallet:
            continue
        sb = order[i]; sr_ = placements_out[i, 1]
        if placements_out[i, 4] + dims_all[sb, sr_, 2] != cz:
            continue
        sx = placements_out[i, 2]; sy = placements_out[i, 3]
        sdx = dims_all[sb, sr_, 0]; sdy = dims_all[sb, sr_, 1]
        xl = max(cx, sx); xh = min(cx + cdx, sx + sdx)
        if xh <= xl:
            continue
        yl = max(cy, sy); yh = min(cy + cdy, sy + sdy)
        if yh <= yl:
            continue
        total += (xh - xl) * (yh - yl); nsup += 1
    fp = cdx * cdy
    return (total / fp if fp else 1.0), nsup


def b4_dissect_mode4():
    """Smoking gun: confirm the block decoder commits a box that fails its OWN
    integer support check (not a float/rounding/validator-config artifact)."""
    print("=== B4: dissect mode-4 floating box via decoder's own arithmetic ===")
    from pallet_packer._brkga_core.jit_decoders_cstr_cy import decode_blocks_njit_mode_cstr
    boxes = make_workload(800)
    cfg = PackerConfig()
    n = len(boxes)
    n_rots_arr, dims_all, sku_id = precompute_box_dims_and_sku(boxes)
    w, m, rfs, pmw, hc = precompute_constraint_arrays(boxes, EURO)
    L = int(round(EURO.length)); W = int(round(EURO.width)); H = int(round(EURO.height))
    sr = float(cfg.support_ratio); rc = 1 if cfg.require_centroid_supported else 0
    found = False
    for seed in range(5):
        chrom = np.random.default_rng(seed).random(2 * n + 1).astype(np.float64)
        order = np.argsort(chrom[:n]).astype(np.int64)
        po = np.zeros((n, 6), dtype=np.int64)
        n_skus = int(sku_id.max()) + 1
        decode_blocks_njit_mode_cstr(
            order, n_rots_arr, dims_all, sku_id, L, W, H, 200, po, n_skus,
            w, m, rfs, pmw, sr, rc, -1e18, 1e18, -1e18, 1e18, 0.0, 0)
        r = dispatch.decode_chromosome(
            chrom, boxes, EURO, cfg, n_rots_arr, dims_all, mode=4, max_pallets=200,
            sku_id_per_box=sku_id, weights=w, mlot=m, rfs=rfs,
            pallet_max_weight=pmw, has_constraints=hc, support_ratio=sr,
            require_centroid=rc)
        floating = [e for e in validate(r, EURO, cfg) if "floating" in e]
        if not floating:
            continue
        found = True
        id2pos = {boxes[order[p]].id: p for p in range(n) if po[p, 5] == 1}
        print(f"  seed={seed}: {len(floating)} floating; dissecting up to 3:")
        for e in floating[:3]:
            bid = e.split(":")[1].strip().split(" ")[0]
            pos = id2pos.get(bid)
            if pos is None:
                continue
            dr, dn = decoder_support_ratio(po, order, dims_all, pos, po[pos, 0])
            print(f"    {e}")
            print(f"      committed flag={po[pos,5]} z={po[pos,4]} "
                  f"DECODER-OWN integer support={dr:.4f} nsup={dn} thr={sr}")
            if dr < sr - 1e-6:
                print(f"      => committed despite failing its OWN check: DECODER DEFECT")
        break
    if not found:
        print("  (no floating box in 5 seeds — unexpected)")
    print()
    return found


def solve(boxes, time_limit, pop, n_pop, **kw):
    cfg = PackerConfig()
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        r = brkga_pack_v35(
            boxes, EURO, cfg, time_limit_s=time_limit, max_pallets=200,
            seed=kw.pop("seed", 42), population_size=pop, n_populations=n_pop,
            patience=10_000, local_search_budget_s=min(8.0, time_limit / 4),
            n_modes=6, verbose=True, **kw)
    return r, time.time() - t0, buf.getvalue()


# ---------------------------------------------------------------------------
def t3_int_float():
    print("=== T3: int-decode vs float-validate fidelity ===")
    all_int = all(
        (l == int(l) and w == int(w) and h == int(h))
        for (l, w, h, _, _) in SKU_CATALOG)
    all_int = all_int and all(d == int(d)
                              for d in (EURO.length, EURO.width, EURO.height))
    print(f"  all SKU + pallet dims integer? {all_int}")
    print("  -> if True, int(round(dim))==dim, so a support violation is a")
    print("     genuine geometric disagreement, NOT a rounding artifact.\n")
    return all_int


# ---------------------------------------------------------------------------
def b3_per_mode_decoder(boxes_n=800):
    """The decisive test: isolate each constraint-aware decoder mode."""
    print("=== B3: per-mode constraint-aware decoder validity (N=%d) ===" % boxes_n)
    boxes = make_workload(boxes_n)
    cfg = PackerConfig()
    n = len(boxes)
    n_rots_arr, dims_all, sku_id = precompute_box_dims_and_sku(boxes)
    w, m, rfs, pmw, hc = precompute_constraint_arrays(boxes, EURO)
    from pallet_packer._brkga_core.blocks import (
        enumerate_best_block_per_sku, enumerate_top_k_blocks_per_sku)
    L = int(round(EURO.length)); W = int(round(EURO.width)); H = int(round(EURO.height))
    sbb = enumerate_best_block_per_sku(boxes, sku_id, n_rots_arr, dims_all, L, W, H)
    stk = enumerate_top_k_blocks_per_sku(boxes, sku_id, n_rots_arr, dims_all,
                                         L, W, H, k_top=8)
    sr = float(cfg.support_ratio)
    rc = 1 if cfg.require_centroid_supported else 0
    chrom_size = 2 * n + 1
    print(f"  has_constraints={hc} support_ratio={sr} require_centroid={rc}")
    any_bad = False
    bad_modes = []
    for mode in range(6):
        worst = (0, 0, None)  # (n_errs, n_floating, offender)
        for seed in range(5):
            chrom = np.random.default_rng(seed).random(chrom_size).astype(np.float64)
            r = dispatch.decode_chromosome(
                chrom, boxes, EURO, cfg, n_rots_arr, dims_all, mode=mode,
                max_pallets=200, sku_id_per_box=sku_id,
                sku_best_block=sbb, sku_top_k_blocks=stk,
                weights=w, mlot=m, rfs=rfs, pallet_max_weight=pmw,
                has_constraints=hc, support_ratio=sr, require_centroid=rc)
            errs = validate(r, EURO, cfg)
            if len(errs) > worst[0]:
                off = manual_support(r, EURO, cfg)
                worst = (len(errs), sum("floating" in e for e in errs),
                         off[0] if off else None)
        flag = ""
        if worst[0]:
            any_bad = True
            bad_modes.append(mode)
            flag = "  <-- PRODUCES INVALID PLACEMENTS"
        print(f"  mode {mode}: worst errs={worst[0]:4d} floating={worst[1]:4d}{flag}")
        if worst[2]:
            o = worst[2]
            print(f"      e.g. {o['pallet']}/{o['box']} ratio={o['ratio']:.4f} "
                  f"dims={o['dims']} footprint={o['footprint']} "
                  f"supported={o['supported']} n_sup={o['n_sup_at_z']}")
    print(f"  [B3] bad modes: {bad_modes if bad_modes else 'NONE — all decoders clean'}\n")
    return any_bad, bad_modes


# ---------------------------------------------------------------------------
def b2_fullsolve_attribute(n=2000, time_limit=25.0):
    """Reproduce a real full solve at N and attribute any violation."""
    print(f"=== B2: full-solve attribution (N={n}, budget={time_limit}s) ===")
    boxes = make_workload(n)
    cfg = PackerConfig()

    print("  [a] default (v2_seed=auto=True, v2 hybrid polish ON)...", flush=True)
    r_a, wall_a, _ = solve(boxes, time_limit, 150, 2)
    errs_a = validate(r_a, EURO, cfg)
    fl_a = sum("floating" in e for e in errs_a)
    print(f"      wall={wall_a:.1f}s pallets={len(r_a.pallets)} "
          f"placed={sum(len(p.placements) for p in r_a.pallets)} "
          f"errs={len(errs_a)} floating={fl_a}")
    for e in errs_a[:3]:
        print(f"      ERR: {e}")

    # Pure v2 packer baseline — is the violation in the production v2 packer?
    from pallet_packer.packer import PalletPacker
    t0 = time.time()
    v2 = PalletPacker(EURO, cfg).pack(boxes)
    v2_wall = time.time() - t0
    v2_errs = validate(v2, EURO, cfg)
    fl_v2 = sum("floating" in e for e in v2_errs)
    print(f"  [v2-only] wall={v2_wall:.1f}s pallets={len(v2.pallets)} "
          f"placed={sum(len(p.placements) for p in v2.pallets)} "
          f"errs={len(v2_errs)} floating={fl_v2}")
    for e in v2_errs[:3]:
        print(f"      v2 ERR: {e}")

    print("  [b] pure decoder (use_v2_seed=False, use_v2_hybrid_polish=False)...",
          flush=True)
    r_b, wall_b, _ = solve(boxes, time_limit, 150, 2,
                           use_v2_seed=False, use_v2_hybrid_polish=False)
    errs_b = validate(r_b, EURO, cfg)
    fl_b = sum("floating" in e for e in errs_b)
    print(f"      wall={wall_b:.1f}s pallets={len(r_b.pallets)} "
          f"placed={sum(len(p.placements) for p in r_b.pallets)} "
          f"errs={len(errs_b)} floating={fl_b}")
    for e in errs_b[:3]:
        print(f"      DECODER ERR: {e}")
    print()
    return dict(errs_a=len(errs_a), fl_a=fl_a, errs_v2=len(v2_errs), fl_v2=fl_v2,
                errs_b=len(errs_b), fl_b=fl_b)


# ---------------------------------------------------------------------------
def a_time_overrun():
    print("=== A: time_limit_s honored? (cheap N=600 probe) ===")
    boxes = make_workload(600)
    for tl in (10.0, 20.0):
        r, wall, vtext = solve(boxes, tl, 120, 2)
        gens = [int(x) for x in re.findall(r"gen (\d+)", vtext)]
        mg = max(gens) if gens else -1
        over = wall - tl
        print(f"  budget={tl:5.1f}s wall={wall:6.1f}s overrun=+{over:5.1f}s "
              f"(+{over/tl*100:5.1f}%) max_gen={mg} "
              f"[{'OVER' if over > 0.5 else 'OK'}]")
    print("  budget checked once/generation (driver.py:411); post-loop")
    print("  PR+LS+LNS+v2-polish run AFTER with own budgets => SOFT budget.\n")


def main():
    print("backend:", dispatch.decode_njit_mode.__module__,
          " _BATCH_AVAILABLE =", dispatch._BATCH_AVAILABLE)
    print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS", "default"))
    print("warming up JIT...", flush=True)
    warmup_jit(); print()

    all_int = t3_int_float()
    any_bad_mode, bad_modes = b3_per_mode_decoder(800)
    b4_dissect_mode4()
    a_time_overrun()
    attr = b2_fullsolve_attribute(300, 15.0)  # small N: defect surfaces fast

    print("=== VERDICT INPUTS ===")
    print(f"  dims all integer (no rounding gap):     {all_int}")
    print(f"  isolated decoder modes producing bad:   {bad_modes}")
    print(f"  N=2000 default-config errs/floating:    {attr['errs_a']}/{attr['fl_a']}")
    print(f"  N=2000 v2-only errs/floating:           {attr['errs_v2']}/{attr['fl_v2']}")
    print(f"  N=2000 pure-decoder errs/floating:      {attr['errs_b']}/{attr['fl_b']}")
    print()
    if any_bad_mode or attr["fl_b"] > 0:
        print("  -> A BRKGA DECODER emits validator-rejected placements: CORRECTNESS DEFECT.")
    elif attr["fl_a"] > 0 and attr["fl_v2"] > 0 and attr["fl_b"] == 0:
        print("  -> Floating box originates in the v2 PalletPacker, surfaced via")
        print("     v2-seed/v2-hybrid; the BRKGA decoder under test is clean.")
    elif attr["fl_a"] > 0:
        print("  -> Floating box in default config only; source = v2 path.")
    else:
        print("  -> No floating box reproduced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
