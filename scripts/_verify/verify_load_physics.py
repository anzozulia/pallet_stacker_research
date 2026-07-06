"""
verify_load_physics.py — randomized decode-level PHYSICS gate.

Backend equivalence (verify_backend_equiv_cstr.py) proves the Cython and
Numba twins agree — it is STRUCTURALLY BLIND to bugs where both twins are
identically wrong (they are kept bit-equivalent on purpose). Hardening
round 3 found exactly that class live (docs/reports/37): the block
decoder committed jointly-overloading sibling columns (F20) and late
"under-fill" placements inherited unchecked load (F21). This gate closes
the hole: thousands of raw constraint decodes across all 6 modes, every
output re-validated against an EXACT independent physics oracle:

  - AABB overlap, container bounds (one-sided overhang), height
  - support contact >= eff_sr * footprint at z>0 (and contact > 0)
  - floor deck-contact rule when overhang > 0 (F17)
  - floor centroid-over-deck when overhang > 0 AND the config requires
    centroid support (round 6, F30 — the toppling rule)
  - load bearing: DIRECT model when the transitive flag is off (the
    historical contract), TRANSITIVE physics when on (F19)
  - pallet weight cap

Three sweeps (seeds fixed — A/B are the EXACT instances that exposed
F20/F21 pre-fix: 51/7200 and 23/7200 violating decodes):
  sweep A: mixed settings   (seed 20260703, 150 inst x 8 chrom x 6 modes)
  sweep B: service defaults (seed 777,      200 inst x 6 chrom x 6 modes)
  sweep C: toppling regime  (seed 20260704, 120 inst x 6 chrom x 6 modes;
           low sr + full overhang — the F30 family, round 6)
plus the deterministic round-3 regression instances.

PASS requires ZERO violations. Run (from repo root):
  docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
    -v "$(pwd)":/app -w /app pallet-packer:dev \
    python scripts/_verify/verify_load_physics.py
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import numpy as np

from pallet_packer.models import Box, Pallet, PackerConfig, ALL_ROTATIONS
import pallet_packer._brkga_core.driver as drv
from pallet_packer._brkga_core.dispatch import decode_chromosome

EPS_A = 1e-6
MODES = [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Instance generators (seeds and draw order are LOAD-BEARING: they reproduce
# the round-3 finding instances — do not reorder the rng draws)
# ---------------------------------------------------------------------------

def rand_instance_mixed(rng: random.Random):
    L = rng.choice([300, 400, 600, 800])
    W = rng.choice([300, 400, 600, 800])
    H = 1200
    ov = rng.choice([0, 0, 100, min(L, W) // 2, min(L, W)])
    sr = rng.choice([0.0, 0.5, 0.8, 1.0])
    transitive = rng.random() < 0.5
    max_pallets = rng.choice([1, 1, 2])
    pmw = rng.choice([float("inf"), float("inf"), 60.0])
    n = rng.randint(4, 14)
    boxes = []
    for i in range(n):
        d = [rng.choice([50, 80, 100, 150, 200, 300, 400]) for _ in range(3)]
        w = rng.choice([0.0, 1.0, 2.5, 5.0, 10.0, 20.0])
        r = rng.random()
        if r < 0.30:
            mlot = float("inf")
        elif r < 0.45:
            mlot = 0.0
        else:
            mlot = rng.uniform(0.5, 3.0) * max(w, 1.0)
        boxes.append(Box(
            id=f"b{i}", length=d[0], width=d[1], height=d[2], weight=w,
            max_load_on_top=mlot,
            allowed_rotations=list(ALL_ROTATIONS),
            requires_full_support=(rng.random() < 0.2)))
    pallet = Pallet(length=L, width=W, height=H, max_weight=pmw,
                    max_overhang=float(ov))
    cfg = PackerConfig(
        support_ratio=sr,
        require_centroid_supported=(rng.random() < 0.5),
        allow_pallet_overhang=(ov > 0),
        enforce_load_bearing=True,
        transitive_load_bearing=transitive,
        seed=rng.randint(0, 10**6))
    return boxes, pallet, cfg, ov, sr, transitive, max_pallets


def rand_instance_topple(rng: random.Random):
    """Round 6 (F30): the toppling regime — low support_ratio + a big
    overhang + require_centroid ON. Boxes are sized near or above the deck
    so overhanging floor rows are forced; pre-fix this family shipped
    validate-clean floor boxes with their centroid past the deck edge in
    105/120 solves. NEW generator on purpose: the existing generators'
    draw order is load-bearing and must not change."""
    L = rng.choice([300, 400, 500])
    W = rng.choice([300, 400, 500])
    H = 1200
    ov = min(L, W)
    sr = rng.choice([0.0, 0.2, 0.3, 0.4])
    transitive = rng.random() < 0.5
    n = rng.randint(3, 10)
    boxes = []
    for i in range(n):
        dl = rng.randint(int(L * 0.6), L)
        dw = rng.randint(int(W * 0.6), W)
        dh = rng.choice([100, 150, 200, 300])
        w = rng.choice([0.0, 1.0, 5.0, 10.0, 20.0])
        mlot = rng.choice([float("inf"), float("inf"), 0.0, 2.0 * max(w, 1.0)])
        boxes.append(Box(id=f"b{i}", length=dl, width=dw, height=dh,
                         weight=w, max_load_on_top=mlot,
                         allowed_rotations=list(ALL_ROTATIONS),
                         requires_full_support=False))
    pallet = Pallet(length=L, width=W, height=H, max_overhang=float(ov))
    cfg = PackerConfig(support_ratio=sr, require_centroid_supported=True,
                       allow_pallet_overhang=True,
                       enforce_load_bearing=True,
                       transitive_load_bearing=transitive,
                       seed=rng.randint(0, 10**6))
    return boxes, pallet, cfg, ov, sr, transitive, 1


def rand_instance_prod(rng: random.Random):
    L = rng.choice([800, 1000, 1200])
    W = rng.choice([600, 800, 1000])
    H = 1500
    ov = rng.choice([0, 0, 0, 200])
    sr = 0.8
    transitive = True
    max_pallets = rng.choice([1, 2])
    n_types = rng.randint(2, 4)
    types = []
    for _ in range(n_types):
        d = [rng.choice([100, 150, 200, 250, 300, 400]) for _ in range(3)]
        w = rng.choice([2.0, 5.0, 8.0, 12.0, 20.0])
        r = rng.random()
        if r < 0.25:
            mlot = float("inf")
        elif r < 0.40:
            mlot = 0.0
        else:
            mlot = rng.uniform(0.8, 3.0) * w
        types.append((d, w, mlot))
    n = rng.randint(8, 20)
    boxes = []
    for i in range(n):
        d, w, mlot = types[rng.randrange(n_types)]
        boxes.append(Box(id=f"b{i}", length=d[0], width=d[1], height=d[2],
                         weight=w, max_load_on_top=mlot,
                         allowed_rotations=list(ALL_ROTATIONS),
                         requires_full_support=False))
    pallet = Pallet(length=L, width=W, height=H, max_overhang=float(ov))
    cfg = PackerConfig(support_ratio=sr, require_centroid_supported=True,
                       allow_pallet_overhang=(ov > 0),
                       enforce_load_bearing=True,
                       transitive_load_bearing=transitive,
                       seed=rng.randint(0, 10**6))
    return boxes, pallet, cfg, ov, sr, transitive, max_pallets


# ---------------------------------------------------------------------------
# Exact physics oracle
# ---------------------------------------------------------------------------

def oracle(res, boxes, pallet, ov, sr, transitive, require_centroid=False):
    """Return violation strings for one PackResult.

    require_centroid (round 6, F30): the floor centroid-over-deck rule is
    gated on the config's require_centroid_supported — an rc=0 decode may
    legally place an overhung floor box with its centroid past the deck
    edge, so the criterion must NOT fire there (sweep A randomises rc).
    """
    v = []
    L, W, H = pallet.length, pallet.width, pallet.height
    for pal in res.pallets:
        geo = [(p.x, p.y, p.z, p.dx, p.dy, p.dz, p.box) for p in pal.placements]
        for (x, y, z, dx, dy, dz, b) in geo:
            if x < -1e-9 or y < -1e-9 or z < -1e-9:
                v.append(f"NEG_COORD {b.id}")
            if x + dx > L + ov + 1e-6 or y + dy > W + ov + 1e-6:
                v.append(f"OOB {b.id}")
            if z + dz > H + 1e-6:
                v.append(f"TOO_TALL {b.id}")
        for i in range(len(geo)):
            xi, yi, zi, dxi, dyi, dzi, bi = geo[i]
            for j in range(i + 1, len(geo)):
                xj, yj, zj, dxj, dyj, dzj, bj = geo[j]
                if (min(xi + dxi, xj + dxj) - max(xi, xj) > 1e-6 and
                        min(yi + dyi, yj + dyj) - max(yi, yj) > 1e-6 and
                        min(zi + dzi, zj + dzj) - max(zi, zj) > 1e-6):
                    v.append(f"OVERLAP {bi.id}x{bj.id}")
        for (x, y, z, dx, dy, dz, b) in geo:
            fp = dx * dy
            eff = 1.0 if b.requires_full_support else sr
            if z <= 1e-9:
                cx = max(0.0, min(x + dx, L) - max(x, 0.0))
                cy = max(0.0, min(y + dy, W) - max(y, 0.0))
                deck = cx * cy
                if ov > 0 and deck < eff * fp - EPS_A - 1e-9 * fp:
                    v.append(f"DECK {b.id}")
                if deck <= 0 and ov > 0:
                    v.append(f"OFF_DECK {b.id}")
                # Round 6 (F30): footprint centroid over the deck-contact
                # rectangle, else the box tips past the deck edge.
                if ov > 0 and require_centroid and (
                        x + dx / 2.0 > min(x + dx, L) + EPS_A
                        or y + dy / 2.0 > min(y + dy, W) + EPS_A):
                    v.append(f"FLOOR_TOPPLE {b.id} "
                             f"com=({x + dx / 2.0},{y + dy / 2.0})")
            else:
                sup = 0.0
                for (x2, y2, z2, dx2, dy2, dz2, b2) in geo:
                    if b2 is b or abs((z2 + dz2) - z) > 1e-6:
                        continue
                    oxx = min(x + dx, x2 + dx2) - max(x, x2)
                    oyy = min(y + dy, y2 + dy2) - max(y, y2)
                    if oxx > 0 and oyy > 0:
                        sup += oxx * oyy
                if sup <= 0:
                    v.append(f"HOVER {b.id}")
                elif sup < eff * fp - EPS_A - 1e-9 * fp:
                    v.append(f"SUPPORT {b.id} {sup / fp:.2%}<{eff:.0%}")
        # load model (direct when transitive flag off, transitive when on)
        inflow = [0.0] * len(geo)
        order = sorted(range(len(geo)), key=lambda k: -geo[k][2])
        for k in order:
            x, y, z, dx, dy, dz, b = geo[k]
            out = b.weight + (inflow[k] if transitive else 0.0)
            if z <= 1e-9 or out <= 0:
                continue
            sups = []
            for m2 in range(len(geo)):
                if m2 == k:
                    continue
                x2, y2, z2, dx2, dy2, dz2, b2 = geo[m2]
                if abs((z2 + dz2) - z) > 1e-6:
                    continue
                oxx = min(x + dx, x2 + dx2) - max(x, x2)
                oyy = min(y + dy, y2 + dy2) - max(y, y2)
                if oxx > 0 and oyy > 0:
                    sups.append((m2, oxx * oyy))
            tot = sum(a for _, a in sups)
            if tot <= 0:
                continue
            for m2, a in sups:
                inflow[m2] += out * (a / tot)
        for k, (x, y, z, dx, dy, dz, b) in enumerate(geo):
            lim = b.max_load_on_top
            if lim != float("inf"):
                # mirror the engine's scale-aware tolerance (round 3, F25)
                if inflow[k] > lim + max(1e-6, 1e-9 * lim) + 1e-9 * max(lim, 1.0):
                    tag = "TRANS" if transitive else "DIRECT"
                    v.append(f"LOAD[{tag}] {b.id} {inflow[k]:.4f}>{lim:.4f}")
        if pallet.max_weight != float("inf"):
            tw = sum(g[6].weight for g in geo)
            if tw > pallet.max_weight + 1e-6:
                v.append(f"PALLET_W {tw}>{pallet.max_weight}")
    return v


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def run_sweep(name, gen, seed, n_instances, k_chroms, chrom_seed_base):
    rng = random.Random(seed)
    total = bad = 0
    examples = []
    for inst in range(n_instances):
        boxes, pallet, cfg, ov, sr, transitive, max_pallets = gen(rng)
        n = len(boxes)
        n_rots_arr, dims_all, sku_id = drv.precompute_box_dims_and_sku(boxes)
        L, W, H = int(pallet.length), int(pallet.width), int(pallet.height)
        best_blk = drv.enumerate_best_block_per_sku(
            boxes, sku_id, n_rots_arr, dims_all, L, W, H)
        topk_blk = drv.enumerate_top_k_blocks_per_sku(
            boxes, sku_id, n_rots_arr, dims_all, L, W, H, k_top=8)
        weights_arr, mlot_arr, rfs_arr, pmw, _ = \
            drv.precompute_constraint_arrays(boxes, pallet)
        nprng = np.random.default_rng(chrom_seed_base + inst)
        for kc in range(k_chroms):
            chrom = nprng.random(2 * n)
            for mode in MODES:
                total += 1
                try:
                    res = decode_chromosome(
                        chrom, boxes, pallet, cfg, n_rots_arr, dims_all,
                        mode=mode, max_pallets=max_pallets,
                        sku_id_per_box=sku_id, sku_best_block=best_blk,
                        sku_top_k_blocks=topk_blk,
                        weights=weights_arr, mlot=mlot_arr, rfs=rfs_arr,
                        pallet_max_weight=pmw, has_constraints=True,
                        support_ratio=sr,
                        require_centroid=int(cfg.require_centroid_supported),
                        max_overhang=float(ov))
                except Exception as e:  # noqa: BLE001
                    bad += 1
                    examples.append(
                        f"[{name} inst {inst} chrom {kc} mode {mode}] "
                        f"RAISED {type(e).__name__}: {e}")
                    continue
                viol = oracle(res, boxes, pallet, ov, sr, transitive,
                              require_centroid=cfg.require_centroid_supported)
                if viol:
                    bad += 1
                    examples.append(
                        f"[{name} inst {inst} chrom {kc} mode {mode} ov={ov} "
                        f"sr={sr} tr={int(transitive)}] " + "; ".join(viol[:4]))
    print(f"  sweep {name}: {total} decodes, {bad} violating")
    for e in examples[:20]:
        print("    " + e)
    if len(examples) > 20:
        print(f"    ... +{len(examples) - 20} more")
    return total, bad


def run_regressions():
    """Deterministic round-3 instances (block siblings / under-fill /
    driver diamond) — each decoded across all modes and oracled."""
    total = bad = 0

    def check(tag, boxes, pallet, cfg, ov, sr, transitive, max_pallets=1):
        nonlocal total, bad
        n = len(boxes)
        n_rots_arr, dims_all, sku_id = drv.precompute_box_dims_and_sku(boxes)
        L, W, H = int(pallet.length), int(pallet.width), int(pallet.height)
        best_blk = drv.enumerate_best_block_per_sku(
            boxes, sku_id, n_rots_arr, dims_all, L, W, H)
        topk_blk = drv.enumerate_top_k_blocks_per_sku(
            boxes, sku_id, n_rots_arr, dims_all, L, W, H, k_top=8)
        weights_arr, mlot_arr, rfs_arr, pmw, _ = \
            drv.precompute_constraint_arrays(boxes, pallet)
        nprng = np.random.default_rng(4242)
        for kc in range(4):
            chrom = nprng.random(2 * n)
            for mode in MODES:
                total += 1
                res = decode_chromosome(
                    chrom, boxes, pallet, cfg, n_rots_arr, dims_all,
                    mode=mode, max_pallets=max_pallets,
                    sku_id_per_box=sku_id, sku_best_block=best_blk,
                    sku_top_k_blocks=topk_blk,
                    weights=weights_arr, mlot=mlot_arr, rfs=rfs_arr,
                    pallet_max_weight=pmw, has_constraints=True,
                    support_ratio=sr,
                    require_centroid=int(cfg.require_centroid_supported),
                    max_overhang=float(ov))
                viol = oracle(res, boxes, pallet, ov, sr, transitive,
                              require_centroid=cfg.require_centroid_supported)
                if viol:
                    bad += 1
                    print(f"    REGRESSION {tag} chrom {kc} mode {mode}: "
                          + "; ".join(viol[:4]))

    def cfg_for(sr, transitive, centroid=True):
        return PackerConfig(support_ratio=sr,
                            require_centroid_supported=centroid,
                            enforce_load_bearing=True,
                            transitive_load_bearing=transitive, seed=1)

    # Round-3 diamond (the live-served 1.6x overload): 2 bases + spanning
    # mid + four 8 kg tops.
    boxes = ([Box(id=f"base{i}", length=400, width=400, height=200,
                  weight=20.0, max_load_on_top=17.0,
                  allowed_rotations=list(ALL_ROTATIONS)) for i in range(2)]
             + [Box(id="mid", length=800, width=400, height=200, weight=16.0,
                    max_load_on_top=20.0,
                    allowed_rotations=list(ALL_ROTATIONS))]
             + [Box(id=f"top{i}", length=200, width=200, height=100,
                    weight=8.0, allowed_rotations=list(ALL_ROTATIONS))
                for i in range(4)])
    for tr in (False, True):
        check(f"diamond tr={int(tr)}", boxes,
              Pallet(length=800, width=400, height=1500),
              cfg_for(0.8, tr), 0, 0.8, tr)

    # Block sibling minimal (weak base, 4 same-SKU tops).
    boxes = ([Box(id="base", length=100, width=100, height=20, weight=50.0,
                  max_load_on_top=25.0,
                  allowed_rotations=list(ALL_ROTATIONS))]
             + [Box(id=f"t{i}", length=50, width=50, height=20, weight=10.0,
                    allowed_rotations=list(ALL_ROTATIONS))
                for i in range(4)])
    for tr in (False, True):
        check(f"block-sib tr={int(tr)}", boxes,
              Pallet(length=100, width=100, height=200),
              cfg_for(0.8, tr), 0, 0.8, tr)

    # Under-fill minimal (pillar + cantilever slab + fragile floor box).
    boxes = [Box(id="pillar", length=200, width=200, height=100, weight=1.0,
                 allowed_rotations=list(ALL_ROTATIONS)),
             Box(id="slab", length=400, width=200, height=100, weight=1.0,
                 allowed_rotations=list(ALL_ROTATIONS)),
             Box(id="fragile", length=200, width=200, height=100, weight=5.0,
                 max_load_on_top=0.0,
                 allowed_rotations=list(ALL_ROTATIONS))]
    for tr in (False, True):
        check(f"underfill tr={int(tr)}", boxes,
              Pallet(length=400, width=200, height=300),
              cfg_for(0.5, tr, centroid=False), 0, 0.5, tr)

    print(f"  regressions: {total} decodes, {bad} violating")
    return total, bad


def main():
    grand_total = grand_bad = 0
    print("=== verify_load_physics: exact-oracle decode sweep ===")
    t, b = run_sweep("A-mixed", rand_instance_mixed, 20260703, 150, 8, 0)
    grand_total += t
    grand_bad += b
    t, b = run_sweep("B-prod", rand_instance_prod, 777, 200, 6, 1000)
    grand_total += t
    grand_bad += b
    # Round 6 (F30): the toppling regime. Appended sweep — each sweep owns
    # its own random.Random(seed) and default_rng(chrom_seed_base + inst),
    # so adding it cannot perturb A/B.
    t, b = run_sweep("C-topple", rand_instance_topple, 20260704, 120, 6, 2000)
    grand_total += t
    grand_bad += b
    print("=== deterministic round-3 regressions ===")
    t, b = run_regressions()
    grand_total += t
    grand_bad += b
    ok = grand_bad == 0
    print(f"\n=== RESULT: {'PASS' if ok else 'FAIL'} "
          f"({grand_total} decodes, {grand_bad} physics violations) ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
