"""recheck_float-int-dims.py — INDEPENDENT counter-probe for the int-decode /
float-validate gap. Re-derived from first principles; does NOT reuse the
original probe's audit code.

Goals (adversarial + skeptical):
  A) Minimal reproduction: TEST 1 (100.49^3-ish cube x64). Confirm validate()
     fails AND the overlap is REAL float geometry, not a validator-EPS artifact.
  B) Control: matched INTEGER-rounded twin must validate clean.
  C) Control: genuinely-integer BR-style input must validate clean (rules out a
     general solver bug; isolates cause to fractional dims).
  D) Realistic (non-adversarial) fractional case: 333.33 x 250.7 x 124.9.
  E) Is the failure a validator-EPS (1e-6) edge effect, or true overlap?
     Recompute every overlap with generous tolerance 1e-3 (100x EPS) and report
     the max penetration depth. If depth >> EPS, it's a real defect not an EPS
     boundary case.
  F) Direct mechanism check: decode one chromosome, read back integer pitch vs
     float box extent, prove the butt-against overlap arithmetically.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pallet_packer import Box, Pallet, PackerConfig, validate, ALL_ROTATIONS
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit


def geom_cfg(max_pallets=1):
    return PackerConfig(
        support_ratio=0.0, require_centroid_supported=False,
        enforce_load_bearing=False, allow_pallet_overhang=False,
        cog_envelope_fraction=1.0, max_pallets=max_pallets, optimize="max_util")


def solve(boxes, pallet, cfg, seed=42, t=4.0, mp=1):
    return brkga_pack_v35(boxes, pallet, cfg, time_limit_s=t, max_pallets=mp,
                          population_size=200, n_populations=3, patience=120,
                          local_search_budget_s=1.0, seed=seed, verbose=False,
                          n_modes=6)


def max_overlap_depth(result, tol=1e-3):
    """Worst pairwise float penetration depth, ignoring sub-tol touches."""
    worst = 0.0; worst_pair = None; n_pairs = 0
    for st in result.pallets:
        pls = st.placements
        for i in range(len(pls)):
            a = pls[i]; ax2, ay2, az2 = a.x2, a.y2, a.z2
            for j in range(i + 1, len(pls)):
                b = pls[j]; bx2, by2, bz2 = b.x2, b.y2, b.z2
                ox = min(ax2, bx2) - max(a.x, b.x)
                oy = min(ay2, by2) - max(a.y, b.y)
                oz = min(az2, bz2) - max(a.z, b.z)
                if ox > tol and oy > tol and oz > tol:
                    pen = min(ox, oy, oz); n_pairs += 1
                    if pen > worst:
                        worst = pen
                        worst_pair = (a.box.id, b.box.id, a.x, a.x2, b.x, ox, oy, oz)
    return worst, worst_pair, n_pairs


def max_oob(result, pallet, tol=1e-3):
    worst = 0.0; worst_box = None
    for st in result.pallets:
        for p in st.placements:
            excess = max(-min(p.x, p.y, p.z),
                         p.x2 - pallet.length, p.y2 - pallet.width,
                         p.z2 - pallet.height)
            if excess > tol and excess > worst:
                worst = excess; worst_box = (p.box.id, p.x2, p.y2, p.z2)
    return worst, worst_box


def report(label, r, pallet, cfg):
    errs = validate(r, pallet, cfg)
    od, op, npairs = max_overlap_depth(r)
    ob, obx = max_oob(r, pallet)
    n = sum(len(s.placements) for s in r.pallets)
    print(f"  {label:<40} placed={n:>3} val_errs={len(errs):>3} "
          f"overlap>{1e-3}={od:.4f}({npairs}prs) oob>{1e-3}={ob:.4f}")
    if op:
        print(f"      worst overlap: {op}")
    if obx:
        print(f"      worst oob:     {obx}")
    return errs, od, ob


def main():
    warmup_jit()
    cfg = geom_cfg()
    print("=" * 72)
    print("RECHECK float-int-dims (independent)")
    print("=" * 72)

    # --- A) minimal repro: 100.49^2 x 100 cubes, x64, pallet 804^2 ---
    print("\n[A] minimal repro 100.49^2 x100 x64 (rounds DOWN -> butt overlap)")
    fb = [Box(id=f"d{i}", length=100.49, width=100.49, height=100.0, weight=0.0,
              allowed_rotations=list(ALL_ROTATIONS)) for i in range(64)]
    fp = Pallet(length=804.0, width=804.0, height=100.0)
    eA, odA, _ = report("FLOAT 100.49", solve(fb, fp, cfg), fp, cfg)

    # --- B) matched integer twin (control) ---
    print("\n[B] matched INTEGER twin (control: should be clean)")
    ib = [Box(id=f"d{i}", length=100.0, width=100.0, height=100.0, weight=0.0,
              allowed_rotations=list(ALL_ROTATIONS)) for i in range(64)]
    ip = Pallet(length=804.0, width=804.0, height=100.0)
    eB, odB, _ = report("INT 100", solve(ib, ip, cfg), ip, cfg)

    # --- C) genuinely-integer realistic mix (control: general solver sane?) ---
    print("\n[C] integer realistic mix (control)")
    cb = [Box(id=f"c{i}", length=300, width=200, height=150, weight=0.0,
              allowed_rotations=list(ALL_ROTATIONS)) for i in range(40)]
    cp = Pallet(length=1200, width=1000, height=1500)
    eC, odC, _ = report("INT mixed", solve(cb, cp, cfg), cp, cfg)

    # --- D) realistic NON-adversarial fractional (333.33 mm) ---
    print("\n[D] realistic fractional 333.33x250.7x124.9 n=33")
    db = [Box(id=f"f{i}", length=333.33, width=250.7, height=124.9, weight=0.0,
              allowed_rotations=list(ALL_ROTATIONS)) for i in range(33)]
    dp = Pallet(length=1200.0, width=1000.0, height=1500.0)
    eD, odD, _ = report("FLOAT 333.33", solve(db, dp, cfg), dp, cfg)

    # --- D2) sub-mm collapse: 0.4^3 -> dim 0 -> all-pairs overlap ---
    print("\n[D2] sub-mm collapse 0.4^3 x50 in Pallet(5^3) (rounds to int dim 0)")
    from pallet_packer.brkga_v3_fast import precompute_box_dims
    zb = [Box(id="z", length=0.4, width=0.4, height=0.4, weight=0.0,
              allowed_rotations=[ALL_ROTATIONS[0]])]
    _, zdims = precompute_box_dims(zb)
    print(f"    precompute: 0.4 -> int dims_all = {int(zdims[0,0,0])} "
          f"({'ZERO-VOLUME' if zdims[0,0,0]==0 else 'nonzero'})")
    sb = [Box(id=f"s{i}", length=0.4, width=0.4, height=0.4, weight=0.0,
              allowed_rotations=list(ALL_ROTATIONS)) for i in range(50)]
    sp = Pallet(length=5.0, width=5.0, height=5.0)
    rS = solve(sb, sp, cfg, t=3.0)
    eS = validate(rS, sp, cfg)
    nS = sum(len(s.placements) for s in rS.pallets)
    ov_errs = [e for e in eS if "overlaps" in e]
    cnk2 = nS * (nS - 1) // 2
    print(f"    placed={nS}  val_errs={len(eS)}  overlap_errs={len(ov_errs)}  "
          f"C(placed,2)={cnk2}")
    collapse = (len(ov_errs) == cnk2 and cnk2 > 0)
    print(f"    -> {'CONFIRMED all-pairs collapse' if collapse else 'no collapse'}")

    # --- D3) INT-SNAP twin: snap float dims to int(round); SAME decode
    # geometry, but Placement.dx now = snapped integer -> validator sees what the
    # decoder saw. If this validates CLEAN, the cause is purely the float dims
    # handed to validate(), NOT the search/decoder placement logic. ---
    print("\n[D3] INT-SNAP twin of [A] (dims=int(round); same decode geometry)")
    snb = [Box(id=f"d{i}", length=float(round(100.49)),
               width=float(round(100.49)), height=100.0, weight=0.0,
               allowed_rotations=list(ALL_ROTATIONS)) for i in range(64)]
    snp = Pallet(length=804.0, width=804.0, height=100.0)
    rSnap = solve(snb, snp, cfg)
    eSnap = validate(rSnap, snp, cfg)
    print(f"    INT-snap(100.49->100): placed="
          f"{sum(len(s.placements) for s in rSnap.pallets)} "
          f"val_errs={len(eSnap)} "
          f"({'CLEAN -> cause isolated to float dims' if not eSnap else 'STILL DIRTY'})")

    # --- E) EPS-artifact check: is overlap >> validator EPS (1e-6)? ---
    print("\n[E] EPS-artifact check (validator EPS=1e-6; we measured at 1e-3 tol)")
    big_overlap = max(odA, odD)
    print(f"    worst true overlap across A,D at 1e-3 tol = {big_overlap:.4f} units")
    print(f"    -> {'REAL DEFECT (>> EPS)' if big_overlap > 1e-2 else 'borderline/EPS'}")

    # --- F) arithmetic mechanism proof (no search) ---
    print("\n[F] arithmetic mechanism on solved FLOAT 100.49 result")
    rF = solve(fb, fp, cfg)
    # find two boxes adjacent along x with integer pitch ~100 but float dx=100.49
    pls = rF.pallets[0].placements if rF.pallets else []
    proven = False
    for a in pls:
        for b in pls:
            if a is b:
                continue
            # same row (y,z equal), b just right of a
            if abs(a.y - b.y) < 1e-9 and abs(a.z - b.z) < 1e-9:
                pitch = b.x - a.x
                if 0 < pitch < a.dx:  # next box starts before this one ends
                    print(f"    a.x={a.x} a.dx={a.dx} a.x2={a.x2}  b.x={b.x} "
                          f"pitch={pitch}  -> butt overlap {a.x2 - b.x:.4f} units")
                    proven = True
                    break
        if proven:
            break
    if not proven:
        print("    (no adjacent-x butt pair found in this solve)")

    # --- VERDICT ---
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    float_fails = bool(eA) or bool(eD)
    int_clean = (not eB) and (not eC)
    print(f"  FLOAT inputs produce validator errors: A={len(eA)} D={len(eD)}")
    print(f"  INT controls clean:                    B={len(eB)} C={len(eC)}")
    print(f"  worst real overlap depth (1e-3 tol):   {big_overlap:.4f} units")
    if float_fails and int_clean and big_overlap > 1e-2:
        print("  => CONFIRMED: fractional dims -> validator-invalid packings; "
              "integer inputs clean.")
        return 1
    if not float_fails:
        print("  => NOT REPRODUCED: float inputs validated clean.")
        return 0
    print("  => MIXED / inconclusive.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
