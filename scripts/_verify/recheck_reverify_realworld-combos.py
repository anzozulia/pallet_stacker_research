"""recheck_reverify_realworld-combos — INDEPENDENT counter-probe.

Claim under test: _pack_with_groups assigns whole groups to pallets by VOLUME
first-fit-decreasing only (never reads pallet.max_weight). When the weight cap
binds, heavy low-volume groups all pass the volume budget and land on ONE
pallet; the single-container solve then can only fit max_weight worth and
silently spills the rest to unpacked while other allowed pallets sit EMPTY.

I independently re-derive each load-bearing fact:
  (1) STRUCTURAL: does the bundle-to-pallet assigner consult max_weight?
      -> re-read source already done; here we PROVE behaviour, not just grep.
  (2) BEHAVIOUR: does grouped drop far more than ungrouped when the weight cap
      binds, with allowed pallets left EMPTY?  (the headline repro)
  (3) HARD SAFETY: do all hard invariants still hold under the defect?
      - conservation exact (placed+unpacked == input)
      - per-pallet weight cap never exceeded
      - validate() == []  (oracle: weight, overlap, support, load-bearing)
      - NO group ever SPANS two pallets (true co-location invariant)
  (4) DISCRIMINATOR: is it weight-CAP-blind specifically? Build a case where
      the VOLUME budget would also bind to make groups separate, and confirm
      grouped packs FINE there -> proves the failure is weight-specific, not a
      generic group-path bug.
  (5) CONTROL: with weight cap = inf (cap does not bind), grouped must NOT
      regress vs ungrouped -> proves the cause is the binding weight cap.
  (6) REGRESSION? the group feature is NEW (commit 6ce064f); compare to the
      pre-group behaviour by running the SAME boxes with group=None (which is
      exactly the multi-pallet path that existed before).

Run in Docker with OMP_NUM_THREADS=1, PYTHONHASHSEED=0.
"""
from __future__ import annotations
import sys
sys.path.insert(0, '.')
from collections import defaultdict

from pallet_packer import (
    Box, Pallet, PackerConfig, THIS_SIDE_UP, ALL_ROTATIONS,
    validate,
)
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit

EPS = 1e-6


def strip_groups(boxes):
    return [Box(id=b.id, length=b.length, width=b.width, height=b.height,
                weight=b.weight, allowed_rotations=b.allowed_rotations,
                max_load_on_top=b.max_load_on_top, group=None)
            for b in boxes]


def analyze(r, boxes, pallet, config):
    placed = sum(len(st.placements) for st in r.pallets)
    n_unp = len(r.unpacked)
    # group -> set of pallet indices
    gp = defaultdict(set)
    g_unpacked = defaultdict(int)
    g_total = defaultdict(int)
    for b in boxes:
        if b.group is not None:
            g_total[b.group] += 1
    for pi, st in enumerate(r.pallets):
        for p in st.placements:
            g = getattr(p.box, "group", None)
            if g is not None:
                gp[g].add(pi)
    unp_ids = {id(b) for b in r.unpacked}
    for b in boxes:
        if b.group is not None and id(b) in unp_ids:
            g_unpacked[b.group] += 1
    spans = {g: sorted(ps) for g, ps in gp.items() if len(ps) > 1}
    partials = {g: (g_unpacked[g], g_total[g]) for g in g_total
                if g_unpacked[g] and len(gp[g]) >= 1}
    overweight = []
    for st in r.pallets:
        w = sum(p.box.weight for p in st.placements)
        if w > pallet.max_weight + 1e-3:
            overweight.append((st.pallet_id, round(w, 1)))
    errs = validate(r, pallet, config)
    return {
        "pallets": len(r.pallets), "placed": placed, "unpacked": n_unp,
        "conserve_ok": placed + n_unp == len(boxes),
        "spans": spans, "partials": partials,
        "overweight": overweight, "validator_errs": errs,
    }


def headline_boxes():
    """5 heavy groups of 18, 10kg each = 180kg/group, 50% fragile, cap=250."""
    boxes = []
    for g in ["H1", "H2", "H3", "H4", "H5"]:
        for i in range(18):
            fragile = (i % 2 == 0)
            boxes.append(Box(
                id=f"{g}-{i:02d}", length=300, width=250, height=200,
                weight=10.0,
                allowed_rotations=THIS_SIDE_UP if fragile else ALL_ROTATIONS,
                max_load_on_top=0.0 if fragile else 35.0, group=g))
    return boxes


def run(boxes, pallet, config, max_pallets, seed=42, t=14):
    return brkga_pack_v35(boxes, pallet, config, time_limit_s=t,
                          max_pallets=max_pallets, seed=seed,
                          population_size=300, n_populations=3, patience=150,
                          n_modes=6, validate_input=True, verbose=False)


def main():
    warmup_jit()
    results = {}

    # === (2)+(3) HEADLINE: weight cap binds, max_pallets=8 ==================
    print("=" * 70)
    print("(A) HEADLINE: 5x18 heavy groups, cap=250kg, max_pallets=8")
    print("=" * 70)
    boxes = headline_boxes()
    pal = Pallet(length=1200, width=1000, height=1500, max_weight=250)
    cfg = PackerConfig()
    for seed in (42, 7, 123):
        rg = run(boxes, pal, cfg, 8, seed=seed)
        ru = run(strip_groups(boxes), pal, cfg, 8, seed=seed)
        ag = analyze(rg, boxes, pal, cfg)
        au = analyze(ru, strip_groups(boxes), pal, cfg)
        print(f"  seed={seed}: GROUPED pallets={ag['pallets']} "
              f"placed={ag['placed']} unp={ag['unpacked']} "
              f"spans={ag['spans']} partials={ag['partials']} "
              f"overweight={ag['overweight']} errs={len(ag['validator_errs'])} "
              f"conserve={ag['conserve_ok']}")
        print(f"           UNGROUPED pallets={au['pallets']} "
              f"placed={au['placed']} unp={au['unpacked']} "
              f"overweight={au['overweight']} errs={len(au['validator_errs'])}")
        results.setdefault("headline", []).append((seed, ag, au))

    # === (4) DISCRIMINATOR: VOLUME-binding (not weight) — should be fine =====
    print("=" * 70)
    print("(B) DISCRIMINATOR: groups separated by VOLUME, cap=inf — "
          "grouped must NOT regress")
    print("=" * 70)
    # Big-volume groups so volume budget forces them apart; weight is trivial.
    vbig = []
    for g in ["V1", "V2", "V3", "V4", "V5"]:
        for i in range(18):
            vbig.append(Box(id=f"{g}-{i:02d}", length=400, width=350,
                            height=300, weight=1.0,
                            allowed_rotations=ALL_ROTATIONS,
                            max_load_on_top=40.0, group=g))
    palv = Pallet(length=1200, width=1000, height=1500, max_weight=float("inf"))
    rgv = run(vbig, palv, cfg, 8)
    ruv = run(strip_groups(vbig), palv, cfg, 8)
    agv = analyze(rgv, vbig, palv, cfg)
    auv = analyze(ruv, strip_groups(vbig), palv, cfg)
    print(f"  GROUPED   pallets={agv['pallets']} placed={agv['placed']} "
          f"unp={agv['unpacked']} spans={agv['spans']} "
          f"errs={len(agv['validator_errs'])}")
    print(f"  UNGROUPED pallets={auv['pallets']} placed={auv['placed']} "
          f"unp={auv['unpacked']} errs={len(auv['validator_errs'])}")
    results["volume_discriminator"] = (agv, auv)

    # === (5) CONTROL: SAME headline boxes but cap=inf — must NOT regress =====
    print("=" * 70)
    print("(C) CONTROL: headline boxes, cap=INF (cap doesn't bind), "
          "max_pallets=8")
    print("=" * 70)
    pal_inf = Pallet(length=1200, width=1000, height=1500,
                     max_weight=float("inf"))
    rgc = run(boxes, pal_inf, cfg, 8)
    ruc = run(strip_groups(boxes), pal_inf, cfg, 8)
    agc = analyze(rgc, boxes, pal_inf, cfg)
    auc = analyze(ruc, strip_groups(boxes), pal_inf, cfg)
    print(f"  GROUPED   pallets={agc['pallets']} placed={agc['placed']} "
          f"unp={agc['unpacked']} spans={agc['spans']} "
          f"partials={agc['partials']} errs={len(agc['validator_errs'])}")
    print(f"  UNGROUPED pallets={auc['pallets']} placed={auc['placed']} "
          f"unp={auc['unpacked']} errs={len(auc['validator_errs'])}")
    results["cap_inf_control"] = (agc, auc)

    # === (6) REGRESSION baseline: pre-group multi-pallet path ===============
    # group=None on these boxes is EXACTLY the path that existed before the
    # group feature. We already have it as UNGROUPED above; restate verdict.

    print("=" * 70)
    print("VERDICT SYNTHESIS")
    print("=" * 70)
    # Defect signature: grouped drops >=10% more than ungrouped AND >=2 pallets
    # left empty AND the cap binds.
    hl = results["headline"]
    any_defect = False
    for seed, ag, au in hl:
        lost_extra = au["placed"] - ag["placed"]
        unused = 8 - ag["pallets"]
        binds = ag["overweight"] == []  # cap respected => it bound
        defect = (lost_extra >= 9 and unused >= 2)
        any_defect = any_defect or defect
        print(f"  headline seed={seed}: lost_extra={lost_extra} "
              f"pallets_unused={unused} -> defect={defect}")
    # Hard safety must hold on every grouped run.
    safety_ok = all(
        ag["conserve_ok"] and ag["overweight"] == []
        and ag["validator_errs"] == [] and ag["spans"] == {}
        for _, ag, _ in hl
    )
    print(f"  HARD SAFETY held on all headline grouped runs: {safety_ok}")
    # Discriminator: volume-binding grouped should be roughly on par.
    agv, auv = results["volume_discriminator"]
    vol_ok = (auv["placed"] - agv["placed"]) < 9
    print(f"  volume-discriminator grouped on par with ungrouped: {vol_ok} "
          f"(grouped={agv['placed']} ungrouped={auv['placed']})")
    # Control: cap=inf grouped on par.
    agc, auc = results["cap_inf_control"]
    ctrl_ok = (auc["placed"] - agc["placed"]) < 9
    print(f"  cap=inf control grouped on par with ungrouped: {ctrl_ok} "
          f"(grouped={agc['placed']} ungrouped={auc['placed']})")

    print()
    print(f"  QUALITY DEFECT REPRODUCED (weight-cap-blind): {any_defect}")
    print(f"  HARD SAFETY INTACT: {safety_ok}")
    print(f"  CAUSE IS WEIGHT-SPECIFIC (vol/inf fine): "
          f"{vol_ok and ctrl_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
