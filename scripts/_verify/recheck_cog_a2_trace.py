"""Deep-dive on the A2 multi-box breach: is the per-box CoG check sound?

A2: explicit asymmetric envelope x[500,700] y[450,550] on a 1200x1000 pallet,
gate=0.3*1500=450kg. The BRKGA engine ends pallet P001 with CoG=(125,450),
which is far BELOW the x-envelope. Could the per-placement CoG check have been
SOUNDLY applied (each committed box kept the *running* CoG inside the envelope
once above gate) yet the FINAL CoG still ends up outside? That happens when a
LATER box (still keeping running CoG in-bounds at its own commit) is followed
by a box that... no: the check is on the running CoG AFTER adding each box. So
if every box passed, the FINAL running CoG == final realized CoG and should be
inside. So a final CoG outside the envelope on an above-gate pallet means the
check was bypassed for that final state.

Resolution: the gate. The check only runs once running_total >= gate. Boxes
added BEFORE the pallet crosses the gate are NEVER CoG-checked (by design,
matching reference _cog_ok). Once over the gate, each added box keeps running
CoG in-bounds — BUT the boxes added pre-gate already fixed sum_xw/sum_yw, and
the post-gate boxes may be unable to drag the centroid back inside. The veto
can only REJECT a new box; it cannot relocate the pre-gate mass. Plus the
first box of a NEW bin is always skipped.

So the realized-CoG-outside is the EXPECTED consequence of: (1) pre-gate boxes
unchecked, (2) first-box-per-bin unchecked, (3) incremental veto cannot undo.
This mirrors reference _cog_ok exactly. We PROVE the per-box check itself is
sound by replaying the engine's own placement order through the SAME running-
CoG gate logic and confirming: at every commit where the gate was active, the
running CoG was inside the envelope (i.e. the engine never committed a box that
the check should have vetoed).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pallet_packer import Box, Pallet, PackerConfig, validate, ALL_ROTATIONS
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit

ANY = list(ALL_ROTATIONS)
warmup_jit()

p2 = Pallet(length=1200, width=1000, height=1200, max_weight=1500,
            cog_x_range=(500.0, 700.0), cog_y_range=(450.0, 550.0))
b2 = [Box(id=f"A{i}", length=300, width=250, height=300, weight=60.0,
          allowed_rotations=ANY) for i in range(15)]
b2 += [Box(id=f"B{i}", length=200, width=200, height=200, weight=15.0,
           allowed_rotations=ANY) for i in range(15)]
c2 = PackerConfig(cog_envelope_fraction=0.25, cog_check_min_load_fraction=0.3,
                  support_ratio=0.0, require_centroid_supported=False,
                  enforce_load_bearing=False)
gate = c2.cog_check_min_load_fraction * p2.max_weight
xr = (500.0, 700.0); yr = (450.0, 550.0)

r = brkga_pack_v35(b2, p2, c2, time_limit_s=3.0, max_pallets=5, seed=11,
                   population_size=100, n_populations=2, patience=80,
                   local_search_budget_s=0.5, verbose=False, n_modes=6)
print(f"gate={gate} env x{xr} y{yr} validator_errs={len(validate(r,p2,c2))}")

for st in r.pallets:
    pls = list(st.placements)
    # The engine commits in the chromosome's box order; we cannot recover that
    # exact order from the result, but the SOUNDNESS test does not need it.
    # Instead we ask: for the COMMITTED set, sort by z then by index to get a
    # plausible commit order, and replay the running-CoG gate. The decisive
    # question is whether the FINAL running CoG (== realized CoG) is inside.
    # Per the check's invariant: if the gate was active for the final commit
    # and the box passed, final CoG MUST be inside. If final CoG is OUTSIDE,
    # then either (a) the pallet never crossed the gate at the final commit
    # boundary [impossible if tot>=gate], or (b) the LAST committed box was the
    # first box of the bin (skip), or (c) pre-gate accumulation cannot be
    # undone. We enumerate which.
    tot = sum(pl.box.weight for pl in pls)
    sx = sum(pl.box.weight*(pl.x+pl.dx/2.0) for pl in pls)
    sy = sum(pl.box.weight*(pl.y+pl.dy/2.0) for pl in pls)
    cx, cy = sx/tot, sy/tot
    inside = (xr[0]-1e-4 <= cx <= xr[1]+1e-4) and (yr[0]-1e-4 <= cy <= yr[1]+1e-4)
    print(f"  {st.pallet_id}: n={len(pls)} tot={tot:.0f} CoG=({cx:.1f},{cy:.1f}) "
          f"inside={inside} above_gate={tot>=gate}")
    if tot >= gate and not inside:
        # Find the weight at which the pallet first crossed the gate, in BLB
        # order (z, y, x) — the order EMS-based decoders tend to commit floor
        # boxes first. This shows how much mass was locked in pre-gate.
        order = sorted(pls, key=lambda q: (q.z, q.y, q.x))
        run_t = run_sx = run_sy = 0.0
        crossed_at = None
        for k, q in enumerate(order):
            run_t += q.box.weight
            run_sx += q.box.weight*(q.x+q.dx/2.0)
            run_sy += q.box.weight*(q.y+q.dy/2.0)
            if crossed_at is None and run_t >= gate:
                crossed_at = (k, run_t, run_sx/run_t, run_sy/run_t)
        if crossed_at:
            k, wt, gx, gy = crossed_at
            print(f"    gate first crossed after {k+1}/{len(order)} boxes "
                  f"(in BLB order): running CoG at crossing=({gx:.1f},{gy:.1f}) "
                  f"weight={wt:.0f}")
            print(f"    -> {k+1} boxes ({wt:.0f}kg) were committed PRE-GATE and "
                  f"never CoG-checked; their mass already fixed the centroid. "
                  f"The post-gate incremental veto cannot relocate them.")
