# 34 — Realism layer: recenter + orientation alignment + secondary fitness

**Status: shipped (2026-07-02), all flags default OFF in the core.**
Service-side (pallet_packer_api_service) they are ON via env kill-switches.

## Problem

A 19-scenario live study against the deployed API showed layouts that are
constraint-correct but real-world wrong:

- **Corner-jamming** (12/18 scenarios): every under-filled load anchors at
  (0,0); worst case two 100 kg parts with the combined CoG 67% off deck
  centre — a forklift hazard.
- **Rotation chaos**: identical cartons get different orientations inside one
  layer (verified numerically: same SKU, same layer, opposite argmax at
  different positions — the DFTRC score's `(H−z−dz)²` term decays with
  height, so tipped rotations overtake flat ones above ~z=700).
- **Heavy-on-top**: 18 kg cartons riding on 4 kg cartons (a crush path);
  `heavy_on_bottom` was inert in the v3.5 path (seed chromosomes order by
  volume only).
- **Dead search**: fitness `1 − used_vol/capacity` TIES for every feasible
  layout once all boxes fit → seeds 42/7/123 byte-identical output, local
  search accepted 0 of 171,388 moves, budgets observational no-ops. Layout
  quality was decided 100% by corner-seeking decoder tie-breaks.

## What shipped

1. **`postprocess.py`** — two replay-validated passes applied exactly once by
   the `brkga_pack_v35` wrapper (internal restart/group recursions call the
   un-postprocessed `_impl`):
   - `recenter_pass`: rigid per-pallet x/y translation targeting the weighted
     CoG at deck centre (bbox midpoint when weightless), integer-floored,
     clamped so the shift never creates overhang and never touches z.
     Idempotent by construction (target derived from translation-invariant
     internal offsets).
   - `align_orientations_pass`: re-rotates same-SKU deviants to the dominant
     dims-tuple of their (SKU, z-level) when feasibility-preserving —
     dz-preserving-only when the box supports others, whole-pallet
     `validate()` with single-field revert per swap, time-boxed 2 s/pallet.
     Per-level dominance deliberately leaves interlocked alternating layers
     alone.
   Gating: `PackerConfig.recenter_layout` / `.align_orientations`.

2. **Secondary realism fitness** (`_brkga_core/realism.py`), gated by
   `PackerConfig.realism_weight` (default 0.0 = off):
   `fitness' = base + eps·(0.5·HM + 0.4·MH + 0.1·OI)` with HM = weight-weighted height
   moment (volume-weighted fallback), MH = max height fraction, OI = per-SKU
   height-class inconsistency (keyed on dz — yaw variation within a
   layer is legitimate interlocking; a tipped box breaks the layer top). The two
   physical terms dominate the cosmetic one: with equal weights, OI
   outvoted HM on the 28-box mix scenario and preferred a heavy-on-top
   layout over the heavy-low alternative (measured: heavy-low HM 0.337 /
   OI 0.214 vs heavy-high HM 0.358 / OI 0.071).
   `eps = 0.5·min_box_volume/capacity·realism_weight` — **bounded loss, not
   lexicographic**: realism can never cost more than half the smallest box's
   volume (never drops a box); eps < 1e-7 disables the term for that solve.
   Threaded through EVERY fitness site — batch
   (`dispatch._compute_fitness_pallet1_batch`), the three polish phases,
   sku_aware, the v2-hybrid comparison, and the (gated) restart comparison —
   so the whole search optimizes one scalar. No CoG-offset term on purpose
   (pre-recenter it measures fill level; the post-pass centres).

3. **Heavy-first seed**: gated `weight_desc` ordering in
   `make_informed_chromosomes` (active only when `heavy_on_bottom` AND
   realism is live AND weights vary — the smart-list length is part of
   default seeding, so it must not change for default callers).

4. **v2 candidate tie-break**: `PalletPacker.pack()`'s quality tuple gains a
   4th realism key when `realism_weight > 0`. Rationale: on dense orders no
   single greedy decode packs everything (measured: best single decode
   27/28), so the v2 warm-start's candidate set decides the final layout —
   and several candidates tie on `(unpacked, pallets, −util)`, where the
   historical first-wins tie is how heavy-on-top winners survived.

## Verification

- **Flag-OFF bit-identity**: fixed-seed default-config solves (geometric /
  constrained / homogeneous) produce byte-identical placements before vs
  after the change (verified in Docker against the pre-change image).
- **Scalar/batch parity**: |Δ| < 1e-12 across constrained and geometric
  decode paths, realism on and off (pytest, 4 parameterizations × 16
  chromosomes).
- **Scenario battery** (pallet_packer_api_service
  `scripts/realism_scenarios.py`, 19 scenarios over the live API):
  corner-jamming and off-centre CoG eliminated in all previously-failing
  scenarios; heavy_light_mix heavy-low; fragile/this-side-up unchanged
  (still perfect). See the service repo's ADR D14.
- `scripts/verify_realism_layer.py` (this repo) covers recenter/align units
  and the realism ordering properties.

## Known residual

In dense mixed layers a single straggler can still be TIPPED by the greedy
per-placement rotation argmax when no flat placement fits its gap; the align
post-pass cannot flatten it in place (footprint change infeasible or a
dependent rests on it). Measured: 3/19 battery scenarios retain exactly one
tipped box. The fix is the deferred decoder-level rotation tie-break below.

## Deferred (next report if pursued)

Decoder-level rotation-consistency tie-breaks and brick-bond interlock —
both require the Numba/Cython twin discipline (df2ea2c/9d9ab07 precedent)
and BR re-validation; the post-pass + fitness route delivered most of the
value without touching the decoders.
