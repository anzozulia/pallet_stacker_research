# 41 — Hardening round 8: per-assembly toppling certifier (F36) + single-box CoG parity (F37)

**Status: shipped (2026-07).** Round-8 adversarial evaluation of the service
stack after round 7 (report 40). The round-7 fix held (config forwards
verbatim into every grouped/multibin decode; the twins are bit-identical; the
`0.0*inf → nan` min-load gate is guarded everywhere; nothing engine-reachable
ships unsafe). Two residuals in the certifying validator, both fixed in
`validate.py` — pure Python, NO decoder twin.

## F36 — per-assembly toppling under overhang (the report-40 deferred residual)

F30 (report 39) keeps each floor box's centroid over the deck; the stacked
rule keeps each box's centroid over its immediate SUPPORTER (which can itself
overhang); D20/F35 (report 40) keeps the WHOLE-pallet CoG over the deck. None
checks a connected sub-ASSEMBLY. A detached sub-tower rooted on an overhanging
floor box tips over the deck edge while a disjoint counterweight keeps the
whole-pallet CoG central. Hand-proven (L=1200, overhang=300): M(200×1000×1000,
w1000)@x0 · A(1000×1000×100, w100)@x400 (deck-contact 0.80, F30-ok) ·
B(200×1000×100, w400) on A's overhang tail @x1200 → whole-pallet CoG x=473.3 ∈
[0,1200] (D20 passes), component {A,B} CoG x=1220 > deck edge 1200 → tips.

### Core change

`validate.py` gains section 8: partition placements into rigid assemblies —
connected components under the VERTICAL resting-on relation (reusing the
`sup_cache` edges already built for the support checks) — and require each
assembly's weighted CoG to lie within the convex hull of its own deck-contact
region (the union of its floor boxes' on-deck rectangles). Gated exactly like
F30 (`allow_pallet_overhang and require_centroid_supported`). No decoder twin.

The corrected model matters: joining by ALL face contacts (the "union-find"
sketched as the report-40 residual) is physically wrong — a vertical side face
gives no tipping restraint, so it would false-merge a side-abutting
counterweight and miss the tip, and miss two side-by-side stacks that each tip
independently. `_contact_area`'s `abs(z−z2)>EPS` guard already restricts edges
to vertical (resting-on) contacts. The convex hull (not the bounding rect) is
the physically-correct support polygon.

### Scope: validate certifier, engine-side reject deferred

The engine already avoids these layouts: the running-CoG reject rejects the
tipping box on every decode order unless the counterweight lands first, and
heavy-high placement is realism-penalised. A round-8 reachability probe (full
`brkga_pack_v35`, service config, ≥180 adversarial instances incl.
counterweight-forcing generators, two independent detectors) measured 0
engine-produced tips — the same standard of evidence as F35's 0/160. So the
validate-only certifier is proportionate (it completes the certifying gate and
protects postprocess edits via the existing revert path); the engine-side
per-assembly running reject in both twins stays deferred, contingent on the
probe ever finding a reachable case (measured 0/210 this round).

### Inert without overhang (proof)

Without overhang a floor box's deck-contact rectangle IS its full footprint, so
the per-box centroid-over-supporter rule inductively puts every box's centre —
hence the weighted CoG, a convex combination — inside the assembly's
deck-contact hull. §8 can never fire → goldens/BR/geometric callers
bit-identical.

## F37 — single-box CoG parity

`validate.py` §7 now skips a single-placement pallet (`len(placements) > 1`).
The engines never CoG-check the first box (v2 `_cog_ok`'s `not self.placements`
bypass; the JIT cstr new-bin path seeds the box with only
`_apply_cog_contribution_njit`, no `_check_cog_envelope_njit`), so flagging a
lone corner box under an explicit `cog_x_range` violated validate's
no-stricter-than-the-engine / empty-error-list contract.

## Gates

No decoder twin changed → the backend-equivalence campaign is unaffected
(2670/0) and the standing physics gate is untouched (CoG/assembly is not a
physics-oracle criterion in the validate-only path). BR smoke + all three
goldens bit-identical (F36/F37 inert without overhang). The new
`SUBASSEMBLY_TOPPLE` oracle criterion + twinned per-assembly reject would only
be added under report 41's contingent engine branch, which the probe did not
trigger.
