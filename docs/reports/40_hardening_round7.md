# 40 — Hardening round 7: sub-assembly toppling under overhang (deck-footprint CoG envelope)

**Status: shipped (2026-07).** Round-7 adversarial evaluation of the service
stack after round 6 (report 39), aimed at the freshest surfaces since rounds
2–6 exhausted the load model. The newest code held (the round-6 multi-pallet
objective is clean under a 300+-instance audit; a suspected cold-start JIT
cost was refuted — the image is Cython AOT). One real gap:

**F35 — sub-assembly toppling under overhang.** F30 (report 39 / ADR D18)
made each floor box keep its centroid over the deck, and the stacked rule
keeps each box's centroid over its immediate supporter — but nothing checks a
connected SUB-ASSEMBLY. A box stacked on the overhanging part of a floor box
has its CoM over the supporter (passes) yet past the deck edge; the combined
`{floor+stacked}` weighted CoG can project off the deck-contact region and the
pair tips as a unit, while `validate()` returns clean (hand-proven at the
default config: combined CoG x=1063.6 > deck edge 1000). NOT engine-reachable
(0/160 forcing solves; recenter + heavy-low realism steer away) — a validator
blind spot. Also corrects a round-6 framing: F30 does not "transitively bound
the whole-pallet CoG" — a box on an overhanging supporter has its CoM over the
supporter, which extends past the deck.

## Core change

The fix reuses the existing CoG-envelope machinery — no new engine code, no
decoder-twin edit. The service adapter auto-activates a **deck-footprint CoG
envelope under overhang** (`cog_envelope_fraction=0.5` → `[0,L]×[0,W]`,
`cog_check_min_load_fraction=0.0`). The CoG envelope is a per-placement
running-CoG reject in both the v2 engine (`_cog_ok`) and the JIT decoders
(`_check_cog_envelope_njit`), so activating it makes the search AVOID
off-deck-CoG layouts during decode (verified: rejects the hand case). The
fraction path keeps the recenter realism pass alive (explicit cog ranges
would disable it).

The single core change is in `validate.py`: its CoG check now also fires for
the config-fraction envelope UNDER OVERHANG (gated `allow_pallet_overhang and
cog_envelope_fraction < 1.0` — exactly the regime where the constraint
decoders enforce it, since overhang forces the cstr path), keeping validate
no-stricter-than-the-engine (a purely geometric no-overhang caller stays
unchecked, matching the geometric decoders that never enforce CoG). No Cython
twin is involved.

Whole-pallet scope is sufficient for every engine-reachable case; the
isolated-overhanging-sub-tower residual (whole-pallet CoG central while a
detached sub-tower tips) is a documented deferred residual. See ADR D20.

## Gates

Inert without overhang (`fraction=1.0` → `cog_active` off) → BR smoke and all
three goldens bit-identical; the cstr CoG path is already in the equivalence
sweep (2670/0 unchanged); the physics gate is unaffected (CoG is not a
physics-oracle criterion). Service suite green; realism battery 19/22
(the single overhang scenario stays balanced → PASS).
