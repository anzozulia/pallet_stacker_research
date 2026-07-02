# 35 — Hardening: post-pass safety, capacity cliff, boundary guards

**Status: shipped (2026-07-02).** Follow-up to report 34 (realism layer): an
adversarial evaluation of the shipped layer — a 29-scenario hostile battery
with independent geometry validation, old-vs-new image timing differentials, a
v2 warm-start scaling sweep, and two code audits with live probes — found four
core defects (two introduced by the realism layer, two pre-existing) plus a
set of boundary holes. This report covers the CORE-touching fixes; the
service-side fixes (schema bounds, body cap, rate-limiter fail-safe) live in
the API repo (`pallet_packer_api_service`, `docs/06_hardening_plan.md` is the
full plan with finding inventory F1–F15).

All fixes preserve the flag-OFF golden: fixed-seed default-config solves are
**bit-identical** to the pre-change image (the realism/postprocess flags
default OFF, `PalletPacker.pack`'s new deadline defaults to `None` = the
historical unbounded behavior).

## Fixes

### 1. Align pass could strand a dependent (F1, HIGH, introduced by report 34)

`align_orientations_pass`'s dependents gate only blocked dz-changing swaps, so
a *yaw* swap could pull the footprint out from under a box resting on top
(probe-confirmed: dependent left at support ratio 0.0) — and the replay
`validate()` was too weak to catch it. Two-part fix:

- **Full dependents skip** (postprocess.py): a box that supports anything is
  never re-rotated. Any dims change under a dependent is locally unrepairable,
  and a deviant by definition has non-dominant dims.
- **`validate()` hardened to the engine's feasibility stack** (validate.py):
  per-box `requires_full_support` (100% support regardless of the global
  ratio), centroid-over-supporter (gated on
  `config.require_centroid_supported`), zero-contact floaters at z > 0 are
  always invalid when any stability constraint is active, and the CoG check
  fires on explicit `pallet.cog_x_range`/`cog_y_range` only (the
  config-fraction default is not enforced on the geometric decoder path, so
  validating it would reject engine-legal packings).

**F15 (documented, deliberately NOT "fixed"):** load bearing in `validate()`
stays DIRECT-supporter. The v2 packer propagates load transitively
(`_propagate_load`) but the BRKGA JIT commit
(`_apply_load_contribution_njit`) charges direct supporters only — a
transitive validate would falsely reject decoder-legal results
(no-stricter-than-the-weakest-engine-path rule). The real fix is transitive
accumulation in the JIT commit — both decoder twins + BR re-validation —
grouped with the deferred decoder work.

### 2. Capacity cliff: unbounded v2 warm-start (F2, HIGH, pre-existing)

The v2 warm-start (auto-enabled by any finite constraint) has superlinear cost
on identical boxes: N=100 → 9 s, 200 → 40 s, 300 → 101 s, 400 → 175 s. Any
homogeneous constrained load ≥ ~300 boxes blew every budget (the API service
hard-killed at 120 s; report 30's "500 boxes is safe" was measured on a MIXED
catalogue only). A 1 s-budget 200-box request ran 21 s.

- **`PalletPacker.pack(boxes, time_limit_s=None)`** (packer.py): optional
  wall-clock deadline checked per-box in the greedy EP loop (remainder →
  unpacked), per-trial in `_multi_start`, and before every optional
  candidate-generation stage (safety net, block-building, MIP, layer
  building, ejection/consolidation/rebuild). `None` = historical behavior.
- **Budget-aware seeding** (driver.py): the v3.5 driver gives the v2 seed
  **half the solve budget** as its deadline (a partial pack still seeds the
  BRKGA) and skips it entirely when the slice is under 1 s.

Measured after: the 500-identical-capped scenario that always hard-killed at
120 s completes in ~44 s end-to-end; the 1 s-budget case honours it within
spawn overhead (~4.5 s wall through the service).

### 3. Bounded-loss guarantee voidable (F3, MED, introduced by report 34)

`realism_weight > 2` silently made dropping the smallest box profitable
(ε unclamped). Now `eps = 0.5·(min_vol/cap)·min(1.0, realism_weight)` —
the weight is a dial in [0, 1], values above 1 clamp (realism.py).

### 4. Boundary guards (C4–C7, LOW/MED)

- **`input_validation.py`**: box `weight ≥ 1e15` rejected — weights at or
  above the decoders' finite `_NO_LIMIT` sentinel (1e18) exceed every cap
  *including the sentinel* and become silently unpackable even on an
  unlimited pallet; 500 × 1e15 = 5e17 keeps a full request under the
  sentinel.
- **`postprocess.py`**: `recenter_pass` skips pallets with explicit
  `cog_x_range`/`cog_y_range` (the envelope owner decides placement; the
  centred fractional envelope needs no guard — a rigid shift toward deck
  centre can only move the weight-CoG deeper into it, and weightless loads
  short-circuit the engine's check). `apply_postprocess` now shares ONE
  global align budget (default 4 s) across all pallets instead of
  2 s/pallet — a many-pallet result can no longer eat a caller's margin;
  over-budget pallets still get the O(N) recenter.
- **`realism.py`**: `build_realism_context` returns `None` (term disabled,
  logged) on duplicate box ids — the scalar fitness path maps placements by
  id while the batch path is positional; duplicates would silently desync
  the two objectives and corrupt the search.

## Files changed

`pallet_packer/validate.py`, `pallet_packer/postprocess.py`,
`pallet_packer/packer.py`, `pallet_packer/input_validation.py`,
`pallet_packer/_brkga_core/driver.py`, `pallet_packer/_brkga_core/realism.py`
(+ `models.py` comment for the clamped dial). No decoder (Numba/Cython twin)
changes — backend equivalence is unaffected.

## Verification

- **Flag-off golden**: fixed-seed default-config fingerprints (geometric,
  constrained, homogeneous cases) bit-identical old image vs new.
- **Scalar/batch fitness parity** suite: |Δ| < 1e-12 with realism on/off.
- API-repo container suite: 93 passed (unit coverage for every fix above:
  yaw-swap-under-dependent, centroid-over-gap, RFS under-support, sr=0
  floater, direct-not-transitive load contract, explicit CoG range, ε clamp
  property, deadline bounds + conservation, sentinel rejection, dup-id
  disable, recenter-skip, global-budget skip).
- 22-scenario realism battery: 19/22 acceptable — unchanged from report 34's
  baseline; the 3 fails are the known deferred TIPPED_IN_LAYER decoder
  residual (see report 34, "Known residual").
