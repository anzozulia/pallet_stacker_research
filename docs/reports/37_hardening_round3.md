# 37 — Hardening round 3: load-model completeness (block overlay, under-fill, v2 diamond flow)

**Status: shipped (2026-07-03).** Follow-up to report 36: a third
adversarial evaluation introduced a NEW verification modality — a
decode-level randomized PHYSICS gate with an exact independent oracle — and
immediately found what two rounds of backend-equivalence testing could not:
the constraint decoder twins were **identically wrong** in two ways, and
bit-identity testing is structurally blind to that. This round is the
project's second modification of the decoder twins.

## Why equivalence testing missed this

`verify_backend_equiv_cstr.py` proves Cython == Numba. The block decoder's
Phase 2c comment even says "kept bit-equivalent on purpose" — both twins
faithfully implemented the same defective model. The new standing gate
`scripts/_verify/verify_load_physics.py` re-validates every decode output
against exact physics (overlap / bounds / support / deck / direct-and-
transitive load): **90 violations across 14,544 decodes pre-fix → 0
post-fix.** Measured end-to-end pre-fix: 3/30 full solves at API-service
defaults returned physically overloaded plans (up to 1.6×), and the volume
fitness actively *selects* them — an overloaded stack packs more, so the GA
converges to the violation whenever it beats the legal optimum.

## Fixes (all unconditional — physics bugs, not flags)

### 1. Block sibling-column overlay (F20, HIGH)

Phase 2c checked each of a k×l block's bottom columns against the SAME
pre-block `placement_top_loads`, then Phase 3 committed all of them: a
shared external supporter never saw the aggregate (four 10 kg columns each
passed against mlot=25; 40 kg landed). Pre-existing in the direct model
(k·l·w vs one w per check); round 2's transitive relay made it m× worse
(k·l·m·w). Fix: the check functions gained a `pending_loads` overlay
argument (all-zero from every non-block caller — legacy results
unchanged); Phase 2c accumulates each ACCEPTED column's contribution into
the overlay (the existing apply routines pointed at `blk_inc` instead of
the real loads) before checking the next sibling, then wipes the overlay
and commits exactly as before. No float-subtraction rollback anywhere.

### 2. Under-fill rider rule (F21, HIGH)

A box placed later with its top plane exactly at an existing box's bottom,
XY-overlapping it, becomes a NEW physical supporter and inherits a
contact-share of that rider's outflow — previously unchecked and unbooked
(confirmed decode trace: a fragile mlot=0 box committed LAST at z=0 under a
loaded slab, silently carrying its share; all 6 modes + the v2 engine).
Fix: new twin helper `_rider_inflow` — per rider R,
`share = out_R · a / (T_old + a)` with `out_R` = R's weight (+ R's booked
load under the transitive model). Every placement path (including floor
placements in occupied bins — the confirmed case) now rejects when the
inherited load exceeds the candidate's own `max_load_on_top`, feeds
`weight + inherited` to the transitive dry-run and commit, and books the
inherited load on the candidate's row. Old supporters are deliberately NOT
debited: strictly conservative, no negative propagation, no float dust.
Blocks with riders on any column top shrink to a single box with exact
semantics. Zero-weight riders contribute nothing (weightless workloads
bit-identical). Mirrored in v2 `feasible`/`_commit`.

### 3. v2 diamond flow drop (F22, MED)

v2's recursive `_propagate_load` used a visited set that ADDED a
re-converged diamond node's second share but BLOCKED its onward
distribution — everything below the junction permanently undercounted, and
v2's round-2 dry-run check (exact) disagreed with v2's own commit
(measured: cache 3.8 vs true 4.6 against a 4.3 limit — over-accepted).
Replaced by the same accumulate-then-distribute worklist the dry-run uses
(descending bottom-z, deterministic ties); `regen_top_load` inherits.

### 4. Scale-aware load tolerance (F25, LOW)

All load/weight comparisons used `+ 1e-6` absolute — below one double ulp
for limits ≥ ~4.5e9 (the API contract allows 1e12), so exactly-at-limit
stacks flipped on accumulation-order noise and the engine and validator
could disagree. All sites (twins, v2, validate, the new repair module) now
use `max(1e-6, 1e-9 · limit)` (`models.load_tol`); identical behavior for
limits ≤ 1e3. `input_validation` also bounds finite `max_load_on_top` to
the same < 1e15 cap as weights (it was previously unbounded — a sentinel-
headroom seam).

### 5. Library QoL

`dispatch._BATCH_AVAILABLE` was set but never consulted — hosts without
compiled Cython crashed with NameError on generation 0 of every constrained
solve, contradicting the module header's Numba-fallback promise.
`decode_population_fitness` now falls back to the docstring's reference
loop (per-chromosome scalar decode + scalar fitness), pinned bit-identical
to the batch path by test.

## Verification

- **Backend equivalence (extended first):** deterministic block-sibling,
  under-fill, and 1e12-epsilon cases added; new coverage counters
  (`block_joint_rejections_seen`, `underfill_rejections_seen`,
  `epsilon_scale_seen`) gate the PASS. Pre-fix run: FAIL with a round-3
  coverage hole (defect proof). Post-fix: **PASS — 2670 comparisons, 0
  mismatches, all counters fired.**
- **NEW `verify_load_physics.py`:** two seeded sweeps (mixed settings +
  API-service defaults, 14,400 decodes) + deterministic regressions (the
  live-served diamond, the block 5-box minimal, the under-fill pillar/slab/
  fragile case). Pre-fix: **90 violations**. Post-fix: **0**. This gate is
  now part of the standing verify battery.
- BR1/3 smoke bit-identical to the tracked checkpoint (geometric decoders
  untouched by construction; BR has no load constraints).
- API-service side: full suite green; flag-off goldens bit-identical
  (including the load-constrained scenario — its winning trajectory never
  crossed the fixed paths); end-to-end violation rate 0/30 (was 3/30);
  production-settings decode fuzz 0/7200 (was 23/7200).

## Files changed

`pallet_packer/models.py` (load_tol), `packer.py` (_apply_load_flow,
_rider_inflow, feasible/_commit, regen_top_load), `validate.py`,
`input_validation.py`, `repair.py` (NEW — deterministic load repair used by
the API service), `_brkga_core/jit_constraints.py` + `jit_constraints_cy.pyx`
+ `.pxd` (pending_loads param, _rider_inflow twin, scale-aware eps),
`_brkga_core/jit_decoders_cstr.py` + `jit_decoders_cstr_cy.pyx` (overlay
Phase 2c, rider checks + booking, batch overlay planes),
`_brkga_core/dispatch.py` (_BATCH_AVAILABLE fallback),
`scripts/_verify/verify_backend_equiv_cstr.py` (round-3 battery +
counters), `scripts/_verify/verify_load_physics.py` (NEW standing gate).

## Deferred (unchanged)

Decoder rotation-score shaping (F18 root; TIPPED residual), symmetric
overhang convention, geometric-decoder deck rule, scalar/batch geometric
overhang-dims unification (unreachable via the driver; noted in dispatch).
