# 36 — Hardening round 2: dims bounds, floor deck rule, transitive load

**Status: shipped (2026-07-03).** Follow-up to report 35: a second adversarial
evaluation of the hardened stack found two HIGH physical/numeric defects and a
quality defect, all in the CORE. This round is the project's **first
modification of the constraint decoder twins** (Numba + Cython), verified by
an extended backend-equivalence campaign.

## Fixes

### 1. Spatial dimension bound (F16, HIGH)

Dims were unbounded; they land in int64 arrays and the JIT hot paths form
int64 products that WRAP SILENTLY: contact-area math corrupts from dims
~3e9 (support/load checks evaluate garbage → physically invalid plans
returned), the batch fitness volume sum corrupts from ~2.1e6 (scalar/batch
parity contract violated — the scalar twin uses exact Python arithmetic),
and ≥2⁶³ raises a raw OverflowError. Fix: `MAX_DIM = 1_000_000` in
`input_validation.py` (with dims ≤ 1e6, every area ≤ 1e12, volume ≤ 1e18,
and per-pallet volume sum ≤ capacity ≤ 1e18 provably fit int64). The API
schema mirrors the bound.

### 2. Floor deck-contact rule under overhang (F17, HIGH)

Every engine path treated z=0 as unconditionally supported — historically
safe, but overhang inflates the decoder container bounds (L_eff = L + ov),
so floor boxes were placed FULLY OFF the deck, floating in air (live probe:
6/12 boxes at 0% deck contact, validator-clean, served `done`). Fix: when
overhang is active, a floor placement needs deck-contact ≥ its effective
support ratio (RFS → 1.0). Enforced in `_check_load_on_top_njit`'s z≤0
branch (raw deck dims passed only under overhang; the 0 sentinel keeps the
legacy instruction path bit-identical), in the decoders' new-bin bypass
paths, in v2 `PalletState.feasible`, and in `validate()`. No
centroid-over-deck on purpose (v2 parity: no centroid rule at floor level).
Inert at overhang == 0 → golden/BR unchanged by construction.

### 3. Transitive load bearing (F19, MED→physical)

Both engines under-enforced `max_load_on_top`: the BRKGA commit was
direct-supporter-only, and v2's check-direct/commit-transitive model NEVER
rejects a fresh pure column (each new box's direct supporter carries only
its immediate rider) — a 10-stack of individually-legal links left the
bottom box at **8.6×** its limit. Fix, gated by NEW
`PackerConfig.transitive_load_bearing` (default OFF = bit-identical
historical behavior): a transitive dry-run CHECK
(`_check_load_transitive_njit` — the essential piece; accumulation alone is
insufficient) plus a transitive commit sibling
(`_apply_load_contribution_transitive_njit`), worklist walk highest
bottom-z first with lowest-row-index tie-break (the twin bit-identity
contract), caller-owned scratch (Cython nogil constraint), O(k·n) per
commit. The block decoder relays each bottom box's full column (m·w)
externally under the flag. Mirrored in v2 `_load_bearing_ok` and in
`validate()` (transitive accumulation before the per-box comparison).
Measured: 10-stack max transitive load ratio 0.95 with the flag (vs 8.57).

### 4. Library-only deadline polish

`pack(time_limit_s=0)` meant unbounded (falsy check) — now expired; stage
budgets (MIP, ejection, consolidation, layer loop) are clamped to the
remaining deadline.

## Twin campaign verification

- `scripts/_verify/verify_backend_equiv_cstr.py` **extended first**: the
  runners now pass raw `pallet_l/pallet_w`, the sweep alternates the
  transitive flag (180/360 instances) and overhang (180/360), and a
  deterministic round-2 battery proves both new branches FIRE (behavior
  differs vs sentinel/flag-off) while staying cy==nb. Coverage counters gate
  the PASS. Result: **2670 comparisons, 0 mismatches; deck branch fired 5×,
  transitive branch fired 5×.**
- BR smoke (thpack1/3) bit-identical to the checkpoint; smoke_baseline +
  reverify_regression_equivalence outputs byte-identical to the pre-change
  baseline; flags-off API golden bit-identical.
- Pre-existing (unchanged, recorded): verify_backend_equiv_geom mode4/5
  exceptions (script drift) and verify_batch_consistency's NON-INT-dims
  divergence — both present before this round, outside the integer input
  contract; `dispatch._BATCH_AVAILABLE` is set but never consulted (no-Cython
  hosts crash on constrained batch decode; Docker unaffected).

## Files changed

`pallet_packer/input_validation.py`, `models.py`, `packer.py`,
`validate.py`, `_brkga_core/jit_constraints.py` + `jit_constraints_cy.pyx`
+ `.pxd`, `_brkga_core/jit_decoders_cstr.py` + `jit_decoders_cstr_cy.pyx`,
`_brkga_core/dispatch.py`, `scripts/_verify/verify_backend_equiv_cstr.py`.

## Deferred

Decoder rotation-score shaping (the F18 root — exact-fit tilings broken by
the greedy rotation argmax; the API mitigates by forcing the v2 seed for
small instances, 7/8 → 8/8), explicit NO-GO this round: it needs its own
twin campaign + BR sensitivity study. Geometric-decoder deck rule
(no support semantics there by design; validate() covers replay).
