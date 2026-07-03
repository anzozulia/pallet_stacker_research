# 38 — Hardening rounds 4–5: evaluation of the round-3 fixes + consistency polish

**Status: shipped (2026-07-04).** Round-4 adversarial evaluation of the
round-3 load-model work (report 37): a 300-case repair fuzzer, a
7,200-decode rider/overlay-hostile physics fuzz (plane-aligned stacking,
same-SKU blocks, overhang, multi-pallet — the geometry engineered to
stress the new code), decoder perf/quality A/Bs against the pre-round-3
build, and two independent line-level audits.

**Result: the round-3 fixes hold — zero physics violations in 7,500
probes; no wrong-accept path exists** (conservative rider booking is
reject-only; `blk_inc` provably all-zero at every rider-scan site; the
repair closure argument is airtight including partial support and
cross-pallet). Round 5 ships the two residual core items:

## Core changes (round 5)

1. **Pallet-cap tolerance unified (R1).** Round 3's `load_tol` was applied
   to v2's `feasible` weight gate but not to `validate()` (still absolute
   `EPS`) or the JIT twins' cap compares (still `1e-6`, sub-ulp above
   ~4.5e9). An engine-legal at-limit heavy plan could fail replay
   validation — the "flagged then shipped with noise" pattern round 3
   eliminated for `max_load_on_top`. Now all four surfaces use
   `max(1e-6, 1e-9·limit)`: `validate.py`, both twins (`eps_w` per decoder,
   6 cap sites per twin file), v2 unchanged.
2. **Guard comments (R8).** The block shrink-fallback's rider re-check is
   LOAD-BEARING (the fallback checks omit `inherited_blk` while Phase 3
   books it) — marked in both twins so it is never removed as "redundant".
   Stale "DIRECT-supporter" docstring in `validate.py` fixed; notes added:
   the no-Cython scalar fallback can resolve realism fitness ties
   differently from the batch path (same seed → possibly different plan
   across host types; per-host determinism holds), and `repair.py`'s
   topmost-feeder order deliberately favors hover-freedom + determinism
   over minimality under the direct model.

## Measured and deferred (R7)

The F21 rider scan costs ~2× RAW decode throughput at n=500 on
plane-aligned worst-case geometry (922→496 scalar decodes/s, 1807→958
batch chrom/s; ~20% at n=200; noise at n≤60) — but ZERO measured
end-to-end effect: identical service wall times and bit-identical
fixed-budget solution quality (196/500 packed, util 0.7458, two seeds,
both builds). Designed fix on file: a per-pallet distinct-bottom-z plane
set consulted before the O(n) rider scan (~#layers entries; the scan then
runs only when the candidate's top matches an existing plane). Deferred
until profiling ever shows decode-bound solves.

## Verification

`verify_backend_equiv_cstr.py` PASS (2670 comparisons, 0 mismatches, all
round-2/3 coverage counters fired — the epsilon battery sweeps the cap
boundary); `verify_load_physics.py` PASS (14,544 decodes, 0 violations);
BR smoke bit-identical; API-service goldens bit-identical (cap ≤ 900 →
tolerance change provably inert); service suite 135 green incl. a new
tolerance-band regression (limit 1e6 + 2e-4 accepted, +10 rejected).

The API-service side of rounds 4–5 (fail-fast config, responsive crash
detection, pre-body rate limiting, poll limiter, env-gated CORS, reporting
honesty in `to_json`) is documented in the service repo,
`docs/09_hardening_round4.md`.
