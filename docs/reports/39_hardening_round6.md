# 39 — Hardening round 6: floor toppling rule + multi-pallet objective

**Status: shipped (2026-07-06).** Round-6 adversarial evaluation of the
service stack after round 5 (report 38), aimed at the least-probed
subsystems since rounds 3–5 had exhausted the load model: a 500-scenario
full-service-path physics fuzzer (postprocess + repair in the loop, oracle
independent of `validate.py`), a targeted floor-toppling probe, and breadth
audits of multi-pallet/group correctness and serve-time reporting.

**Result: the load/support/geometry model held (0 violations / 0 crashes in
500 full-path scenarios), but two engine-side gaps surfaced.** Both fixed
here.

## Core changes

1. **Floor toppling rule (F30 / service ADR D18).** Round 2's F17 gave
   overhanging floor boxes the deck-contact RATIO rule but not the toppling
   half: at `support_ratio < 0.5` a floor box can pass the ratio while its
   footprint centroid projects PAST the deck edge (majority of the mass
   cantilevered off — it tips on placement), because the stacked-box
   centroid rule explicitly exempts floor boxes. Reachable at will
   (105/120 engine solves at sr<0.5 produced ≥1 validate-clean toppling
   floor box). The fix completes F17 on all four surfaces — v2
   `PalletState.feasible`, both JIT twins
   (`_check_load_on_top_njit` / `_ck_load_on_top`, the single shared floor
   branch every cstr decoder path already calls), and `validate()` —
   gated on `require_centroid` and inert unless overhang inflates the
   container. The twin form is integer-exact (`2*x + dx > 2*x_hi`);
   a centroid EXACTLY on the deck edge accepts (mirrors the stacked rule).

   **Inertness boundary.** Categorically inert with overhang off (the
   `pallet_l <= 0` sentinel returns before the check). Provably inert for
   `sr ≥ 0.5 + 1e-6` (area ratio ≤ min per-axis ratio). At sr = 0.5 exactly
   it is exact on integer grids for dims ≤ 5·10⁵; only float-dim v2 at
   exactly sr = 0.5 under overhang sits on a physical knife edge and is not
   bit-identity-promised. BR (geometric path, never calls the constraint
   core), the flags-off goldens, and the equivalence/physics batteries are
   provably unaffected.

2. **Multi-pallet objective (F31 / service ADR D19).** The BRKGA fitness —
   base AND realism, scalar (`_fitness_pallet1`) AND batch
   (`_compute_fitness_pallet1_batch`) — masked to `pallets[0]`. For
   `max_pallets > 1` without groups the search was blind past the first
   pallet: packing 10 boxes on pallet 1 scored identically to dropping
   them (0.975 both ways, reproduced). Fix: gate on `max_pallets != 1`
   (includes the library `<= 0` auto-bins mode) and score
   `v_unpacked/cap + β·n_pallets + 0.5·eps·R_over_all_pallets`,
   `β = 0.5·min_vol/cap` — strict dominance (a packed box always beats a
   drop by ≥ 0.5β; fewer pallets win at equal volume; realism a pure
   tiebreak; the eps halving is required so a full realism swing can't flip
   a pallet-count decision at realism_weight=1). `max_pallets == 1` runs
   the exact historical statements byte-for-byte. Realism gained an
   `all_pallets` flag on both paths; no Cython twin is involved
   (fitness/realism/dispatch-fitness are pure Python). `_pack_with_groups`
   also stopped silently dropping ~10 tuning params for its per-pallet
   sub-solves.

   **Measured:** +173 boxes packed across 16 heterogeneous no-group
   `max_pallets=3` A/B solves (0 worse, identical pallet counts, all
   valid); scalar/batch parity holds at |Δ| < 1e-12 on multi-bin
   populations.

## Gate extensions

- `scripts/_verify/verify_load_physics.py`: a FLOOR_TOPPLE oracle
  criterion (gated on the config's centroid flag — an rc=0 decode may
  legally topple) and a new low-sr/full-overhang scenario family
  (`rand_instance_topple`, appended sweep C — existing generators' RNG
  draw order untouched). Pre-fix FAIL (2,402 violations / 18,864 decodes;
  the sr=0.8 production sweep B stays at 0), post-fix 0.
- `scripts/_verify/verify_backend_equiv_cstr.py`: `run_round6_battery`
  (mode-2 deterministic topple/boundary/one-past/rc0 cases, cy vs nb) and
  a `floor_com_rejections_seen` coverage counter gating PASS.

BR smoke bit-identical; the F31 change is verified for validity/conservation
by the existing `verify_multipallet_conservation.py`.
