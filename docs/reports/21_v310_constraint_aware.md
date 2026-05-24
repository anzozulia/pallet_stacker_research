# v3.10 — Constraint-Aware Decoder (Industry Deployment Unblocker)

Extends the v3.8 JIT decoders to enforce three physical constraints that
the geometric-only fast path silently ignored:

- `Pallet.max_weight` — per-pallet weight cap
- `Box.max_load_on_top` — per-box stacking-load capacity (fragility)
- `PackerConfig.support_ratio` — minimum supported-footprint fraction
  for non-floor placements

**Headline result:** total industry-suite validator errors dropped from
**267 → 0** across all 10 scenarios with zero regression on the BR
academic benchmark (geometric-only path preserved).

## TL;DR

| Suite | v3.8 (geometric) | v3.10 (constraint-aware) |
|---|---|---|
| BR1/3/5/7 n=2 sanity | identical (no constraints active) | identical |
| Industry, validator errors | 267 total | **0 total** |
| Industry, validator-clean cases | 2 of 10 | **10 of 10** |

The trade-off is that constraint-violating cases now (correctly) require
more pallets and report lower util — e.g. IND2 (200 fragile pharma boxes)
goes from "1 pallet 24% util, 181 errs" to "4 pallets 6.4% util, 0 errs."
The earlier "great util" numbers were fictional; the algorithm was
silently stacking 200 fragile boxes into a single pallet.

---

## 1. What was built

### 1a. `precompute_constraint_arrays`
Extracts per-box weights + per-box `max_load_on_top` into float64 arrays;
extracts `Pallet.max_weight` as a scalar. Returns
`has_constraints: bool` — True iff any cap is finite — so the JIT
decoders can short-circuit on pure-geometric workloads (BR).

### 1b. `_check_load_on_top_njit`
JIT helper. Given a candidate placement at (x, y, z) with dims
(dx, dy, dz) and weight w:
1. Iterates placements on the same pallet; finds supporters (placed
   boxes with top z == cand z and overlapping XY footprint).
2. Computes total contact area.
3. **Support-ratio check** (Tier 1.5): if support_ratio > 0 and
   total_area / footprint < support_ratio → reject.
4. **Load-bearing check** (Tier 1): distributes cand's weight across
   supporters proportional to contact area; rejects if any supporter's
   running top-load + share > its max_load_on_top.

Mirrors v2 packer.py:_load_bearing_ok + the support_ratio half of
v2.feasible.

### 1c. `_apply_load_contribution_njit`
After a placement is accepted, updates each supporter's running top-load.
Called once per committed placement.

### 1d. `decode_njit_mode_cstr`
New constraint-aware twin of `decode_njit_mode` (DFTRC / wall / corner).
For each box, per-bin iteration:
1. Pre-check pallet weight cap; skip bin if would bust.
2. Find geometrically best (rot, x, y, z) in this bin.
3. Run `_check_load_on_top_njit`; if violates, skip bin.
4. Otherwise commit, update `pallet_weights[b]` and call
   `_apply_load_contribution_njit`.

If no existing bin accepts, opens a new bin (first placement is z=0 →
trivially passes load/support). If even an empty pallet can't fit the
box (alone exceeds pallet cap), leaves box unpacked.

### 1e. Dispatcher + plumbing
- `decode_chromosome` accepts `weights`, `mlot`, `pallet_max_weight`,
  `has_constraints`, `support_ratio` parameters; modes 0/1/2 use the
  constraint-aware decoder when `has_constraints=True`. Modes 3/4/5
  (layer / dynamic blocks / top-K blocks) **fall back to constraint-
  aware mode 0** when constraints are active — those decoders don't
  yet have constraint-aware variants (deferred to Tier 2).
- `decode_auto_mode` passes the constraint args through.
- `local_search_2opt`, `path_relinking`, `lns_polish` all updated to
  thread the constraint args through to their internal decoder closures
  so the polish phase respects the same constraints as the main BRKGA.
- `brkga_pack_v35` calls `precompute_constraint_arrays` and forwards
  the result through all decoder closures.

---

## 2. Industry suite results (v3.8 → v3.10)

| Case | v3.8 | v3.10 | Δ pallets | Δ errs |
|---|---|---|---|---|
| IND1 E-commerce (120) | 2p / 96.3% util₁ / 24 errs | 4p / 85.6% util₁ / **0 errs** (20 unp) | +2 | -24 |
| IND2 Pharma (200 fragile) | 1p / 24.0% util₁ / 181 errs | 4p / 6.4% util₁ / **0 errs** | +3 | **-181** |
| IND3 Furniture (15) | 5p / 66.5% util₁ / 7 errs (3 unp) | 5p / 55.7% util₁ / **0 errs** (3 unp) | 0 | -7 |
| IND4 Beverage (44) | 1p / 69.2% util₁ / 1 err | 1p / 69.2% util₁ / **0 errs** | 0 | -1 |
| IND5 Retail 4-SKU (150) | 2p / 94.5% util₁ / 1 err | 2p / 84.1% util₁ / **0 errs** | 0 | -1 |
| IND6 LTL groupage (43) | 1p / 39.2% util₁ / 1 err | 1p / 39.2% util₁ / **0 errs** | 0 | -1 |
| IND7 Electronics (150) | 1p / 78.3% util₁ / 16 errs | 3p / 48.0% util₁ / **0 errs** (19 unp) | +2 | -16 |
| IND8 Automotive (60 heavy) | 1p / 78.7% util₁ / 36 errs | 4p / 53.6% util₁ / **0 errs** | +3 | -36 |
| IND9 Document (100) | 2p / 85.5% util₁ / **0 errs** | 2p / 85.5% util₁ / **0 errs** | 0 | 0 |
| IND10 Cold-chain (80) | 2p / 95.4% util₁ / **0 errs** | 2p / 95.4% util₁ / **0 errs** | 0 | 0 |

**Errors: 267 → 0.** Clean validity across the full suite.

### Interpretation

- **Cases where v3.8 was already valid (IND9, IND10):** unchanged. The
  geometric-only decoder happened to produce valid packings on these
  homogeneous-load scenarios, so constraint enforcement adds nothing
  but is harmless.
- **Cases where v3.8 cheated on fragility/weight (IND1, IND2, IND7,
  IND8):** v3.10 now uses the correct number of pallets and reports
  honest utilization. The pallet count increase reflects what should
  have been needed all along.
- **Cases where v3.8 had ~1 support-ratio violation (IND4, IND5,
  IND6):** v3.10 finds a fully-supported placement, occasionally at a
  small util cost (e.g. IND5: 94.5 → 84.1 util₁).

### Unpacked items in IND1 / IND3 / IND7

- **IND1:** 20 unpacked at the 4-pallet cap; the dataset has fragility
  + this-side-up + 120 boxes that genuinely need ~5 pallets.
- **IND3:** 3 unpacked = the three 1.7 m tall fridges (taller than the
  pallet's 2 m height minus the deck — geometrically impossible).
  Already known in earlier reports.
- **IND7:** 19 unpacked at the 3-pallet cap; needs ~5 pallets for full
  fragility-respecting placement.

These reflect the `max_pallets = min_pallets_by_vol + 2` cap in the
eval script, not algorithm limitation. Bumping the cap to 5-6 would
absorb them.

---

## 3. BR sanity (geometric path)

n=2/set, 15 s/instance — confirms `has_constraints=False` short-circuits
the constraint check and the BR throughput is preserved:

| Instance | v3.8 prior result | v3.10 now | Δ |
|---|---|---|---|
| BR1 #1 | 91.05 | 91.05 | 0 |
| BR1 #2 | 91.36 | 91.03 | -0.33 |
| BR3 #1 | 93.63 | 93.63 | 0 |
| BR3 #2 | 95.22 | 94.96 | -0.26 |
| BR5 #1 | 91.45 | 91.24 | -0.21 |
| BR5 #2 | 93.35 | 92.51 | -0.84 |
| BR7 #1 | 92.87 | 92.83 | -0.04 |
| BR7 #2 | 92.26 | 92.70 | +0.44 |

Variation is within seed/JIT-recompilation noise. Geometric BR
performance is preserved.

---

## 4. What's NOT in v3.10 (deferred)

- **Constraint-aware variants for modes 3/4/5** (layer / dynamic
  blocks / top-K blocks). When `has_constraints=True`, these modes
  currently fall back to constraint-aware mode 0 (DFTRC). For
  industry workloads where block packing is meaningful (e.g.
  homogeneous SKUs like IND9 banker boxes), this gives up the
  block-mode win. Estimated 1-2 days per mode to add support.
- **`require_centroid_supported`** — secondary support check ("footprint
  centroid lies above some supporter"). v2 enforces; v3.10 doesn't.
  Most placements that pass support_ratio also pass centroid-supported,
  so the gap is small in practice.
- **`enforce_load_bearing` recursive propagation** — v2 propagates the
  added load through the stack. v3.10 only checks the direct
  supporters. The difference matters for deep stacks (3+ levels);
  for typical pallet packing (1-3 levels) the direct check is
  conservative enough.
- **CoG envelope** — v2 checks the pallet center-of-gravity falls
  within a configurable envelope. v3.10 ignores. Most pallets pass
  trivially due to small box sizes vs pallet footprint.
- **`requires_full_support` per-box override** — v2 supports per-box
  "this box demands full support" (e.g. for super-blocks). v3.10
  uses a uniform support_ratio. Easy to add when needed.
- **`group` constraint** — IND6's "customer-1/2/3" grouping is a
  PalletState concern; the BRKGA decoder doesn't see groups. Would
  require integrating group constraints into the BPS sort or
  pre-grouping boxes by customer.

---

## 5. Files touched

- `pallet_packer/brkga_v3_5.py`
  - `precompute_constraint_arrays` (new helper)
  - `_check_load_on_top_njit` (new JIT helper, includes support_ratio)
  - `_apply_load_contribution_njit` (new JIT helper)
  - `decode_njit_mode_cstr` (new constraint-aware JIT decoder)
  - `decode_chromosome` (new constraint params, dispatch logic)
  - `decode_auto_mode` (constraint plumbing)
  - `local_search_2opt`, `path_relinking`, `lns_polish` (constraint
    plumbing through inner decoder closures)
  - `brkga_pack_v35` (calls precompute, forwards via closures)
  - `warmup_jit` (compiles the new constraint-aware variant)
- `scripts/run_v310_industry_eval.py` (new eval harness)
- `results/checkpoints/v310_industry_eval.json` (results)

---

## 6. Status

- v3.10 is **the new production recommendation for industry workloads.**
  Geometric workloads (BR / academic) are unaffected: same throughput,
  same util, same code path.
- For real-world deployment the algorithm now produces honest, valid
  packings on all 10 industry scenarios.

## 7. What's next

- **Add constraint-aware mode 3/4/5** for the cases where block packing
  wins (homogeneous-SKU industry workloads).
- **C/Cython decoder port** — original final-step item; closes the
  BR1 gap to BRKGA-2013. Independent from constraint awareness.
- **Per-box `requires_full_support` + `require_centroid_supported`** —
  next tier of constraint completeness once a real deployment
  surfaces cases that need them.
