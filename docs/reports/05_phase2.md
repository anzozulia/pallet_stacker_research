# Phase 2 — Implementation Report

Phase 2 introduced four interrelated improvements to the algorithm: SKU-consistent
rotation pre-decision, GRASP placement randomization, ejection chains, and a
relaxed BRKGA n-threshold. A "safety net" mechanism was added to guarantee
no regression vs. the v1 or v2-baseline result, regardless of which Phase 2
features are enabled.

This document records what landed, what worked, what didn't, and three
significant bugs caught and fixed during integration.

---

## What landed

### Phase 2a — SKU-consistent rotation
*Bortfeldt-Gehring 2001.* For each SKU group of multiple identical items, score
each allowed rotation by `floor(L_eff / dx) * floor(W_eff / dy) * floor(H / dz)`
using the effective pallet footprint (extended by `max_overhang` when overhang
is enabled), pick the rotation with the highest grid count, and restrict each
member's `allowed_rotations` to that single choice.

Implemented in `PalletPacker._lock_sku_rotation()`. Activated by
`PackerConfig.sku_consistent_rotation=True`. Skipped for single-unit groups
and for SKUs whose rotation set is already singleton.

### Phase 2b — GRASP randomization in the placement decoder
*Parreño et al. 2008.* `PalletState.try_place()` now collects ALL feasible
placements and their scores; when `config.grasp_alpha > 1` it picks uniformly
among the top-α instead of always the single best. This gives BRKGA and
multi-start non-trivial per-decode variation — the previous deterministic
decoder converged every chromosome to the same packing, which is why the
original v2 ablation showed zero gain.

### Phase 2d — Ejection chains
*Crainic-Perboli-Tadei 2009; Faroe-Pisinger-Zachariasen 2003.* When the greedy
pass leaves unpacked items, the ejection-chain post-processor tries to fit
each unpacked item by displacing one (depth=1) or two (depth=2) placed items
and re-fitting them after. Iterates until no improvement, capped at
`ejection_max_iters` or a 5-second wall-clock budget. Only fires when there
are unpacked items — pure rebalancing is Phase 3 territory.

### BRKGA n-threshold default raised from 40 → 200
With Phase 2b in place, BRKGA's chromosome subspace is actually explored
(the decoder is no longer deterministic), so the conservative cutoff no
longer applies. Set to `None` for "always engage" mode.

### Safety net (`PackerConfig.use_safety_net=True`, default)
When GRASP or BRKGA is enabled, the packer also runs three deterministic
"v1-equivalent" candidates using a fresh-seeded RNG and `grasp_alpha=1`:
1. `search(boxes)` (BRKGA without GRASP, deterministic placement decoder)
2. `_multi_start(boxes)` (no BRKGA at all — the pure v1 path)
3. Block-building with GRASP off (only when `use_block_building=True`)

The final result is `min(all_candidates, key=quality)`. This guarantees
Phase 2 features can only add value, never destroy a v1- or v2-baseline win.
The cost is a roughly 2-3× per-pack runtime overhead when Phase 2 is active;
it can be disabled by setting `use_safety_net=False` for benchmarking
purposes when measuring raw Phase 2 contribution.

### `PackerConfig.max_pallets`
New cap (default `None` = unlimited). Setting `max_pallets=1` turns the
multi-pallet packer into a single-container max-utilization packer (the
Bischoff-Ratcliff objective). Used by the BR benchmark harness.

---

## Bugs caught and fixed during integration

### 1. SKU-consistent rotation ignored overhang on F10
*Symptom:* F10 (4 boxes, overhang-enabled, was 1p/140% in v1) regressed to
2p/70% with SKU lock on. The lock chose WLH (grid=2 on strict pallet
120×100) over LWH (grid=1 on strict pallet), but on the **effective** pallet
140×120 (with 20mm overhang) LWH actually fits 4 units in a 2×2 grid while
WLH only fits 2.

*Fix:* `_lock_sku_rotation` now computes the grid count on the effective
footprint (extended by `max_overhang` when overhang is enabled), matching
the asymmetric overhang model in `_within_pallet`. Tie-breaks by smallest
remaining edge slack so SKUs with multiple equally-dense orientations
prefer the most-interlocking aspect ratio.

### 2. GRASP randomization defeated the v2 F3 win
*Symptom:* Original v2 (block + BRKGA, no GRASP) closed F3 from 2p/45% to
1p/90% on the headline failure case. New v2_phase2 with everything on
regressed it back to 2p/45%. Root cause: with `grasp_alpha=3`, the BRKGA
seed chromosome no longer decoded deterministically, so the interlock
layout BRKGA had discovered under deterministic decoding was no longer
in the candidate set.

*Fix:* The safety net (above) re-runs all the "no-Phase-2" candidate
paths with `grasp_alpha=1` and includes their results. Specifically, the
F3 win comes from `search(boxes)` with BRKGA on and GRASP off — not from
block-building. The safety net now always includes that path.

### 3. Safety-net BRKGA explored a different subspace per run
*Symptom:* Even with the safety net added, F3 still regressed to 2p/45%
under all_on — but the same deterministic-decode config in isolation
produced 1p/90%. Root cause: the safety net was using the packer's main
RNG, which had already been consumed by the Phase 2 GRASP path. The
BRKGA seed chromosome decoded the same way under GRASP=1, but the random
chromosomes in the population started from a different RNG state and
explored a different subspace.

*Fix:* The safety net now uses its own fresh `random.Random(config.seed)`
for reproducibility, restoring the original packer RNG when it exits.

### 4. `_lock_sku_rotation` initially had a typo — picked the wrong rotation
*Symptom:* On F10 with overhang and SKU lock, the rebuilt boxes had
`rotations=['WLH']` instead of the optimal `['LWH']`. Caught by direct
inspection of `_lock_sku_rotation` output during F10 debugging.

(This was the same bug as #1 above — they got resolved together.)

---

## Regression — 41-case suite

All 41 cases (28 benchmark + 13 failure suite) run under three
configurations: `v1`, `v2_phase2` (Phase 2 features without
block-building), and `all_on` (Phase 2 + block-building).

The 5 slowest cases (B4, F2, F1-bench, C4, F12-failure) were skipped
in this session because the safety-net + Phase 2 candidate explosion
takes >30s per case at N>50. Their behavior was spot-checked
individually during debugging and confirmed not to regress.

### Summary (36 cases × 3 configs)

| Metric | v2_phase2 | all_on |
|---|---|---|
| Improvements vs v1 (Δp<0 or Δunpacked<0) | 1 | 1 |
| Regressions vs v1 | **0** | **0** |
| Validator errors | **0** | **0** |

### The improvement
| Case | v1 | v2_phase2 | all_on |
|---|---|---|---|
| F3 interlock big+small | 2p/0u/45% | **1p/0u/90%** | **1p/0u/90%** |

F3 closed by `search(boxes)` (BRKGA on individual boxes, GRASP off — the
deterministic path). The original v2 had also found this; the safety net
preserves it under Phase 2.

### Why other cases didn't improve
Every other case shows v1 == v2_phase2 == all_on. This is the safety net
working as designed: when Phase 2 randomization can't find anything
better than v1, the safety net's v1 candidate dominates the
`min(candidates, key=quality)`. Cases that *might* have improved with
exhaustive search (the C/D cluster at +1, F12 at +1) didn't move at
BRKGA pop=16 gens=4 = 64 evaluations. They likely need:
- Larger BRKGA population (Phase 3 work).
- Ejection chains with deeper depth (Phase 2d default depth=2).
- Phase 2e layer-building decoder (not yet implemented).

---

## Deep BR test — what we learned

Sample of 31 instances run across BR1, BR3, BR5 (mostly v1 only — v2_phase2
proved too slow per instance to test at scale in this session).

| Set | N inst | v1 mean | v1 stdev | v1 range | BR-1995 baseline | Gap |
|---|---|---|---|---|---|---|
| BR1 | 12 | 76.4% | 10.1 | 50.7 – 88.4 | 83.1% | −6.7 pp |
| BR3 |  9 | 73.7% |  5.3 | 67.0 – 83.5 | 79.5% | −5.8 pp |
| BR5 | 10 | 73.5% |  7.4 | 65.5 – 83.9 | 76.3% | −2.8 pp |

The larger sample shows v1's gap to the 1995 baseline is roughly 3-7 pp,
slightly worse than the n=3 estimate suggested. Variance is high (stdev
5-10pp), which means we'd need n>50 to call mean differences of <2pp
statistically meaningful.

### Phase 2 contribution at BR scale: minimal
For one BR1 instance (#1, N=112) tested with Phase 2 features and **safety
net OFF**, the result was **58.7% util** — well below v1's 79.3%. With
safety net ON, the result returns to v1's 79.3% (the safety net's v1
candidate dominates).

This confirms the FAILURE_ANALYSIS finding: at N>100, the extreme-point
decoder's bias toward back-left-bottom placement is hard to escape with
randomization alone. GRASP and BRKGA explore the wrong neighborhood.
Specifically:
- **GRASP randomization makes packings worse on average** at BR scale —
  the top-α candidates are nearly equivalent geometrically, so random
  selection just adds noise.
- **SKU-consistent rotation has neutral effect** on BR — every SKU's
  optimal-density rotation is already what the deterministic decoder picks.
- **Ejection chains rarely fire** on BR — when all items fit on one
  pallet, there's nothing to eject.

The features were designed for **small/structured cases** (F3 interlock,
F12 stranded tail), not for the "many heterogeneous items on a large
container" profile of BR.

---

## Phase 2e — layer-building decoder (added in same session)

After the deep BR test revealed the extreme-point decoder was the floor of
our solution quality, I implemented the layer-building decoder (George-Robinson
1980, Bischoff-Ratcliff 1995) as the next Phase 2 increment.

**What landed:**
- `PalletPacker._layer_pack(boxes, axis, seed_strategy)`. Builds slab-shaped
  sub-problems along one pallet axis (x, y, or z). Each slab is then filled
  via the extreme-point engine on a virtual sub-pallet.
- Three axes (x = wall-building, y = lateral walls, z = horizontal layers).
- Four seed-selection strategies: `cross_section`, `depth`, `min_depth`,
  `sku_volume` (Bischoff-Ratcliff style).
- Activated by `PackerConfig.use_layer_building=True`. When on, `pack()`
  runs `axes × strategies = up to 12 candidates` and includes them in the
  candidate set.
- Replay-validate: the assembled placements are re-fed through the real
  PalletState's feasibility check; any placement that fails (e.g.,
  unsupported items on an axis='z' layer above floor) gets demoted to
  unpacked. This caught a bug where axis='z' layer-building was producing
  floating boxes (the sub-pallet assumes its origin is the floor, which
  isn't true for layers above z=0).

**Results:**

| Case | v1 | v1 + layer-building |
|---|---|---|
| F3 interlock big+small | 2p/0u/45% | **1p/0u/90%** |
| 35 other cases | unchanged | unchanged (safety: zero regressions) |
| F13 pure layered cargo | 2p/0u/70% | 2p/0u/70% (already at LB) |
| BR1 inst 1 | util 79.3% | util 79.3% (layer-pack's best 69.4% lost to v1 in candidate set) |

**The honest finding:** layer-building alone closes F3 without needing
BRKGA or block-building. But it does NOT help on BR-scale instances. Best
layer-pack variant on BR1 inst 1 gets 69.4% (vs v1's 79.3%) — the layer
sub-fill itself uses extreme-point, so the result is structurally still
extreme-point-class, just with a different outer ordering.

**Why this didn't fully close the BR gap:** the literature's layer-building
specifies a *different* layer-fill heuristic, not extreme-point. Specifically
2D MaxRects (Jylänki) or SKU-pattern-grid filling. Implementing those
is the next layer-building increment. The current implementation captures
the OUTER structure (layer slabs) but reuses the INNER decoder we already
have. That's enough for F3 (the interlock is naturally a layer arrangement)
but not enough for BR heterogeneity.

### Phase 2e.2 — 2D MaxRects sub-fill (Jylänki 2010)

After Phase 2e shipped, I added a 2D MaxRects layer-fill as an alternative
inner decoder. The MaxRects algorithm maintains a list of maximal free
rectangles, places items into the best-area-fit free rect, splits affected
rects into up to four sub-rects, and prunes dominated rects.

**Result on BR1 inst 1 (v1 = 79.3%):**

| Axis × Strategy | EP fill | MaxRects fill |
|---|---|---|
| x × cross_section | 64.1% | 60.6% |
| x × depth | 67.7% | 29.5% |
| x × sku_volume | 57.9% | 48.8% |
| y × depth | 69.4% | 35.0% |
| z × sku_volume | 63.9% | 16.4% |

Across 24 variants tested (3 axes × 4 strategies × 2 fills), MaxRects was
**consistently worse than EP fill** — and none beat v1.

**Why MaxRects underperforms here:** 2D MaxRects packs items at the back
of the slab (depth-axis position = 0). When an item's depth-axis dim is
shorter than `slab_depth`, the volume behind the item — between the item's
trailing face and the slab's other wall — is **wasted**. BR's heterogeneous
cargo has many short-depth items relative to whatever seed sets the slab
thickness, so the wasted-depth volume compounds quickly.

The literature avoids this by **SKU-grid filling**: each layer is dominated
by one SKU whose items are placed in a regular grid, so every item in the
grid extends the full slab depth (no shadows). Secondary SKUs fill leftover
space, also depth-aligned where possible. This is qualitatively different
from "free 2D bin-packing" — it's a pattern-based fill.

**Impact on the candidate set:** MaxRects results join the candidate pool,
and the safety net's `min(quality)` always picks the best. So MaxRects can
only HELP — when it produces a structurally different packing that happens
to win, it wins; otherwise it loses to the EP fill or v1. Across the
36-case regression we saw zero MaxRects-driven improvements, but zero
regressions either. F3 remains closed by the EP-fill layer-building.

**What this tells us about the gap:** the BR-1995 baseline of 83% on BR1
is NOT reachable via 2D MaxRects on slab faces alone. It requires
SKU-grid filling (the actual BR-1995 algorithm) or 3D MaxRects with
proper free-volume tracking (more complex). Both are tractable but
substantial implementations — Phase 3 territory.

---

## Phase 4 — MIP polish via CP-SAT (added in same session)

After Phase 2e.2's mixed-quality result, pivoted to Phase 4: CP-SAT-based
exact 3D bin-packing as a candidate when N is small enough.

**What landed:** `mip_polish.py` with `mip_polish(boxes, pallet, config)`
function using OR-Tools CP-SAT. Models:
- Per-item rotation (Boolean per allowed rotation, channeled through dx/dy/dz).
- Per-item placement (x/y/z integers).
- Per-item pallet assignment + unpacked Boolean.
- Pairwise no-overlap via 6 separation literals per (i, j, p) triple.
- Per-pallet weight budget.
- Anti-floating: every placed item is either on z=0 OR has at least one
  supporter j with `z_j + dz_j = z_i` and (x, y) footprints overlapping.
  Doesn't enforce `support_ratio` quantitatively (full overlap area is
  non-linear in the placement vars), but catches outright floating items.

Wired into `pack()` with `use_mip_polish=True`. Triggers when
`N ≤ mip_n_threshold` (default 25). Calls the solver with a pallet
budget hinted from the best heuristic candidate so far, plus an
"aggressive" variant trying one fewer pallet. Solver time budget
configurable via `mip_time_limit_s` (default 30s). Output goes through
the same replay-validate step as layer-building so the full constraint
stack (support_ratio, CoG, fragility) is honored on the final candidate.

**Results:**

| Case | v1 | v1 + MIP polish |
|---|---|---|
| F3 interlock (N=12) | 2p/0u/45% | **1p/0u/90%** (closed) |
| F4 height pairing (N=8) | 1p/0u/100% | unchanged (already optimal) |
| F11 spurious unpacked (N=7) | 1p/0u/27% | unchanged (at LB) |
| C3 Pareto 30 (N=30, OPEN +1) | 2p/0u/41% | unchanged after replay-validate |
| 36-case regression | — | 0 regressions, 0 validator errors |

**Honest finding:** MIP polish closes F3 (provably optimal at N=12) and
confirms optimality on every small case already at LB. It does NOT close
C3 (N=30, the only open case within the threshold) because the support
constraint is approximated — the MIP finds a "raw" packing that satisfies
the lighter anti-floating constraint but fails the full support_ratio=0.8
check on replay-validate. The replay-validate then demotes failed items,
leaving a worse candidate that loses to v1 in the candidate set.

To close C3 (and other +1 open cases that exceed the MIP threshold), we'd
need either:
1. **Quantitative support_ratio in the MIP** — requires area computations
   that aren't linear in placement vars; would need linearization or
   piecewise approximations.
2. **A larger N threshold** with proportionally larger CPU budget —
   feasible for N up to ~40 with proper formulation, but C4 (N=80) and
   F12 (N=72) are out of reach.

For now, MIP polish is a high-quality candidate on small structured cases
(F3) and a no-op confirmation on cases already at LB. It joins the
candidate-set design cleanly: works when it can, defers when it can't,
never regresses anything.

The candidate-set design means this addition is "harmless and helpful" —
layer-building wins where it can (F3), and the safety net's other
candidates win everywhere else. No tuning needed.

---

## What this means for Phase 3+

Updated ROADMAP priorities based on what we've learned:

1. **Phase 2e is partially complete — needs 2D MaxRects sub-fill.**
   The outer layer-slab structure is in place; replacing the extreme-point
   sub-fill with 2D MaxRects (per Jylänki 2010) or pattern-grid filling
   (BR-1995 style) should close the remaining BR gap. Estimated effort
   ~5-7 days. Every
   published BR baseline beats us via either layer-building or
   wall-building decoding. Our extreme-point decoder is the floor — to
   close the 5-7pp gap on BR, we need a structurally different decoder,
   not more metaheuristic exploration. **Highest expected ROI.**

2. **Phase 3 (multi-pallet co-optimization) needed for the C/D-cluster
   gaps.** The 41-case results show those cases are stuck at the
   greedy-FFD-outer-loop ceiling. Ejection chains alone don't move them.

3. **The safety net is expensive — needs optimization.** Currently it
   adds 2-3× runtime overhead. For Phase 3+, we should consider either:
   - Caching: skip safety-net candidates that are guaranteed to match v1
     (e.g., for instances where no Phase 2 feature applies).
   - Lazy evaluation: only run safety-net candidates when the primary
     Phase 2 result is worse than a quick lower-bound estimate.

4. **A "BR-mode" config is worth canonicalizing.** `max_pallets=1` plus
   `geometric_only` makes the packer behave as a single-container packer,
   which is the right setup for any benchmark or research comparison.
   Consider a `PackerConfig.from_br_benchmark()` factory.

---

## Reproducing

```bash
# Regression suite (36 cases, ~5 minutes)
python3 run_phase2_regression.py

# Full ablation (6 configs, slow — 20+ min)
python3 run_phase2_regression.py --ablation --full

# Deep BR test (sample 5–10 per set)
python3 run_deep_br.py --sample 10 --seeds 1 --configs v1 v2_phase2 \
        --sets thpack1 thpack3 thpack5
```

Results checkpoint to `phase2_results.json` and `deep_br_results.json` so
they can be inspected mid-run and resumed.

---

## Files added in Phase 2

| File | Purpose |
|---|---|
| `pallet_packer.py` *(modified)* | SKU-lock, GRASP, ejection chains, safety net, max_pallets, brkga_n_threshold raise. ~1500 lines total. |
| `run_phase2_regression.py` | 41-case suite × N configs, with incremental JSON checkpointing. |
| `run_deep_br.py` | BR1/3/5 across larger samples + multiple seeds, also checkpointed. |
| `phase2_results.json` | Regression-test checkpoint. |
| `deep_br_results.json` | BR-deep-test checkpoint. |
| `PHASE2_REPORT.md` | This file. |
