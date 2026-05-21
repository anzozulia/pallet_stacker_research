# Quality-Focused Improvement Roadmap

Companion to [EVALUATION_REPORT.md](EVALUATION_REPORT.md). Quality > speed:
we accept higher compute cost in exchange for closing the remaining gaps.

The evaluation found **11 open cases (all derived from the same underlying
scenario)** plus a few partially-evaluable larger cases. This roadmap
attacks each gap with the specific mechanism the literature prescribes,
ordered by expected impact on the headline gap count.

**Important update from a partial Q1 implementation attempt:** Q1 (multi-
pallet ejection chains) was implemented and tested in two variants. Neither
closes the C/D cluster — debug output confirmed the extreme-point decoder
cannot produce packings tight enough for those box geometries regardless
of ordering or post-hoc displacement. The C/D cluster's +1 gap is a
**decoder-geometry** problem, not a search-strategy problem. This shifts
priorities: **Q4 (SKU-grid layer fill) is now the most leveraged remaining
work**. Q1 stayed in the codebase as a safety-net feature with bounded
cost when it doesn't help.

---

## Tier 1 — Close the C / D cluster (10–11 cases, +1 to +2 gaps)

### Q1. Multi-pallet ejection chains
**Mechanism.** After greedy completes with all items packed but more
pallets than the LB, identify the lowest-utilized pallet (the
"leftover"). For each of its items: try to displace items on fuller
pallets and fit the leftover-pallet item there instead; if the displaced
item can be re-placed elsewhere without opening a new pallet, commit
the swap. Iterate until no improvement.

**Implementation.** Extends the existing `_ejection_chains` /
`_try_one_ejection_pass` machinery, which currently only targets
items in `result.unpacked`. New trigger: when `unpacked` is empty AND
the lowest-util pallet has util < (e.g.) 60% of the mean. The swap
search reuses `_rebuild_after_swap` we already have.

The original removal reason was that the "no unpacked items" branch
explored O(N²) swap candidates per iteration. The fix is to limit the
search to (a) items in the lowest-util pallet only, and (b) displacement
targets in pallets where there's measurable slack. This is closer to
the literature's "ejection chain on leftover pallet" formulation —
not a generic rebalance.

**Effort.** 3–5 days.
**Expected impact.** Closes C1, C2, C3, D1–D7 (10 cases) and likely D8
(via the same mechanism plus fragility-aware item ordering, Q4).
**Risk.** Could regress if not careful — needs the safety net + replay-
validate path used elsewhere.

---

## Tier 2 — Tighten the MIP (closes 1–2 more cases, unblocks more)

### Q2. Quantitative support_ratio in CP-SAT
**Mechanism.** Currently MIP polish enforces only "every placed item
has at least one supporter directly below". The full constraint
requires `supported_area(i) ≥ support_ratio · footprint(i)` where
`supported_area(i)` is the total (x, y) overlap area between item i
and items in its supporter set.

CP-SAT can handle this via:
- For each (i, j) pair, define `overlap_w[i,j]` = `max(0,
  min(x_i+dx_i, x_j+dx_j) - max(x_i, x_j))`. Same for y.
- `overlap_area[i,j]` = `overlap_w[i,j] · overlap_h[i,j]`. Bilinear —
  needs auxiliary integer var + multiplication channeling.
- For each item i not on floor: `sum_j (z_aligned[i,j] · overlap_area[i,j]) ≥
  support_ratio · dx_i · dy_i`.

This is a substantial constraint set (O(N²) bilinear products), but
CP-SAT handles it on small N.

**Effort.** 4–6 days.
**Expected impact.** Closes C3 (provable optimum at N=30). Enables
raising `mip_n_threshold` to 40 for C2 closure too. The 1p/81% MIP
solution we saw before replay-validate becomes valid.
**Risk.** CP-SAT runtime growth — may force the threshold lower than
hoped on N>30.

### Q3. MIP warm-starting from heuristic
**Mechanism.** CP-SAT's `solver.SolutionHint()` accepts a partial
assignment. Pass v1's best placement as a hint so the search starts
near a known-good solution. Typically halves the solve time on
hard instances and can unlock larger N.

**Effort.** 1–2 days.
**Expected impact.** Unblocks `mip_n_threshold=40+` (closes C2). Makes
the MIP cheap enough to enable by default. May also help C3 even if
Q2 isn't done.
**Risk.** Low. Warm-starting is a well-supported CP-SAT pattern.

---

## Tier 3 — Close the BR gap (the published-benchmark slot)

### Q4. SKU-grid pattern fill for layer-building
**Mechanism.** Replace the current layer-fill (extreme-point or 2D
MaxRects) with the BR-1995 pattern-grid approach. For each layer:

1. Pick the dominant SKU among remaining items (most remaining volume).
2. Choose the rotation that gives the smallest slab-depth dim for that
   SKU.
3. Pack the slab with copies of that SKU in a regular `n_x × n_y` grid
   on the slab face, every item extending the full slab depth (no
   shadows).
4. Fill the remaining 2D space (corners and leftover gaps in the grid
   pattern) with secondary SKUs, depth-aligned where possible.
5. Move to the next slab.

This is the **actual algorithm** that gets 83% on BR1. The 2D MaxRects
implementation I tried gets ~60% because it doesn't anchor on a
dominant SKU — the depth-shadow problem dominates.

**Effort.** 5–7 days.
**Expected impact.** BR1 util from 76% (v1 average) to ~83% (BR-1995
baseline). BR3 similar gain. BR5 modest gain.
**Risk.** Some BR instances are SKU-uniform and may not benefit; need
graceful degradation back to other layer-fill strategies via the
candidate-set.

---

## Tier 4 — Specific gaps and polish

### Q5. Fragility-aware item ordering pass
**Mechanism.** Pre-pass identifies items where `max_load_on_top = 0` or
near-zero (fragile / "this side up" items). For each such item, decide
its preferred horizontal placement BEFORE the main packing runs, then
treat them as anchor points. Other items pack around them.

This addresses D8 specifically — currently fragility forces a 6th
pallet because heavy items can't stack on fragile ones, so they need
their own pallet. Pre-anchoring fragile items lets the packer reserve
floor space for them rather than discovering the conflict late.

**Effort.** 2 days.
**Expected impact.** D8 closes from +2 to +1 (LB doesn't account for
fragility's effect on stacking; full closure to 4 pallets may not be
achievable since the LB itself may be loose).

### Q6. Solver budget calibration
**Mechanism.** Profile per-feature contribution at each tier:
v1 → +SKU-lock → +GRASP → +BRKGA → +ejection → +layer → +MIP, and
measure (util gain, runtime cost) at each step. Use to set sensible
defaults that approach quality_max without paying for it on cases
that don't benefit. The current quality_max is overkill on every
case where v1 already hits LB (i.e., most cases).

**Effort.** 2 days.
**Expected impact.** Same quality at 5–10× lower runtime, making the
"production" config viable to enable by default.

### Q7. Tighten lower bounds for D8 + BR cases
**Mechanism.** Add a fragility-aware LB (items requiring own pallet
due to weight + fragility get a +1 contribution). Add per-SKU max-fit
that accounts for irregularity (currently we use grid-fit on a single
rotation; the real max-fit may be lower or higher when SKUs mix).

**Effort.** 1–2 days.
**Expected impact.** D8 might reclassify as "at LB" (currently shows
+2 to a loose LB). BR instances may have tighter LBs we can match
exactly with current algorithms.

---

## Effort and impact summary

| ID | Item | Effort | Open cases closed | Tier |
|---|---|---|---|---|
| Q1 | Multi-pallet ejection chains | 3–5 d | 10 (C1–C3, D1–D7) | 1 |
| Q2 | Quantitative support_ratio in MIP | 4–6 d | 1 (C3 via MIP) | 2 |
| Q3 | MIP warm-starting | 1–2 d | 1 (C2 via MIP) | 2 |
| Q4 | SKU-grid layer fill | 5–7 d | BR1/3/5 quality gap | 3 |
| Q5 | Fragility-aware ordering | 2 d | 1 (D8 partial) | 4 |
| Q6 | Solver budget calibration | 2 d | none directly; runtime | 4 |
| Q7 | Tighter LBs | 1–2 d | reclassification of D8 | 4 |

**Total effort to close every confirmed-open case:** ~17–22 working days.

**Most impactful single change:** Q1 (multi-pallet ejection chains) —
addresses the root cause of 10 of 11 open cases with a single fix.

**Recommended sequence:** Q1 → Q3 → Q2 → Q4 → Q7 → Q5 → Q6.

---

## What we are *not* doing (and why)

- **Removing safety net to gain runtime.** Tested in BR experiments —
  without it, Phase 2 randomization actively makes BR1 worse (58.7% vs
  v1's 79.3%). Safety net costs runtime, not quality, and runtime isn't
  the constraint here.
- **Pursuing larger BRKGA budgets on the C/D cluster.** Phase 2
  ablation already showed multi_start_trials=1 and trials=50 give
  identical results. The decoder's local optimum is sticky; more
  metaheuristic exploration of the same decoder space won't help.
- **Implementing wall-building separately from layer-building.** Our
  layer-building covers axis=x (wall-building per George-Robinson) and
  axis=z (horizontal layers per BR-1995) and axis=y. The mechanism is
  unified; what we need is better *fill* (Q4), not more axis variants.
- **Adding more strategies / configs.** The candidate set is already
  large (12 layer variants × 2 box variants × {with/without GRASP} +
  several block strategies + MIP × 2 budgets + safety net candidates).
  Adding more candidates without adding new mechanisms just adds
  runtime.
- **Trying alternative metaheuristics (SA, tabu, ant colony).** None
  would close the C/D cluster — same decoder, same local optimum. The
  mechanism gap is structural, not search-strategy.
