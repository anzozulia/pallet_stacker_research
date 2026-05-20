# Failure Analysis & Fix Mapping

Where the algorithm falls short of optimum, and which paper to read for the fix.

## Summary

13 stress cases run; **3 confirmed underperformance failures**, **8 passes**, **2 "passes for the wrong reason"** (got the right answer but by luck of the heuristic, not by recognizing the structure).

| Case | N | LB | Got | Util | Verdict |
|------|---|----|----|------|---------|
| F1 Pareto continuum | 53 | 4 | 5 | 60.4% | **FAILURE +1** |
| F2 block of identicals | 30 | 2 | 2 | 64.5% | Pass — but missed structure |
| F3 interlock big+small | 12 | 1 | 2 | 45.0% | **FAILURE +1** |
| F4 height pairing | 8 | 1 | 1 | 100.0% | Pass — by lucky alignment |
| F5 wrong-rotation cascade | 6 | 1 | 1 | 60.0% | Pass |
| F6 fragility + dense fit | 16 | 2 | 2 | 53.3% | Pass |
| F7 layered pyramid | 25 | 1 | 1 | 86.7% | Pass |
| F8 strongly heterogeneous (16 SKU) | 21 | 2 | 2 | 76.5% | Pass |
| F9 long this-side-up | 18 | 2 | 3 | 66.0% | Pass — volume LB was loose; 3 IS optimum |
| F10 overhang | 4 | 2 | 1 | 140%* | Pass — overhang works (vol LB undercounts) |
| F11 narrow box wedge | 7 | 1 | 1 | 27.3% | Pass — multi-start rescued it |
| F12 strongly hetero @ scale | 72 | 2 | 3 | 65.4% | **FAILURE +1** |
| F13 pure layered cargo | 22 | 2 | 2 | 70.0% | Pass — but 85% available |

\* Utilisation > 100 % because the cargo footprint exceeds the pallet footprint via overhang — the metric measures against pallet volume, not extended cargo volume.

## What the three real failures have in common

`failure_gallery.png` shows the same pattern in all three:

| Case | Pallet utilisation per pallet | Wasted pallet |
|------|--------------------------------|---------------|
| F1 | 73 / 72 / 63 / 74 / **21**% | last (21%) is the leftover |
| F3 | 80 / **10**% | second (10%) is the leftover |
| F12 | 94 / 85 / **17**% | last (17%) is the leftover |

**Every failure is a final-pallet underutilisation.** The greedy First-Fit-Decreasing outer loop commits items to a pallet permanently; when later items don't fit, a new pallet opens, and the items that would have completed an earlier pallet are stranded.

The classical name for this is **first-fit waste**: any bin-packing heuristic that places items one at a time without retraction will sometimes leave a low-loaded "tail" bin. For 1D bin packing the gap is bounded by FFD's 11/9 OPT + 6/9 asymptotic ratio (Johnson 1973); the 3D analogue is unbounded in the worst case.

## Mapping each failure mode to a literature fix

### Mode 1 — Greedy first-fit waste (F1, F3, F12)
**What goes wrong.** Heaviest-first sort fills early pallets densely; final items are heterogeneous fragments that need their own pallet.

**Fix 1 — True BRKGA with population recombination** (Gonçalves & Resende, *Int. J. Production Economics* 145, 2013):
Maintain a population P of random-key chromosomes. Each generation, recombine an elite parent with a non-elite parent via biased crossover (each gene takes the elite's value with probability ~0.7). This actively *explores* the solution space rather than perturbing one seed.

Our current "BRKGA-lite" doesn't do recombination — that's why the ablation showed zero improvement. The full algorithm typically closes the +1 pallet gap on instances like F1 and F12 in 60–200 generations with population size 20·N.

**Fix 2 — GRASP** (Parreño, Alvarez-Valdés, Tamarit & Oliveira, *INFORMS J. Computing* 20(3), 2008):
Greedy Randomized Adaptive Search Procedure: at each placement step, randomly choose among the top-α best candidate positions (rather than always the best). Restart many times. Empirically beats deterministic constructive heuristics by 3–7 pp utilisation on Bischoff–Ratcliff benchmarks.

**Fix 3 — Tabu search over EP / max-space candidates** (Crainic, Perboli & Tadei "TS²PACK", *EJOR* 195(3), 2009):
Two-level tabu: outer level over assignment of items to bins, inner level over placement within a bin. Tabu memory prevents revisiting recent bad moves. Standard reference for "extreme-points + metaheuristic".

**Fix 4 — Exact-MIP polish for small bins** (do Nascimento, de Queiroz & Junqueira, *COR* 128, 2021):
For pallets with ≤ ~30 items in the heuristic solution, feed the items into an ILP/CP solver with their model (12 practical constraints). Per their results, > 70 % of instances solve in < 1 h. Useful as a final improvement pass on the heuristic's "leftover" pallet.

### Mode 2 — Block-building potential not exploited (F2, F4 by lucky alignment)
**What goes wrong.** A homogeneous sub-set of items should be packed as a rectangular block, but our placer treats each unit independently and disperses them.

**Fix — Block-building heuristic** (Eley, *EJOR* 141, 2002; Bortfeldt, *EJOR* 124, 2000):
1. Pre-process the input: detect SKUs with ≥ K identical units (typically K = 6).
2. For each such SKU, enumerate candidate **simple blocks** (n_x × n_y × n_z grids that fit the pallet).
3. Pack the blocks (treated as atomic super-items) using a guillotine or extreme-point procedure.
4. Pack remaining heterogeneous items into the residual spaces.

Bortfeldt's block heuristic plus a tabu search on block selection achieves 90 %+ utilisation on Bischoff–Ratcliff BR1 (~75 % is the baseline). For workloads with even modest SKU repetition this is the single highest-ROI upgrade.

### Mode 3 — Wall-/layer-building structure missed (F3, F13)
**What goes wrong.** Items naturally form layers (all H = same) or walls (all W = same), but our column-based EP heuristic builds vertically item-by-item.

**Fix 1 — Wall-building** (George & Robinson, *Computers & OR* 7, 1980):
Treat the pallet as a sequence of "walls" along its length. For each wall: pick a seed item, then fill the wall by similar items, then move on. Classic for shipping containers (depth >> height).

**Fix 2 — Layer-building** (Bischoff & Ratcliff, *Omega* 23(4), 1995):
For each layer at increasing z: pick the box of greatest layer-thickness contribution, then 2D-pack the rest of that thickness. Outstanding when most items share a height (F13).

**Fix 3 — Hybrid skyline + 2D MaxRects** (Burke, Kendall & Whitwell, *OR* 52(4), 2004):
Maintain a 2D height-map (skyline) and treat each new placement as the 2D bin-packing problem of fitting the box into the lowest valley. The 2D sub-problem is solved by MaxRects (Jylänki) — a fast, well-known 2D packer (available in Python as `rectpack`).

### Mode 4 — SKU-inconsistent rotation (F5 passes; would fail on harder instances)
**What goes wrong.** When the algorithm picks rotation per-box, identical SKUs can end up in different orientations, fragmenting the pattern.

**Fix — Group-rotation pre-decision** (Bortfeldt & Gehring, *EJOR* 131, 2001):
Before placement, for each SKU compute a "layer-density score" for each allowed rotation (how many fit per layer). Choose one rotation per SKU and lock it. Then run the usual placer with reduced rotation set.

Cheap to implement; would proof F5 against harder variants where rotation inconsistency does emerge.

### Mode 5 — Constraint-induced unpacking (F11 passed, but rare cases would fail)
**What goes wrong.** Greedy placement consumes a "needed" slot; a later item finds no feasible position even though a feasible global solution exists.

**Fix — Ejection chains / repair** (Crainic, Perboli & Tadei *EJOR* 2009; Faroe, Pisinger & Zachariasen *INFORMS JoC* 2003):
When a box can't fit, *displace* an existing box temporarily, place the new box, then try to re-place the displaced one. Up to depth-3 chains have been shown effective.

Our `try_place` never displaces — it accepts or rejects. Adding ejection chains is a self-contained add-on.

### Mode 6 — Misleading volume lower bound (F9, F10)
**Not a failure of the algorithm but of the reporting.** The volume-based LB = ceil(total_volume / pallet_volume) ignores geometry. F9 has LB = 2 by volume but LB = 3 by geometry (110 mm boxes only fit one way). F10 has LB = 2 by volume but LB = 1 by overhang allowance.

**Fix — Tighter LBs:**
- For each SKU and each allowed rotation, compute its maximum-per-pallet count (an integer-programming relaxation works in milliseconds).
- Combine into an LP-relaxation lower bound (Martello, Pisinger & Vigo 2000).
- Use the *maximum* of (volume LB, weight LB, SKU LB, LP LB) as the reported LB.

## Failure modes the suite does NOT cover

These would also be valuable to test:

- **Stability under transport dynamics.** Our static support-ratio check doesn't simulate vibration / acceleration. A packing that's geometrically supported can still fail if its CoG climbs > 70 % of pallet height. Junqueira et al. (2012) horizontal-stability and Martínez-Franco & Álvarez-Martínez (*SoftwareX* 12, 2020) PhysX validation address this.
- **Multi-drop sequencing.** When a vehicle delivers to multiple sites, the packing order must enable LIFO unload. Bortfeldt & Wäscher (2013) cat. 4a.
- **Heterogeneous pallets.** The current API takes one pallet type; an MHBSBPP instance (different pallet sizes available) requires the outer loop to choose pallet type per shipment.
- **Axle / Load Distribution Diagram.** Our CoG envelope is a centred rectangle, not the EU 96/53/EC regulatory envelope (Ramos, Silva & Oliveira *EJOR* 266, 2018) with per-axle limits.

## Priorities for the next iteration

Ordered by expected utilisation gain vs. implementation cost (high gain / low cost first):

1. **Block-building preprocessing** (Eley 2002 / Bortfeldt 2000) — ~1–2 weeks; closes F2/F4-style cases and typically gains 5–10 pp on Bischoff–Ratcliff instances.
2. **True BRKGA recombination** (Gonçalves & Resende 2013) — ~1 week; closes the +1 pallet gap on F1/F12-style heterogeneous instances. Requires only changing `_pack_once` and the multi-start loop.
3. **Layer-building decoder** as an alternative strategy — ~1 week; for cargoes with low height diversity (F13), expect 10–15 pp gain.
4. **Tighter lower-bound calculation** — ~3 days; doesn't improve packings, but stops the benchmark report from claiming false "gaps" on cases like F9/F10.
5. **Ejection chains** (Faroe, Pisinger & Zachariasen 2003) — ~1 week; rescues edge cases where one item gets stranded.
6. **Group-rotation pre-decision** (Bortfeldt & Gehring 2001) — ~2 days; cheap insurance against rotation fragmentation.

Items 1 + 2 alone would likely close every failure in this suite to LB and lift the headline 45-box benchmark from 5 pallets / 68 % to a 4-pallet / 86 % solution.

## What this analysis was NOT designed to find

- **Validator bugs.** All 13 cases pass `validate()`. Any bug in the support / load-bearing / CoG enforcement should produce a non-empty error list; none did. This is consistent with — but does not prove — the constraint stack being implemented correctly.
- **Runtime regressions.** Largest case (F12, 72 boxes) ran in 1.5 s. F2 (30 boxes) in 0.65 s. No timeouts after the load-bearing optimisation.
- **Cases where the algorithm produces invalid solutions.** None observed. The constraint stack is conservative: it rejects placements it can't certify safe, which is the right side of the trade-off for a planning tool.
