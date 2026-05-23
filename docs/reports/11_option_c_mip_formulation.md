# Option C — MIP Formulation Tweaks (Honest Negative-Result Report)

A targeted attempt to improve CP-SAT MIP performance via formulation
changes. Goal was 2-5× speedup on N≤50 cases and possibly pushing the
threshold to N=60-80 (closing F12 N=72 and C4 N=80).

**Outcome: net algorithmic improvement = 0.** Two of three attempted
changes regressed in some way and were reverted; the third (pair-pruning)
was kept as safe-but-inert infrastructure. F12 remained at 3p/65% even
with a 120s MIP budget. Documenting honestly so future attempts know what
not to repeat.

## What was tried

### 1. SKU-symmetry breaking — ❌ REVERTED

For each group of truly-identical items (same dims, weight, allowed
rotations, fragility), added:
- Packed-ness ordering: if item k is unpacked, k+1 must be unpacked too
- Position lex order among packed items: composite position
  `pallet * L*W*H + z * L*W + y * L + x` must be non-decreasing

**Result on baseline cases (use_mip_polish=True, mip_time_limit_s=30):**

| Case | Baseline | After SKU sym | Verdict |
|------|----------|---------------|---------|
| B1 (N=8) | 0.40s, 1p/100% | 0.24s, 1p/100% | minor speedup |
| C1 (N=45) | 19.51s, **4p/85.6%** | 44.37s, **5p/68.5%** | **2× slower + QUALITY REGRESSION** |
| C2 (N=40) | 35.65s, 4p/65% | 46.83s, 4p/65% | 1.3× slower |
| C3 (N=30) | 23.56s, 2p/40.5% | 11.52s, 2p/40.5% | 2× faster |
| D3 (N=45) | 13.58s, 5p/68.5% | 41.58s, 5p/68.5% | **3× slower** |
| D8 (N=45) | 12.53s, 6p/57.1% | 20.14s, 6p/57.1% | 1.6× slower |
| F3 (N=8) | 0.14s, 2p/40% | 0.11s, 2p/40% | similar |

**Root cause**: lex-order constraints invalidate the warm-start hint
when the heuristic's placements don't pre-satisfy the canonical order.
CP-SAT then spends search budget reconciling the hint with the new
constraints, often failing to improve over v1 within budget. The C1
regression (4p/85.6% → 5p/68.5%) loses the headline +17pp MIP win
documented in `07_evaluation.md`.

### 2. Tight placement variable bounds — ❌ REVERTED

Replaced `x_var = NewIntVar(0, L)` with `x_var = NewIntVar(0, L - min_dx_i)`
(and similarly for y, z). The geometric containment constraint
`x + dx ≤ L when placed` is unchanged; this just tightens the variable
domain so CP-SAT prunes "obviously out-of-bounds" placements at
domain-propagation level.

**Result after revert of SKU sym, tight bounds only:**

| Case | Baseline | After tight bounds | Verdict |
|------|----------|-------------------|---------|
| B1 | 0.40s, 1p/100% | 0.26s, 1p/100% | minor speedup |
| C1 | 19.51s, **4p/85.6%** | 11.24s, **5p/68.5%** | **faster but QUALITY REGRESSION** |
| C2 | 35.65s, 4p/65% | 34.95s, 4p/65% | unchanged |
| C3 | 23.56s, 2p/40.5% | 11.24s, 2p/40.5% | 2× faster |
| D3 | 13.58s, 5p/68.5% | 12.24s, **4p/85.6%** | **accidental QUALITY GAIN** |
| D8 | 12.53s, 6p/57.1% | 11.25s, 6p/57.1% | unchanged |
| F3 | 0.14s, 2p/40% | 0.15s, 2p/40% | unchanged |

**Root cause**: CP-SAT search-path variance. Bound changes alter the
search tree, leading to different solutions at the same time limit. C1
regressed (4p win lost); D3 accidentally found a 4p/0u/85.6% packing
that — while validator-clean — likely violates support_ratio=1.0 in
ways the MIP's corner-coverage approximation doesn't catch. The
unpredictability is the killer for tool reliability.

### 3. Pair-pruning — ✓ KEPT (neutral but safe)

Pre-computed pairs (i, j) that can never share a pallet:
- Combined weight exceeds pallet capacity, OR
- Can't fit side-by-side in any axis (using min possible dims)

For incompatible pairs, replaced the O(P · 7) BoolVars per pair (6 sep
literals + 1 same-pallet) with a single linear `assigned[i][p] +
assigned[j][p] ≤ 1` per pallet.

**Result on baseline cases:**

| Case | Baseline | After pair-pruning |
|------|----------|--------------------|
| B1 (N=8) | 0.40s, 1p/100% | 0.28s, 1p/100% |
| C1 (N=45) | 19.51s, **4p/85.6%** | 19.68s, **4p/85.6%** (preserved!) |
| C2 (N=40) | 35.65s, 4p/65% | 35.65s, 4p/65% |
| C3 (N=30) | 23.56s, 2p/40.5% | 23.37s, 2p/40.5% |
| D3 (N=45) | 13.58s, 5p/68.5% | 13.52s, 5p/68.5% |
| D8 (N=45) | 12.53s, 6p/57.1% | 12.43s, 6p/57.1% |

All within noise (<1% delta). **Zero regressions, including C1 closure
preserved.** No improvement either — our test cases have no
incompatible pairs (boxes are small relative to the pallet, weights
fit easily). The code is kept because it can't hurt and might help on
different cargo profiles (highly heterogeneous, weight-constrained,
high-fragility scenarios).

### Bonus test: F12 (N=72) with raised threshold + 120s budget

```
F12: N=72 boxes
Heuristic only: 3p/0u util=65.4% in 0.76s
+MIP(120s):     3p/0u util=65.4% in 133.87s errs=0
```

With pair-pruning + threshold raised to 80 + 120s MIP budget, F12 did
not improve. Either:
- 2 pallets is infeasible for F12 (likely — this would be a +1 closure)
- 2 pallets is feasible but MIP can't find it in 120s

Either way, our formulation tweaks haven't moved the N>50 threshold.
The HANDOFF's existing finding stands: F12/C4/F2 require either column
generation, much larger compute (1+ hours per case), or a different
solver paradigm.

## Why nothing worked: the CP-SAT-formulation paradox

CP-SAT is a sophisticated solver with strong internal heuristics for
propagation, variable ordering, and conflict learning. "Obvious"
improvements like symmetry breaking and tight bounds **interfere with
its internal reasoning** in ways that are hard to predict without
careful measurement across many seeds.

Specifically:
- **Symmetry-breaking constraints add variables** (composite positions,
  both-placed booleans) which themselves slow propagation.
- **They invalidate warm-start hints** that don't pre-respect the new
  canonical order. CP-SAT's warm-start is a *constraint-strengthening*
  hint, not a soft preference — incompatible hints get rejected and
  search starts from scratch in the reduced space.
- **Tight bounds change the search tree** in unpredictable ways. A bound
  that "feels" like it should help may eliminate paths CP-SAT was using
  to find good solutions quickly.

This is a known phenomenon in CP-SAT optimization work and matches
common community guidance: **measure relentlessly, never assume an
"obvious" improvement helps**.

## What would actually help (out of scope for this work)

For N>50 closure, the realistic paths require substantial investment:

1. **Column generation / branch-and-price**. Generate pattern columns
   (single-pallet packings) via the MIP, master problem chooses which
   patterns to use. Mature literature for 1D-BPP, less for 3D-BPP but
   tractable. Effort: 3-6 weeks. High probability of pushing N to 70-100.

2. **Hybrid heuristic + LP relaxation**. Use LP bounds to guide the
   heuristic decoder (e.g., LP-favored items go first). Effort: 1-2
   weeks. Modest impact.

3. **Tighter LP-relaxation cuts in CP-SAT**. Add valid inequalities
   (volume cuts, knapsack cuts) that CP-SAT can leverage. Requires
   deep CP-SAT expertise. Effort: 2-3 weeks. Uncertain impact.

4. **Hours-long MIP per case**. Just throw compute at it. Effort: 0
   developer time. Direct conflict with the tool-usability constraint.

## What landed

- `pallet_packer/mip.py`: pair-pruning logic (incompatible pairs get
  mutual exclusion; compatible pairs keep the full no-overlap
  formulation). Net code: +20 lines.
- Two NOTE comments documenting the failed experiments so future
  attempts don't repeat them.

## Validation

Final state of the algorithm:

```
Full regression: Pass 28 | Fail 0 | Validator errors 0 | Total runtime 25.92s
```

Identical to pre-Option-C baseline. D3 and D8 still pass their
`expected_pallets` assertions (5p and 6p respectively). MIP-enabled
quality preserved: C1 → 4p/85.6% (the +17pp win still works).

## Recommendation

**Stop investing in CP-SAT formulation tweaks for this codebase.**
The 30s budget cases already converge or stay at their constraint-bound
LB. The N>50 cases need different paradigms entirely. Future
algorithmic work should target either:
- Smarter heuristic decoders (the algorithmic *floor* — current
  extreme-point decoder may have known better alternatives), OR
- Deployment / tool integration work (where leverage is much higher
  for users).
