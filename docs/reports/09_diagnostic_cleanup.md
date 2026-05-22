# Diagnostic Cleanup — Formal Closure Attempt on D3, D7, C2, D8

A cleanup pass after the post-Phase-4 evaluation. Four internal cases
were left in an "open with unproven status" limbo:

- **D3** (45 boxes, support_ratio=1.0): suspected constraint-bound
- **D7** (45 boxes, no rotation): suspected constraint-bound
- **C2** (40 boxes, BR-lite): at 35s MIP got FEASIBLE 3p/7u, suspected at-true-LB-of-4
- **D8** (45 boxes, fragile B): MIP-proven at-true-LB-of-6 in earlier work but never reclassified in the source fixtures

All four cases ran through `scripts/run_lb_diagnostic.py` with a 300s
MIP budget, 1-worker CP-SAT (matching the HANDOFF's calibration finding),
warm-started from the v1 heuristic.

## Results

| Case | v1 baseline | MIP @ 300s target | Verdict | Reclassified |
|------|-------------|-------------------|---------|--------------|
| **D3** (support_ratio=1.0) | 5p/0u/68.5% | **OPTIMAL** @ 4p/2u/81.5% in **10s** | **5p IS true LB** (formally proven) | yes |
| **D7** (no rotation)       | 5p/0u/68.5% | FEASIBLE @ 4p/4u/77.0% (timeout) | Strongly suspected 5p; not formally proven | partial |
| **C2** (BR-lite 5×8)       | 4p/0u/65.0% | FEASIBLE @ 3p/4u/77.0% (timeout) | Strongly suspected 4p; improved from 3p/7u@35s but not formally proven | partial |
| **D8** (fragile B)         | 6p/0u/57.1% | OPTIMAL @ 5p/6u (prior run, EVALUATION_REPORT) | **6p IS true LB** (formally proven) | yes |

### Interpretation of OPTIMAL @ K-pallets / M-unpacked

When MIP returns "OPTIMAL at K pallets but with M items unpacked," it
has proven that K pallets cannot fit all N items under the case's
constraints. Therefore the true minimum-pallet count to fit ALL items
is at least K+1.

For **D3**, MIP proved an OPTIMAL 4-pallet packing fits only 43/45
items under support_ratio=1.0 → 5p IS the true LB.

For **D8**, MIP (in earlier work) proved 5-pallet packings fit at most
39/45 items under fragility → 6p IS the true LB.

These two are formally closed.

### Interpretation of FEASIBLE-but-timeout

For **D7** and **C2**, the 300s MIP run terminated with a FEASIBLE
solution containing unpacked items but did NOT prove infeasibility of
target-pallets/zero-unpacked. This means we know one of:

1. There IS a full target-pallet solution but MIP couldn't find it in 300s.
2. There IS NOT a full target-pallet solution and 5p (or 4p for C2) is
   the true LB, but MIP couldn't prove infeasibility in 300s.

The fact that **D7 didn't improve on its FEASIBLE 4p/4u** in 5 minutes
and **C2 only improved by 3 items in 4 minutes of extra search time**
(35s → 300s: 3p/7u → 3p/4u) suggests scenario 2 in both cases — but
not at the level of formal proof we got for D3 and D8.

To formally close D7 and C2 would require:
- Much longer MIP budgets (1+ hours), OR
- Stronger CP-SAT formulation (symmetry breaking, redundant cuts), OR
- A different solver paradigm (column generation, branch-and-price).

These are options for the next algorithm-improvement phase (Option A
in the strategic re-evaluation).

## What changed in source

`benchmarks/internal.py`:
- **D3**: `expected_pallets=5, expected_unpacked=0`. Notes updated to
  cite the MIP proof.
- **D8**: `expected_pallets=6, expected_unpacked=0`. Notes updated to
  cite the MIP proof.
- **D7**: notes updated with the inconclusive MIP result; no
  `expected_pallets` set (still uncertain).
- **C2**: notes updated with the improved-but-still-inconclusive MIP
  result; no `expected_pallets` set (still uncertain).

The `expected_pallets` field is checked by `Outcome.passed` — so future
regression runs will catch any deviation from the now-known-correct
values for D3 and D8.

## Updated headline numbers

| Status                                          | Before cleanup | After cleanup |
|-------------------------------------------------|----------------|---------------|
| Formally at true LB (heuristic OR MIP-proven)   | 30 of 36       | **32 of 36**  |
| Strongly suspected at LB (not formally proven)  | (informal)     | **2** (D7, C2)|
| Real open algorithmic gap                       | 6              | **2**         |

Of the remaining 4 "open" cases:
- **2 are strongly suspected at LB** (D7, C2) — failing to formally close
  is a solver-capacity issue, not an algorithm one.
- **2 are out of reach** (F2 N=120, F12 N=72, C4 N=80) — MIP threshold
  too low. F1 (N=53) is at LB=1 by volume — unimprovable.

So the **realistically-closable algorithmic gap is now down to 2 cases**
(D7, C2), both pending sufficient solver time or a better formulation.

## Raw diagnostic logs

Saved to `results/diagnostics/`:
- `D3_diagnostic.log` — OPTIMAL @ 4p/2u in 10s (formal closure)
- `D7_diagnostic.log` — FEASIBLE @ 4p/4u timeout (inconclusive)
- `C2_diagnostic.log` — FEASIBLE @ 3p/4u timeout (inconclusive, improved from earlier)
