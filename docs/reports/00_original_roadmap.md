# Roadmap — Algorithmic Improvements

Math-algorithm-focused improvement plan. Physics-layer extensions (axle load,
multi-drop, road-vibration sim) are deliberately out of scope here; they are a
separate workstream once the algorithmic side is at literature parity.

The thread running through this roadmap: **measure first, then optimize the
heuristic, then prove optimality on the small slice, then make it fast.** Every
phase ends with a regression gate — re-run all 41 existing cases plus BR1-BR7
and confirm no case got worse on `(unpacked, pallets, -util)`.

OSS solvers are on the table: HiGHS (via PuLP) for LP relaxation, OR-Tools
CP-SAT for MIP polish. No commercial-solver dependencies.

---

## Phase 0 — Tighten the lower bound (prerequisite)

Before optimizing anything, know which "failures" are actually open. F9 was
already optimal once rotation constraints were considered; F1 and F12 are
suspect.

Build `lower_bounds.py` with three independent LBs per instance:

1. **Volume LB** — the existing `ceil(sum_vol / pallet_vol)`. Keep as baseline.
2. **Per-SKU max-fit LP LB** — for each SKU under its rotation set, enumerate
   the maximum integer count that fits one pallet by orientation (1D-knapsack
   on dimension occupancy). Then solve an LP relaxation in HiGHS minimizing
   total pallet count subject to per-SKU coverage. Report `ceil(LP_obj)`.
3. **Martello-Pisinger L0/L1/L2** — combinatorial bounds (Martello, Pisinger,
   Vigo 2000). L1 considers items too large to share a dimension; L2
   strengthens by pairwise infeasibility.

Headline LB = `max(volume, LP, L2)` per case.

Exit criterion: ANALYSIS.md gains a "true LB" column and we have an
`LB_REPORT.md` listing every case where achieved > true LB (real headroom)
vs achieved == true LB (already optimal — stop chasing).

Effort: 1–2 days.

---

## Phase 1 — BR1-BR7 benchmark integration

Anchor against the published literature before optimizing. Load thpack1.txt
through thpack7.txt from OR-Library (1000 instances, Bischoff–Ratcliff sets);
add `br_benchmark.py` that runs them with our physical constraints disabled
(pure geometric packing — these benchmarks don't have weight/CoG/fragility);
report mean volume utilization per set against the published baselines from
Bortfeldt 2000, Crainic-Perboli-Tadei 2008, Gonçalves-Resende 2013.

Exit criterion: a one-line-per-BR-set table showing v1 mean util, v2 mean
util, and published baseline. This becomes the moving benchmark for Phases 2+.

Effort: 1 day (parsing + harness + a `PackerConfig.geometric_only` flag).

---

## Phase 2 — Quality fixes (close the +1 gap)

Five sub-tracks, ordered by ROI per effort. Commit and re-benchmark after
each. If 2a-2c don't move the BR numbers we revisit before sinking time into
2d/2e.

### 2a — SKU-consistent rotation pre-decision
Bortfeldt-Gehring 2001. Per-SKU layer-density score per rotation; lock the
winner; remove other rotations from `box.allowed_rotations` before search.
Cuts BRKGA search space, helps F9-shape cases. 1–2 days.

### 2b — GRASP randomization in the placement decoder
Parreño et al. 2008. Placement step picks uniformly among top-α (α≈3-5)
instead of single best. Gives BRKGA real diversity to exploit. 1 day.

### 2c — Adversarial block shapes in BRKGA chromosome
Extend chromosome to encode block shape per SKU group rather than trying
three fixed strategies post-hoc. Lets the search learn contextual shape
preference. 2–3 days.

### 2d — Ejection chains
Crainic-Perboli-Tadei 2009; Faroe-Pisinger-Zachariasen 2003. After greedy
leaves K boxes unpacked, displace 1–3 placed items, retry, iterate. Standard
escape from "leftover pallet" failure mode. Single most likely fix for F1/F12.
3–5 days.

### 2e — Layer-building decoder
George-Robinson 1980; Bischoff-Ratcliff 1995. Alternative to extreme-point;
expose as third decoder in BRKGA. Helps F13 and gives the search
qualitatively different solutions to recombine. 3–5 days.

---

## Phase 3 — Multi-pallet co-optimization

The greedy first-fit-decreasing outer loop is the structural reason F1/F12
strand a low-loaded final pallet. Lift it by either (a) extending BRKGA
chromosomes to encode pallet assignment, or (b) wrapping the per-pallet
packer in an outer tabu/SA loop that swaps boxes between pallets and re-runs
the placement decoder.

Biggest structural win available on heterogeneous cargo. 5–7 days.

---

## Phase 4 — MIP polish for small N

Implement do Nascimento-Queiroz-Junqueira 2021 formulation (12 side
constraints — aligns with our stack) using OR-Tools CP-SAT (its
no-overlap-2d propagator is competitive with commercial MIP on these sizes).

Two uses:
- Whole-problem optimality when N ≤ ~30.
- Leftover-pallet polish: after Phase 3, take the under-loaded final pallet
  + nearby contents + unpacked boxes, resolve the local sub-problem exactly.

5–7 days.

---

## Phase 5 — Performance

Done last on purpose. After 0–4 we'll be slower; address it once the
algorithm has stabilized.

- Cap or prune the extreme-point set (unbounded growth on tiny-box cases —
  see B4, F2).
- Spatial index for supporter lookup (grid hash or simple R-tree). Cuts
  per-placement support computation from O(N) to O(log N) or O(1) amortized.
- Make support-area calculation incremental, like `_top_load` already is.

Target: 100 boxes in <5 s (currently ~30 s).

3–5 days.

---

## Effort summary

| Phase | What | Effort | Solver | Primary payoff |
|---|---|---|---|---|
| 0 | Tighter LBs (volume + per-SKU LP + Martello-Pisinger) | 1–2 d | HiGHS | Know what's actually open |
| 1 | BR1-BR7 benchmark harness | 1 d | — | Literature-anchored baseline |
| 2a | SKU-consistent rotation | 1–2 d | — | Search-space reduction; F9-shape cases |
| 2b | GRASP randomization | 1 d | — | Real BRKGA diversity |
| 2c | Adversarial block shapes | 2–3 d | — | Contextual block-shape choice |
| 2d | Ejection chains | 3–5 d | — | Closes F1, F12 (most likely) |
| 2e | Layer-building decoder | 3–5 d | — | F13 + qualitatively new packings |
| 3 | Multi-pallet co-optimization | 5–7 d | — | Structural fix to FFD outer-loop waste |
| 4 | MIP polish (do Nascimento 2021) | 5–7 d | CP-SAT | Provable optimality on small N |
| 5 | EP capping + spatial-indexed support | 3–5 d | — | Make all of the above fast |

Total: roughly 25–40 working days end to end. Phase 0+1 (≤3 days) are the
must-do prerequisite — they prevent the rest from being directionless.
