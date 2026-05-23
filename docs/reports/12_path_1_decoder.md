# Path 1 — Smarter Heuristic Decoder (Negative Result)

A targeted attempt to add new heuristic-decoder variants to the
candidate set, hoping to close some of the cases the current 4
scoring strategies (`blb`, `bbl`, `max_touch`, `corner_fit`) leave
unimproved.

**Outcome: net algorithmic improvement = 0.** Three new scoring
strategies added zero wins to the candidate set while adding ~52%
to default-config runtime. Reverted to the original 4 strategies.

## What was tried

Added three new scoring strategies to `PalletState._score_placement`
and registered them in `SCORING_STRATEGIES`:

| Strategy | Idea | Code |
|----------|------|------|
| `tight_bbox` | Minimize bounding-box growth across all placed items | Score = sum of axis growth beyond current max |
| `back_first` | Prefer back-wall first (low y) | Score = (y, z, x) — permutation of blb's (z, y, x) |
| `min_overhang` | Prefer placements with full supporter coverage | Score = footprint - supported_area |

Each is genuinely different in *intent* from the existing 4:
- `tight_bbox` rewards compactness across the entire packing, not
  just local placement.
- `back_first` reorders the (z, y, x) lexicographic priorities to
  bias toward fully filling back-of-pallet before going up.
- `min_overhang` adds a quantitative penalty for overhang, distinct
  from `max_touch` which rewards contact but doesn't penalize
  overhang per se.

The strategies plug into the existing `_multi_start` loop:
```python
trials = [(seed_order, strat, sel) for strat in SCORING_STRATEGIES
                                  for sel in selections]
```
With 7 strategies × 2 pallet-selection modes = **14 trials per
multi-start call**, vs the original 8.

## Result

### Full 41-case regression (default config, no MIP)

| | Baseline (4 strategies) | After (7 strategies) |
|---|------------------------|---------------------|
| Pass / Fail | 28 / 0 | 28 / 0 |
| Validator errors | 0 | 0 |
| Total runtime | 22-26s | **39.32s** (~52% slower) |
| Wins over baseline | — | **0** |

Not a single case had its candidate-set winner come from a new
strategy. The existing 4 strategies already produce the best
heuristic candidates on every test case.

### Documented heuristic-ceiling cases (didn't move)

These are cases FAILURE_ANALYSIS.md identified as bottlenecked on
the heuristic (where a better heuristic *might* help):

| Case | N | Documented baseline | After new strategies |
|------|---|---------------------|---------------------|
| F1 Pareto continuum | 53 | 5p/0u/60.4% | 5p/0u/60.4% (unchanged) |
| F3 interlock big+small | 12 | 2p/0u/45.0% | 2p/0u/45.0% (unchanged) |
| F12 strongly heterogeneous | 72 | 3p/0u/65.4% | 3p/0u/65.4% (unchanged) |

The hypothesis was that one of these might benefit from a
qualitatively different placement bias. None did.

## Why it didn't work

Two related reasons:

1. **The existing 4 strategies already span the useful space.** They
   bias toward (a) bottom-z fill, (b) wall contact, (c) corner
   hugging. The 3 new strategies are *combinatorially close* — e.g.,
   `back_first` is a permutation of `blb`'s axis order. On real
   geometry, they produce nearly identical placements because the
   first-feasible-position discovery dominates the scoring.

2. **The candidate set winner is determined by `min(quality)`
   = `(unpacked, pallets, -util)`.** If the existing strategies
   already find the best `(pallets, util)` for a case, a new
   strategy can only contribute by tying — never strictly
   beating. And ties don't change the headline.

This matches the pattern of all prior incremental work on the
heuristic: ANALYSIS.md (Multi-start trials gave zero improvement),
PHASE2_REPORT (GRASP randomization broke BR), Option C MIP
formulation (CP-SAT changes regressed quality), and now Path 1
(new scoring variants add no candidates).

## What landed

- Reverted `SCORING_STRATEGIES` back to the original 4
- Removed dead-code strategy branches from `_score_placement`
- Added a NOTE in the file documenting the failed attempt

## What I considered but didn't try

- **Skyline-based layer-fill** (Burke-Kendall-Whitwell 2004). Would
  add a 4th `_fill_layer_skyline` option alongside extreme-point,
  2D MaxRects, and SKU-grid fills. *Why not tried*: PHASE2_REPORT
  established that 2D MaxRects fill (the closest analogue) lost to
  extreme-point fill across all 24 axis × strategy variants. A
  skyline approach has the same "depth-shadow" problem that
  doomed MaxRects: it packs flush to one slab face, leaving
  potentially wasted depth behind shorter items.

- **3D MaxRects with proper free-volume tracking**. Genuinely
  different from extreme-point in that it maintains the free-space
  data structure explicitly. *Why not tried*: 5-10 day implementation
  with uncertain payoff. Given the pattern of negative results
  from cheaper attempts, the expected value didn't justify the
  cost.

- **Reinforcement-learning placement policy** (Hu et al. 2017
  and similar). *Why not tried*: requires ML infrastructure that
  doesn't fit the "pure Python, lightweight" character of this
  codebase, and recent RL-for-BPP results show only modest gains
  over strong heuristics like ours.

## Broader pattern

After 4 explored options (B + C cleanup + C MIP + Path 1), the
honest picture is:

- **Option B** (tuned defaults): real 2-4× speedup, kept. Algorithmic
  capability unchanged.
- **Option C cleanup** (diagnostics): D3/D8 formally proven at-LB,
  pure documentation. Algorithmic capability unchanged.
- **Option C MIP** (formulation tweaks): tried 3 changes, 2 caused
  regressions, 1 was neutral. Algorithmic capability unchanged.
- **Path 1** (new decoder strategies): tried 3 strategies, 0 wins.
  Reverted. Algorithmic capability unchanged.

**The algorithm is at its engineering ceiling for "pure Python,
candidate-set design, well-understood literature."** Further
algorithmic improvement requires substantial commitment to:
- Column generation / branch-and-price (3-6 weeks, would close N>50)
- Full BRKGA-2013 reproduction at literature compute (4+ weeks,
  would close BR gap to SOTA)
- Reinforcement learning (months, ML infrastructure)

None of these fit the "tool we will use" constraint that prompted
Option B's speed-focused tuning.

## Recommendation

**Stop investing in algorithmic improvement work.** The algorithm
is competitive with year-2000 literature, gives the right answer on
89% of internal cases (32/36 at LB), beats BR-1995 on every set
and matches Bortfeldt-2000 on BR7. Further algorithm gains are
costly and uncertain.

**The high-leverage next step is deployment**: CLI, real-cargo
ingestion, web/desktop UI, multi-pallet-type / truck modes, WMS
integration. These have:
- High and quantifiable practical value (people actually using
  the algorithm vs. it sitting in Python files)
- High confidence (no probabilistic "might converge" risk)
- Direct alignment with "tool we will use"

See `NEXT_STEPS.md` (Option D — Build out the deployment side) for
the strategic framing.
