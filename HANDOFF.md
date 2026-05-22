# Handoff — 3D Pallet Packer

Baseline document for continuing work on this codebase. Written for an
engineer (or another agent) picking this up cold.

---

## What this is

A 3D bin-packing / pallet-loading algorithm in pure Python. Takes a list
of boxes (with weight, allowed rotations, fragility, etc.) and a pallet
spec, returns a placement of every box that respects:

- Geometric fit (within pallet bounds, no inter-box overlap)
- Support / no-floating (every box rests on something below)
- Per-box weight limits (stacking pressure)
- Per-pallet weight budget
- Centre-of-gravity envelope
- Rotation restrictions ("this side up", fragile, etc.)
- Optional edge overhang

The codebase has two heuristic engines (an extreme-point engine and a
layer-building engine) plus a CP-SAT-based exact solver (OR-Tools) that
runs on small subproblems. They feed into a single candidate set; the
final answer is `min(candidates, key=quality)`.

---

## How good is it (as of this commit)

Tested on 41 internal cases + 86 Bischoff-Ratcliff benchmark instances.

**Internal suite:** 32 of 36 testable cases at provable lower bound
(89%). After the diagnostic cleanup (see
[09_diagnostic_cleanup.md](docs/reports/09_diagnostic_cleanup.md)),
the remaining 4 split into:
- 2 are *strongly suspected* at-true-LB but not formally proven:
  D7 (no rotation — MIP at 300s found only 4p/4u) and C2 (BR-lite —
  MIP at 300s found 3p/4u, improved from 3p/7u at 35s). Both would
  need much longer MIP budgets or stronger CP-SAT formulation to
  formally close.
- 2 (well, 3) exceed our MIP threshold and need bigger compute or
  different algorithms (F2 N=120, F12 N=72, C4 N=80).
- (F1 N=53 is at LB=1 by volume — unimprovable.)

D3 and D8 were formally MIP-proven at-true-LB during the cleanup pass.

**BR1-7 (academic benchmark):** beats the 1995 baseline on every set,
matches the 2000 baseline on BR7 (the hardest set). Gap to BRKGA-2013
(modern state of the art): 5-8pp, would require substantially more
sophisticated search to close.

The non-trivial wins came from two changes:
1. **SKU-grid layer fill** (BR-1995 pattern): +2.6 to +3.4pp on BR.
2. **MIP polish with correct calibration** (single-threaded, n_threshold=50,
   warm-started from heuristic): closes 7 internal cases that the
   heuristic couldn't.

Validator-clean across all configurations.

---

## Quick start

```bash
cd /Users/anzozulia/Desktop/pallet_stacker_research

# Smoke test: pack a small mix using all defaults
python3 example.py

# Run the 41-case regression (~3-5 minutes)
python3 run_phase2_regression.py

# Run the BR benchmark (small sample, ~1 minute)
python3 br_benchmark.py --sample 3 --skip-v2

# Run the BR deep evaluation (~30 min for 86 instances)
python3 run_br_deep_eval.py --max-time 1800

# Run a diagnostic to check if a specific case is at true LB via MIP
python3 run_lb_diagnostic.py --case C3 --target 1 --time-limit 30
```

The validator is independent and always-on:

```python
from pallet_packer import validate
errors = validate(result, pallet, config)
assert not errors
```

---

## Code map

```
pallet_packer.py        ~2600 lines. The main module.
                        - Data models: Box, Pallet, Placement, PackResult,
                          PackerConfig (~50 config fields)
                        - PalletState: per-pallet state with extreme-point
                          engine, constraint stack, GRASP-aware try_place
                        - PalletPacker: top-level packer with:
                          - multi-start search
                          - BRKGA (population-based search)
                          - block-building preprocessor
                          - layer-building decoder (3 axes × 4 seed
                            strategies × 3 fills incl. SKU-grid)
                          - ejection chains (single + multi-pallet)
                          - last-pallet consolidation
                          - safety-net candidate generation
                          - MIP polish invocation
                        - validate(): independent constraint checker

mip_polish.py           ~400 lines. CP-SAT 3D-BPP solver.
                        - Per-item rotation choice as Booleans
                        - Pairwise no-overlap with 6 separation literals
                        - Corner-sampling support constraint
                        - Fragility (max_load_on_top=0)
                        - Warm-starting from heuristic
                        - Fixed obstacles (subproblem extraction)
                        - Replay-validate output through real constraint
                          stack

lower_bounds.py         LB computation: volume, per-SKU-max, geometric
                        Martello-Pisinger, LP relaxation. Used by reports
                        and diagnostics.

br_benchmark.py         OR-Library thpack parser + harness.
                        geometric_only_config() helper.

example.py              Headline scenario (the user's original 45-box mix).
visualize.py            Renders a packing as 3D matplotlib figure from
                        packing_result.json.
benchmark.py            28-case internal benchmark suite.
failure_cases.py        13-case "designed to expose weakness" suite.

run_*.py                Test/eval harnesses (all JSON-checkpointed):
  run_phase2_regression.py    41-case suite × N configs ablation
  run_evaluation.py           v1 vs quality_max evaluation
  run_lb_diagnostic.py        MIP-based "is this case at true LB?" check
  run_br_deep_eval.py         BR1/3/5/7 large-sample evaluation
  run_lb_report.py            41-case LB report generator
  run_deep_br.py              earlier BR seed-variance harness

Reports (markdown, one per phase / topic):
  ROADMAP.md                  initial phased plan
  ANALYSIS.md                 v1 deep dive
  FAILURE_ANALYSIS.md         failure-mode to literature mapping
  README.md                   full feature reference (gets stale fast)
  PHASE2_REPORT.md            cumulative Phase 2/2e/4 findings
  EVALUATION_REPORT.md        41-case + diagnostic story
  QUALITY_ROADMAP.md          Q1-Q7 sequenced improvements
  BR_REPORT.md / BR_DEEP_REPORT.md   BR results
  LB_REPORT.md                LB-vs-achieved per case
  NEXT_STEPS.md               non-technical analysis of directions
  HANDOFF.md                  this file
```

---

## Gotchas — things that took hours to figure out

1. **CP-SAT: `num_workers=1` is FASTER than parallel for this model.**
   Multi-threaded mode interferes with the warm-start hint. C1 with 4
   workers: no convergence in 30s. With 1 worker: OPTIMAL in 12s. This
   is now the default (`mip_num_workers: int = 1`).

2. **Safety net is non-negotiable.** Randomized features (GRASP, BRKGA)
   can actively destroy v1 quality on certain inputs (BR1 inst 1 went
   from 79% to 58% without safety net). The safety net re-runs paths
   with `grasp_alpha=1` from a fresh-seeded RNG and includes results
   as fallback candidates. `use_safety_net=True` is the default; only
   disable for raw benchmarking.

3. **"Open +N" gaps reported against volume LB are often false
   positives.** The volume LB ignores support, fragility, rotation.
   Three cases (C3, D8, F3 bench) were originally flagged as "+1 gap"
   but turned out to be at the true LB once support / fragility were
   accounted for. Always sanity-check open gaps with a tighter LB or
   with a MIP diagnostic before treating them as headroom.

4. **The `optimize` config field matters for BR.** Default
   `optimize="min_unpacked"` is right for multi-pallet logistics
   (minimize pallets, secondarily maximize util). For BR's
   single-container objective, `optimize="max_util"` is right
   (maximize util, secondarily minimize unpacked). With the wrong
   setting on BR you can get measurably worse benchmark scores even
   when the algorithm is generating better packings — the candidate
   selector picks the wrong one.

5. **SKU-consistent rotation must be overhang-aware.** Original
   implementation picked rotations based on strict pallet dims; with
   overhang enabled this picked the wrong rotation and regressed F10
   from 1p/140% to 2p/70%. Fixed by computing grid count on effective
   footprint.

6. **Replay-validate every algorithmic output.** Layer-building MIP
   and subproblem MIP can produce results that fail the full
   constraint stack (e.g., axis='z' layers above floor produce
   floating boxes; MIP without explicit support modeling can place
   unsupported items). Always replay through `PalletState.feasible`
   and demote failed placements to `.unpacked` before returning.

---

## Architecture invariants to preserve

- **`PackResult.validate()` returns 0 errors** is non-negotiable. Every
  return path must pass validation.
- **Candidate set + `min(quality)` design**: features compete by
  generating candidates. Quality function (`(unpacked, pallets, -util)`
  or its `max_util` variant) selects best. New features should produce
  candidates, not mutate the "current best" in place.
- **`self._rng` is shared across the packer.** When a path needs
  reproducibility independent of prior RNG consumption (e.g., safety
  net), save it, replace with a fresh `random.Random(config.seed)`,
  restore after.
- **No feature should regress v1 by design.** If a feature can in
  principle produce worse results than v1, it MUST be paired with a
  candidate-set entry that re-runs v1 (or its closest equivalent) so
  `min(quality)` falls back.

---

## What's left to do

In order of (effort, expected value):

### Small, well-defined work

- ~~**D8 cleanup**~~ — done. D8 reclassified to `expected_pallets=6`
  in `benchmarks/internal.py` with MIP-proof citation.
- ~~**MIP for D3**~~ — done. MIP @ 300s returned OPTIMAL 4p/2u →
  formally proves 5p is the true LB. Reclassified to
  `expected_pallets=5`.
- **MIP for D7 (not formally closed)**. MIP @ 300s returned FEASIBLE
  4p/4u without proving either way. Strongly suspected at 5p but
  needs hours-scale MIP or better formulation to formalize.
- **C2 (not formally closed)**. MIP @ 300s improved to FEASIBLE
  3p/4u (vs 3p/7u @ 35s) but still didn't prove 3p/0u feasible nor
  4p as true LB. Strongly suspected at 4p; same blocker as D7.

The realistic-closeable algorithmic gap is now 2 cases (D7, C2);
both wait on more solver time or formulation work.

### Medium, real-impact work

- **Scale MIP to N=60-80.** F12 (N=72) and C4 (N=80) are the largest
  open cases. Currently MIP threshold is 50. The blockers are
  solve-time growth (CP-SAT runtime ~quadratic in N) and the
  subproblem extraction approach not finding improvements at this
  scale. Possible directions: smarter subproblem selection (e.g.,
  let MIP rearrange a full pallet without obstacles), column
  generation, or a different decomposition. ~1-2 weeks.
- **BRKGA at much larger budgets for BR.** Our quality_max BR gain is
  +3pp; BRKGA-2013 published is +10pp. Reaching the 2013 number
  requires pop=20·N (e.g., 2000) generations=100-200 vs our
  pop=12 gens=3. That's a 10000x larger budget per instance.
  Tractable if integrated with layer-building (so each BRKGA evaluation
  is a layer-decoder run, not a full extreme-point run). ~2-4 weeks.
- **3D MaxRects sub-fill for layer-building.** Current sub-fill is
  either extreme-point (3D) or 2D MaxRects (within slab face). A
  proper 3D MaxRects that tracks free volumes within a slab could
  beat both. ~1 week.

### Larger, structural work

- **Multi-pallet co-optimization (Phase 3).** The outer FFD loop is
  greedy. For multi-pallet objectives, a proper co-optimization
  (BRKGA with chromosome encoding pallet assignment + placement, or
  tabu-search over assignments) could close cases where the FFD
  outer loop strands a low-util tail. We attempted a limited version
  (`_consolidate_via_rebuild`); a full version would do more. ~2-3
  weeks.
- **Deployment layer.** This is the highest-leverage practical work.
  The algorithm is competitive; the friction is everything around it:
  no CLI for non-Python users, no visualization beyond matplotlib, no
  integration with WMS systems, no truck/multi-pallet-type support,
  no scheduling / order-flow integration. Effort wildly variable
  depending on what you build for. See NEXT_STEPS.md for the
  non-technical framing.

---

## Things NOT worth doing

- More metaheuristic exploration features (more BRKGA variants, more
  perturbation schemes, etc.). These add defensive value but don't
  move headline cases. We tried several; none helped beyond F3.
- Heuristic tweaks (different sort orders, scoring strategies). The
  heuristic is well-tuned. The remaining wins are on the solver side
  or the deployment side.
- Adding more internal benchmark cases. The 41-case suite plus 86 BR
  instances are sufficient for any development decision. Adding more
  test cases won't reveal anything new about the algorithm.

---

## How to continue

1. Read this file.
2. Skim [EVALUATION_REPORT.md](EVALUATION_REPORT.md) for the per-case
   results and the diagnostic narrative.
3. Skim [BR_DEEP_REPORT.md](BR_DEEP_REPORT.md) for the academic
   benchmark story.
4. Run `python3 run_phase2_regression.py` to validate the current
   state matches the documented results.
5. Pick a direction from "What's left to do" — or decide on the
   strategic question (algorithm improvement vs. deployment) per
   [NEXT_STEPS.md](NEXT_STEPS.md).

For algorithmic work, always:
- Start with a TEST that exercises the change (extend the regression
  or write a one-off in the diagnostic style).
- Use the candidate-set pattern, don't bypass it.
- Validate every output.
- Compare against v1 explicitly.

For deployment work, the algorithm side is stable enough to treat as
an API.
