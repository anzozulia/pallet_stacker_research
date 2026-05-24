# 3D Pallet Packer — Research Repository

A 3D bin-packing / pallet-loading algorithm in pure Python that handles real
physical constraints: per-box weight, configurable rotation, edge overhang,
anti-floating support, fragility / load-bearing limits, and per-pallet
centre-of-gravity envelope.

Built over multiple phases of algorithmic research, anchored against the
academic Bischoff–Ratcliff benchmark. The codebase is at a natural pause
point — algorithm work has reached the realistic ceiling for pure Python +
CP-SAT, and the next decision is strategic (see `NEXT_STEPS.md`).

---

## Status at a glance

| Benchmark                             | v2 (production) | v3.5 | **v3.6** (research, n=5) |
|---------------------------------------|-----------------|------|--------------------------|
| Internal 41-case suite                | 32 / 36 LB (89%) | —   | —                        |
| BR1 (3 SKUs)                          | 84.2%           | 88.7% | 88.85%                   |
| BR3 (8 SKUs)                          | 82.2%           | 87.9% | **92.17%** (BEATS SOTA 90.5) |
| BR5 (12 SKUs)                         | 81.0%           | 87.9% | **92.54%** (BEATS SOTA 88.7) |
| BR7 (20 SKUs, hardest)                | 79.8%           | 86.7% | **92.77%** (BEATS SOTA 85.4 by +7.4pp) |
| Validator errors across all configs   | 0               | 0   | 0                        |

**v3.6 (`pallet_packer/brkga_v3_5.py` with `n_modes=5, use_lns=True`)** is
the current best algorithm: v3.5 + composite blocks (Bischoff-Ratcliff
1995, decoder mode 4) + Large Neighborhood Search polish. **Beats
BRKGA-2013 SOTA on BR3/5/7.** Only BR1 still has gap to SOTA (3.75pp).
See [docs/reports/16_v36_blocks.md](docs/reports/16_v36_blocks.md).

**v3.5** (without blocks): hybrid BRKGA with multi-decoder (DFTRC + wall
+ corner + layer-build), v2 warm-start seed, position-based local search,
and path relinking. JIT-compiled core for 1.6ms/decode. See
[docs/reports/15_v35_breakthrough.md](docs/reports/15_v35_breakthrough.md).

**v2 (`pallet_packer/packer.py`)** remains as the trusted production
reference with the full constraint stack (support, weight, fragility,
CoG envelope). Use v2 for deployment; use v3.6 for BR-style academic
benchmarking or pure-geometric optimization.

Of the 4 remaining "open" internal cases, 2 are strongly suspected at-true-LB
(D7, C2 — MIP at 300s couldn't formally close them but found no
target-pallet solution either) and the other 3 are beyond current MIP capability
(F2 N=120, F12 N=72, C4 N=80). F1 is at LB=1 by volume — unimprovable.
See [docs/reports/09_diagnostic_cleanup.md](docs/reports/09_diagnostic_cleanup.md).

---

## Repository layout

```
pallet_packer/         The algorithm package (importable).
  models.py            Box, Pallet, Placement, PackerConfig, Rotation + constants.
  packer.py            PalletState (per-pallet engine) + PalletPacker + PackResult.
  mip.py               CP-SAT exact 3D-BPP polish (OR-Tools).
  lower_bounds.py      Volume / per-SKU / Martello-Pisinger / LP bounds.
  validate.py          Independent constraint validator.
  io.py                JSON serialization (to_json / save_json).

benchmarks/            Test fixtures.
  internal.py          28-case internal benchmark suite.
  failure_cases.py     13-case "designed to expose weakness" suite.
  br.py                Bischoff-Ratcliff thpack parser + harness.
  data/                BR1/3/5/7 instance files (thpack1.txt etc.).

scripts/               Runnable entry points (run from repo root).
  example.py                 Quickstart demo.
  visualize.py               Render a packing JSON as 3D figures.
  visualize_cases.py         Multi-panel gallery of benchmark cases.
  visualize_failures.py      Render the failure-case scenarios.
  run_phase2_regression.py   41-case regression with config ablation.
  run_evaluation.py          v1 vs quality_max evaluation (per-case).
  run_lb_diagnostic.py       MIP-based "is this case at true LB?" check.
  run_lb_report.py           41-case LB report generator.
  run_br_deep_eval.py        BR1/3/5/7 large-sample evaluation.

docs/reports/          Historical research reports — the journey.
  00_original_roadmap.md     Initial phased plan.
  01_v1_analysis.md          Deep dive on v1 baseline (60% at LB).
  02_failure_analysis.md     Failure-mode → literature mapping.
  03_lb_phase0.md            Tightening lower bounds (Phase 0).
  04_br_baseline.md          BR1-7 baseline anchoring (Phase 1).
  05_phase2.md               Phase 2 implementation findings.
  06_quality_roadmap.md      Q1-Q7 sequenced improvements.
  07_evaluation.md           Canonical per-case evaluation + MIP discovery.
  08_br_deep_eval.md         BR1-7 deep evaluation (86 instances).
  09_diagnostic_cleanup.md   Post-cleanup: D3/D8 formally proven at LB.

results/               Re-run artifacts (gitignored).
  checkpoints/         JSON eval checkpoints.
  galleries/           Visualization PNGs.
  examples/            example.py output.

archive/               Legacy code preserved for diff but no longer maintained.
  v1_snapshot/         Pre-Phase-0 v1 baseline.
  compare_v1_v2.py     Superseded v1-vs-v2 comparison harness.
  visualize_v1_v2.py   Superseded v1-vs-v2 visualizer.
  run_deep_br.py       Superseded BR harness (use run_br_deep_eval.py).
  README_legacy.md     Original v1+v2-era README.

HANDOFF.md             Engineering continuation guide. Read this first.
NEXT_STEPS.md          Strategic direction options.
```

---

## Quick start

```bash
# Smoke test: pack a 45-box mix using all defaults
python3 scripts/example.py

# Run the 41-case regression (~3–5 minutes)
python3 scripts/run_phase2_regression.py

# Run a small BR sample
python3 scripts/run_br_deep_eval.py --sets thpack1 --max-time 60

# Run an MIP-based diagnostic to check if a case is at the true LB
python3 scripts/run_lb_diagnostic.py --case C3 --target 1 --time-limit 30
```

Library use:

```python
from pallet_packer import (
    Box, Pallet, PalletPacker, PackerConfig,
    THIS_SIDE_UP, save_json, validate,
)

boxes = [Box(id=f"A-{i}", length=30, width=40, height=70, weight=5.0,
             allowed_rotations=THIS_SIDE_UP, max_load_on_top=50.0)
         for i in range(20)]

pallet = Pallet(length=120, width=100, height=100, max_weight=500)
result = PalletPacker(pallet, PackerConfig()).pack(boxes)

errors = validate(result, pallet, PackerConfig())
assert not errors
```

---

## Where to read next

1. **`HANDOFF.md`** — Engineering continuation guide: state, gotchas
   (hard-won lessons like CP-SAT `num_workers=1` being faster than
   parallel), architecture invariants, and a sequenced list of remaining
   work.
2. **`docs/reports/07_evaluation.md`** — Canonical "where are we now"
   report, including the per-case results and the MIP-calibration
   discovery that closed 7 cases at once.
3. **`docs/reports/08_br_deep_eval.md`** — Academic benchmark story:
   how SKU-grid layer fill brought us above the 1995 baseline.
4. **`NEXT_STEPS.md`** — Non-technical strategic framing of the decision
   ahead: algorithm improvement vs. deployment vs. academic gap-closing.

---

## What the algorithm does (one paragraph)

Takes a list of boxes (each with weight, allowed rotations, fragility,
etc.) and a pallet spec. Returns a placement of every box that respects:
geometric fit within pallet bounds, no inter-box overlap, support /
no-floating constraint per box, per-box load-bearing limits, per-pallet
weight budget, centre-of-gravity envelope, rotation restrictions ("this
side up", "no rotation"), and optional edge overhang. The packer combines
two heuristic engines (extreme-point and layer-building) with a CP-SAT
exact solver (OR-Tools) for small subproblems, all feeding a single
candidate pool. The final answer is `min(candidates, key=quality)` —
features compete by adding candidates, never by mutating state. The
independent validator (`pallet_packer.validate`) re-checks every
returned result against the full constraint stack.
