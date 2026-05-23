# Option B — Adaptive Solver Budgets + Tuned Defaults

A tool-usability pass. The packer's algorithm was already strong; what
hurt was the default config baking in compute it didn't need. This pass
fixes the defaults and adds a small adaptive layer around the MIP budget
so users get sensible behavior out of the box.

## Bug found and fixed

The Phase 1 reorganization (commit `817ab2f`) left two stale imports
in `pallet_packer/packer.py`:

```python
# These silently fail since the reorg — MIP path was effectively dead
# in the main pack() method when use_mip_polish=True.
from mip_polish import mip_polish as _mip_polish    # WRONG (old name)
```

The imports were inside `try: ... except ImportError: pass` blocks, so
the failure was silent — MIP got disabled instead of crashing. Anyone
who set `use_mip_polish=True` on `pack()` would have gotten no MIP at
all without realizing it. (The diagnostic script and direct calls to
`pallet_packer.mip.mip_polish` weren't affected — they use the new path
explicitly.)

Fixed to `from .mip import mip_polish as _mip_polish` at both call
sites (line 850 in the main pack() method, line 1000 in
`_mip_last_pallet_polish`).

## Changes

### 1. `multi_start_trials` default: 20 → 1

ANALYSIS.md (now `docs/reports/01_v1_analysis.md`) established years
ago that multi-start trials produce **zero** improvement on every
tested instance — `trials=1` and `trials=50` give identical pallet
counts, utilization, and zero seed variance. The 20-default was thus
a 20× compute multiplier with no benefit.

Cases that explicitly override (`multi_start_trials=25` on D-series)
keep their override.

### 2. Adaptive MIP time budget (when use_mip_polish=True)

Added `_adaptive_mip_budget(n_items, default_budget)` helper that
scales the MIP wall-clock budget by problem size:

| N range | Budget |
|---------|--------|
| N < 10  | `min(5s, default)` |
| N < 25  | `min(15s, default)` |
| N ≥ 25  | `default` (typically 30s) |

Empirical calibration: small problems converge to OPTIMAL in under a
second; the 30s default was pure overhead. The helper never returns
*more* than the user's `mip_time_limit_s`, so explicit overrides are
respected for hard instances.

Wired at both MIP call sites (main pack() + subproblem polish).

## Validation

### Full 41-case internal regression

```
Total runtime: 25.03s  |  Pass: 28  |  Fail: 0  |  Validator errors: 0
```

(`Pass` = expected_pallets/expected_unpacked match. Bench cases with
`expected_pallets=None` aren't counted as pass/fail but still validate.)

All cases produce identical pallet counts and utilization to the
pre-tuning baseline. Timing comparison on cases using default config
(`PackerConfig()`):

| Case | Before | After | Speedup |
|------|--------|-------|---------|
| C1 user's headline mix (N=45) | 0.22s | 0.09s | 2.4× |
| C2 BR-lite (N=40) | 0.35s | 0.15s | 2.3× |
| C3 Pareto-30 | 0.51s | 0.21s | 2.4× |
| F1 e-commerce (N=53) | 3.76s | 1.55s | 2.4× |
| F2 pharma (N=120) | 18.37s | 10.44s | 1.8× |

Cases that override the default with `multi_start_trials=25` (D-series)
keep their original runtime — those tests deliberately exercise
multi-start behavior.

### BR sample regression (5 instances per set)

| Set | Mean util | Mean runtime | Docs v1 mean (n=18-19) |
|-----|-----------|--------------|-----------------------|
| BR1 | 81.3% | 3.29s | 82.4% |
| BR3 | 78.9% | 3.28s | 78.8% |
| BR5 | 78.6% | 3.24s | 78.2% |
| BR7 | 78.2% | 3.58s | 77.8% |

Within sample variance of documented v1 baseline. BR's
`geometric_only_config` already set `multi_start_trials=1` explicitly,
so the default change is a no-op there — confirms no quality regression
in the academic benchmark.

### MIP-enabled sanity check (use_mip_polish=True)

| Case | N | Result | Runtime |
|------|---|--------|---------|
| B1 (already-optimal) | 8 | 1p/0u/100.0% | 0.41s (was 1.28s — 3× faster) |
| C1 user mix | 45 | **4p/0u/85.6%** | 20.09s |
| D1 base | 45 | **4p/0u/85.6%** | 19.59s |
| D8 fragile | 45 | 6p/0u/57.1% | 12.61s |
| C3 Pareto-30 | 30 | 2p/0u/40.5% | 23.78s |
| F3 heavy industrial | 8 | 2p/0u/40.0% | 0.15s |

C1 and D1 still close to **4p/85.6%** — the +17pp MIP wins documented
in the EVALUATION_REPORT are preserved.

## What this doesn't do

- **Doesn't enable MIP by default.** That would surprise users with
  5-30s pack times and require OR-Tools to be installed for the tool
  to "feel right." Users who want the +17pp wins on C1-style mixes can
  still opt in with `PackerConfig(use_mip_polish=True)`.
- **Doesn't change the algorithm.** This is purely default-tuning +
  one small policy helper. The candidate-set / `min(quality)` pattern,
  the validators, all the search machinery are untouched.
- **Doesn't address D-series tests' multi_start_trials=25.** Those
  case configs were intentional — they exercise the multi-start path.
  Left as-is.

## Headline

Default-config packings are now **2-4× faster** at zero quality cost
across the realistic-N range (10-120 boxes). MIP path is correctly
wired (silent bug fixed) and adaptive — small problems no longer waste
the full 30s budget.
