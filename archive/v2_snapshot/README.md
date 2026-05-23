# v2 Snapshot — May 2026

Frozen copy of the pallet_packer package at the v2.0 git tag.

This represents the state after Options B, C, and Path 1 were
explored — the algorithm at its engineering ceiling for the
extreme-point + CP-SAT + layer-building family.

Use this snapshot to:
- Diff against newer versions (v3 = BRKGA-2013 reproduction)
- Reproduce v2 benchmark numbers (32/36 internal at LB,
  84.2% BR1, 82.2% BR3, 81.0% BR5, 79.8% BR7 with +layer)
- Build deployment tooling (CLI, API) on a stable algorithm

Files:
  __init__.py, models.py, packer.py, mip.py, lower_bounds.py,
  validate.py, io.py — the v2 package (full algorithm)
  benchmarks_at_v2/ — the benchmark fixtures + industry dataset
                     as of v2

If you need to revert: `git checkout v2.0` from the root.
