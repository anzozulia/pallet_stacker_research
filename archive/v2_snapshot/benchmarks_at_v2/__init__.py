"""
benchmarks — test fixtures and benchmark harnesses for the pallet packer.

  - internal:       28-case internal benchmark suite (B-, C-, D-, E-, F-series)
  - failure_cases:  13-case "designed to expose weakness" suite (F-series)
  - br:             OR-Library Bischoff-Ratcliff thpack parser + harness
  - data/:          BR1/3/5/7 instance files (thpack1.txt etc.)

The .py files import from `pallet_packer` and assume the repo root is on
sys.path. Use `python -m benchmarks.<name>` from the repo root to run a
benchmark's __main__ entry point directly.
"""
