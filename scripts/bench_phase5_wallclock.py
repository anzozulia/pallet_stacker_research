"""
bench_phase5_wallclock.py — measure end-to-end speedup from Phase 5.

Times one BR instance at a fixed budget and reports decodes/sec + best
util. Run on the current HEAD vs a pre-Phase-5 commit (5a0664b) to
quantify the wall-clock improvement.

Run inside Docker:
    docker run --rm -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/bench_phase5_wallclock.py
"""
from __future__ import annotations

import io
import os
import re
import sys
import time
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.br import parse_thpack, geometric_only_config
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit


def run_one(set_name: str, instance_no: int, time_budget_s: float, seed: int):
    """Run BRKGA on one BR instance, return (best_util, decodes, gens, wall)."""
    set_id = set_name.upper().replace("THPACK", "BR")
    instances = parse_thpack(f"benchmarks/data/{set_name}.txt", set_id=set_id)
    inst = instances[instance_no - 1]
    boxes, pallet = inst.boxes, inst.pallet
    config = geometric_only_config(max_pallets=1)
    t0 = time.time()
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = brkga_pack_v35(
            boxes, pallet, config,
            time_limit_s=time_budget_s, max_pallets=1,
            population_size=600, n_populations=3,
            patience=200, local_search_budget_s=4.0,
            seed=seed, verbose=True,
            n_modes=6,
        )
    wall = time.time() - t0
    out = buf.getvalue()

    # Parse the verbose output for decode count and last reported gen.
    m_decodes = re.findall(r"decodes=(\d+)", out)
    decodes = int(m_decodes[-1]) if m_decodes else 0
    m_gen = re.findall(r"gen (\d+)", out)
    gens = int(m_gen[-1]) if m_gen else 0

    # Util from result.
    if result.pallets:
        cap = pallet.length * pallet.width * pallet.height
        used = sum(p.box.volume for p in result.pallets[0].placements)
        util = used / cap * 100
    else:
        util = 0.0
    return util, decodes, gens, wall


def main():
    sets = [("thpack1", 1), ("thpack3", 1)]
    budget = 20.0
    seed = 43

    print("warming up JIT...", flush=True)
    warmup_jit()
    print(f"=== Phase 5 wall-clock benchmark "
          f"(budget={budget}s, seed={seed}) ===")
    print(f"{'instance':12s}  {'util %':>8s}  {'decodes':>10s}  "
          f"{'gens':>6s}  {'wall s':>8s}  {'decodes/s':>10s}")
    for set_name, inst in sets:
        util, decodes, gens, wall = run_one(set_name, inst, budget, seed)
        ds = decodes / wall if wall > 0 else 0.0
        print(f"{set_name}#{inst:<4d}  {util:8.2f}  {decodes:10d}  "
              f"{gens:6d}  {wall:8.2f}  {ds:10.0f}")


if __name__ == "__main__":
    main()
