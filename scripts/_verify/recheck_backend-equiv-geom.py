"""recheck_backend-equiv-geom.py — independent re-derivation of the claimed
backend-equiv-geom defect.

Claim under test: Cython geom decoders are bit-identical to Numba twins on all
positive-dim inputs; the ONLY divergence is degenerate zero-dim boxes, where
Numba raises ZeroDivisionError in the block path (modes 4/5) but Cython returns
a result. Probe asserts this degenerate input is "never produced by the real
pipeline".

I independently check:
  T1. Confirm the per-chromosome divergence mechanism: zero-dim box -> Numba
      ZeroDivisionError in mode 4/5, Cython returns SOMETHING. Print what
      Cython actually returns (garbage / sane / crash).
  T2. Is a zero-dim dims_all entry REACHABLE from the public Box API? The Box
      dataclass has NO validation. int(round(dx)) maps any dim in [0, 0.5) -> 0.
      Build real Box objects with (a) length=0.0 and (b) length=0.3 and run
      precompute_box_dims; show the resulting dims_all rows.
  T3. END-TO-END: run the public solver brkga_pack_v35 on a workload containing
      a zero-dim box and a sub-0.5-dim box. Does it crash? Diverge? Does the
      independent validator flag the result? Compare to a sane control.
  T4. Does the BATCH hot loop (the path the driver actually uses) hit the same
      division? Drive a population with a zero-dim box via decode_batch_blocks.

Run inside Docker (OMP_NUM_THREADS=1).
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pallet_packer._brkga_core import jit_decoders_geom_cy as cy
from pallet_packer._brkga_core import jit_decoders_geom as nb


def banner(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


# ---------------------------------------------------------------------------
# T1: per-chromosome divergence mechanism on a zero-dim box (mode 4 blocks)
# ---------------------------------------------------------------------------
def t1():
    banner("T1: per-chromosome mode-4 divergence on a zero-dim box")
    # One box, rotation 0 has dz = 0 (degenerate). Pallet 50x50x50, mp=1.
    n_boxes = 1
    n_rots = np.array([1], dtype=np.int64)   # only rot 0 used
    dims = np.zeros((1, 6, 3), dtype=np.int64)
    # base dims: 10 x 10 x 0  -> a zero-height "box"
    dims[0, 0] = (10, 10, 0)
    order = np.array([0], dtype=np.int64)
    sku = np.array([0], dtype=np.int64)
    L = W = H = 50
    mp = 1
    n_skus = 1

    def run(backend):
        po = np.zeros((n_boxes, 6), dtype=np.int64)
        cnt = backend.decode_blocks_njit_mode(order, n_rots, dims, sku, L, W, H, mp, po, n_skus)
        return po, int(cnt)

    nb_res = None
    nb_exc = None
    try:
        nb_res = run(nb)
    except Exception as e:  # noqa: BLE001
        nb_exc = type(e).__name__

    cy_res = None
    cy_exc = None
    try:
        cy_res = run(cy)
    except Exception as e:  # noqa: BLE001
        cy_exc = type(e).__name__

    print(f"  Numba : exc={nb_exc}  res={nb_res}")
    print(f"  Cython: exc={cy_exc}  res={cy_res}")
    diverged = (nb_exc != cy_exc) or (
        nb_exc is None and cy_exc is None and not (
            np.array_equal(nb_res[0], cy_res[0]) and nb_res[1] == cy_res[1]))
    print(f"  DIVERGENCE: {diverged}")
    if cy_res is not None:
        # is the cython block count sane? block region best_x+k*dx etc.
        po = cy_res[0]
        print(f"  Cython placement row 0: {po[0].tolist()} "
              f"(bin,rot,x,y,z,placed)")
    return diverged


# ---------------------------------------------------------------------------
# T2: reachability of zero-dim dims_all from the public Box API
# ---------------------------------------------------------------------------
def t2():
    banner("T2: zero-dim dims_all REACHABLE from public Box API?")
    from pallet_packer import Box
    from pallet_packer.brkga_v3_fast import precompute_box_dims

    cases = [
        ("zero-height length=0.0", Box(id="z", length=10.0, width=10.0, height=0.0)),
        ("sub-half height=0.3", Box(id="h", length=10.0, width=10.0, height=0.3)),
        ("sub-half length=0.49", Box(id="l", length=0.49, width=10.0, height=10.0)),
        ("control 10x10x10", Box(id="c", length=10.0, width=10.0, height=10.0)),
    ]
    any_zero = False
    for name, b in cases:
        nrots, dims = precompute_box_dims([b])
        has_zero = bool((dims[0] == 0).any())
        any_zero = any_zero or has_zero
        print(f"  {name:28s}: dims[0,0]={dims[0,0].tolist()}  has_zero_entry={has_zero}")
    print(f"  ==> A real Box can produce a zero-dim dims_all entry: {any_zero}")
    print("  (Box dataclass has NO dimension validation / no __post_init__.)")
    return any_zero


# ---------------------------------------------------------------------------
# T3: END-TO-END full solve via public API with a degenerate box
# ---------------------------------------------------------------------------
def t3():
    banner("T3: END-TO-END brkga_pack_v35 with a degenerate box (public API)")
    from pallet_packer import Box, Pallet, PackerConfig, validate
    from pallet_packer.brkga_v3_5 import brkga_pack_v35

    pallet = Pallet(length=100.0, width=100.0, height=100.0)
    cfg = PackerConfig()  # geometric defaults

    # Workload: many identical 20x20x20 boxes (forms a SKU -> triggers block
    # modes 4/5), PLUS one degenerate zero-height box of the SAME footprint so
    # it shares a SKU bucket OR forms its own. Use two scenarios.
    def identical(n, dims, idprefix):
        return [Box(id=f"{idprefix}{i}", length=dims[0], width=dims[1], height=dims[2])
                for i in range(n)]

    scenarios = {
        "control_sane": identical(20, (20.0, 20.0, 20.0), "c"),
        "with_zero_height": identical(19, (20.0, 20.0, 20.0), "c") +
                            [Box(id="zero", length=20.0, width=20.0, height=0.0)],
        "with_subhalf": identical(19, (20.0, 20.0, 20.0), "c") +
                        [Box(id="tiny", length=0.3, width=20.0, height=20.0)],
        "all_zero_sku": identical(20, (20.0, 20.0, 0.0), "z"),  # whole SKU degenerate
    }

    results = {}
    for name, boxes in scenarios.items():
        outcome = {"crash": None, "n_pallets": None, "viol": None}
        try:
            res = brkga_pack_v35(
                boxes, pallet, cfg,
                time_limit_s=2.0, max_pallets=1, seed=12345,
                population_size=20, n_populations=1, patience=20,
                local_search_budget_s=0.0, n_modes=6, verbose=False,
            )
            outcome["n_pallets"] = len(res.pallets)
            outcome["n_unpacked"] = len(res.unpacked)
            viol = validate(res, pallet, cfg)
            outcome["viol"] = viol
        except Exception as e:  # noqa: BLE001
            outcome["crash"] = f"{type(e).__name__}: {e}"
            outcome["tb"] = traceback.format_exc().splitlines()[-3:]
        results[name] = outcome
        print(f"\n  [{name}]")
        if outcome["crash"]:
            print(f"    CRASH: {outcome['crash']}")
            for ln in outcome.get("tb", []):
                print(f"      {ln}")
        else:
            print(f"    pallets={outcome['n_pallets']} unpacked={outcome['n_unpacked']} "
                  f"validator_violations={len(outcome['viol'])}")
            if outcome["viol"]:
                for v in outcome["viol"][:5]:
                    print(f"      VIOLATION: {v}")
    return results


# ---------------------------------------------------------------------------
# T4: batch hot-loop path with a zero-dim box (the path the driver uses)
# ---------------------------------------------------------------------------
def t4():
    banner("T4: BATCH decode_batch_blocks with a zero-dim box (driver hot path)")
    from pallet_packer._brkga_core import dispatch
    print(f"  _BATCH_AVAILABLE = {getattr(dispatch, '_BATCH_AVAILABLE', '??')}")

    n_boxes = 3
    n_rots = np.array([1, 1, 1], dtype=np.int64)
    dims = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    dims[0, 0] = (20, 20, 20)
    dims[1, 0] = (20, 20, 20)
    dims[2, 0] = (20, 20, 0)   # zero-height degenerate, same SKU footprint-ish
    sku = np.array([0, 0, 0], dtype=np.int64)
    L = W = H = 100
    mp = 1
    n_skus = 1

    # Build a small population of orders.
    pop = 4
    orders = np.zeros((pop, n_boxes), dtype=np.int64)
    rng = np.random.default_rng(7)
    for p in range(pop):
        orders[p] = rng.permutation(n_boxes)

    if not hasattr(cy, "decode_batch_blocks_njit_mode"):
        print("  (cython has no decode_batch_blocks_njit_mode export; skipping)")
        return None

    po_batch = np.zeros((pop, n_boxes, 6), dtype=np.int64)
    nbins = np.zeros(pop, dtype=np.int64)
    cy_exc = None
    try:
        # signature discovery: try the common batch signature
        cy.decode_batch_blocks_njit_mode(
            orders, n_rots, dims, sku, L, W, H, mp, po_batch, nbins, n_skus)
        print(f"  Cython batch returned: n_bins per chrom = {nbins.tolist()}")
        print(f"  placement row for box2 (zero-dim), chrom0: "
              f"{po_batch[0, 2].tolist()}")
    except TypeError as e:
        print(f"  (batch signature mismatch, skipping: {e})")
        return None
    except Exception as e:  # noqa: BLE001
        cy_exc = f"{type(e).__name__}: {e}"
        print(f"  Cython batch CRASH: {cy_exc}")
    return cy_exc


def main():
    print("recheck backend-equiv-geom: re-derive the claimed defect from scratch")
    d1 = t1()
    z2 = t2()
    r3 = t3()
    e4 = t4()

    banner("VERDICT INPUTS")
    print(f"  T1 per-chrom mode4 divergence on zero-dim box : {d1}")
    print(f"  T2 zero-dim dims reachable from public Box API: {z2}")
    print("  T3 end-to-end outcomes:")
    crashed = []
    invalid = []
    for name, o in (r3 or {}).items():
        if o["crash"]:
            crashed.append(name)
        elif o["viol"]:
            invalid.append(name)
    print(f"      scenarios that CRASHED         : {crashed}")
    print(f"      scenarios with VALIDATOR VIOLS : {invalid}")
    print(f"  T4 batch path crash/result          : {e4!r}")

    print("\n  INTERPRETATION:")
    print("   - The bit-identity finding (0 real divergences on positive dims) is")
    print("     CORROBORATED by the original sweep + T1.")
    print("   - The divergence is REACHABLE iff a zero-dim survives to mode 4/5.")
    print("   - Whether this is a real defect hinges on T2 (reachability) and T3")
    print("     (does the full public solve crash or emit an INVALID result).")


if __name__ == "__main__":
    main()
