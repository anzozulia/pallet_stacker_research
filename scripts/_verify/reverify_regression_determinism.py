"""reverify_regression_determinism.py — fresh adversarial determinism re-check
after the driver changes (use_cstr_path, group dispatch, D2 decoupling,
max_pallets propagation into v2 packer).

Targets the SPECIFIC code paths the existing verify_determinism.py does NOT
exercise, because that probe deliberately disables LS/LNS/v2-hybrid and forces
the geometric/explicit config. Here we instead hit the PRODUCTION-DEFAULT paths:

  (a) A CONSTRAINED industry solve repeated 3x with the *default* driver
      settings (use_v2_seed auto-enabled because has_constraints=True;
      use_v2_hybrid_polish default True) -> bit-identical placement signature.
      Bounded by patience (low) + generous time so termination is by patience,
      not the wall clock -> wall-clock-independent.
  (c) D2 path: a WEIGHTLESS workload with support_ratio=0.8 + require_centroid
      -> this is the NEW use_cstr_path / stability_active branch (constraint
      decoder active WITHOUT real weight constraints; v2-seed must stay OFF
      because has_constraints is False). Repeated 3x -> identical.
  (d) Box.group multi-pallet solve (IND6) -> hits driver _pack_with_groups
      dispatch (max_pallets>1 + any group). Repeated 3x -> identical AND every
      group co-located on a single pallet (the fix's invariant) AND validator
      clean.

The script prints REVERIFY_SHA256= over the concatenation of all converged
signatures. The orchestrator runs this twice (OMP=1, OMP=8, PYTHONHASHSEED=0)
and compares the digest for thread-invariance.
"""
from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pallet_packer import (
    validate, Box, Pallet, PackerConfig, ALL_ROTATIONS,
)
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer._brkga_core import dispatch
from pallet_packer._brkga_core.precompute import precompute_constraint_arrays
from benchmarks.industry import cases as industry_cases


def placements_signature(result):
    rows = []
    for pi, st in enumerate(result.pallets):
        for p in st.placements:
            rows.append((pi, str(p.box.id), p.rotation.name,
                         round(float(p.x), 6), round(float(p.y), 6), round(float(p.z), 6)))
    rows.sort()
    return tuple(rows)


def sig_to_str(sig):
    return "\n".join("|".join(str(c) for c in row) for row in sig)


def sha256_of(*sig_strings):
    h = hashlib.sha256()
    for s in sig_strings:
        h.update(s.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def group_of(box_id):
    # IND6 ids look like "C1-03"
    return str(box_id).split("-")[0]


def repeat(label, solve_fn, n=3, extract=None):
    sigs, extra, last = [], "", None
    for _ in range(n):
        r = solve_fn()
        sigs.append(placements_signature(r))
        last = r
    if extract is not None:
        extra = extract(last)
    distinct = len(set(sigs))
    ok = distinct == 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {n}x distinct_sigs={distinct} "
          f"n_placed={[len(s) for s in sigs]}{(' ' + extra) if extra else ''}")
    if not ok:
        si, sj = set(sigs[0]), set(sigs[1])
        print(f"        run0!=run1: |only0|={len(si - sj)} |only1|={len(sj - si)} "
              f"first_diff={sorted(si ^ sj)[:3]}")
    return ok, sig_to_str(sigs[0]), last


def main():
    print("=" * 72)
    print("REVERIFY REGRESSION-DETERMINISM  (OMP_NUM_THREADS=%s PYTHONHASHSEED=%s)"
          % (os.environ.get("OMP_NUM_THREADS", "default"),
             os.environ.get("PYTHONHASHSEED", "unset")))
    print("=" * 72)
    warmup_jit()
    mod = dispatch.decode_njit_mode.__module__
    print(f"[backend] {mod}  CYTHON={'cy' in mod}  BATCH={dispatch._BATCH_AVAILABLE}")

    overall_ok = True
    blocks = []
    cs = industry_cases()
    by = {c.name.split()[0]: c for c in cs}

    # -----------------------------------------------------------------
    # (a) CONSTRAINED industry solve, PRODUCTION-DEFAULT settings (v2-seed
    #     auto-on, v2-hybrid-polish default-on), 3x bit-identical.
    #     Converged regime: low patience + big time so patience wins.
    # -----------------------------------------------------------------
    print("\n=== (a) constrained industry, DEFAULT driver (v2-seed ON) x3 [converged] ===")
    for key in ("IND2", "IND4", "IND10"):
        if key not in by:
            continue
        c = by[key]
        *_, has_cstr = precompute_constraint_arrays(c.boxes, c.pallet)

        def solve(case=c):
            return brkga_pack_v35(
                case.boxes, case.pallet, case.config,
                time_limit_s=600.0, max_pallets=10,
                population_size=80, n_populations=2, patience=10,
                # production defaults: do NOT force-disable v2 paths.
                seed=42, verbose=False, n_modes=6,
            )

        ok, sigstr, last = repeat(
            f"{key} N={len(c.boxes)} has_cstr={has_cstr} v2seed=auto",
            solve,
            extract=lambda r: f"pallets={len(r.pallets)} unp={len(r.unpacked)} "
                              f"valid_errs={len(validate(r, c.pallet, c.config))}",
        )
        if not has_cstr:
            print(f"        NOTE: {key} has_constraints=False (v2-seed would stay off)")
        overall_ok &= ok
        blocks.append(sigstr)

    # -----------------------------------------------------------------
    # (c) D2 WEIGHTLESS + support_ratio=0.8 + require_centroid_supported.
    #     New use_cstr_path/stability_active branch. has_constraints must be
    #     False (no weights/mlot/rfs) so v2-seed stays OFF, but the constraint
    #     decoder must run for stability. 3x bit-identical.
    # -----------------------------------------------------------------
    print("\n=== (c) D2 weightless + support_ratio=0.8 + centroid x3 [converged] ===")
    d2_boxes = []
    for i in range(60):
        L, W, H = (220, 180, 140) if i % 3 == 0 else \
                  (300, 200, 160) if i % 3 == 1 else (160, 160, 120)
        d2_boxes.append(Box(id=f"D2-{i:02d}", length=L, width=W, height=H,
                            weight=0.0, allowed_rotations=ALL_ROTATIONS))
    # GENUINELY weightless: max_weight=inf (the Pallet default). A finite
    # max_weight (even 0.0) is itself a weight constraint -> has_constraints
    # would correctly be True, which is NOT what "weightless" means here.
    d2_pallet = Pallet(length=1200, width=1000, height=1500, max_weight=float("inf"))
    d2_cfg = PackerConfig(support_ratio=0.8, require_centroid_supported=True)
    _, _, _, _, d2_has_cstr = precompute_constraint_arrays(d2_boxes, d2_pallet)

    # DIRECT invariant check: replicate driver.py's use_v2_seed resolution
    # (use_v2_seed is None -> bool(real has_constraints)). Weightless must
    # leave v2-seed OFF even though support_ratio=0.8 turns the cstr decoder on.
    d2_v2_seed_would_enable = bool(d2_has_cstr)
    print(f"        [invariant] weightless+support_ratio=0.8: has_constraints="
          f"{d2_has_cstr} -> use_v2_seed(auto)={d2_v2_seed_would_enable} "
          f"(expected False)")

    def solve_d2():
        return brkga_pack_v35(
            d2_boxes, d2_pallet, d2_cfg,
            time_limit_s=600.0, max_pallets=4,
            population_size=80, n_populations=2, patience=10,
            seed=123, verbose=False, n_modes=6,
        )

    ok, sigstr, last = repeat(
        f"D2 weightless N={len(d2_boxes)} has_cstr={d2_has_cstr} (must be False)",
        solve_d2,
        extract=lambda r: f"pallets={len(r.pallets)} unp={len(r.unpacked)} "
                          f"valid_errs={len(validate(r, d2_pallet, d2_cfg))}",
    )
    invariant_ok = (d2_has_cstr is False)
    if not invariant_ok:
        print("        ** REGRESSION: weightless workload reports has_constraints=True "
              "-> v2-seed would auto-enable, violating the documented invariant **")
    overall_ok &= ok and invariant_ok
    blocks.append(sigstr)

    # -----------------------------------------------------------------
    # (d) Box.group MULTI-pallet (IND6) -> _pack_with_groups dispatch.
    #     3x bit-identical + group co-location invariant + validator clean.
    # -----------------------------------------------------------------
    print("\n=== (d) Box.group multi-pallet (IND6) x3 [converged] + co-location ===")
    c6 = by.get("IND6")
    if c6 is None:
        print("  [FAIL] IND6 not found")
        overall_ok = False
    else:
        def solve_grp():
            return brkga_pack_v35(
                c6.boxes, c6.pallet, c6.config,
                time_limit_s=600.0, max_pallets=6,
                population_size=80, n_populations=2, patience=10,
                seed=99, verbose=False, n_modes=6,
            )

        def grp_extract(r):
            # group -> set of pallet indices
            gloc = {}
            for pi, st in enumerate(r.pallets):
                for p in st.placements:
                    gloc.setdefault(group_of(p.box.id), set()).add(pi)
            split = {g: sorted(pis) for g, pis in gloc.items() if len(pis) > 1}
            return (gloc, split, validate(r, c6.pallet, c6.config))

        ok, sigstr, last = repeat(
            f"IND6 N={len(c6.boxes)} groups",
            solve_grp,
            extract=lambda r: (lambda g, s, e:
                f"pallets={len(r.pallets)} unp={len(r.unpacked)} "
                f"groups_seen={sorted(g)} SPLIT_GROUPS={s} valid_errs={len(e)}"
            )(*grp_extract(r)),
        )
        # Hard-assert the co-location invariant on the last run.
        gloc, split, errs = grp_extract(last)
        coloc_ok = (len(split) == 0)
        valid_ok = (len(errs) == 0)
        # also: every PLACED box's group lands on exactly one pallet
        print(f"        co_location_ok={coloc_ok} validator_clean={valid_ok}")
        if not coloc_ok:
            print(f"        ** GROUP SPLIT ACROSS PALLETS: {split} **")
        overall_ok &= ok and coloc_ok and valid_ok
        blocks.append(sigstr)

    # -----------------------------------------------------------------
    digest = sha256_of(*blocks)
    print("\n" + "=" * 72)
    print(f"REVERIFY_SHA256={digest}")
    print(f"N_BLOCKS={len(blocks)}")
    print(f"=== RESULT: {'PASS' if overall_ok else 'FAIL'} ===")
    print("=" * 72)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
