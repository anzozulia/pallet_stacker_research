"""Test the input-validation gate. Run in Docker (pure Python, no rebuild)."""
from __future__ import annotations
import sys, math
sys.path.insert(0, '.')
from pallet_packer import (Box, Pallet, PackerConfig, ALL_ROTATIONS,
                           validate_packing_input, check_packing_input,
                           PackingInputError)
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from benchmarks.br import parse_thpack
from benchmarks.industry import cases as industry_cases


def ok_box(i=0):
    return Box(id=f"B{i}", length=100, width=100, height=100, weight=1.0,
               allowed_rotations=ALL_ROTATIONS)


def ok_pallet():
    return Pallet(length=1000, width=800, height=600, max_weight=500)


PASS = FAIL = 0


def expect(label, problems, should_have):
    global PASS, FAIL
    got = len(problems) > 0
    good = got == should_have
    print(f"  [{'PASS' if good else 'FAIL'}] {label}: "
          f"{'problems=' + str(problems[:1]) if got else 'clean'}")
    PASS += good
    FAIL += (not good)


def main():
    # valid baseline
    expect("valid integer instance", validate_packing_input([ok_box()], ok_pallet()), False)

    # --- malformed cases: each must be CAUGHT (should_have=True) ---
    p = ok_pallet()
    expect("non-integer box dim 100.49",
           validate_packing_input([Box(id="x", length=100.49, width=100, height=100,
                                       allowed_rotations=ALL_ROTATIONS)], p), True)
    expect("sub-0.5 dim 0.4 (would round to 0)",
           validate_packing_input([Box(id="x", length=0.4, width=0.4, height=0.4,
                                       allowed_rotations=ALL_ROTATIONS)], p), True)
    expect("negative dim",
           validate_packing_input([Box(id="x", length=-30, width=30, height=30,
                                       allowed_rotations=ALL_ROTATIONS)], p), True)
    expect("zero dim",
           validate_packing_input([Box(id="x", length=0, width=30, height=30,
                                       allowed_rotations=ALL_ROTATIONS)], p), True)
    expect("NaN box dim",
           validate_packing_input([Box(id="x", length=float('nan'), width=30, height=30,
                                       allowed_rotations=ALL_ROTATIONS)], p), True)
    expect("negative weight",
           validate_packing_input([Box(id="x", length=10, width=10, height=10, weight=-1,
                                       allowed_rotations=ALL_ROTATIONS)], p), True)
    expect("NaN max_load_on_top",
           validate_packing_input([Box(id="x", length=10, width=10, height=10,
                                       max_load_on_top=float('nan'),
                                       allowed_rotations=ALL_ROTATIONS)], p), True)
    expect("empty allowed_rotations",
           validate_packing_input([Box(id="x", length=10, width=10, height=10,
                                       allowed_rotations=[])], p), True)
    expect("duplicate ids",
           validate_packing_input([ok_box(0), Box(id="B0", length=10, width=10, height=10,
                                                  allowed_rotations=ALL_ROTATIONS)], p), True)
    # pallet
    expect("NaN pallet.max_weight",
           validate_packing_input([ok_box()],
                                  Pallet(length=1000, width=800, height=600,
                                         max_weight=float('nan'))), True)
    expect("zero pallet.max_weight",
           validate_packing_input([ok_box()],
                                  Pallet(length=1000, width=800, height=600, max_weight=0)), True)
    expect("non-integer pallet dim",
           validate_packing_input([ok_box()],
                                  Pallet(length=1000.5, width=800, height=600)), True)
    expect("empty box list",
           validate_packing_input([], ok_pallet()), True)
    expect("over the N cap (501 > 500)",
           validate_packing_input([ok_box(i) for i in range(501)], ok_pallet()), True)
    expect("N cap disabled (max_boxes=None)",
           validate_packing_input([ok_box(i) for i in range(501)], ok_pallet(), max_boxes=None), False)

    # valid floats allowed for weights (not dims)
    expect("fractional weight 2.5 is OK",
           validate_packing_input([Box(id="x", length=10, width=10, height=10, weight=2.5,
                                       allowed_rotations=ALL_ROTATIONS)], p), False)
    # integer-valued float dim 100.0 is OK
    expect("integer-valued float dim 100.0 is OK",
           validate_packing_input([Box(id="x", length=100.0, width=100.0, height=100.0,
                                       allowed_rotations=ALL_ROTATIONS)], p), False)

    # real datasets must all pass
    print("\n  Real datasets pass the gate:")
    bad = 0
    for inst in parse_thpack('benchmarks/data/thpack1.txt', set_id='BR1')[:5]:
        pr = validate_packing_input(inst.boxes, inst.pallet, max_boxes=None)
        if pr:
            bad += 1; print(f"    BR1#{inst.instance_id}: {pr[:1]}")
    for c in industry_cases():
        pr = validate_packing_input(c.boxes, c.pallet, max_boxes=None)
        if pr:
            bad += 1; print(f"    {c.name}: {pr[:1]}")
    expect("all BR(5)+industry(10) datasets clean", [str(bad)] if bad else [], False)

    # raising variant + brkga opt-in
    print("\n  Raising variant + brkga opt-in:")
    global PASS, FAIL
    try:
        check_packing_input([Box(id="x", length=100.49, width=100, height=100,
                                 allowed_rotations=ALL_ROTATIONS)], ok_pallet())
        print("  [FAIL] check_packing_input did not raise"); FAIL += 1
    except PackingInputError as e:
        print(f"  [PASS] check_packing_input raised: {len(e.problems)} problem(s)"); PASS += 1
    warmup_jit()
    try:
        brkga_pack_v35([Box(id="x", length=-5, width=5, height=5,
                            allowed_rotations=ALL_ROTATIONS)], ok_pallet(),
                       PackerConfig(), time_limit_s=2, validate_input=True)
        print("  [FAIL] brkga_pack_v35(validate_input=True) did not raise"); FAIL += 1
    except PackingInputError:
        print("  [PASS] brkga_pack_v35(validate_input=True) raised on bad input"); PASS += 1

    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    return 0 if FAIL == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
