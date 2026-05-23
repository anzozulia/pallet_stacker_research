"""
example.py — Demonstration of the pallet_packer package on the user's scenario:
  - 20 boxes of 30 x 40 x 70
  - 15 boxes of 40 x 45 x 50
  - 10 boxes of 60 x 60 x 30
  - Pallet: 120 x 100 x 100
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pallet_packer import (
    Box, Pallet, PalletPacker, PackerConfig,
    ALL_ROTATIONS, THIS_SIDE_UP, NO_ROTATION,
    to_json, save_json, validate,
)


def build_boxes():
    boxes = []
    # 20 boxes of 30x40x70, weight 5kg, this-side-up only.
    for i in range(20):
        boxes.append(Box(
            id=f"A-{i+1:02d}",
            length=30, width=40, height=70,
            weight=5.0,
            allowed_rotations=THIS_SIDE_UP,
            max_load_on_top=50.0,  # can bear 50 kg on top
        ))
    # 15 boxes of 40x45x50, weight 8 kg, full rotation allowed.
    for i in range(15):
        boxes.append(Box(
            id=f"B-{i+1:02d}",
            length=40, width=45, height=50,
            weight=8.0,
            allowed_rotations=ALL_ROTATIONS,
            max_load_on_top=80.0,
        ))
    # 10 boxes of 60x60x30, weight 12 kg, sturdy base (good for bottom layer).
    for i in range(10):
        boxes.append(Box(
            id=f"C-{i+1:02d}",
            length=60, width=60, height=30,
            weight=12.0,
            allowed_rotations=THIS_SIDE_UP,
            max_load_on_top=150.0,
        ))
    return boxes


def main():
    pallet = Pallet(
        length=120, width=100, height=100,
        max_weight=500.0,        # 500 kg total per pallet
        max_overhang=0.0,        # no overhang of pallet edges
    )

    config = PackerConfig(
        support_ratio=0.8,             # 80% of base must be supported
        require_centroid_supported=True,
        heavy_on_bottom=True,
        cog_envelope_fraction=0.25,    # CoG within +/-25% of centre
        enforce_load_bearing=True,
        multi_start_trials=25,         # 25 randomized trials, pick best
        seed=42,
    )

    boxes = build_boxes()
    print(f"Input: {len(boxes)} boxes")
    print(f"  Total volume: {sum(b.volume for b in boxes):,.0f} mm³")
    print(f"  Total weight: {sum(b.weight for b in boxes):.1f} kg")
    print(f"  Pallet capacity: "
          f"{pallet.length * pallet.width * pallet.height:,.0f} mm³, "
          f"{pallet.max_weight:.1f} kg")
    print()

    packer = PalletPacker(pallet, config)
    result = packer.pack(boxes)

    print(f"Result:")
    print(f"  Pallets used: {result.num_pallets}")
    print(f"  Boxes packed: {sum(len(st.placements) for st in result.pallets)}")
    print(f"  Boxes unpacked: {len(result.unpacked)}")
    print(f"  Overall volume utilisation: {result.total_volume_utilisation:.1%}")
    print()
    for st in result.pallets:
        used = sum(p.box.volume for p in st.placements)
        cap = pallet.length * pallet.width * pallet.height
        print(f"  {st.pallet_id}: {len(st.placements):2d} boxes, "
              f"{st.total_weight:5.1f} kg, "
              f"{used / cap:5.1%} utilised")

    # Save JSON output for the visualiser.
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "examples")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "packing_result.json")
    save_json(result, pallet, out_path)
    print(f"\nWrote {out_path}")

    # Independent validation of all constraints.
    errors = validate(result, pallet, config)
    if errors:
        print(f"\n❌ Validation found {len(errors)} violation(s):")
        for e in errors[:10]:
            print(f"   - {e}")
    else:
        print("\n✅ Validation passed: all constraints satisfied.")


if __name__ == "__main__":
    main()
