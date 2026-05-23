"""
benchmarks.industry — realistic-shipment scenarios for the pallet packer.

Designed to exercise the algorithm against the kinds of cargo profiles
a real tool would actually see. Each scenario is grounded in observable
business constraints: SKU heterogeneity, rotation restrictions, fragility,
weight distribution, multi-pallet orders.

Profiles (10 cases):

  IND1  E-commerce fulfillment (mixed box sizes, ~60 boxes)
  IND2  Pharma distribution (many fragile small items, weight-constrained)
  IND3  Furniture/appliances (few large boxes, rotation-locked)
  IND4  Beverage cases (uniform stacking, heavy, weight-binding)
  IND5  Mixed retail order (3-5 SKUs at moderate scale)
  IND6  LTL groupage (heterogeneous customer pallets combined)
  IND7  Electronics warehouse (this-side-up, light, low-fragility)
  IND8  Automotive parts (heavy, irregular sizing)
  IND9  Document/files shipping (uniform small boxes, dense pack)
  IND10 Cold-chain reefer (insulated boxes, rotation-locked, stacking limits)

Each scenario returns a Case (matching benchmarks/internal.py's dataclass)
so it plugs into the existing harness.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from typing import List

from pallet_packer import (
    Box, Pallet, PackerConfig,
    ALL_ROTATIONS, THIS_SIDE_UP, NO_ROTATION,
)
from benchmarks.internal import Case

# Standard Euro pallet, 1200 x 1000 x 1500 mm interior stack height
# (the 0.5m above the deck is typical for stacking).
EURO_PALLET = Pallet(length=1200, width=1000, height=1500, max_weight=1000)


def case_ecommerce_fulfillment() -> Case:
    """IND1 — D2C fulfillment center, ~120 boxes (forces multi-pallet).

    Mix of small/medium boxes from different SKUs going to mixed
    destinations. Most are this-side-up due to product orientation.
    """
    boxes: List[Box] = []
    # 30 small (200x150x100)
    for i in range(30):
        boxes.append(Box(
            id=f"S-{i:02d}", length=200, width=150, height=100, weight=0.6,
            allowed_rotations=THIS_SIDE_UP, max_load_on_top=15.0,
        ))
    # 40 medium (300x250x200)
    for i in range(40):
        boxes.append(Box(
            id=f"M-{i:02d}", length=300, width=250, height=200, weight=1.5,
            allowed_rotations=THIS_SIDE_UP, max_load_on_top=20.0,
        ))
    # 30 larger (400x300x300)
    for i in range(30):
        boxes.append(Box(
            id=f"L-{i:02d}", length=400, width=300, height=300, weight=3.0,
            allowed_rotations=THIS_SIDE_UP, max_load_on_top=10.0,
        ))
    # 20 small fragile (150x150x100) — electronics
    for i in range(20):
        boxes.append(Box(
            id=f"F-{i:02d}", length=150, width=150, height=100, weight=0.8,
            allowed_rotations=THIS_SIDE_UP, max_load_on_top=0.0,
        ))
    return Case(
        name="IND1 E-commerce fulfillment (120 mixed)",
        boxes=boxes, pallet=EURO_PALLET,
        config=PackerConfig(),
        notes="120 boxes across 4 SKUs; mix of fragile/normal; "
              "this-side-up. Typical D2C fulfillment daily run; "
              "forces 2-3 pallets.",
    )


def case_pharma_distribution() -> Case:
    """IND2 — pharma distribution, all fragile, forces flat layout."""
    boxes: List[Box] = []
    # 200 small fragile boxes (180x120x100), light but max_load_on_top=0
    # All fragile → can't stack, so all must fit on a floor layout.
    # Each box footprint = 0.0216 m², pallet floor = 1.2 m² = 55 boxes max.
    # With overhang allowed → 56-60. So ~3-4 pallets needed.
    for i in range(200):
        boxes.append(Box(
            id=f"P-{i:03d}", length=180, width=120, height=100, weight=2.5,
            allowed_rotations=THIS_SIDE_UP, max_load_on_top=0.0,
        ))
    pallet = Pallet(length=1200, width=1000, height=1500, max_weight=800)
    return Case(
        name="IND2 Pharma distribution (200 all-fragile)",
        boxes=boxes, pallet=pallet,
        config=PackerConfig(),
        notes="200 fragile boxes (no stacking). Forces all items onto "
              "the floor, requiring multiple pallets. Tests fragility "
              "constraint at scale and how well floor area is utilized.",
    )


def case_furniture_appliances() -> Case:
    """IND3 — furniture/large appliances, 15 boxes, tall pallet (2m)."""
    boxes = []
    # 4 LCD TV boxes (laid flat, fragile)
    for i in range(4):
        boxes.append(Box(
            id=f"LCD-{i:02d}", length=900, width=600, height=120, weight=25.0,
            allowed_rotations=THIS_SIDE_UP, max_load_on_top=0.0))
    # 3 fridges (tall, can't lie down, can't have anything on top)
    for i in range(3):
        boxes.append(Box(
            id=f"Fridge-{i:02d}", length=800, width=700, height=1700,
            weight=80.0, allowed_rotations=THIS_SIDE_UP, max_load_on_top=0.0))
    # 3 washer/dryers (tall, can support some weight)
    for i in range(3):
        boxes.append(Box(
            id=f"WD-{i:02d}", length=700, width=650, height=1000,
            weight=65.0, allowed_rotations=THIS_SIDE_UP, max_load_on_top=30.0))
    # 5 microwaves
    for i in range(5):
        boxes.append(Box(
            id=f"MW-{i:02d}", length=550, width=400, height=350, weight=15.0,
            allowed_rotations=THIS_SIDE_UP, max_load_on_top=20.0))
    # 2m-tall pallet to accommodate the 1.7m fridges (e.g., "high-cap" GMA).
    tall_pallet = Pallet(length=1200, width=1000, height=2000, max_weight=1000)
    return Case(
        name="IND3 Furniture/appliances (15 large, tall pallet)",
        boxes=boxes, pallet=tall_pallet,
        config=PackerConfig(),
        notes="Mix of tall (1.7m fridges) and short appliances on a tall "
              "pallet. Tests rotation-locked + fragility at large item sizes.",
    )


def case_beverage_cases() -> Case:
    """IND4 — beverage cases (uniform stacking, heavy, weight-binding)."""
    boxes = []
    # 24 cases of beer (400x300x250, 12kg each = 288kg total)
    # Pallet capped at 500kg total
    for i in range(24):
        boxes.append(Box(
            id=f"Beer-{i:02d}", length=400, width=300, height=250,
            weight=12.0,
            allowed_rotations=THIS_SIDE_UP,
            max_load_on_top=60.0,  # can stack 5 cases up
        ))
    # 20 cases of wine (350x250x300, 15kg each)
    for i in range(20):
        boxes.append(Box(
            id=f"Wine-{i:02d}", length=350, width=250, height=300,
            weight=15.0,
            allowed_rotations=THIS_SIDE_UP,
            max_load_on_top=45.0,  # can stack 3 cases up
        ))
    pallet = Pallet(length=1200, width=1000, height=1500, max_weight=600)
    return Case(
        name="IND4 Beverage cases (44, weight-binding)",
        boxes=boxes, pallet=pallet,
        config=PackerConfig(),
        notes="Beverage distribution. Pallet weight cap forces 2+ pallets "
              "even though volume would fit on 1. Tests weight-constraint "
              "binding.",
    )


def case_mixed_retail_order() -> Case:
    """IND5 — moderate-scale retail order, ~150 boxes, 4 SKUs."""
    boxes = []
    rotations = THIS_SIDE_UP
    # Bumped quantities to force ~2-3 pallets.
    skus = [
        ("A", 250, 200, 150, 1.5, 20, 40),
        ("B", 350, 280, 100, 2.0, 18, 35),
        ("C", 200, 200, 200, 1.8, 30, 50),
        ("D", 500, 350, 200, 4.0, 8, 25),
    ]
    for sku, l, w, h, wt, mlot, qty in skus:
        for i in range(qty):
            boxes.append(Box(
                id=f"{sku}-{i:02d}", length=l, width=w, height=h, weight=wt,
                allowed_rotations=rotations, max_load_on_top=mlot,
            ))
    return Case(
        name="IND5 Retail order 4-SKU (150 boxes)",
        boxes=boxes, pallet=EURO_PALLET,
        config=PackerConfig(),
        notes="Larger retail order, 150 boxes across 4 SKUs. Forces "
              "2-3 pallets; favors block-building for the high-qty SKUs.",
    )


def case_ltl_groupage() -> Case:
    """IND6 — LTL groupage: multiple customers' partial shipments mixed."""
    rng = random.Random(102)
    boxes = []
    # Customer 1: 20 boxes of one size
    for i in range(20):
        boxes.append(Box(
            id=f"C1-{i:02d}", length=300, width=250, height=200, weight=2.0,
            allowed_rotations=THIS_SIDE_UP, max_load_on_top=15.0,
            group="C1",
        ))
    # Customer 2: 15 boxes of another size
    for i in range(15):
        boxes.append(Box(
            id=f"C2-{i:02d}", length=400, width=300, height=180, weight=2.5,
            allowed_rotations=THIS_SIDE_UP, max_load_on_top=12.0,
            group="C2",
        ))
    # Customer 3: 8 mixed
    for i in range(8):
        l = rng.choice([200, 250, 300])
        w = rng.choice([150, 200, 250])
        h = rng.choice([100, 150, 200])
        boxes.append(Box(
            id=f"C3-{i:02d}", length=l, width=w, height=h,
            weight=rng.uniform(0.5, 3.0),
            allowed_rotations=THIS_SIDE_UP,
            max_load_on_top=rng.choice([0.0, 10.0, 20.0]),
            group="C3",
        ))
    return Case(
        name="IND6 LTL groupage (3 customers, 43 boxes)",
        boxes=boxes, pallet=EURO_PALLET,
        config=PackerConfig(),
        notes="LTL (less-than-truckload) groupage. Boxes from 3 customers "
              "with grouping constraint — each customer's items must "
              "stay together (e.g., for sequenced unloading).",
    )


def case_electronics_warehouse() -> Case:
    """IND7 — electronics, ~150 boxes, this-side-up, low fragility."""
    boxes = []
    # 80 medium boxes
    for i in range(80):
        boxes.append(Box(
            id=f"EL-{i:02d}", length=300, width=200, height=150, weight=2.0,
            allowed_rotations=THIS_SIDE_UP, max_load_on_top=20.0,
        ))
    # 50 small accessory boxes
    for i in range(50):
        boxes.append(Box(
            id=f"AC-{i:02d}", length=150, width=100, height=80, weight=0.5,
            allowed_rotations=THIS_SIDE_UP, max_load_on_top=10.0,
        ))
    # 20 large server boxes (heavy, fragile)
    for i in range(20):
        boxes.append(Box(
            id=f"SV-{i:02d}", length=450, width=350, height=200, weight=8.0,
            allowed_rotations=THIS_SIDE_UP, max_load_on_top=0.0,
        ))
    return Case(
        name="IND7 Electronics warehouse (150 mixed)",
        boxes=boxes, pallet=EURO_PALLET,
        config=PackerConfig(),
        notes="Three SKU sizes including fragile server boxes. Tests "
              "heterogeneous mixing at scale. Expect 1-2 pallets.",
    )


def case_automotive_parts() -> Case:
    """IND8 — automotive parts: ~60 heavy, irregular, full rotation OK."""
    rng = random.Random(103)
    boxes = []
    for i in range(60):
        l = rng.randint(200, 500)
        w = rng.randint(150, 400)
        h = rng.randint(100, 350)
        weight = rng.uniform(5.0, 25.0)
        boxes.append(Box(
            id=f"AUTO-{i:02d}", length=l, width=w, height=h, weight=weight,
            allowed_rotations=ALL_ROTATIONS,  # parts can be rotated
            max_load_on_top=weight * 3,  # roughly proportional
        ))
    pallet = Pallet(length=1200, width=1000, height=1500, max_weight=500)
    return Case(
        name="IND8 Automotive parts (60 heavy heterogeneous)",
        boxes=boxes, pallet=pallet,
        config=PackerConfig(),
        notes="Heavy industrial parts; full rotation allowed but weight "
              "will bind (total ~900kg vs 500kg pallet cap). Forces "
              "2-3 pallets. Tests heterogeneous + weight-binding.",
    )


def case_document_shipping() -> Case:
    """IND9 — document boxes, ~100 uniform, dense block-building."""
    boxes = []
    # 100 banker's boxes (380x300x250), 12kg each → 1200kg total
    for i in range(100):
        boxes.append(Box(
            id=f"DOC-{i:03d}", length=380, width=300, height=250,
            weight=12.0,
            allowed_rotations=THIS_SIDE_UP,
            max_load_on_top=50.0,  # solid boxes — can stack high
        ))
    pallet = Pallet(length=1200, width=1000, height=1500, max_weight=900)
    return Case(
        name="IND9 Document shipping (100 banker boxes)",
        boxes=boxes, pallet=pallet,
        config=PackerConfig(),
        notes="100 uniform boxes, 1200kg total → ~2 pallets by weight. "
              "Should pack densely. Tests homogeneous block-building.",
    )


def case_cold_chain_reefer() -> Case:
    """IND10 — insulated reefer boxes, ~80 with stacking limits."""
    boxes = []
    # 50 insulated cooler boxes (350x300x300) — 8kg each
    for i in range(50):
        boxes.append(Box(
            id=f"COOL-{i:02d}", length=350, width=300, height=300,
            weight=8.0,
            allowed_rotations=THIS_SIDE_UP,
            max_load_on_top=24.0,  # limited stacking to preserve insulation
        ))
    # 30 small specialty (200x200x250)
    for i in range(30):
        boxes.append(Box(
            id=f"SPEC-{i:02d}", length=200, width=200, height=250,
            weight=4.0,
            allowed_rotations=THIS_SIDE_UP,
            max_load_on_top=16.0,
        ))
    return Case(
        name="IND10 Cold-chain reefer (80 insulated)",
        boxes=boxes, pallet=EURO_PALLET,
        config=PackerConfig(),
        notes="80 insulated boxes for cold chain. Rotation-locked; "
              "stacking limited (load_on_top=24kg = 3 boxes max). Tests "
              "load-bearing limits + multi-pallet planning.",
    )


def cases() -> List[Case]:
    """Return all 10 industry scenarios."""
    return [
        case_ecommerce_fulfillment(),
        case_pharma_distribution(),
        case_furniture_appliances(),
        case_beverage_cases(),
        case_mixed_retail_order(),
        case_ltl_groupage(),
        case_electronics_warehouse(),
        case_automotive_parts(),
        case_document_shipping(),
        case_cold_chain_reefer(),
    ]


if __name__ == "__main__":
    for c in cases():
        n = len(c.boxes)
        total_vol = sum(b.length * b.width * b.height for b in c.boxes) / 1e9
        total_weight = sum(b.weight for b in c.boxes)
        pallet_vol = c.pallet.length * c.pallet.width * c.pallet.height / 1e9
        min_pallets_vol = max(1, int(total_vol / pallet_vol + 0.9999))
        min_pallets_wt = max(1, int(total_weight / c.pallet.max_weight + 0.9999))
        print(f"{c.name}")
        print(f"  N={n} boxes  total_vol={total_vol:.3f}m^3  total_weight={total_weight:.1f}kg")
        print(f"  Pallet: {c.pallet.length}x{c.pallet.width}x{c.pallet.height} "
              f"max_wt={c.pallet.max_weight}kg  pallet_vol={pallet_vol:.3f}m^3")
        print(f"  Volume LB: {min_pallets_vol}  Weight LB: {min_pallets_wt}  "
              f"max_lb: {max(min_pallets_vol, min_pallets_wt)}")
        print()
