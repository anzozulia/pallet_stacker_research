"""
pallet_packer — 3D bin-packing / pallet-loading with real-world physical
constraints (weight, fragility, support, CoG, rotation limits, overhang).

Public API
----------
Data layer (data classes, no algorithm dependencies):
    Box, Pallet, Placement, PackerConfig, Rotation
    ALL_ROTATIONS, THIS_SIDE_UP, NO_ROTATION
    EPS

Algorithm:
    PalletState       per-pallet engine
    PackResult        multi-pallet output
    PalletPacker      top-level orchestrator with .pack(boxes)

Validation:
    validate(result, pallet, config) -> List[str]

Serialization:
    to_json(result, pallet) -> dict
    save_json(result, pallet, path) -> None

Submodules (importable explicitly):
    pallet_packer.mip            CP-SAT exact polish (mip_polish())
    pallet_packer.lower_bounds   LB calculations (compute_lower_bounds())

References live in `packer.py` and `mip.py` module docstrings. See
the project HANDOFF.md for usage notes and the EVALUATION_REPORT.md for
where each feature does and doesn't help.
"""
from .models import (
    EPS,
    Rotation,
    ALL_ROTATIONS,
    THIS_SIDE_UP,
    NO_ROTATION,
    Box,
    Pallet,
    Placement,
    PackerConfig,
)
from .packer import PackResult, PalletState, PalletPacker
from .validate import validate
from .io import to_json, save_json

__all__ = [
    "EPS",
    "Rotation",
    "ALL_ROTATIONS",
    "THIS_SIDE_UP",
    "NO_ROTATION",
    "Box",
    "Pallet",
    "Placement",
    "PackerConfig",
    "PackResult",
    "PalletState",
    "PalletPacker",
    "validate",
    "to_json",
    "save_json",
]
