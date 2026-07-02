"""
pallet_packer.input_validation — boundary gate for packing requests.

The packing engine works on an INTEGER coordinate grid and assumes well-formed
geometry. Malformed or non-integer input does NOT raise inside the engine — it
silently produces physically-impossible-but-validator-clean packings:

  - non-integer dimensions  -> boxes butt at the rounded pitch but overlap by up
    to ~0.5 units in real geometry (85.7% of fractional inputs fail the output
    validator; sub-0.5 dims collapse to zero-volume points);
  - NaN pallet.max_weight    -> math.isinf(NaN) is False, so the weight cap
    becomes NaN and every comparison is False -> cap silently disabled;
  - negative / zero dims      -> validator-clean but physically out-of-bounds.

(See docs/reports/30_verification.md, blockers 1/2 + defect D7.) This module is
the gate that must run BEFORE the solver. The contract:

  - All SPATIAL dimensions (box length/width/height, pallet length/width/height,
    pallet.max_overhang) must be POSITIVE INTEGERS. Units are caller-defined and
    must be consistent (mm, cm, inch — the engine does not care, but every value
    shares the same unit). Integer-valued floats (e.g. 100.0) are accepted.
  - WEIGHTS are physical and may be fractional: box.weight (>= 0, finite),
    box.max_load_on_top (>= 0 or +inf), pallet.max_weight (> 0 or +inf).
  - Each box needs at least one allowed rotation and (for result mapping) a
    unique id.
  - The number of boxes must be within [1, max_boxes].

Usage:
    problems = validate_packing_input(boxes, pallet)       # -> List[str]
    check_packing_input(boxes, pallet)                     # raises on problems
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

from .models import Box, Pallet

# Feasible-solve ceiling. Constrained workloads solve cleanly within a ~60s
# budget up to ~500 boxes; beyond that the v2-seed + per-box decoders go
# super-linear and the time budget stops being honored (see the scaling table
# in docs/reports/30_verification.md). Overridable per call / per deployment.
DEFAULT_MAX_BOXES = 500

# The BRKGA decoders encode "no cap" as the finite sentinel 1e18
# (_brkga_core.precompute._NO_LIMIT) so the JIT hot loops stay nan/inf-free.
# A box weighing >= 1e18 exceeds every cap INCLUDING the sentinel and becomes
# silently unpackable even on an unlimited pallet. Rejecting weights >= 1e15
# keeps even a full 500-box request (500 x 1e15 = 5e17) safely under the
# sentinel. (Hardening plan C4/F5.)
MAX_BOX_WEIGHT = 1e15


class PackingInputError(ValueError):
    """Raised by check_packing_input when the request is malformed.

    `.problems` holds the full list of human-readable issues.
    """

    def __init__(self, problems: List[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _is_positive_integer_dim(v) -> bool:
    """True iff v is finite, integer-valued, and strictly positive."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f > 0.0 and f == int(f)


def _is_nonneg_integer(v) -> bool:
    """True iff v is finite, integer-valued, and >= 0."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f >= 0.0 and f == int(f)


def _is_finite_nonneg(v) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f >= 0.0


def _is_cap(v) -> bool:
    """Valid weight/load cap: a positive finite number OR +inf. NaN rejected."""
    if v is None:
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    if math.isnan(f):
        return False
    if math.isinf(f):
        return f > 0          # +inf ok, -inf not
    return f > 0.0


def validate_packing_input(
    boxes: Sequence[Box],
    pallet: Pallet,
    *,
    max_boxes: Optional[int] = DEFAULT_MAX_BOXES,
) -> List[str]:
    """Return a list of problems with the request (empty list = valid).

    Pass max_boxes=None to disable the box-count cap (e.g. for offline research
    runs that knowingly exceed the service limit).
    """
    problems: List[str] = []

    # ---- pallet ----
    if pallet is None:
        return ["pallet is required"]
    for name in ("length", "width", "height"):
        v = getattr(pallet, name, None)
        if not _is_positive_integer_dim(v):
            problems.append(
                f"pallet.{name} must be a positive integer (got {v!r})")
    mw = getattr(pallet, "max_weight", math.inf)
    if not _is_cap(mw):
        problems.append(
            f"pallet.max_weight must be a positive number or infinity "
            f"(got {mw!r})")
    ov = getattr(pallet, "max_overhang", 0)
    if not _is_nonneg_integer(ov):
        problems.append(
            f"pallet.max_overhang must be a non-negative integer (got {ov!r})")

    # ---- boxes ----
    if boxes is None or len(boxes) == 0:
        problems.append("at least one box is required")
        return problems
    if max_boxes is not None and len(boxes) > max_boxes:
        problems.append(
            f"too many boxes: {len(boxes)} exceeds the limit of {max_boxes}")

    seen_ids: dict = {}
    for i, b in enumerate(boxes):
        bid = getattr(b, "id", None)
        label = f"box[{i}]" + (f" (id={bid!r})" if bid is not None else "")
        for name in ("length", "width", "height"):
            v = getattr(b, name, None)
            if not _is_positive_integer_dim(v):
                problems.append(
                    f"{label}.{name} must be a positive integer (got {v!r})")
        w = getattr(b, "weight", 0.0)
        if not _is_finite_nonneg(w):
            problems.append(
                f"{label}.weight must be a finite, non-negative number "
                f"(got {w!r})")
        elif float(w) >= MAX_BOX_WEIGHT:
            problems.append(
                f"{label}.weight is out of the supported range "
                f"(must be < {MAX_BOX_WEIGHT:.0e}, got {w!r})")
        m = getattr(b, "max_load_on_top", math.inf)
        # max_load_on_top: 0 (fragile) is valid; +inf (unlimited) is valid;
        # negative / NaN is not.
        m_ok = False
        try:
            mf = float(m)
            m_ok = (not math.isnan(mf)) and (math.isinf(mf) and mf > 0 or mf >= 0.0)
        except (TypeError, ValueError):
            m_ok = False
        if not m_ok:
            problems.append(
                f"{label}.max_load_on_top must be >= 0 or infinity (got {m!r})")
        rots = getattr(b, "allowed_rotations", None)
        if not rots:
            problems.append(f"{label}.allowed_rotations must be non-empty")
        if bid is None:
            problems.append(f"box[{i}].id is required (used to map results)")
        elif bid in seen_ids:
            problems.append(
                f"duplicate box id {bid!r} (box[{seen_ids[bid]}] and box[{i}]) "
                f"— ids must be unique")
        else:
            seen_ids[bid] = i

    return problems


def check_packing_input(
    boxes: Sequence[Box],
    pallet: Pallet,
    *,
    max_boxes: Optional[int] = DEFAULT_MAX_BOXES,
) -> None:
    """Raise PackingInputError if the request is malformed; else return None."""
    problems = validate_packing_input(boxes, pallet, max_boxes=max_boxes)
    if problems:
        raise PackingInputError(problems)
