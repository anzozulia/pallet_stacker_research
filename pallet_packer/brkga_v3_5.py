"""
pallet_packer.brkga_v3_5 — backward-compatibility shim.

The v3.5 → v3.12 hybrid BRKGA was refactored from a 4154-line monolith
into the `pallet_packer._brkga_core` package (12 modules). This shim
preserves the historical import path so existing scripts and tests
continue to work:

    from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit

New code should prefer:

    from pallet_packer._brkga_core import brkga_pack_v35, warmup_jit

See `pallet_packer/_brkga_core/__init__.py` for the module map and
`docs/reports/25_refactor.md` for the refactor rationale.
"""
from ._brkga_core import *  # noqa: F401,F403
from ._brkga_core import (  # noqa: F401  explicit re-export for IDEs
    brkga_pack_v35,
    warmup_jit,
    decode_chromosome,
    decode_auto_mode,
    _selector_to_mode,
    precompute_box_dims_and_sku,
    precompute_constraint_arrays,
    precompute_cog_envelope,
    chromosome_from_order,
    make_informed_chromosomes,
    chromosome_from_v2_result,
    enumerate_best_block_per_sku,
    enumerate_top_k_blocks_per_sku,
    resolve_sku_blocks_from_chrom,
    local_search_2opt,
    path_relinking,
    lns_polish,
    probe_decoder_modes,
    mode_weights_from_fitness,
    mode_weights_to_cdf,
    brkga_sku_aware_search,
    decode_njit_mode,
    decode_layer_njit,
    decode_blocks_njit_mode,
    decode_precomputed_blocks_njit_mode,
    find_best_block_at_pos_njit,
    decode_njit_mode_cstr,
    decode_blocks_njit_mode_cstr,
    decode_layer_njit_cstr,
    _check_load_on_top_njit,
    _check_cog_envelope_njit,
    _apply_load_contribution_njit,
    _apply_cog_contribution_njit,
    find_best_wall_njit,
    find_best_corner_njit,
    find_best_in_slab_njit,
)
