"""
pallet_packer._brkga_core — refactored implementation of the v3.5 → v3.12
hybrid BRKGA. Split from the monolithic 4154-line `brkga_v3_5.py` into
12 logically-grouped modules so each one becomes a natural compile unit
for the planned C/Cython port.

Module map
----------

  precompute.py         per-box / per-pallet array prep (SKU, weights, CoG)
  jit_primitives.py     geometric scoring helpers (find_best_wall, corner,
                        in_slab) used by every decoder
  jit_constraints.py    feasibility + bookkeeping helpers
                        (_check_load_on_top, _check_cog_envelope, _apply_*)
  jit_decoders_geom.py  JIT decoders 0/1/2/3/4/5 — pure geometric
  jit_decoders_cstr.py  JIT decoders 0/1/2/3/4 — constraint-aware variants
                        + block helpers (_commit_block_placements_njit etc.)
  dispatch.py           Python wrappers: warmup_jit, decode_chromosome,
                        decode_auto_mode, _selector_to_mode
  chromosome.py         chromosome construction (from_order, from_v2_result,
                        make_informed_chromosomes)
  blocks.py             block enumeration (enumerate_best, enumerate_top_k,
                        resolve_sku_blocks_from_chrom)
  polish.py             post-BRKGA polish (local_search_2opt,
                        path_relinking, lns_polish)
  adaptive.py           (experimental, OFF) adaptive mode selector
  sku_aware.py          (experimental, OFF) SKU-aware mini-BRKGA
  driver.py             brkga_pack_v35 — main orchestrator

Public API
----------

The two symbols any caller actually needs:

  brkga_pack_v35   main entry point
  warmup_jit       pre-compile the JIT decoders

The other re-exports are for legacy script compatibility.
"""
from .driver import brkga_pack_v35
from .dispatch import (
    warmup_jit,
    decode_chromosome,
    decode_auto_mode,
    decode_population_fitness,
    _selector_to_mode,
)
from .precompute import (
    precompute_box_dims_and_sku,
    precompute_constraint_arrays,
    precompute_cog_envelope,
)
from .chromosome import (
    chromosome_from_order,
    make_informed_chromosomes,
    chromosome_from_v2_result,
)
from .blocks import (
    enumerate_best_block_per_sku,
    enumerate_top_k_blocks_per_sku,
    resolve_sku_blocks_from_chrom,
)
from .polish import local_search_2opt, path_relinking, lns_polish
from .adaptive import (
    probe_decoder_modes,
    mode_weights_from_fitness,
    mode_weights_to_cdf,
)
from .sku_aware import (
    brkga_sku_aware_search,
    compute_sku_groups,
    sku_chrom_to_full_chrom,
)
from .jit_decoders_geom import (
    decode_njit_mode,
    decode_layer_njit,
    decode_blocks_njit_mode,
    decode_precomputed_blocks_njit_mode,
    find_best_block_at_pos_njit,
)
from .jit_decoders_cstr import (
    decode_njit_mode_cstr,
    decode_blocks_njit_mode_cstr,
    decode_layer_njit_cstr,
)
from .jit_constraints import (
    _check_load_on_top_njit,
    _check_cog_envelope_njit,
    _apply_load_contribution_njit,
    _apply_cog_contribution_njit,
)
from .jit_primitives import (
    find_best_wall_njit,
    find_best_corner_njit,
    find_best_in_slab_njit,
)

__all__ = [
    "brkga_pack_v35",
    "warmup_jit",
    "decode_chromosome",
    "decode_auto_mode",
    "precompute_box_dims_and_sku",
    "precompute_constraint_arrays",
    "precompute_cog_envelope",
    "chromosome_from_order",
    "make_informed_chromosomes",
    "chromosome_from_v2_result",
    "enumerate_best_block_per_sku",
    "enumerate_top_k_blocks_per_sku",
    "resolve_sku_blocks_from_chrom",
    "local_search_2opt",
    "path_relinking",
    "lns_polish",
    "probe_decoder_modes",
    "mode_weights_from_fitness",
    "mode_weights_to_cdf",
    "brkga_sku_aware_search",
]
