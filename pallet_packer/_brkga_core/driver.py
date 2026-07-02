"""
pallet_packer._brkga_core.driver — brkga_pack_v35 main orchestrator.

Coordinates every phase of the v3.5 → v3.12 hybrid BRKGA:

  Phase 0   v2 PalletPacker warm-start (cheap, gives a strong baseline)
  Phase 1   smart-init populations (5 informed orderings × 6 modes)
  Phase 1b  optional SKU-aware mini-BRKGA (off by default)
  Phase 1c  optional adaptive mode probe (off by default)
  Phase 2   main BRKGA loop with migration + selection pressure
  Phase 3a  path relinking between top-2 elites
  Phase 3b  position-based local search
  Phase 3c  optional LNS polish
  Phase 4   v2 hybrid polish (return max of BRKGA vs v2 baseline)

The driver also handles n_restarts > 1 (re-enters itself K times with
different seeds, returns the best result).
"""
from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import List, Optional

import numpy as np

from ..models import Box, Pallet, PackerConfig
from ..packer import PackResult, PalletState
from ..brkga_v3_fast import _fitness_pallet1
from .precompute import (
    _NO_LIMIT,
    precompute_box_dims_and_sku,
    precompute_constraint_arrays,
    precompute_cog_envelope,
)
from .blocks import (
    enumerate_best_block_per_sku,
    enumerate_top_k_blocks_per_sku,
)
from .chromosome import (
    chromosome_from_order,
    make_informed_chromosomes,
    chromosome_from_v2_result,
)
from .dispatch import (
    warmup_jit,
    decode_chromosome,
    decode_auto_mode,
    decode_population_fitness,
)
from .polish import local_search_2opt, path_relinking, lns_polish
from .adaptive import (
    probe_decoder_modes,
    mode_weights_from_fitness,
    mode_weights_to_cdf,
)
from .sku_aware import brkga_sku_aware_search
from .realism import build_realism_context
from ..postprocess import apply_postprocess


def brkga_pack_v35(
    boxes: List[Box],
    pallet: Pallet,
    config: PackerConfig,
    **kwargs,
) -> PackResult:
    """Public v3.5 entry point: the hybrid BRKGA solve plus the optional
    realism post-passes (config.align_orientations / config.recenter_layout),
    applied exactly once on the finished result. The internal restart and
    group recursions call _brkga_pack_v35_impl directly, so a nested solve is
    never postprocessed twice. See _brkga_pack_v35_impl for the full
    parameter list (this wrapper forwards everything verbatim)."""
    result = _brkga_pack_v35_impl(boxes, pallet, config, **kwargs)
    if config.align_orientations or config.recenter_layout:
        apply_postprocess(result, pallet, config,
                          verbose=bool(kwargs.get("verbose", False)))
    return result


def _brkga_pack_v35_impl(
    boxes: List[Box],
    pallet: Pallet,
    config: PackerConfig,
    *,
    time_limit_s: float = 30.0,
    max_pallets: int = 1,
    seed: int = 42,
    # BRKGA params
    population_size: int = 600,
    generations: int = 1000,
    n_populations: int = 3,
    elite_fraction: float = 0.20,
    mutant_fraction: float = 0.15,
    p_elite_inherit: float = 0.70,
    migration_interval: int = 15,
    migrants_per_swap: int = 3,
    patience: int = 100,
    # v3.5 features
    use_multi_decoder: bool = True,
    n_modes: int = 6,  # 6 = DFTRC + wall + corner + layer + DFTRC+blocks + precomp-blocks
    # Default behavior is "auto" (None): v2 seed used iff the workload has
    # finite constraints. Reasoning: v2 seed gives an immediately feasible
    # baseline that helps the constraint path enormously (industry IND2:
    # 0 unp with v2 → 144 unp without), but on pure-geometric BR it traps
    # BRKGA in the v2 basin and forfeits better exploration (BR1#8: 96.4%
    # without v2 → 94.3% with v2). Explicit True/False overrides.
    use_v2_seed: Optional[bool] = None,
    use_smart_init: bool = True,
    use_local_search: bool = True,
    local_search_budget_s: float = 4.0,
    use_lns: bool = False,
    lns_budget_s: float = 3.0,
    # SKU-aware mini-BRKGA (v3.7, experimental): runs first with small budget,
    # uses 2S+1 chromosome (S=#SKUs). Designed to help homogeneous loads, but
    # empirical n=3 test showed neutral-to-negative impact (-0.10 to +0.70pp
    # depending on set, -0.59 on BR7). Default OFF; enable via flag for
    # experimentation. See docs/reports/17_v37_sku_aware.md.
    use_sku_aware: bool = False,
    sku_aware_budget_s: float = 5.0,
    use_v2_hybrid_polish: bool = True,
    # Adaptive mode selector (v3.9, experimental): probe each decoder
    # mode on a small seed set, then bias the selector keyspace toward
    # better modes via inverse-CDF lookup. Empirically NEUTRAL on BR
    # (n=5 mean Δ ≈ -0.08pp; W/T/L = 3/12/5 across BR1/3/5/7) — the
    # probe only sees performance on smart-init chromosomes and fails
    # to predict mode performance on random/crossover chromosomes that
    # dominate BRKGA's exploration. BRKGA's elite-survival mechanism
    # already biases toward good modes implicitly. Default OFF; kept
    # for future experimentation with multi-seed probing or mid-run
    # adaptation. See docs/reports/20_v39_adaptive_selector.md.
    use_adaptive_mode_selector: bool = False,
    adaptive_probe_seeds: int = 4,
    # Multi-restart: run K times with different seeds, take best.
    # Reduces variance at the cost of less compute per run.
    n_restarts: int = 1,
    # Opt-in boundary check. When True, raises PackingInputError on malformed
    # input (non-integer/negative dims, NaN weight cap, etc.) instead of
    # silently producing an invalid packing. Off by default to preserve the
    # behaviour of existing callers; the service layer passes True. The
    # box-count cap is a deployment concern, so it is NOT applied here
    # (max_boxes=None) — callers enforce it via check_packing_input directly.
    validate_input: bool = False,
    verbose: bool = False,
) -> PackResult:
    """v3.5: hybrid BRKGA with all quality improvements."""
    if validate_input:
        from ..input_validation import check_packing_input
        check_packing_input(boxes, pallet, max_boxes=None)
    # Box.group co-location: boxes sharing a non-None group must land on the
    # same pallet (e.g. LTL groupage — a customer's items stay together). The
    # decoders are group-unaware, so when more than one pallet is allowed we
    # assign whole groups to pallets up front and pack each pallet on its own,
    # where co-location is automatic. Single-pallet solves (max_pallets == 1)
    # are inherently group-safe and skip this. See _pack_with_groups.
    if max_pallets > 1 and any(getattr(b, "group", None) is not None
                               for b in boxes):
        return _pack_with_groups(
            boxes, pallet, config,
            time_limit_s=time_limit_s, max_pallets=max_pallets, seed=seed,
            population_size=population_size, n_populations=n_populations,
            patience=patience, n_modes=n_modes,
            use_multi_decoder=use_multi_decoder, use_v2_seed=use_v2_seed,
            use_smart_init=use_smart_init, use_local_search=use_local_search,
            local_search_budget_s=local_search_budget_s, use_lns=use_lns,
            lns_budget_s=lns_budget_s,
            use_v2_hybrid_polish=use_v2_hybrid_polish, verbose=verbose)
    # Resolve the use_v2_seed auto-default before any branch. v2 seed gives
    # a feasible-and-anchored baseline that's worth its slow runtime on
    # constrained workloads but actively traps BRKGA in a suboptimal basin
    # on pure-geometric ones. Peek at the constraints (cheap) to decide.
    if use_v2_seed is None:
        _, _, _, _, _has_cstr = precompute_constraint_arrays(boxes, pallet)
        use_v2_seed = bool(_has_cstr)
    # If multi-restart, recurse into single-run with split budget.
    if n_restarts > 1:
        time_per = time_limit_s / n_restarts
        best_result: Optional[PackResult] = None
        best_util = -1.0
        cap = pallet.length * pallet.width * pallet.height
        # Realism-aware restart selection (None keeps the historical raw-util
        # comparison verbatim — its first-wins tie semantics differ from a
        # fitness rewrite, so it must not change when the flag is off).
        restart_ctx = build_realism_context(boxes, pallet, config)
        best_restart_fit = float('inf')
        for r_idx in range(n_restarts):
            res = _brkga_pack_v35_impl(
                boxes, pallet, config,
                time_limit_s=time_per, max_pallets=max_pallets,
                seed=seed + r_idx * 17,
                population_size=population_size, generations=generations,
                n_populations=n_populations,
                elite_fraction=elite_fraction, mutant_fraction=mutant_fraction,
                p_elite_inherit=p_elite_inherit,
                migration_interval=migration_interval,
                migrants_per_swap=migrants_per_swap,
                patience=patience,
                use_multi_decoder=use_multi_decoder,
                n_modes=n_modes,
                use_v2_seed=use_v2_seed and r_idx == 0,  # only first run uses v2 (slow)
                use_smart_init=use_smart_init,
                use_local_search=use_local_search,
                local_search_budget_s=local_search_budget_s,
                use_lns=use_lns,
                lns_budget_s=lns_budget_s,
                use_sku_aware=use_sku_aware,
                sku_aware_budget_s=sku_aware_budget_s,
                use_v2_hybrid_polish=use_v2_hybrid_polish and r_idx == 0,
                use_adaptive_mode_selector=use_adaptive_mode_selector,
                adaptive_probe_seeds=adaptive_probe_seeds,
                n_restarts=1,
                verbose=verbose,
            )
            used = (sum(p.box.volume for p in res.pallets[0].placements)
                    if res.pallets else 0)
            util = used / cap if cap > 0 else 0.0
            if verbose:
                print(f"  [v3.5 multi-restart] run {r_idx+1}/{n_restarts}: util={util*100:.2f}%")
            if restart_ctx is not None:
                fit = _fitness_pallet1(res, pallet, realism=restart_ctx)
                if fit < best_restart_fit - 1e-9:
                    best_restart_fit = fit
                    best_util = util
                    best_result = res
            elif util > best_util:
                best_util = util
                best_result = res
        return best_result if best_result is not None else PackResult(
            pallets=[], unpacked=list(boxes))
    # ----------------------- single-run path -----------------------
    n = len(boxes)
    if n == 0:
        return PackResult(pallets=[], unpacked=[])

    warmup_jit()
    n_rots_arr, dims_all, sku_id_per_box = precompute_box_dims_and_sku(boxes)
    L = int(round(pallet.length))
    W = int(round(pallet.width))
    H = int(round(pallet.height))
    sku_best_block = enumerate_best_block_per_sku(
        boxes, sku_id_per_box, n_rots_arr, dims_all, L, W, H)
    sku_top_k_blocks = enumerate_top_k_blocks_per_sku(
        boxes, sku_id_per_box, n_rots_arr, dims_all, L, W, H, k_top=8)
    weights_arr, mlot_arr, rfs_arr, pallet_max_weight, has_constraints = \
        precompute_constraint_arrays(boxes, pallet)
    # Secondary realism fitness (None unless config.realism_weight > 0).
    # Threaded into EVERY fitness evaluation — batch, polish, sku-aware, and
    # the v2-hybrid comparison — so the whole search optimizes one scalar.
    realism_ctx = build_realism_context(
        boxes, pallet, config, dims_all=dims_all,
        sku_id_per_box=sku_id_per_box, weights_arr=weights_arr,
        n_rots_arr=n_rots_arr)
    # Physical stability (support_ratio / centroid), the CoG envelope, and
    # pallet overhang are requirements of the CONFIG + PALLET, not of the
    # cargo's weights — a weightless box can still float. Previously all of
    # them were gated by has_constraints, so a weightless workload silently
    # packed with NO support check (floating placements the validator rejects;
    # see docs/reports/30_verification.md, defect D2). Drive them from config
    # and route through the constraint-aware decoders whenever ANY physical
    # requirement is active — independent of whether the load carries weight.
    cog_x_min_value, cog_x_max_value, \
        cog_y_min_value, cog_y_max_value, \
        cog_min_load_frac_value, _cog_active_bool = \
        precompute_cog_envelope(pallet, config)
    stability_active = (float(config.support_ratio) > 0.0
                        or bool(config.require_centroid_supported))
    overhang_active = (bool(config.allow_pallet_overhang)
                       and float(pallet.max_overhang) > 0.0)
    # use_cstr_path drives the constraint-aware decoder path (support, centroid,
    # CoG, load, weight). Kept DISTINCT from has_constraints on purpose: the
    # v2-seed auto-default (above) must stay keyed on real weight/load limits,
    # because v2-seeding traps pure-geometric workloads in a worse basin
    # (Phase 7a). Stability-only workloads must NOT auto-enable v2-seed.
    use_cstr_path = bool(has_constraints or stability_active
                         or _cog_active_bool or overhang_active)
    if use_cstr_path:
        support_ratio_value = float(config.support_ratio)
        require_centroid_value = 1 if config.require_centroid_supported else 0
        cog_active_value = 1 if _cog_active_bool else 0
        max_overhang_value = (
            float(pallet.max_overhang) if config.allow_pallet_overhang else 0.0)
    else:
        support_ratio_value = 0.0
        require_centroid_value = 0
        cog_x_min_value, cog_x_max_value = -1e18, 1e18
        cog_y_min_value, cog_y_max_value = -1e18, 1e18
        cog_min_load_frac_value = 0.0
        cog_active_value = 0
        max_overhang_value = 0.0
    chrom_size = 2 * n + (1 if use_multi_decoder else 0)
    # mode_cdf is filled in by the adaptive-selector probe (Phase 1c, below).
    # While None, decode_auto_mode falls back to uniform mode allocation.
    adaptive_state = {"mode_cdf": None, "mode_weights": None}
    if use_multi_decoder:
        def decoder(c, b, p, cfg, na, da, max_pallets=1):
            return decode_auto_mode(
                c, b, p, cfg, na, da, max_pallets=max_pallets, n_modes=n_modes,
                sku_id_per_box=sku_id_per_box, sku_best_block=sku_best_block,
                sku_top_k_blocks=sku_top_k_blocks,
                mode_cdf=adaptive_state["mode_cdf"],
                weights=weights_arr, mlot=mlot_arr,
                pallet_max_weight=pallet_max_weight,
                has_constraints=use_cstr_path,
                support_ratio=support_ratio_value,
                rfs=rfs_arr,
                require_centroid=require_centroid_value,
                cog_x_min=cog_x_min_value, cog_x_max=cog_x_max_value,
                cog_y_min=cog_y_min_value, cog_y_max=cog_y_max_value,
                cog_min_load_frac=cog_min_load_frac_value,
                cog_active=cog_active_value,
                max_overhang=max_overhang_value)
    else:
        def decoder(c, b, p, cfg, na, da, max_pallets=1):
            return decode_chromosome(
                c, b, p, cfg, na, da, mode=0, max_pallets=max_pallets,
                weights=weights_arr, mlot=mlot_arr,
                pallet_max_weight=pallet_max_weight,
                has_constraints=use_cstr_path,
                support_ratio=support_ratio_value,
                rfs=rfs_arr,
                require_centroid=require_centroid_value,
                cog_x_min=cog_x_min_value, cog_x_max=cog_x_max_value,
                cog_y_min=cog_y_min_value, cog_y_max=cog_y_max_value,
                cog_min_load_frac=cog_min_load_frac_value,
                cog_active=cog_active_value,
                max_overhang=max_overhang_value)

    t_start = time.time()

    # ----------------------------------------------------------------------
    # Phase 0: v2 seed (cheap, gives us a strong baseline chromosome)
    # ----------------------------------------------------------------------
    v2_result: Optional[PackResult] = None
    v2_seed_chrom: Optional[np.ndarray] = None
    v2_seed_chroms_per_mode: List[np.ndarray] = []
    if use_v2_seed:
        try:
            from ..packer import PalletPacker
            # Honor the function-arg max_pallets in the v2 seed/hybrid packer.
            # PalletPacker reads only config.max_pallets, so without this the
            # v2 result — which can win the hybrid comparison (Phase 4) and be
            # returned directly, or be the fallback return — would breach the
            # cap (verification defect: IND2 with max_pallets=1 -> 4 pallets).
            # The whole BRKGA path already uses the function arg; align v2 too.
            v2_config = replace(config, max_pallets=max_pallets)
            v2_packer = PalletPacker(pallet, v2_config)
            v2_result = v2_packer.pack(boxes)
            if use_multi_decoder:
                for m in range(n_modes):
                    cs = chromosome_from_v2_result(v2_result, boxes, n_rots_arr,
                                                    decoder_mode=m,
                                                    n_modes=n_modes,
                                                    rng=np.random.default_rng(seed + 7 + m))
                    if cs is not None:
                        v2_seed_chroms_per_mode.append(cs)
            else:
                v2_seed_chrom = chromosome_from_v2_result(
                    v2_result, boxes, n_rots_arr,
                    rng=np.random.default_rng(seed + 7))
        except Exception as e:
            if verbose:
                print(f"  [v3.5] v2 seed failed: {e}")
    v2_time = time.time() - t_start
    if verbose:
        u = 0
        if v2_result and v2_result.pallets:
            cap = pallet.length * pallet.width * pallet.height
            u = sum(p.box.volume for p in v2_result.pallets[0].placements) / cap * 100
        print(f"  [v3.5] v2 seed: {v2_time:.2f}s, v2 util={u:.2f}%")

    # ----------------------------------------------------------------------
    # Phase 1: Smart init + random fill of populations
    # ----------------------------------------------------------------------
    K = max(1, n_populations)
    pop_size = max(20, population_size // K)
    n_elite = max(1, int(pop_size * elite_fraction))
    n_mutant = max(1, int(pop_size * mutant_fraction))
    n_cross = max(0, pop_size - n_elite - n_mutant)

    rngs = [np.random.default_rng(seed + i) for i in range(K)]
    pops = [rngs[i].random((pop_size, chrom_size)) for i in range(K)]

    # Heavy-first seed gate: only when the realism term is live (so the
    # search can PREFER heavy-low layouts), heavy_on_bottom is on, and the
    # load actually has weight variation. Gated so default-config callers
    # (BR benchmarks) keep an unchanged smart-list.
    include_weight_order = (bool(config.heavy_on_bottom)
                            and realism_ctx is not None
                            and float(weights_arr.max()) > float(weights_arr.min()))
    if use_smart_init or use_v2_seed:
        # Inject informed chromosomes into the front of pop 0
        smart = []
        if use_smart_init:
            if use_multi_decoder:
                for m in range(n_modes):
                    smart.extend(make_informed_chromosomes(
                        boxes, n_rots_arr, decoder_mode=m,
                        n_modes=n_modes, seed=seed + 100 + m,
                        weights=weights_arr,
                        include_weight_order=include_weight_order))
            else:
                smart.extend(make_informed_chromosomes(
                    boxes, n_rots_arr, decoder_mode=None, seed=seed + 100,
                    weights=weights_arr,
                    include_weight_order=include_weight_order))
        if v2_seed_chrom is not None:
            smart.insert(0, v2_seed_chrom)
        if v2_seed_chroms_per_mode:
            for c in v2_seed_chroms_per_mode:
                smart.insert(0, c)
        # Insert into front of pop 0 (and spread across pops if K > 1)
        for i, c in enumerate(smart[:pop_size]):
            pops[0][i] = c[:chrom_size]
        # Also seed pop 1 with one smart chromosome if K > 1
        if K > 1 and smart:
            for i, c in enumerate(smart[:pop_size // 2]):
                idx_in_pop1 = i % pop_size
                pops[1 % K][idx_in_pop1] = c[:chrom_size]

    # ----------------------------------------------------------------------
    # Phase 1b: SKU-aware mini-BRKGA (v3.7)
    # ----------------------------------------------------------------------
    # Runs early with small budget. Best result becomes a strong seed for
    # main BRKGA (Phase 2). Especially crucial when S << N (BR1, real-world
    # homogeneous stock).
    sku_aware_result: Optional[PackResult] = None
    sku_aware_chrom: Optional[np.ndarray] = None
    sku_aware_time = 0.0
    if use_sku_aware and sku_aware_budget_s > 0:
        t_sku = time.time()
        sku_aware_result, sku_aware_chrom, sku_aware_decodes = brkga_sku_aware_search(
            boxes, pallet, config, n_rots_arr, dims_all, sku_id_per_box,
            time_budget_s=sku_aware_budget_s,
            seed=seed + 11, max_pallets=max_pallets, verbose=verbose,
            realism=realism_ctx,
        )
        sku_aware_time = time.time() - t_sku
        if verbose and sku_aware_result is not None:
            sku_u = (1 - _fitness_pallet1(sku_aware_result, pallet)) * 100
            print(f"  [v3.5] SKU-aware: util={sku_u:.2f}% in {sku_aware_time:.2f}s "
                  f"({sku_aware_decodes} decodes)")

    # ----------------------------------------------------------------------
    # Phase 1c: Adaptive mode selector probe (v3.9)
    # ----------------------------------------------------------------------
    adaptive_time = 0.0
    if use_adaptive_mode_selector and use_multi_decoder and n_modes > 1:
        t_ad = time.time()
        # Pick up to adaptive_probe_seeds diverse chromosomes from pop[0]
        # (which already contains v2 seed + smart-init chromosomes).
        k_seeds = max(1, min(adaptive_probe_seeds, pops[0].shape[0]))
        probe_chroms = [pops[0][i].copy() for i in range(k_seeds)]
        mode_fits = probe_decoder_modes(
            probe_chroms, boxes, pallet, config,
            n_rots_arr, dims_all, n_modes,
            max_pallets=max_pallets,
            sku_id_per_box=sku_id_per_box,
            sku_best_block=sku_best_block,
            sku_top_k_blocks=sku_top_k_blocks,
        )
        # Only adapt if there is a meaningful spread; otherwise stay uniform.
        if mode_fits.max() - mode_fits.min() > 0.005:  # > 0.5 pp spread
            w = mode_weights_from_fitness(mode_fits)
            adaptive_state["mode_weights"] = w
            adaptive_state["mode_cdf"] = mode_weights_to_cdf(w)
        adaptive_time = time.time() - t_ad
        if verbose:
            mfp = [f"m{m}={mode_fits[m]*100:.2f}%" for m in range(n_modes)]
            wp = (None if adaptive_state["mode_weights"] is None
                  else [f"m{m}={adaptive_state['mode_weights'][m]:.2f}"
                        for m in range(n_modes)])
            print(f"  [v3.5 adaptive] probe: {' '.join(mfp)} "
                  f"in {adaptive_time:.2f}s")
            if wp is not None:
                print(f"  [v3.5 adaptive] weights: {' '.join(wp)}")

    # ----------------------------------------------------------------------
    # Phase 2: BRKGA
    # ----------------------------------------------------------------------
    # Reserve budget for LS + PR (if enabled)
    polish_budget = ((local_search_budget_s if use_local_search else 0)
                     + (lns_budget_s if use_lns else 0))
    brkga_budget = (time_limit_s - v2_time - sku_aware_time
                    - adaptive_time - polish_budget)
    brkga_budget = max(1.0, brkga_budget)

    best_fitness = float('inf')
    best_result: Optional[PackResult] = None
    best_chrom: Optional[np.ndarray] = None
    # If SKU-aware found a result, seed best with it
    if sku_aware_result is not None and sku_aware_chrom is not None:
        sku_fit = _fitness_pallet1(sku_aware_result, pallet,
                                   realism=realism_ctx)
        best_fitness = float(sku_fit)
        best_result = sku_aware_result
        best_chrom = sku_aware_chrom
        # Also inject SKU-aware chromosome into pop[0] as seed
        if (sku_aware_chrom is not None and chrom_size == len(sku_aware_chrom)
                and pops[0].shape[0] > 0):
            pops[0][0] = sku_aware_chrom.copy()
    # Track top-2 distinct elites for path relinking
    second_best_fitness = float('inf')
    second_best_chrom: Optional[np.ndarray] = None
    gens_no_improve = 0
    t0 = time.time()
    total_decodes = 0
    pop_fits = [np.zeros(pop_size) for _ in range(K)]

    for gen in range(generations):
        if time.time() - t0 > brkga_budget:
            if verbose:
                print(f"  [v3.5 BRKGA] gen {gen}: time hit, decodes={total_decodes}")
            break

        for k in range(K):
            fits = pop_fits[k]
            # Phase 5c: batch-evaluate the entire sub-population in parallel.
            # Returns fitness for every chromosome via vectorised math —
            # no PackResult is constructed unless an individual improves on
            # current best (rare after the first few generations). For BR
            # workloads (mostly mode 4/5 hot path), this cuts per-generation
            # time substantially even before the prange multiplier kicks in,
            # because the wasted PackResult construction is eliminated.
            batch_fits = decode_population_fitness(
                pops[k], boxes, pallet, config,
                n_rots_arr, dims_all, max_pallets=max_pallets,
                use_multi_decoder=use_multi_decoder, n_modes=n_modes,
                sku_id_per_box=sku_id_per_box,
                sku_best_block=sku_best_block,
                sku_top_k_blocks=sku_top_k_blocks,
                mode_cdf=adaptive_state["mode_cdf"],
                weights=weights_arr, mlot=mlot_arr,
                pallet_max_weight=pallet_max_weight,
                has_constraints=use_cstr_path,
                support_ratio=support_ratio_value,
                rfs=rfs_arr,
                require_centroid=require_centroid_value,
                cog_x_min=cog_x_min_value, cog_x_max=cog_x_max_value,
                cog_y_min=cog_y_min_value, cog_y_max=cog_y_max_value,
                cog_min_load_frac=cog_min_load_frac_value,
                cog_active=cog_active_value,
                max_overhang=max_overhang_value,
                realism=realism_ctx,
            )
            fits[:] = batch_fits
            total_decodes += pop_size
            # Per-individual best-update path. Order-preserving: same trail
            # of best/second-best updates as the per-chromosome decode loop.
            for i in range(pop_size):
                if fits[i] < best_fitness - 1e-9:
                    # New best — pay for PackResult construction now.
                    res = decoder(pops[k][i], boxes, pallet, config,
                                  n_rots_arr, dims_all, max_pallets=max_pallets)
                    second_best_fitness = best_fitness
                    second_best_chrom = best_chrom
                    best_fitness = float(fits[i])
                    best_result = res
                    best_chrom = pops[k][i].copy()
                    gens_no_improve = 0
                    if verbose:
                        print(f"  [v3.5 BRKGA] gen {gen} pop {k}: util={(1-best_fitness)*100:.2f}%")
                elif (fits[i] < second_best_fitness - 1e-9 and
                      fits[i] > best_fitness + 1e-9):
                    # Update second-best (distinct from best) — no PackResult
                    # needed; path-relinking re-decodes via the closure.
                    second_best_fitness = float(fits[i])
                    second_best_chrom = pops[k][i].copy()

        gens_no_improve += 1
        if gens_no_improve >= patience:
            if verbose:
                print(f"  [v3.5 BRKGA] gen {gen}: patience, decodes={total_decodes}")
            break

        # Migration
        if K > 1 and gen > 0 and gen % migration_interval == 0:
            for k in range(K):
                src = k
                dst = (k + 1) % K
                src_sorted = np.argsort(pop_fits[src])[:migrants_per_swap]
                dst_sorted = np.argsort(pop_fits[dst])[-migrants_per_swap:]
                for s, d in zip(src_sorted, dst_sorted):
                    pops[dst][d] = pops[src][s].copy()

        # Evolve
        new_pops = []
        for k in range(K):
            sorted_idx = np.argsort(pop_fits[k])
            new_pop = np.zeros_like(pops[k])
            new_pop[:n_elite] = pops[k][sorted_idx[:n_elite]]
            new_pop[n_elite:n_elite + n_mutant] = rngs[k].random((n_mutant, chrom_size))
            elite_pool = pops[k][sorted_idx[:n_elite]]
            non_elite_pool = pops[k][sorted_idx[n_elite:]]
            if n_cross > 0 and len(non_elite_pool) > 0:
                pa_idx = rngs[k].integers(0, n_elite, size=n_cross)
                pb_idx = rngs[k].integers(0, len(non_elite_pool), size=n_cross)
                pa_parents = elite_pool[pa_idx]
                pb_parents = non_elite_pool[pb_idx]
                mask = rngs[k].random((n_cross, chrom_size)) < p_elite_inherit
                new_pop[n_elite + n_mutant:] = np.where(mask, pa_parents, pb_parents)
            new_pops.append(new_pop)
        pops = new_pops

    # ----------------------------------------------------------------------
    # Phase 3a: Path relinking between top-2 elites (cheap)
    # ----------------------------------------------------------------------
    if (best_chrom is not None and second_best_chrom is not None
            and use_local_search):
        pr_res, pr_chrom, pr_fit = path_relinking(
            best_chrom, second_best_chrom,
            boxes, pallet, config, n_rots_arr, dims_all,
            multi_decoder=use_multi_decoder, n_modes=n_modes,
            sku_id_per_box=sku_id_per_box,
            sku_best_block=sku_best_block,
            sku_top_k_blocks=sku_top_k_blocks,
            mode_cdf=adaptive_state["mode_cdf"],
            weights=weights_arr, mlot=mlot_arr,
            pallet_max_weight=pallet_max_weight,
            has_constraints=use_cstr_path,
            support_ratio=support_ratio_value,
            rfs=rfs_arr,
            require_centroid=require_centroid_value,
            cog_x_min=cog_x_min_value, cog_x_max=cog_x_max_value,
            cog_y_min=cog_y_min_value, cog_y_max=cog_y_max_value,
            cog_min_load_frac=cog_min_load_frac_value,
            cog_active=cog_active_value,
            max_overhang=max_overhang_value,
            max_pallets=max_pallets, max_evals=50, verbose=verbose,
            realism=realism_ctx,
        )
        if pr_fit < best_fitness - 1e-9:
            best_fitness = pr_fit
            best_result = pr_res
            best_chrom = pr_chrom
            if verbose:
                print(f"  [v3.5] path relinking improved: util={(1-best_fitness)*100:.2f}%")

    # ----------------------------------------------------------------------
    # Phase 3b: Local search polish
    # ----------------------------------------------------------------------
    if use_local_search and best_chrom is not None:
        ls_res, ls_chrom, accepts = local_search_2opt(
            best_chrom, boxes, pallet, config, n_rots_arr, dims_all,
            time_budget_s=local_search_budget_s,
            multi_decoder=use_multi_decoder, n_modes=n_modes,
            sku_id_per_box=sku_id_per_box,
            sku_best_block=sku_best_block,
            sku_top_k_blocks=sku_top_k_blocks,
            mode_cdf=adaptive_state["mode_cdf"],
            weights=weights_arr, mlot=mlot_arr,
            pallet_max_weight=pallet_max_weight,
            has_constraints=use_cstr_path,
            support_ratio=support_ratio_value,
            rfs=rfs_arr,
            require_centroid=require_centroid_value,
            cog_x_min=cog_x_min_value, cog_x_max=cog_x_max_value,
            cog_y_min=cog_y_min_value, cog_y_max=cog_y_max_value,
            cog_min_load_frac=cog_min_load_frac_value,
            cog_active=cog_active_value,
            max_overhang=max_overhang_value,
            max_pallets=max_pallets,
            seed=seed + 999, verbose=verbose,
            realism=realism_ctx,
        )
        ls_fit = _fitness_pallet1(ls_res, pallet, realism=realism_ctx)
        if ls_fit < best_fitness - 1e-9:
            best_fitness = ls_fit
            best_result = ls_res
            best_chrom = ls_chrom
            if verbose:
                print(f"  [v3.5] local search improved: util={(1-best_fitness)*100:.2f}% ({accepts} accepts)")

    # ----------------------------------------------------------------------
    # Phase 3c: Large Neighborhood Search (LNS) polish
    # ----------------------------------------------------------------------
    if use_lns and best_chrom is not None:
        lns_res, lns_chrom, lns_accepts = lns_polish(
            best_chrom, boxes, pallet, config, n_rots_arr, dims_all,
            multi_decoder=use_multi_decoder, n_modes=n_modes,
            sku_id_per_box=sku_id_per_box,
            sku_best_block=sku_best_block,
            sku_top_k_blocks=sku_top_k_blocks,
            mode_cdf=adaptive_state["mode_cdf"],
            weights=weights_arr, mlot=mlot_arr,
            pallet_max_weight=pallet_max_weight,
            has_constraints=use_cstr_path,
            support_ratio=support_ratio_value,
            rfs=rfs_arr,
            require_centroid=require_centroid_value,
            cog_x_min=cog_x_min_value, cog_x_max=cog_x_max_value,
            cog_y_min=cog_y_min_value, cog_y_max=cog_y_max_value,
            cog_min_load_frac=cog_min_load_frac_value,
            cog_active=cog_active_value,
            max_overhang=max_overhang_value,
            max_pallets=max_pallets,
            time_budget_s=lns_budget_s,
            seed=seed + 1234, verbose=verbose,
            realism=realism_ctx,
        )
        lns_fit = _fitness_pallet1(lns_res, pallet, realism=realism_ctx)
        if lns_fit < best_fitness - 1e-9:
            best_fitness = lns_fit
            best_result = lns_res
            best_chrom = lns_chrom
            if verbose:
                print(f"  [v3.5] LNS improved: util={(1-best_fitness)*100:.2f}% ({lns_accepts} accepts)")

    # ----------------------------------------------------------------------
    # Phase 4: Compare with v2 hybrid (if enabled)
    # ----------------------------------------------------------------------
    if use_v2_hybrid_polish and v2_result is not None and v2_result.pallets:
        v2_fit = _fitness_pallet1(v2_result, pallet, realism=realism_ctx)
        if v2_fit < best_fitness - 1e-9:
            if verbose:
                print(f"  [v3.5] v2 hybrid wins: util={(1-v2_fit)*100:.2f}%")
            return v2_result

    if best_result is None:
        if v2_result is not None:
            return v2_result
        return PackResult(pallets=[], unpacked=list(boxes))
    return best_result


def _pack_with_groups(
    boxes: List[Box],
    pallet: Pallet,
    config: PackerConfig,
    *,
    time_limit_s: float,
    max_pallets: int,
    seed: int,
    population_size: int,
    n_populations: int,
    patience: int,
    n_modes: int,
    use_multi_decoder: bool,
    use_v2_seed: Optional[bool],
    use_smart_init: bool,
    use_local_search: bool,
    local_search_budget_s: float,
    use_lns: bool,
    lns_budget_s: float,
    use_v2_hybrid_polish: bool,
    verbose: bool,
) -> PackResult:
    """Pack with the Box.group co-location constraint guaranteed.

    Boxes sharing a non-None ``group`` must all end up on the same pallet. The
    geometric / constraint decoders assign boxes to bins purely by fit and are
    group-unaware, so we enforce co-location structurally:

      1. Bundle the cargo — each group is one indivisible unit; ungrouped boxes
         are free singletons.
      2. Assign whole bundles to pallets first-fit-decreasing by volume (a
         group never spans two pallets by construction; a bundle bigger than a
         pallet just lands alone and its overflow becomes unpacked — still no
         split).
      3. Pack each pallet independently with a single-container solve, where
         co-location is automatic, and concatenate the results.

    The volume FILL factor leaves headroom for 3D packing inefficiency; any
    overflow on a pallet spills to ``unpacked`` (never to another pallet, so
    the co-location invariant holds even when capacity binds).
    """
    groups: "dict" = {}
    group_order: List = []
    singletons: List[Box] = []
    for b in boxes:
        g = getattr(b, "group", None)
        if g is None:
            singletons.append(b)
        else:
            if g not in groups:
                groups[g] = []
                group_order.append(g)
            groups[g].append(b)

    bundles: List[List[Box]] = [groups[g] for g in group_order]
    bundles.extend([bx] for bx in singletons)

    # First-fit-decreasing assignment of whole bundles to pallets, constrained
    # on THREE budgets — not volume alone. A volume-only proxy badly
    # over-estimates capacity and dumps groups onto too few pallets (spilling
    # boxes to unpacked while allowed pallets sit empty) whenever:
    #   - weight binds (heavy low-volume groups), or
    #   - boxes are non-stackable (fragile, max_load_on_top=0 -> single layer,
    #     so they consume FLOOR area, not full container volume).
    # Tracking volume + weight + floor-area per pallet fixes both. FILL leaves
    # headroom for 3D packing inefficiency; any residual overflow spills to
    # unpacked (never to another pallet -> co-location invariant preserved).
    FILL = 0.85
    cap_vol = float(pallet.length * pallet.width * pallet.height) * FILL
    cap_floor = float(pallet.length * pallet.width) * FILL
    _pmw = getattr(pallet, "max_weight", math.inf)
    cap_wt = float(_pmw) if (_pmw is not None and math.isfinite(_pmw)) else math.inf

    def _metrics(bundle: List[Box]):
        vol = wt = floor = 0.0
        for bx in bundle:
            vol += float(bx.volume)
            w = float(getattr(bx, "weight", 0.0) or 0.0)
            wt += w
            m = getattr(bx, "max_load_on_top", math.inf)
            mf = float(m) if m is not None else math.inf
            # Unstackable (nothing may rest on top, or it can't bear even one
            # peer) -> it needs its own floor footprint (smallest face).
            if mf <= 0.0 or (math.isfinite(mf) and mf < w):
                l, ww, h = float(bx.length), float(bx.width), float(bx.height)
                floor += min(l * ww, l * h, ww * h)
        return vol, wt, floor

    scored = [( _metrics(b), b) for b in bundles]
    scored.sort(key=lambda t: t[0][0], reverse=True)   # by volume desc

    assigned: List[dict] = []
    for (v, wt, fl), bundle in scored:
        target = None
        for a in assigned:
            if (a["vol"] + v <= cap_vol and a["wt"] + wt <= cap_wt
                    and a["floor"] + fl <= cap_floor):
                target = a
                break
        if target is None:
            if len(assigned) < max_pallets:
                assigned.append({"boxes": [], "vol": 0.0, "wt": 0.0, "floor": 0.0})
                target = assigned[-1]
            elif assigned:
                # At the pallet cap: drop onto the emptiest pallet (best effort).
                # Whatever doesn't fit becomes unpacked — never a group split.
                target = min(assigned, key=lambda a: a["vol"])
        if target is None:
            continue
        target["boxes"].extend(bundle)
        target["vol"] += v
        target["wt"] += wt
        target["floor"] += fl

    non_empty = [a for a in assigned if a["boxes"]]
    n_used = max(1, len(non_empty))
    per_time = max(1.0, float(time_limit_s) / n_used)

    out_pallets: List[PalletState] = []
    out_unpacked: List[Box] = []
    for idx, a in enumerate(non_empty):
        res = _brkga_pack_v35_impl(
            a["boxes"], pallet, config,
            time_limit_s=per_time, max_pallets=1, seed=seed + idx * 31,
            population_size=population_size, n_populations=n_populations,
            patience=patience, n_modes=n_modes,
            use_multi_decoder=use_multi_decoder, use_v2_seed=use_v2_seed,
            use_smart_init=use_smart_init, use_local_search=use_local_search,
            local_search_budget_s=local_search_budget_s, use_lns=use_lns,
            lns_budget_s=lns_budget_s,
            use_v2_hybrid_polish=use_v2_hybrid_polish, verbose=verbose,
        )
        out_pallets.extend(res.pallets)
        out_unpacked.extend(res.unpacked)

    # Re-id pallets sequentially so the concatenated result has unique ids.
    for i, st in enumerate(out_pallets):
        st.pallet_id = f"P{i + 1:03d}"
    return PackResult(pallets=out_pallets, unpacked=out_unpacked)
