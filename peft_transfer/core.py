"""PEFT-robust spectral consensus attack (PRSCA).

The victim models are never used here.  Unknown fine tuning is represented by
an uncertainty set containing the source encoder and fixed, zero-mean low-rank
residual views with LoRA-like and Adapter-like placement.  The image update is
the coordinate-wise consensus of the gradients from those views.
"""
from __future__ import annotations

import contextlib
import math
import random
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass
class AttackConfig:
    epsilon: float = 10.0 / 255.0
    step_size: float = 2.0 / 255.0
    steps: int = 10
    momentum: float = 0.5
    ranks: Tuple[int, ...] = (4, 8)
    virtual_scale: float = 0.08
    selected_layers: int = 4
    output_weight: float = 1.0
    layer_weight: float = 0.5
    moment_weight: float = 0.15
    cosine_weight: float = 0.5
    ads_weight: float = 1.0
    source_weight: float = 2.0
    robust_weight: float = 0.35
    edge_gamma: float = 3.0
    consensus_floor: float = 0.25
    spectral_low: float = 0.04
    spectral_high: float = 0.42
    spectral_weight: float = 0.35
    ti_kernel: int = 5
    random_start: bool = False
    use_ads: bool = True
    use_multilevel: bool = False
    use_peft_consensus: bool = False
    use_spectral: bool = False
    use_edge: bool = False
    use_readout_coverage: bool = False
    use_logdet_coverage: bool = False
    use_amplitude_eot: bool = False
    use_contrast_eot: bool = False
    use_block_shuffle_eot: bool = False
    use_qk_bypass: bool = False
    use_patchout: bool = False
    use_cross_spectral_anchor: bool = False
    use_adapter_tail: bool = False
    use_frequency_rank_tail: bool = False
    use_skip_gradient: bool = False
    use_skip_ensemble: bool = False
    use_dual_path: bool = False
    use_spectral_uncertainty: bool = False
    use_phase_robust: bool = False
    use_feature_phase: bool = False
    use_grat: bool = False
    use_patch_loss: bool = False
    use_patch_boundary: bool = False
    use_input_phase: bool = False
    use_input_amplitude: bool = False
    use_polar_amplitude_gradient: bool = False
    polar_projection_coordinate: str = ""
    polar_projection_replace: bool = False
    use_ilpd: bool = False
    use_token_collapse: bool = False
    use_dc_shift: bool = False
    use_low_level_lift: bool = False
    use_pasd: bool = False
    use_atcs: bool = False
    use_token_gram: bool = False
    use_depth_layer: bool = False
    use_dual_preprocess: bool = False
    use_depth_polar: bool = False
    use_depth_boundary: bool = False
    use_depth_robust: bool = False
    use_phase_collision: bool = False
    use_graph_collapse: bool = False
    use_qk_topology: bool = False
    task_after_switch: bool = False
    use_quadrature_collision: bool = False
    use_translation_collision: bool = False
    use_input_translation_collision: bool = False
    use_spatial_principal: bool = False
    use_semantic_partition: bool = False
    use_cancellation_gap: bool = False
    use_final_ads_anchor: bool = False
    ads_all_views: bool = False
    asymmetric_step: bool = True
    initial_jitter: float = 0.0
    first_step_jitter: float = 0.0
    coverage_weight: float = 1.0
    coverage_temperature: float = 0.01
    logdet_coverage_weight: float = 0.0
    logdet_coverage_temperature: float = 0.01
    amplitude_eot_samples: int = 2
    amplitude_eot_scale: float = 0.2
    contrast_eot_scales: Tuple[float, ...] = (0.8, 1.2)
    block_shuffle_samples: int = 3
    block_shuffle_grid: int = 4
    block_shuffle_mode: str = "grid"
    qk_gradient_scale: float = 0.0
    patchout_keep_ratio: float = 0.66
    patchout_patch_size: int = 16
    anchor_floor: float = 0.5
    auxiliary_weight: float = 1.0
    adapter_rank: int = 192
    adapter_tail_weight: float = 1.0
    adapter_tail_layers: Tuple[int, ...] = ()
    adapter_tail_sites: Tuple[str, ...] = ("attn", "block")
    frequency_tail_weight: float = 1.0
    skip_gamma: float = 0.5
    skip_gammas: Tuple[float, ...] = (0.0, 0.75)
    skip_weights: Tuple[float, ...] = (0.25, 0.75)
    uncertainty_weight: float = 1.0
    uncertainty_floor: float = 0.1
    phase_softmin_beta: float = 8.0
    phase_reliability_floor: float = 0.05
    feature_phase_weight: float = 1.0
    grat_noise_sd: float = 0.5
    patch_weight: float = 1.0
    patch_boundary_weight: float = 1.0
    input_phase_weight: float = 0.0
    input_phase_low: float = 0.0
    input_phase_high: float = 1.0
    input_amplitude_weight: float = 0.0
    polar_amplitude_weight: float = 0.0
    polar_amplitude_low: float = 0.0
    polar_amplitude_high: float = 0.2
    polar_projection_low: float = 0.0
    polar_projection_high: float = 1.0
    polar_projection_weight: float = 1.0
    ilpd_layer: int = 5
    ilpd_gamma: float = 0.1
    collapse_weight: float = 1.0
    dc_shift_weight: float = 1.0
    lift_layers: int = 2
    lift_weight: float = 1.0
    pasd_mode: str = "shuffled_amplitude"
    pasd_samples: int = 2
    pasd_probe_scale: float = 2.0
    pasd_weight: float = 1.0
    grat_weight: float = 1.0
    pasd_reduce: str = "mean"
    pasd_softmin_beta: float = 8.0
    pasd_amplitude_grid: Tuple[float, ...] = ()
    pasd_depth_robust: bool = False
    pasd_depth_beta: float = 8.0
    pasd_region: str = "late"
    pasd_layer_count: int = 6
    pasd_early_weight: float = 0.0
    pasd_final_weight: float = 0.0
    atcs_weight: float = 1.0
    token_gram_weight: float = 0.25
    token_gram_pool: int = 16
    token_gram_region: str = "late"
    token_gram_layer_count: int = 6
    depth_layer_index: int = 0
    depth_layer_weight: float = 1.0
    depth_schedule: str = ""
    depth_phase_weight: float = 1.0
    depth_amplitude_weight: float = 1.0
    depth_boundary_weight: float = 1.0
    depth_robust_weight: float = 1.0
    depth_robust_count: int = 6
    depth_robust_beta: float = 8.0
    phase_collision_layer: int = 0
    phase_collision_weight: float = 1.0
    graph_collapse_weight: float = 1.0
    graph_collapse_count: int = 1
    qk_topology_layer: int = 0
    qk_topology_weight: float = 1.0
    quadrature_collision_weight: float = 1.0
    quadrature_schedule: str = ""
    translation_collision_shift: Tuple[int, int] = (4, 4)
    translation_collision_weight: float = 1.0
    input_translation_shift: Tuple[int, int] = (8, 8)
    input_translation_weight: float = 1.0
    spatial_principal_rank: int = 8
    spatial_principal_weight: float = 1.0
    semantic_partition_clusters: int = 8
    semantic_partition_temperature: float = 0.1
    semantic_partition_weight: float = 1.0
    cancellation_layer_count: int = 4
    cancellation_final_penalty: float = 0.25
    cancellation_weight: float = 1.0
    prior_guidance_weight: float = 0.0
    prior_guidance_schedule: str = "constant"
    prior_spectral_floor: float = -1.0
    prior_spectral_weight: float = 1.0
    post_ti_kernel: int = 1
    update_channel: int = -1
    pasd_schedule: str = ""
    pasd_switch_step: int = -1
    adapter_view_count: int = 0
    adapter_view_rank: int = 192
    adapter_view_scale: float = 0.25
    peft_reduce: str = "consensus"
    peft_softmin_beta: float = 50.0
    seed: int = 42


def ablation_config(name: str, **overrides) -> AttackConfig:
    """Return one member of the ordered A--F sanity chain."""
    name = name.upper()
    if name == "CTEOT":
        values = vars(ablation_config("B")).copy()
        values.update(use_contrast_eot=True, contrast_eot_scales=(0.8, 1.2))
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"CTEARLY", "CTEARLY010", "CTEARLY025", "CTEARLY050"}:
        depth_weight = {
            "CTEARLY": 1.0,
            "CTEARLY010": 0.10,
            "CTEARLY025": 0.25,
            "CTEARLY050": 0.50,
        }[name]
        values = vars(ablation_config("B")).copy()
        values.update(use_depth_layer=True, depth_layer_index=0,
                      depth_layer_weight=depth_weight, depth_schedule="joint")
        values.update(overrides)
        return AttackConfig(**values)
    if name == "CTEARLYSTAGE":
        values = vars(ablation_config("B")).copy()
        values.update(
            use_depth_layer=True,
            depth_layer_index=0,
            depth_layer_weight=1.0,
            depth_schedule="early_then_final",
        )
        values.update(overrides)
        return AttackConfig(**values)
    order = {key: index for index, key in enumerate("ABCDEF")}
    if name == "G":
        values = dict(
            use_ads=True, use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=False, use_readout_coverage=True,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=1e-4,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "H":
        values = dict(
            use_ads=True, use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=False, use_readout_coverage=False,
            use_cross_spectral_anchor=True, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=1e-5,
            first_step_jitter=1e-4, anchor_floor=0.5,
            auxiliary_weight=1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "I":
        values = dict(
            use_ads=True, use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=False, use_readout_coverage=False,
            use_cross_spectral_anchor=True, use_adapter_tail=True,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=1e-4,
            anchor_floor=0.5, auxiliary_weight=1.0,
            adapter_rank=192, adapter_tail_weight=1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "J":
        values = dict(
            use_ads=True, use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=False, use_readout_coverage=False,
            use_cross_spectral_anchor=True, use_adapter_tail=False,
            use_frequency_rank_tail=True, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=1e-5,
            first_step_jitter=1e-4, anchor_floor=0.5,
            auxiliary_weight=1.0, adapter_rank=192,
            adapter_tail_sites=("attn",), frequency_tail_weight=1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "K":
        values = dict(
            use_ads=True, use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=False, use_readout_coverage=False,
            use_cross_spectral_anchor=True, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=True,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=1e-4,
            anchor_floor=0.5, auxiliary_weight=1.0, skip_gamma=0.5,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "L":
        values = dict(
            use_ads=True, use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=False, use_readout_coverage=False,
            use_cross_spectral_anchor=True, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=True, asymmetric_step=True, random_start=False,
            momentum=0.5, initial_jitter=1e-5, first_step_jitter=1e-4,
            anchor_floor=0.5, auxiliary_weight=1.0,
            skip_gammas=(0.0, 0.75), skip_weights=(0.25, 0.75),
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "M":
        values = dict(
            use_ads=True, use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=False, use_readout_coverage=False,
            use_cross_spectral_anchor=True, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=True,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=1e-4,
            anchor_floor=0.5, auxiliary_weight=1.0,
            skip_gammas=(0.0, 0.75), skip_weights=(0.25, 0.75),
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "N":
        values = dict(
            use_ads=False, use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=False, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=True,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=True, skip_gamma=0.0,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=1e-4,
            uncertainty_weight=1.0, uncertainty_floor=0.1,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "O":
        values = dict(
            use_ads=True, use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=False, use_readout_coverage=False,
            use_cross_spectral_anchor=True, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=True, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=True,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=1e-4,
            anchor_floor=0.5, auxiliary_weight=1.0,
            skip_gammas=(0.0, 0.5, 1.0),
            skip_weights=(1.0 / 3, 1.0 / 3, 1.0 / 3),
            phase_softmin_beta=8.0, phase_reliability_floor=0.05,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"P", "Q"}:
        values = dict(
            use_ads=True, use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=False, use_readout_coverage=False,
            use_cross_spectral_anchor=True, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=True,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=True, asymmetric_step=True, random_start=False,
            momentum=0.5, initial_jitter=1e-5, first_step_jitter=1e-4,
            anchor_floor=0.5, auxiliary_weight=1.0,
            skip_gamma=0.0 if name == "P" else 0.5,
            feature_phase_weight=1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"R", "S", "T"}:
        values = dict(
            use_ads=False, use_grat=True, use_multilevel=False,
            use_peft_consensus=False, use_spectral=False, use_edge=False,
            use_readout_coverage=False, use_cross_spectral_anchor=False,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=name in {"S", "T"}, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=0.0, first_step_jitter=1e-4,
            skip_gamma=0.0 if name == "S" else 0.5,
            grat_noise_sd=0.5,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "U":
        values = dict(
            use_ads=True, use_grat=True, use_multilevel=False,
            use_peft_consensus=False, use_spectral=False, use_edge=False,
            use_readout_coverage=False, use_cross_spectral_anchor=True,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=True, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=0.0, first_step_jitter=1e-4,
            skip_gamma=0.5, grat_noise_sd=0.5,
            anchor_floor=0.5, auxiliary_weight=1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"V", "W"}:
        values = dict(
            use_ads=True, use_grat=True, use_multilevel=False,
            use_peft_consensus=False, use_spectral=False, use_edge=False,
            use_readout_coverage=False, use_cross_spectral_anchor=True,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=True, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=0.0, first_step_jitter=1e-4,
            skip_gamma=0.5, grat_noise_sd=0.5,
            anchor_floor=0.5, auxiliary_weight=2.0 if name == "V" else 4.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"X", "Y", "Z"}:
        values = dict(
            use_ads=name != "X", use_grat=False, use_patch_loss=True,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=False, use_readout_coverage=False,
            use_cross_spectral_anchor=name != "X", use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=name != "X",
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=0.0 if name == "X" else 1.0,
            skip_gamma=0.0 if name == "Y" else 0.5,
            anchor_floor=0.5, auxiliary_weight=1.0, patch_weight=1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"AA", "AB"}:
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_token_collapse=True, use_multilevel=False,
            use_peft_consensus=False, use_spectral=False, use_edge=False,
            use_readout_coverage=False, use_cross_spectral_anchor=False,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=True, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=0.0, first_step_jitter=1e-4,
            output_weight=0.0, skip_gamma=0.0 if name == "AA" else 0.5,
            collapse_weight=1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"AC", "AD", "AE"}:
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=True,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=False, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=name != "AC",
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=0.0,
            skip_gamma=0.0 if name == "AD" else 0.5,
            lift_layers=2, lift_weight=1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"AF", "AG", "AH"}:
        modes = {"AF": "shuffled_amplitude", "AG": "shuffled_white", "AH": "clean_amplitude"}
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False, use_pasd=True,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=1e-5,
            first_step_jitter=1e-4, output_weight=1.0,
            pasd_mode=modes[name], pasd_samples=2, pasd_probe_scale=2.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"AI", "AJ", "AK", "AL"}:
        white = name in {"AK", "AL"}
        half = name in {"AJ", "AL"}
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False, use_pasd=True,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=True,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=1e-5,
            first_step_jitter=1e-4, output_weight=1.0,
            skip_gamma=0.5 if half else 0.0,
            pasd_mode="shuffled_white" if white else "shuffled_amplitude",
            pasd_samples=2, pasd_probe_scale=2.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"AM", "AN", "AO", "AP"}:
        regions = {"AM": "early", "AN": "early", "AO": "mixed", "AP": "mixed"}
        counts = {"AM": 2, "AN": 4, "AO": 4, "AP": 6}
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False, use_pasd=True,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=1e-5,
            first_step_jitter=1e-4, output_weight=1.0,
            pasd_mode="shuffled_white", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_region=regions[name],
            pasd_layer_count=counts[name],
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"AQ", "AR", "AS", "AT"}:
        weights = {"AQ": 1.0, "AR": 0.25, "AS": 0.5, "AT": 1.0}
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=name != "AQ", use_atcs=True,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=1e-5,
            first_step_jitter=1e-4, output_weight=0.0 if name == "AQ" else 1.0,
            pasd_mode="shuffled_white", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_region="late", pasd_layer_count=6,
            atcs_weight=weights[name],
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"AU", "AV", "AW", "AX"}:
        counts = {"AU": 2, "AV": 4, "AW": 6, "AX": 12}
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=1e-5,
            first_step_jitter=1e-4, output_weight=1.0,
            pasd_mode="shuffled_white", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_region="final",
            pasd_layer_count=counts[name],
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"AY", "AZ", "BA", "BB", "BC", "BD"}:
        final = name in {"BA", "BB", "BC", "BD"}
        counts = {"BA": 1, "BB": 1, "BC": 2, "BD": 2}
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=name in {"BA"},
            use_readout_coverage=False, use_cross_spectral_anchor=False,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=False, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=1e-4,
            output_weight=0.0 if name in {"AZ", "BD"} else 1.0,
            pasd_mode="shuffled_white", pasd_samples=2,
            pasd_probe_scale=2.0,
            pasd_region="final" if final else "late",
            pasd_layer_count=counts.get(name, 6),
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"BE", "BF", "BG", "BH"}:
        final = name in {"BG", "BH"}
        uniform = name in {"BF", "BH"}
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=1e-5,
            first_step_jitter=1e-4, output_weight=1.0,
            pasd_mode="spatial_uniform_white" if uniform else "spatial_shuffled_white",
            pasd_samples=2, pasd_probe_scale=2.0,
            pasd_region="final" if final else "late",
            pasd_layer_count=2 if final else 6,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"BI", "BJ", "BK", "BL", "BM"}:
        weights = {"BI": 0.25, "BJ": 0.5, "BK": 1.0, "BL": 0.5, "BM": 1.0}
        spatial = name in {"BL", "BM"}
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=1e-5,
            first_step_jitter=1e-4, output_weight=1.0,
            pasd_mode="spatial_uniform_white" if spatial else "shuffled_white",
            pasd_samples=2, pasd_probe_scale=2.0,
            pasd_region="dual", pasd_layer_count=6,
            pasd_early_weight=weights[name],
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"BN", "BO", "BP", "BQ", "BR", "BS"}:
        weights = {"BN": 0.25, "BO": 0.5, "BP": 1.0, "BQ": 2.0, "BR": 0.5, "BS": 1.0}
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=name in {"BR", "BS"},
            use_readout_coverage=False, use_cross_spectral_anchor=False,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=False, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=1e-4,
            output_weight=1.0, pasd_mode="shuffled_white",
            pasd_samples=2, pasd_probe_scale=2.0,
            pasd_region="late_final", pasd_layer_count=6,
            pasd_final_weight=weights[name],
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"BT", "BU", "BV", "BW"}:
        kernels = {"BT": 3, "BU": 5, "BV": 9, "BW": 15}
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=False, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=1e-5,
            first_step_jitter=1e-4, output_weight=1.0,
            pasd_mode="shuffled_white", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_region="late", pasd_layer_count=6,
            post_ti_kernel=kernels[name],
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"BX", "BY", "BZ", "CA"}:
        schedules = {
            "BX": "late_then_final", "BY": "final_then_late",
            "BZ": "alternate", "CA": "alternate",
        }
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=name == "CA",
            use_readout_coverage=False, use_cross_spectral_anchor=False,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=False, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=1e-4,
            output_weight=1.0, pasd_mode="shuffled_white",
            pasd_samples=2, pasd_probe_scale=2.0,
            pasd_region="late", pasd_layer_count=6,
            pasd_schedule=schedules[name],
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"CS", "CT", "CU", "CV", "CW"}:
        second_region = "mixed" if name in {"CS", "CT", "CW"} else "early"
        switch_steps = {"CS": 4, "CT": 5, "CU": 4, "CV": 5, "CW": 6}
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=False,
            use_readout_coverage=False, use_cross_spectral_anchor=False,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=False, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=1e-4,
            output_weight=1.0, pasd_mode="shuffled_white",
            pasd_samples=2, pasd_probe_scale=2.0,
            pasd_region="final", pasd_layer_count=2,
            pasd_schedule=f"final_then_{second_region}",
            pasd_switch_step=switch_steps[name],
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"CX", "CY", "CZ", "DA"}:
        scheduled = name in {"CX", "CY"}
        samples = 4 if name in {"CX", "CZ"} else 8
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=True, use_atcs=False, use_multilevel=False,
            use_peft_consensus=False, use_spectral=False, use_edge=True,
            use_readout_coverage=False, use_cross_spectral_anchor=False,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=False, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=1e-4,
            output_weight=1.0, pasd_mode="shuffled_white",
            pasd_samples=samples, pasd_probe_scale=2.0,
            pasd_region="late", pasd_layer_count=6,
            pasd_schedule="final_then_late" if scheduled else "",
            pasd_switch_step=5 if scheduled else -1,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"DB", "DC", "DD"}:
        beta = {"DB": 4.0, "DC": 8.0, "DD": 16.0}[name]
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=True, use_atcs=False, use_multilevel=False,
            use_peft_consensus=False, use_spectral=False, use_edge=True,
            use_readout_coverage=False, use_cross_spectral_anchor=False,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=False, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=1e-4,
            output_weight=1.0, pasd_mode="shuffled_white",
            pasd_samples=8, pasd_probe_scale=2.0,
            pasd_reduce="softmin", pasd_softmin_beta=beta,
            pasd_region="late", pasd_layer_count=6,
            pasd_schedule="final_then_late", pasd_switch_step=5,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"DE", "DF", "DG", "DH"}:
        phase_weight = {"DE": 0.05, "DF": 0.1, "DG": 0.25, "DH": 0.5}[name]
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=True,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=True, use_atcs=False, use_multilevel=False,
            use_peft_consensus=False, use_spectral=False, use_edge=True,
            use_readout_coverage=False, use_cross_spectral_anchor=False,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=False, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=1e-4,
            output_weight=1.0, pasd_mode="shuffled_white",
            pasd_samples=2, pasd_probe_scale=2.0,
            pasd_region="late", pasd_layer_count=6,
            pasd_schedule="final_then_late", pasd_switch_step=5,
            input_phase_weight=phase_weight,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"DI", "DJ", "DK"}:
        amplitude_weight = {"DI": 0.01, "DJ": 0.05, "DK": 0.1}[name]
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=True,
            use_input_amplitude=True, use_token_collapse=False,
            use_low_level_lift=False, use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=1e-5,
            first_step_jitter=1e-4, output_weight=1.0,
            pasd_mode="shuffled_white", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_region="late", pasd_layer_count=6,
            pasd_schedule="final_then_late", pasd_switch_step=5,
            input_phase_weight=0.05,
            input_amplitude_weight=amplitude_weight,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"DL", "DM", "DN", "DO", "DP", "DQ", "DR", "LPDC"}:
        bands = {
            "DL": (0.0, 0.2), "DM": (0.05, 0.4), "DN": (0.2, 1.0),
            "DO": (0.0, 0.2), "DP": (0.0, 0.2), "DQ": (0.0, 0.2),
            "DR": (0.0, 0.2), "LPDC": (0.0, 0.2),
        }
        phase_weights = {
            "DL": 0.05, "DM": 0.05, "DN": 0.05, "DO": 0.1,
            "DP": 0.25, "DQ": 0.5, "DR": 0.02, "LPDC": 0.05,
        }
        low, high = bands[name]
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=True,
            use_input_amplitude=False, use_token_collapse=False,
            use_low_level_lift=False, use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=1e-5,
            first_step_jitter=1e-4, output_weight=1.0,
            pasd_mode="shuffled_white", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_region="late", pasd_layer_count=6,
            pasd_schedule="final_then_late", pasd_switch_step=5,
            input_phase_weight=phase_weights[name], input_phase_low=low,
            input_phase_high=high,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"DS", "DT"}:
        values = dict(
            use_ads=False, use_grat=False, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=name == "DT",
            use_input_amplitude=False, use_token_collapse=False,
            use_low_level_lift=False, use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=1e-5,
            first_step_jitter=1e-4, output_weight=1.0,
            pasd_mode="shuffled_white", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_region="late", pasd_layer_count=6,
            pasd_depth_robust=True, pasd_depth_beta=8.0,
            input_phase_weight=0.05, input_phase_low=0.0,
            input_phase_high=0.2,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"DU", "DV", "DW"}:
        schedule = {
            "DU": "grat_then_pasd", "DV": "pasd_then_grat", "DW": "",
        }[name]
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_token_collapse=False,
            use_low_level_lift=False, use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=1.0,
            grat_noise_sd=0.5, pasd_mode="shuffled_white",
            pasd_samples=2, pasd_probe_scale=2.0,
            pasd_region="late", pasd_layer_count=6,
            pasd_schedule=schedule, pasd_switch_step=5 if schedule else -1,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"DX", "DY", "DZ"}:
        pasd_weight = {"DX": 0.25, "DY": 0.5, "DZ": 0.75}[name]
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_token_collapse=False,
            use_low_level_lift=False, use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=1.0,
            grat_noise_sd=0.5, grat_weight=1.0,
            pasd_mode="shuffled_white", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_weight=pasd_weight,
            pasd_region="late", pasd_layer_count=6,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"EA", "EB", "EC"}:
        pasd_weight = {"EA": 0.05, "EB": 0.1, "EC": 0.15}[name]
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_token_collapse=False,
            use_low_level_lift=False, use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=1.0,
            grat_noise_sd=0.5, grat_weight=1.0,
            pasd_mode="shuffled_white", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_weight=pasd_weight,
            pasd_region="late", pasd_layer_count=6,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"ED", "EE", "EF"}:
        switch = {"ED": 9, "EE": 8, "EF": 7}[name]
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_token_collapse=False,
            use_low_level_lift=False, use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=1.0,
            grat_noise_sd=0.5, grat_weight=1.0,
            pasd_mode="shuffled_white", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_weight=0.25,
            pasd_region="late", pasd_layer_count=6,
            pasd_schedule="grat_then_joint", pasd_switch_step=switch,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"EG", "EH", "EI"}:
        low, high = {
            "EG": (0.0, 0.2), "EH": (0.05, 0.4), "EI": (0.2, 1.0),
        }[name]
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=True,
            use_input_amplitude=False, use_token_collapse=False,
            use_low_level_lift=False, use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=1.0,
            grat_noise_sd=0.5, grat_weight=1.0,
            pasd_mode="shuffled_white", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_weight=0.25,
            pasd_region="late", pasd_layer_count=6,
            input_phase_weight=0.05, input_phase_low=low,
            input_phase_high=high,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"EJ", "EK"}:
        # Spectral-probe control: preserve empirical amplitude and vary whether
        # phase is permuted, with every other joint-attack setting fixed.
        pasd_mode = {
            "EJ": "shuffled_amplitude", "EK": "clean_amplitude",
        }[name]
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_token_collapse=False,
            use_low_level_lift=False, use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=1.0,
            grat_noise_sd=0.5, grat_weight=1.0,
            pasd_mode=pasd_mode, pasd_samples=2,
            pasd_probe_scale=2.0, pasd_weight=0.25,
            pasd_region="late", pasd_layer_count=6,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"EL", "EM", "EN"}:
        # Depth-balanced PASD: late blocks provide semantic disruption while
        # the first two blocks anchor the direction shared before PEFT drift
        # accumulates through the encoder.
        early_weight = {"EL": 0.1, "EM": 0.25, "EN": 0.5}[name]
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_token_collapse=False,
            use_low_level_lift=False, use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=1.0,
            grat_noise_sd=0.5, grat_weight=1.0,
            pasd_mode="shuffled_amplitude", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_weight=0.25,
            pasd_region="dual", pasd_layer_count=6,
            pasd_early_weight=early_weight,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"EO", "EP", "EQ"}:
        polar_weight = {"EO": 0.1, "EP": 0.25, "EQ": 0.5}[name]
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_polar_amplitude_gradient=True,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=True, use_atcs=False, use_multilevel=False,
            use_peft_consensus=False, use_spectral=False, use_edge=True,
            use_readout_coverage=False, use_cross_spectral_anchor=False,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=False, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=0.0, first_step_jitter=1e-4,
            output_weight=1.0, grat_noise_sd=0.5, grat_weight=1.0,
            pasd_mode="shuffled_amplitude", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_weight=0.25,
            pasd_region="late", pasd_layer_count=6,
            polar_amplitude_weight=polar_weight,
            polar_amplitude_low=0.0, polar_amplitude_high=0.2,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"ER", "ES", "ET"}:
        coverage_weight = {"ER": 0.05, "ES": 0.1, "ET": 0.25}[name]
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_polar_amplitude_gradient=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=True, use_atcs=False, use_multilevel=False,
            use_peft_consensus=False, use_spectral=False, use_edge=True,
            use_readout_coverage=False, use_logdet_coverage=True,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=1.0,
            grat_noise_sd=0.5, grat_weight=1.0,
            pasd_mode="shuffled_amplitude", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_weight=0.25,
            pasd_region="late", pasd_layer_count=6,
            logdet_coverage_weight=coverage_weight,
            logdet_coverage_temperature=0.01,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"EU", "EV", "EW"}:
        collapse_weight = {"EU": 0.05, "EV": 0.1, "EW": 0.25}[name]
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_polar_amplitude_gradient=False,
            use_token_collapse=True, use_low_level_lift=False,
            use_pasd=True, use_atcs=False, use_multilevel=False,
            use_peft_consensus=False, use_spectral=False, use_edge=True,
            use_readout_coverage=False, use_logdet_coverage=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=1.0,
            grat_noise_sd=0.5, grat_weight=1.0,
            pasd_mode="shuffled_amplitude", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_weight=0.25,
            pasd_region="late", pasd_layer_count=6,
            collapse_weight=collapse_weight,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"EX", "EY", "EZ"}:
        amplitude_scale = {"EX": 0.1, "EY": 0.2, "EZ": 0.3}[name]
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_polar_amplitude_gradient=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=True, use_atcs=False, use_multilevel=False,
            use_peft_consensus=False, use_spectral=False, use_edge=True,
            use_readout_coverage=False, use_logdet_coverage=False,
            use_amplitude_eot=True, use_cross_spectral_anchor=False,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=False, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=0.0, first_step_jitter=1e-4,
            output_weight=1.0, grat_noise_sd=0.5, grat_weight=1.0,
            pasd_mode="shuffled_amplitude", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_weight=0.25,
            pasd_region="late", pasd_layer_count=6,
            amplitude_eot_samples=2, amplitude_eot_scale=amplitude_scale,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"FA", "FB", "FC", "FD", "FE"}:
        # Adapter-invariant token paths on top of the strongest spectral carrier
        # attack. FA/FB isolate QK bypass, FC isolates PatchOut, and FD/FE
        # combine them while preserving the exact UMI-GRAT/PASD objective.
        qk_scales = {"FA": 0.0, "FB": 0.2, "FD": 0.0, "FE": 0.2}
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_token_collapse=False,
            use_low_level_lift=False, use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_logdet_coverage=False, use_amplitude_eot=False,
            use_qk_bypass=name in {"FA", "FB", "FD", "FE"},
            use_patchout=name in {"FC", "FD", "FE"},
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=1.0,
            grat_noise_sd=0.5, grat_weight=1.0,
            pasd_mode="shuffled_amplitude", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_weight=0.25,
            pasd_region="late", pasd_layer_count=6,
            qk_gradient_scale=qk_scales.get(name, 1.0),
            patchout_keep_ratio=0.66, patchout_patch_size=16,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"FF", "FG", "FH"}:
        # First-order PEFT lower bound: maximize displacement energy outside
        # every rank-192 adapter correction subspace while retaining EJ.
        tail_weight = {"FF": 0.25, "FG": 0.5, "FH": 1.0}[name]
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_token_collapse=False,
            use_low_level_lift=False, use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_logdet_coverage=False, use_amplitude_eot=False,
            use_qk_bypass=False, use_patchout=False,
            use_cross_spectral_anchor=False, use_adapter_tail=True,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=1.0,
            grat_noise_sd=0.5, grat_weight=1.0,
            pasd_mode="shuffled_amplitude", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_weight=0.25,
            pasd_region="late", pasd_layer_count=6,
            adapter_rank=192, adapter_tail_weight=tail_weight,
            adapter_tail_sites=("attn", "block"),
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"FI", "FJ", "FK"}:
        # Zero-frequency amplitude displacement encourages a global image-token
        # bias that remains active for absent-class negative prompts.
        dc_weight = {"FI": 0.1, "FJ": 0.25, "FK": 0.5}[name]
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_token_collapse=False,
            use_dc_shift=True, use_low_level_lift=False, use_pasd=True,
            use_atcs=False, use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_logdet_coverage=False, use_amplitude_eot=False,
            use_qk_bypass=False, use_patchout=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=1.0,
            grat_noise_sd=0.5, grat_weight=1.0,
            pasd_mode="shuffled_amplitude", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_weight=0.25,
            pasd_region="late", pasd_layer_count=6,
            dc_shift_weight=dc_weight,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "FL":
        # Robustify the randomized-phase carrier probe over the convex envelope
        # between white and empirical clean-feature amplitudes.
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_token_collapse=False,
            use_dc_shift=False, use_low_level_lift=False, use_pasd=True,
            use_atcs=False, use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_logdet_coverage=False, use_amplitude_eot=False,
            use_qk_bypass=False, use_patchout=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=1.0,
            grat_noise_sd=0.5, grat_weight=1.0,
            pasd_mode="shuffled_amplitude_hull", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_weight=0.25,
            pasd_softmin_beta=8.0,
            pasd_amplitude_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
            pasd_region="late", pasd_layer_count=6,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"FM", "FN"}:
        # Channel-basis-invariant spatial correlation displacement.  The
        # normalized token Gram matrix couples spatial amplitude and relative
        # phase without selecting a fixed Fourier band.
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_token_collapse=False,
            use_dc_shift=False, use_low_level_lift=False, use_pasd=False,
            use_atcs=False, use_token_gram=True,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_logdet_coverage=False, use_amplitude_eot=False,
            use_qk_bypass=False, use_patchout=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=1.0,
            grat_noise_sd=0.5, grat_weight=1.0,
            token_gram_weight=0.25 if name == "FM" else 1.0,
            token_gram_pool=16, token_gram_region="late",
            token_gram_layer_count=6,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "FO":
        # Source-only constrained ascent in the blue sensor channel.  This is
        # used only for RGB modality diagnostics; grayscale inputs should keep
        # the unconstrained baseline.
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_token_collapse=False,
            use_dc_shift=False, use_low_level_lift=False, use_pasd=False,
            use_atcs=False, use_token_gram=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_logdet_coverage=False, use_amplitude_eot=False,
            use_qk_bypass=False, use_patchout=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=1.0,
            grat_noise_sd=0.5, grat_weight=1.0, update_channel=2,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "FP":
        values = dict(
            use_ads=False, use_grat=True, use_patch_loss=False,
            use_patch_boundary=False, use_input_phase=False,
            use_input_amplitude=False, use_token_collapse=False,
            use_dc_shift=False, use_low_level_lift=False, use_pasd=False,
            use_atcs=False, use_token_gram=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=True, use_readout_coverage=False,
            use_logdet_coverage=False, use_amplitude_eot=False,
            use_block_shuffle_eot=True,
            use_qk_bypass=False, use_patchout=False,
            use_cross_spectral_anchor=False, use_adapter_tail=False,
            use_frequency_rank_tail=False, use_skip_gradient=False,
            use_skip_ensemble=False, use_dual_path=False,
            use_spectral_uncertainty=False, use_phase_robust=False,
            use_feature_phase=False, asymmetric_step=True,
            random_start=False, momentum=0.5, initial_jitter=0.0,
            first_step_jitter=1e-4, output_weight=1.0,
            grat_noise_sd=0.5, grat_weight=1.0,
            block_shuffle_samples=3, block_shuffle_grid=4,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "FQ":
        values = vars(ablation_config("FP")).copy()
        values["update_channel"] = 2
        values.update(overrides)
        return AttackConfig(**values)
    if name == "FR":
        values = vars(ablation_config("FP")).copy()
        values.update(
            block_shuffle_samples=7,
            block_shuffle_mode="bsr",
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"FS", "FT", "FU"}:
        # Depth-continuation variants.  Block 0 is the earliest shared
        # transformer representation and is upstream of both LoRA and adapter
        # modifications in the evaluated downstream models.
        values = vars(ablation_config("R")).copy()
        values.update(
            use_depth_layer=True,
            depth_layer_index=0,
            depth_layer_weight=1.0,
            depth_schedule={
                "FS": "early_only",
                "FT": "early_then_grat",
                "FU": "joint",
            }[name],
            pasd_switch_step=5,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"FV", "FW"}:
        values = vars(ablation_config("FS" if name == "FV" else "FU")).copy()
        values["use_dual_preprocess"] = True
        values.update(overrides)
        return AttackConfig(**values)
    if name == "FX":
        # Dimensionless spatial polar displacement at the earliest shared
        # transformer block.  FFT axes are HxW after converting BHWC to BCHW.
        values = vars(ablation_config("FS")).copy()
        values.update(
            use_depth_polar=True,
            depth_layer_weight=0.0,
            depth_phase_weight=1.0,
            depth_amplitude_weight=1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "FY":
        values = vars(ablation_config("FS")).copy()
        values.update(use_depth_boundary=True, depth_boundary_weight=1.0)
        values.update(overrides)
        return AttackConfig(**values)
    if name == "FZ":
        values = vars(ablation_config("FS")).copy()
        values.update(use_patch_loss=True, patch_weight=1.0)
        values.update(overrides)
        return AttackConfig(**values)
    if name == "GA":
        values = vars(ablation_config("R")).copy()
        values.update(
            use_grat=False,
            output_weight=0.0,
            use_depth_robust=True,
            depth_robust_weight=1.0,
            depth_robust_count=6,
            depth_robust_beta=8.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "GB":
        values = vars(ablation_config("FS")).copy()
        values["use_edge"] = True
        values.update(overrides)
        return AttackConfig(**values)
    if name == "GC":
        values = vars(ablation_config("FS")).copy()
        values.update(
            use_block_shuffle_eot=True,
            block_shuffle_samples=7,
            block_shuffle_mode="bsr",
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"GD", "GE"}:
        values = vars(ablation_config("R" if name == "GD" else "FS")).copy()
        values.update(
            use_grat=False,
            output_weight=0.0,
            use_phase_collision=True,
            phase_collision_layer=0,
            phase_collision_weight=1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"GF", "GG", "GH"}:
        # Decoder-independent spatial information collapse.  GF isolates the
        # earliest shared block, GG keeps the late GRAT displacement, and GH
        # constrains a uniformly sampled path through the frozen backbone.
        values = vars(ablation_config("FS" if name == "GF" else "FU")).copy()
        values.update(
            use_graph_collapse=True,
            graph_collapse_weight=1.0,
            graph_collapse_count=1 if name in {"GF", "GG"} else 3,
            depth_layer_weight=0.0 if name == "GF" else 1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"GI", "GJ"}:
        values = vars(ablation_config("FS" if name == "GI" else "FU")).copy()
        values.update(
            use_qk_topology=True,
            qk_topology_layer=0,
            qk_topology_weight=1.0,
            depth_layer_weight=0.0 if name == "GI" else 1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"GK", "GL"}:
        # Phase/amplitude exploration followed by a task-sensitive shared-depth
        # correction on the same optimization trajectory.
        values = vars(ablation_config("CH")).copy()
        values.update(
            use_depth_layer=True,
            depth_layer_index=0,
            depth_layer_weight=1.0,
            depth_schedule="joint",
            use_grat=True,
            grat_weight=1.0,
            task_after_switch=True,
            use_adapter_tail=name == "GL",
            adapter_rank=192,
            adapter_tail_weight=1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "GM":
        values = vars(ablation_config("CH")).copy()
        values.update(
            use_block_shuffle_eot=True,
            block_shuffle_samples=7,
            block_shuffle_mode="bsr",
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"GN", "GO", "GP", "GQ"}:
        values = vars(ablation_config("R" if name == "GN" else "CH")).copy()
        collision_weight = {"GN": 1.0, "GO": 1.0, "GP": 0.1, "GQ": 0.25}[name]
        values.update(
            use_grat=False,
            output_weight=0.0 if name == "GN" else values["output_weight"],
            use_quadrature_collision=True,
            quadrature_collision_weight=collision_weight,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "GR":
        values = vars(ablation_config("CH")).copy()
        values.update(
            use_pasd=False,
            use_quadrature_collision=True,
            quadrature_collision_weight=1.0,
            quadrature_schedule="quadrature_then_ads",
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "GS":
        values = vars(ablation_config("CH")).copy()
        values.update(
            use_ads=False,
            use_quadrature_collision=True,
            quadrature_collision_weight=1.0,
            quadrature_schedule="after_switch",
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"GT", "GU"}:
        # Polar tangent directions selected by a held-out, white-box diagnostic.
        # GT isolates the downstream-aligned quadrature amplitude tangent;
        # GU tests the independently identified final-feature amplitude tangent.
        values = vars(ablation_config("GN" if name == "GT" else "A")).copy()
        values.update(
            polar_projection_coordinate="amplitude",
            polar_projection_replace=True,
            polar_projection_low=0.05 if name == "GT" else 0.10,
            polar_projection_high=0.10 if name == "GT" else 0.20,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "GV":
        values = vars(ablation_config("R")).copy()
        values.update(
            use_grat=False,
            output_weight=0.0,
            use_translation_collision=True,
            translation_collision_shift=(4, 4),
            translation_collision_weight=1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "GW":
        values = vars(ablation_config("CH")).copy()
        values.update(
            use_translation_collision=True,
            translation_collision_shift=(4, 4),
            translation_collision_weight=0.25,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "GX":
        values = vars(ablation_config("R")).copy()
        values.update(
            use_grat=False,
            output_weight=0.0,
            use_input_translation_collision=True,
            input_translation_shift=(8, 8),
            input_translation_weight=1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "GY":
        values = vars(ablation_config("R")).copy()
        values.update(
            use_grat=False, output_weight=0.0,
            use_spatial_principal=True, spatial_principal_rank=8,
            spatial_principal_weight=1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"GZ", "HA"}:
        values = vars(ablation_config("CH")).copy()
        values.update(
            use_spatial_principal=True,
            spatial_principal_rank=8 if name == "GZ" else 32,
            spatial_principal_weight=1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "HB":
        values = vars(ablation_config("R")).copy()
        values.update(
            use_grat=False, output_weight=0.0,
            use_semantic_partition=True, semantic_partition_weight=1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"HC", "HD"}:
        values = vars(ablation_config("CH")).copy()
        values.update(
            use_semantic_partition=True,
            semantic_partition_weight=0.25 if name == "HC" else 1.0,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"HE", "HF", "HG"}:
        penalties = {"HE": 0.0, "HF": 0.25, "HG": 0.5}
        values = vars(ablation_config("R")).copy()
        values.update(
            use_grat=False, output_weight=0.0, use_cancellation_gap=True,
            cancellation_layer_count=4,
            cancellation_final_penalty=penalties[name],
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name == "HH":
        values = vars(ablation_config("CH")).copy()
        values.update(
            use_cancellation_gap=True, cancellation_layer_count=4,
            cancellation_final_penalty=0.25, cancellation_weight=0.25,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"HI", "HJ", "HK"}:
        values = vars(ablation_config("A")).copy()
        values.update(
            use_peft_consensus=True, adapter_view_count=8,
            adapter_view_rank=192,
            adapter_view_scale=0.25 if name != "HK" else 0.4,
            peft_reduce="mean" if name == "HI" else "softmin",
            peft_softmin_beta=50.0,
            momentum=0.5, asymmetric_step=True,
            initial_jitter=1e-5, first_step_jitter=1e-4,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"CB", "CC", "CD"}:
        scheduled = name in {"CC", "CD"}
        values = dict(
            use_ads=True, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=scheduled, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=name in {"CB", "CD"},
            use_readout_coverage=False, use_cross_spectral_anchor=False,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=False, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=1e-4,
            output_weight=1.0, ads_weight=1.0,
            pasd_mode="shuffled_white", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_region="final", pasd_layer_count=2,
            pasd_schedule="final_then_ads" if scheduled else "",
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"CE", "CF", "CG", "CH"}:
        switch_steps = {"CE": 3, "CF": 4, "CG": 6, "CH": 4}
        values = dict(
            use_ads=True, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=True, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=name == "CH",
            use_readout_coverage=False, use_cross_spectral_anchor=False,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=False, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=1e-4,
            output_weight=1.0, ads_weight=1.0,
            pasd_mode="shuffled_white", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_region="final", pasd_layer_count=2,
            pasd_schedule="final_then_ads", pasd_switch_step=switch_steps[name],
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"CI", "CJ", "CK"}:
        scheduled = name in {"CI", "CJ"}
        values = dict(
            use_ads=True, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=scheduled, use_atcs=False,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=name in {"CI", "CK"},
            use_readout_coverage=False, use_cross_spectral_anchor=False,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=False, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=0.0,
            output_weight=1.0, ads_weight=1.0,
            pasd_mode="shuffled_white", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_region="final", pasd_layer_count=2,
            pasd_schedule="final_then_ads" if scheduled else "",
            pasd_switch_step=4 if scheduled else -1,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"CL", "CM", "CN", "CO"}:
        weights = {"CL": 0.25, "CM": 0.5, "CN": 1.0, "CO": 0.5}
        values = dict(
            use_ads=True, use_grat=False, use_patch_loss=False,
            use_token_collapse=False, use_low_level_lift=False,
            use_pasd=True, use_atcs=False, use_final_ads_anchor=True,
            use_multilevel=False, use_peft_consensus=False,
            use_spectral=False, use_edge=name != "CO",
            use_readout_coverage=False, use_cross_spectral_anchor=False,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=False, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=0.0,
            output_weight=1.0, ads_weight=1.0,
            anchor_floor=1.0, auxiliary_weight=weights[name],
            pasd_mode="shuffled_white", pasd_samples=2,
            pasd_probe_scale=2.0, pasd_region="final", pasd_layer_count=2,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name in {"CP", "CQ", "CR"}:
        weights = {"CP": 0.1, "CQ": 0.25, "CR": 0.5}
        values = dict(
            use_ads=True, ads_all_views=True, use_grat=False,
            use_patch_loss=False, use_token_collapse=False,
            use_low_level_lift=False, use_pasd=False, use_atcs=False,
            use_multilevel=False, use_peft_consensus=True,
            use_spectral=False, use_edge=True,
            use_readout_coverage=False, use_cross_spectral_anchor=False,
            use_adapter_tail=False, use_frequency_rank_tail=False,
            use_skip_gradient=False, use_skip_ensemble=False,
            use_dual_path=False, use_spectral_uncertainty=False,
            use_phase_robust=False, use_feature_phase=False,
            asymmetric_step=True, random_start=False, momentum=0.5,
            initial_jitter=1e-5, first_step_jitter=0.0,
            output_weight=1.0, ads_weight=1.0,
            robust_weight=weights[name], adapter_view_count=4,
            adapter_view_rank=192, adapter_view_scale=0.25,
        )
        values.update(overrides)
        return AttackConfig(**values)
    if name not in order:
        raise ValueError(f"unknown ablation {name!r}; expected A--Z or AA--CR")
    level = order[name]
    values = dict(
        use_ads=level >= 1,
        use_multilevel=level >= 2,
        use_peft_consensus=level >= 3,
        use_spectral=level >= 4,
        use_edge=level >= 5,
        asymmetric_step=level >= 1,
        random_start=False,
        momentum=1.0 if level == 0 else 0.5,
        initial_jitter=0.0 if level == 0 else 1e-5,
        first_step_jitter=0.0 if level == 0 else 1e-4,
    )
    values.update(overrides)
    return AttackConfig(**values)


def _tensor_output(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, tuple) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"expected tensor output, got {type(value)}")


class _LowRankResidual(nn.Module):
    def __init__(self, dim: int, rank: int, scale: float, seed: int, nonlinear: bool):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.scale = float(scale)
        self.nonlinear = nonlinear
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            down = torch.randn(self.down.weight.shape, generator=generator) / math.sqrt(dim)
            up = torch.randn(self.up.weight.shape, generator=generator) / math.sqrt(rank)
            self.down.weight.copy_(down)
            self.up.weight.copy_(up)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.down(x)
        if self.nonlinear:
            z = F.gelu(z)
        return self.scale * self.up(z)


class VirtualPEFTView:
    """Fixed low-rank residuals placed where LoRA or adapters modify SAM."""

    def __init__(self, encoder: nn.Module, kind: str, rank: int, scale: float, seed: int):
        if kind not in {"source", "lora", "adapter", "adapter_exact"}:
            raise ValueError(f"unknown virtual view: {kind}")
        self.encoder = encoder
        self.kind = kind
        self.modules: List = []
        self.handles = []
        if kind == "source":
            return
        dim = int(encoder.blocks[0].attn.qkv.in_features)
        device = encoder.blocks[0].attn.qkv.weight.device
        for block_index, block in enumerate(encoder.blocks):
            if kind == "adapter_exact":
                space = _LowRankResidual(
                    dim, rank, scale, seed + 2017 * block_index, nonlinear=True
                ).to(device)
                mlp = _LowRankResidual(
                    dim, rank, 0.5 * scale, seed + 2017 * block_index + 1, nonlinear=True
                ).to(device)
                self.modules.append((space, mlp))
            else:
                residual = _LowRankResidual(
                    dim, rank, scale, seed + 1009 * block_index,
                    nonlinear=(kind == "adapter"),
                ).to(device)
                self.modules.append(residual)

    @contextlib.contextmanager
    def active(self):
        if self.kind == "source":
            yield
            return
        try:
            for block, residual in zip(self.encoder.blocks, self.modules):
                if self.kind == "adapter_exact":
                    space_residual, mlp_residual = residual

                    def space_hook(_module, _inputs, output, residual=space_residual):
                        base = _tensor_output(output)
                        updated = base + residual(base)
                        return updated if torch.is_tensor(output) else (updated, *output[1:])

                    def mlp_hook(_module, inputs, output, residual=mlp_residual):
                        base = _tensor_output(output)
                        updated = base + residual(inputs[0])
                        return updated if torch.is_tensor(output) else (updated, *output[1:])

                    self.handles.append(block.attn.register_forward_hook(space_hook))
                    self.handles.append(block.mlp.register_forward_hook(mlp_hook))
                    continue
                target = block.attn if self.kind == "lora" else block.mlp

                def hook(_module, inputs, output, residual=residual):
                    base = _tensor_output(output)
                    updated = base + residual(inputs[0])
                    if torch.is_tensor(output):
                        return updated
                    return (updated, *output[1:])

                self.handles.append(target.register_forward_hook(hook))
            yield
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()


class AdapterSiteCapture:
    """Capture source features immediately before low-rank PEFT corrections."""

    def __init__(self, encoder: nn.Module, layer_indices: Sequence[int], sites: Sequence[str]):
        self.encoder = encoder
        self.layer_indices = tuple(layer_indices)
        self.sites = tuple(sites)
        self.outputs: Dict[Tuple[int, str], torch.Tensor] = {}
        self.handles = []

    @contextlib.contextmanager
    def active(self):
        try:
            for index in self.layer_indices:
                block = self.encoder.blocks[index]
                modules = {"attn": block.attn, "block": block}
                for site in self.sites:
                    if site not in modules:
                        raise ValueError(f"unknown adapter-tail site {site!r}")

                    def hook(_module, _inputs, output, key=(index, site)):
                        self.outputs[key] = _tensor_output(output)

                    self.handles.append(modules[site].register_forward_hook(hook))
            yield self
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()


@contextlib.contextmanager
def intermediate_perturbation_decay(
    encoder: nn.Module, layer_index: int, clean_output: Optional[torch.Tensor], gamma: float
):
    """Official ILPD feature hook: clean + gamma * (adversarial - clean)."""
    if clean_output is None:
        yield
        return
    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError(f"ILPD gamma must be in (0,1], got {gamma}")
    block = encoder.blocks[int(layer_index)]

    def hook(_module, _inputs, output):
        value = _tensor_output(output)
        decayed = clean_output + float(gamma) * (value - clean_output)
        return decayed if torch.is_tensor(output) else (decayed, *output[1:])

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class SkipGradientView:
    """Preserve the forward pass while contracting mutable branch Jacobians."""

    def __init__(self, encoder: nn.Module, gamma: float):
        self.encoder = encoder
        self.gamma = float(gamma)
        self.handles = []

    @contextlib.contextmanager
    def active(self):
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"skip-gradient gamma must be in [0,1], got {self.gamma}")
        try:
            for block in self.encoder.blocks:
                for module in (block.attn, block.mlp):
                    def hook(_module, _inputs, output, gamma=self.gamma):
                        value = _tensor_output(output)
                        scaled = value.detach() + gamma * (value - value.detach())
                        if torch.is_tensor(output):
                            return scaled
                        return (scaled, *output[1:])

                    self.handles.append(module.register_forward_hook(hook))
            yield
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()


class QKGradientBypassView:
    """Keep the attention forward pass while suppressing Q/K-specific gradients.

    SAM concatenates Q, K, and V along the last axis of every ``qkv`` output.
    Scaling only the first two thirds removes gradients through the learned
    attention map while retaining the value path, which is the PNA backward
    rule adapted to SAM's attention implementation.
    """

    def __init__(self, encoder: nn.Module, scale: float):
        self.encoder = encoder
        self.scale = float(scale)
        self.handles = []

    @contextlib.contextmanager
    def active(self):
        if not 0.0 <= self.scale <= 1.0:
            raise ValueError(f"QK gradient scale must be in [0,1], got {self.scale}")

        def capture_qkv(_module, _inputs, output):
            value = _tensor_output(output)
            if value.shape[-1] % 3:
                raise ValueError(f"qkv width must be divisible by three, got {value.shape[-1]}")
            channels = value.shape[-1] // 3
            multiplier = torch.ones_like(value)
            multiplier[..., :2 * channels] = self.scale
            transformed = value.detach() + multiplier * (value - value.detach())
            if torch.is_tensor(output):
                return transformed
            return (transformed, *output[1:])

        try:
            for block in self.encoder.blocks:
                self.handles.append(block.attn.qkv.register_forward_hook(capture_qkv))
            yield
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()


class QKSiteCapture:
    """Capture the pre-attention QKV projection at one encoder block."""

    def __init__(self, encoder: nn.Module, layer: int):
        if not 0 <= int(layer) < len(encoder.blocks):
            raise IndexError(f"QK topology layer {layer} outside encoder depth")
        self.module = encoder.blocks[int(layer)].attn.qkv
        self.output: Optional[torch.Tensor] = None
        self.handle = None

    @contextlib.contextmanager
    def active(self):
        def capture(_module, _inputs, output):
            self.output = _tensor_output(output)

        self.handle = self.module.register_forward_hook(capture)
        try:
            yield
        finally:
            self.handle.remove()
            self.handle = None


def qk_topology_disruption(
    adv_qkv: torch.Tensor,
    clean_qkv: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Disrupt normalized token relations induced by Q and K.

    QKV projections are BHWC (or window-batched BHWC).  The relation matrix
    uses unit channel vectors, so feature scale cannot inflate the objective.
    """
    if adv_qkv.ndim != 4 or adv_qkv.shape != clean_qkv.shape:
        raise ValueError(f"expected matching BHWC QKV tensors, got {adv_qkv.shape} and {clean_qkv.shape}")
    channels3 = adv_qkv.shape[-1]
    if channels3 % (3 * int(num_heads)):
        raise ValueError(f"QKV width {channels3} is incompatible with {num_heads} heads")
    channels = channels3 // 3
    head_dim = channels // int(num_heads)

    def relations(value: torch.Tensor) -> torch.Tensor:
        batch, height, width, _ = value.shape
        qkv = value.reshape(batch, height * width, 3, int(num_heads), head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key = qkv[0], qkv[1]
        query = F.normalize(query.float(), dim=-1, eps=1e-8)
        key = F.normalize(key.float(), dim=-1, eps=1e-8)
        return query @ key.transpose(-2, -1)

    adv_relation = relations(adv_qkv)
    clean_relation = relations(clean_qkv.detach())
    return F.l1_loss(adv_relation, clean_relation)


def patchout_mask(
    image: torch.Tensor,
    keep_ratio: float,
    patch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample a channel-shared, without-replacement mask on an image grid."""
    if not 0.0 < float(keep_ratio) <= 1.0:
        raise ValueError(f"PatchOut keep ratio must be in (0,1], got {keep_ratio}")
    patch_size = int(patch_size)
    if patch_size <= 0:
        raise ValueError(f"PatchOut patch size must be positive, got {patch_size}")
    height, width = image.shape[-2:]
    rows = math.ceil(height / patch_size)
    cols = math.ceil(width / patch_size)
    count = rows * cols
    kept = max(1, min(count, round(float(keep_ratio) * count)))
    selected = torch.randperm(count, generator=generator, device=image.device)[:kept]
    coarse = torch.zeros((1, 1, rows, cols), device=image.device, dtype=image.dtype)
    coarse.view(-1)[selected] = 1.0
    return coarse.repeat_interleave(patch_size, -2).repeat_interleave(
        patch_size, -1
    )[..., :height, :width]


def _adapter_layer_indices(encoder: nn.Module, cfg: AttackConfig) -> Tuple[int, ...]:
    depth = len(encoder.blocks)
    if cfg.adapter_tail_layers:
        indices = tuple(int(index) for index in cfg.adapter_tail_layers)
    else:
        global_indices = tuple(
            index for index, block in enumerate(encoder.blocks)
            if int(getattr(block, "window_size", 0)) == 0
        )
        indices = global_indices[-3:] if len(global_indices) >= 3 else tuple(
            round(fraction * (depth - 1)) for fraction in (0.5, 0.75, 1.0)
        )
    if not indices or min(indices) < 0 or max(indices) >= depth:
        raise ValueError(f"adapter-tail layers {indices} invalid for encoder depth {depth}")
    return tuple(dict.fromkeys(indices))


def adapter_irreducible_tail(
    adv_sites: Dict[Tuple[int, str], torch.Tensor],
    clean_sites: Dict[Tuple[int, str], torch.Tensor],
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return normalized energy no rank-``rank`` channel correction can cancel.

    Each site is flattened to a token-by-channel displacement matrix D. By the
    Eckart--Young theorem, min_{rank(E)<=r} ||D+E||_F^2 is exactly the squared
    singular-value tail beyond r. The second return value is its energy ratio.
    """
    losses, ratios = [], []
    for key, adv in adv_sites.items():
        clean = clean_sites[key].detach()
        channels = adv.shape[-1]
        displacement = (adv - clean).reshape(-1, channels).float()
        singular_values = torch.linalg.svdvals(displacement)
        squared = singular_values.square()
        cutoff = min(max(int(rank), 0), squared.numel())
        tail = squared[cutoff:].sum()
        clean_scale = clean.float().square().mean().clamp_min(1e-8)
        losses.append(tail / (displacement.numel() * clean_scale))
        ratios.append(tail / squared.sum().clamp_min(1e-12))
    if not losses:
        raise ValueError("no adapter sites were captured")
    return torch.stack(losses).mean(), torch.stack(ratios).mean()


def _haar_bands(features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """One-level orthonormal 2-D Haar bands for a BHWC feature map."""
    if features.ndim != 4 or features.shape[1] % 2 or features.shape[2] % 2:
        raise ValueError(f"Haar features must be BHWC with even H/W, got {tuple(features.shape)}")
    x00 = features[:, 0::2, 0::2]
    x01 = features[:, 0::2, 1::2]
    x10 = features[:, 1::2, 0::2]
    x11 = features[:, 1::2, 1::2]
    return (
        (x00 + x01 + x10 + x11) * 0.5,
        (x00 - x01 + x10 - x11) * 0.5,
        (x00 + x01 - x10 - x11) * 0.5,
        (x00 - x01 - x10 + x11) * 0.5,
    )


def frequency_balanced_irreducible_tail(
    adv: torch.Tensor,
    clean: torch.Tensor,
    rank: int,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Geometric mean of rank-tail energies in orthogonal spatial bands.

    For each Haar projector P_b and any rank-r correction E,
    rank(P_b E) <= r. Parseval therefore gives
    ||D+E||_F^2 >= sum_b sum_{i>r} sigma_i(P_b D)^2.
    The log objective prevents this certified energy from collapsing into one
    spatial band; it is monotone in every bandwise lower bound.
    """
    clean = clean.detach()
    clean_scale = clean.float().square().mean().clamp_min(epsilon)
    energies, ratios = [], []
    for adv_band, clean_band in zip(_haar_bands(adv), _haar_bands(clean)):
        channels = adv_band.shape[-1]
        displacement = (adv_band - clean_band).reshape(-1, channels).float()
        squared = torch.linalg.svdvals(displacement).square()
        cutoff = min(max(int(rank), 0), squared.numel())
        tail = squared[cutoff:].sum()
        normalized = tail / (displacement.numel() * clean_scale)
        energies.append(normalized)
        ratios.append(tail / squared.sum().clamp_min(epsilon))
    log_geometric_mean = torch.stack([
        torch.log(value + epsilon) for value in energies
    ]).mean()
    return log_geometric_mean, torch.stack(ratios).mean()


def _unpack_encoder(output) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if torch.is_tensor(output):
        return output, []
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        final = output[0]
        layers = output[1]
        if torch.is_tensor(layers):
            layers = [layers]
        return final, list(layers)
    raise TypeError("encoder must return Tensor or (final, intermediate_layers)")


def _feature_disruption(adv: torch.Tensor, clean: torch.Tensor, cfg: AttackConfig) -> torch.Tensor:
    clean = clean.detach()
    scale = clean.abs().mean().clamp_min(1e-6)
    relative_l1 = (adv - clean).abs().mean() / scale
    adv_flat = adv.flatten(1)
    clean_flat = clean.flatten(1)
    cosine = 1.0 - F.cosine_similarity(adv_flat, clean_flat, dim=1).mean()
    spatial_dims = tuple(range(2, adv.ndim))
    if spatial_dims:
        mean_shift = (adv.mean(spatial_dims) - clean.mean(spatial_dims)).abs().mean() / scale
        std_shift = (adv.std(spatial_dims, unbiased=False) - clean.std(spatial_dims, unbiased=False)).abs().mean() / scale
        moments = mean_shift + std_shift
    else:
        moments = adv.new_zeros(())
    return relative_l1 + cfg.cosine_weight * cosine + cfg.moment_weight * moments


def multilevel_disruption(
    final_adv: torch.Tensor,
    final_clean: torch.Tensor,
    layers_adv: Sequence[torch.Tensor],
    layers_clean: Sequence[torch.Tensor],
    cfg: AttackConfig,
) -> torch.Tensor:
    loss = cfg.output_weight * _feature_disruption(final_adv, final_clean, cfg)
    count = min(cfg.selected_layers, len(layers_adv), len(layers_clean))
    if count:
        indices = torch.linspace(0, len(layers_adv) - 1, count, device=final_adv.device).round().long().tolist()
        layer_losses = [_feature_disruption(layers_adv[i], layers_clean[i], cfg) for i in indices]
        loss = loss + cfg.layer_weight * torch.stack(layer_losses).mean()
    return loss


def intermediate_disruption(
    layers_adv: Sequence[torch.Tensor],
    layers_clean: Sequence[torch.Tensor],
    cfg: AttackConfig,
) -> torch.Tensor:
    """Disrupt intermediate features without reweighting the final output."""
    count = min(cfg.selected_layers, len(layers_adv), len(layers_clean))
    if not count:
        return layers_adv[0].new_zeros(()) if layers_adv else torch.tensor(0.0)
    indices = torch.linspace(0, len(layers_adv) - 1, count).round().long().tolist()
    return torch.stack([
        _feature_disruption(layers_adv[index], layers_clean[index], cfg) for index in indices
    ]).mean()


def readout_coverage_lower_bound(
    final_adv: torch.Tensor,
    final_clean: torch.Tensor,
    cfg: AttackConfig,
) -> torch.Tensor:
    """Smooth lower bound on the least disrupted channel readout.

    For X = tokens(final_adv - final_clean), every unit channel readout w
    satisfies ||Xw||^2 / N >= lambda_min(X^T X / N). The unnormalized
    soft-min below is no larger than lambda_min and has stable gradients when
    several eigenvalues coincide.
    """
    displacement = (final_adv - final_clean.detach()).flatten(2).transpose(1, 2)
    clean_scale = final_clean.detach().square().mean().clamp_min(1e-8)
    covariance = displacement.transpose(1, 2).matmul(displacement)
    covariance = covariance / (displacement.shape[1] * clean_scale)
    eigenvalues = torch.linalg.eigvalsh(covariance.float())
    tau = max(float(cfg.coverage_temperature), 1e-6)
    return (-tau * torch.logsumexp(-eigenvalues / tau, dim=-1)).mean()


def logdet_readout_coverage(
    final_adv: torch.Tensor,
    final_clean: torch.Tensor,
    temperature: float = 0.01,
) -> torch.Tensor:
    """Spread normalized token displacement across channel readout directions.

    The mean log-volume emphasizes currently weak covariance eigen-directions,
    producing a high-rank displacement that cannot be canceled by one small
    low-rank downstream correction.
    """
    displacement = (final_adv.float() - final_clean.detach().float()).flatten(2).transpose(1, 2)
    clean_scale = final_clean.detach().float().square().mean().clamp_min(1e-8)
    covariance = displacement.transpose(1, 2).matmul(displacement)
    covariance = covariance / (displacement.shape[1] * clean_scale)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    tau = max(float(temperature), 1e-8)
    return torch.log1p(eigenvalues / tau).mean()


def spatial_principal_basis(clean: torch.Tensor, rank: int) -> torch.Tensor:
    """Return leading clean spatial modes of a BCHW image embedding."""
    if clean.ndim != 4:
        raise ValueError("spatial principal basis expects BCHW features")
    tokens = clean.detach().float().permute(0, 2, 3, 1).reshape(-1, clean.shape[1])
    tokens = tokens - tokens.mean(0, keepdim=True)
    left, _singular, _right_h = torch.linalg.svd(tokens, full_matrices=False)
    effective_rank = min(max(int(rank), 1), left.shape[1])
    return left[:, :effective_rank].to(dtype=clean.dtype)


def spatial_principal_disruption(
    current: torch.Tensor,
    clean: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Disrupt the spatial modes retained by tokenwise PEFT channel maps."""
    displacement = (current - clean.detach()).permute(0, 2, 3, 1)
    displacement = displacement.reshape(-1, current.shape[1])
    coefficients = basis.T @ displacement
    return coefficients.square().mean().clamp_min(1e-12).sqrt()


def semantic_partition(
    clean: torch.Tensor,
    clusters: int,
    iterations: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic prompt-free semantic partition of SAM tokens."""
    tokens = clean.detach().float().permute(0, 2, 3, 1).reshape(-1, clean.shape[1])
    tokens = F.normalize(tokens, dim=1)
    count = min(max(int(clusters), 2), tokens.shape[0])
    # Farthest-point initialization avoids random or victim-dependent choices.
    chosen = [int(tokens.square().sum(1).argmax())]
    distance = 1.0 - tokens @ tokens[chosen[0]]
    for _ in range(1, count):
        index = int(distance.argmax())
        chosen.append(index)
        distance = torch.minimum(distance, 1.0 - tokens @ tokens[index])
    centroids = tokens[chosen]
    labels = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
    for _ in range(iterations):
        labels = (tokens @ centroids.T).argmax(1)
        updated = []
        for index in range(count):
            members = tokens[labels == index]
            updated.append(centroids[index] if members.numel() == 0 else members.mean(0))
        centroids = F.normalize(torch.stack(updated), dim=1)
    return centroids.to(clean.dtype), labels


def semantic_partition_disruption(
    current: torch.Tensor,
    centroids: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Cross the clean SAM token partition without a prompt or task label."""
    tokens = current.permute(0, 2, 3, 1).reshape(-1, current.shape[1])
    logits = F.normalize(tokens, dim=1) @ centroids.T
    return F.cross_entropy(logits / max(float(temperature), 1e-4), labels)


def cancellation_gap_disruption(
    final: torch.Tensor,
    clean_final: torch.Tensor,
    layers: Sequence[torch.Tensor],
    clean_layers: Sequence[torch.Tensor],
    layer_count: int,
    final_penalty: float,
) -> torch.Tensor:
    """Expose perturbations that the frozen base path cancels at its output."""
    count = min(max(int(layer_count), 1), len(layers), len(clean_layers))
    early = torch.stack([
        (layers[index] - clean_layers[index].detach()).abs().mean()
        / clean_layers[index].detach().abs().mean().clamp_min(1e-6)
        for index in range(count)
    ]).mean()
    final_distance = (
        (final - clean_final.detach()).abs().mean()
        / clean_final.detach().abs().mean().clamp_min(1e-6)
    )
    return early - float(final_penalty) * final_distance


def carrier_stochastic_disruption(
    layers_adv: Sequence[torch.Tensor],
    layers_clean: Sequence[torch.Tensor],
    cfg: AttackConfig,
) -> torch.Tensor:
    """SE-CGR/ADS backbone retained for its strong source-to-LoRA transfer."""
    if not layers_adv or not layers_clean:
        raise ValueError("ADS requires intermediate encoder features")
    pre_layer = int(len(layers_adv) * 0.9)
    pool = list(range(max(0, pre_layer - 6), pre_layer))
    count = min(max(1, cfg.selected_layers), len(pool))
    indices = sorted(random.sample(pool, count))
    residuals, scales = [], []
    for index in indices:
        clean = layers_clean[index].detach()
        adv = layers_adv[index]
        clean_fft = torch.fft.fftn(clean, dim=(-2, -1), norm="ortho")
        amplitude = clean_fft.abs()
        phase = clean_fft.angle()
        difference = adv * 0.999 - clean
        lam = difference.abs().mean() / amplitude.mean().clamp_min(1e-6)
        sigma = lam * amplitude + 1e-4
        probe_fft = torch.randn_like(amplitude) * sigma * torch.exp(1j * phase)
        probe = torch.fft.ifftn(probe_fft, dim=(-2, -1), norm="ortho").real
        residuals.append(difference - probe)
        scales.append(clean.abs().mean().clamp_min(1e-6))
    normalized = torch.stack([value / scale for value, scale in zip(residuals, scales)])
    return normalized.mean(0).abs().mean()


def gradient_robust_disruption(
    layers_adv: Sequence[torch.Tensor],
    layers_clean: Sequence[torch.Tensor],
    cfg: AttackConfig,
) -> torch.Tensor:
    """Official GRAT gradient-noise feature loss."""
    if not layers_adv or not layers_clean:
        raise ValueError("GRAT requires intermediate encoder features")
    pre_layer = int(len(layers_adv) * 0.9)
    pool = list(range(max(0, pre_layer - 6), pre_layer))
    count = min(4, len(pool))
    indices = random.sample(pool, count)
    adv = torch.stack([layers_adv[index] for index in indices])
    clean = torch.stack([layers_clean[index].detach() for index in indices])
    clean_mean = clean.abs().flatten(1).mean(1)
    view_shape = (len(indices),) + (1,) * (adv.ndim - 1)
    difference = (adv * 0.999 - clean) / clean_mean.view(view_shape).clamp_min(1e-12)
    noise = torch.randn_like(difference) * float(cfg.grat_noise_sd) + 1.0
    return (difference * noise).mean(0).abs().mean()


def spectral_uncertainty_separation(
    layers_adv: Sequence[torch.Tensor],
    layers_clean: Sequence[torch.Tensor],
    cfg: AttackConfig,
) -> torch.Tensor:
    """Certified separation in a clean-amplitude spectral uncertainty norm.

    If feature drift C belongs to ||C / sigma||_F <= rho, the reverse triangle
    inequality gives ||(D-C)/sigma||_F >= ||D/sigma||_F-rho. Maximizing the
    first term therefore maximizes a lower bound uniformly over the ellipsoid.
    """
    if not layers_adv or not layers_clean:
        raise ValueError("spectral uncertainty requires intermediate features")
    count = min(max(1, cfg.selected_layers), len(layers_adv), len(layers_clean))
    indices = torch.linspace(
        max(0, len(layers_adv) // 2), len(layers_adv) - 1, count
    ).round().long().tolist()
    terms = []
    for index in indices:
        clean = layers_clean[index].detach()
        difference = layers_adv[index] * 0.999 - clean
        clean_spectrum = torch.fft.fft2(clean, norm="ortho")
        difference_spectrum = torch.fft.fft2(difference, norm="ortho")
        amplitude = clean_spectrum.abs()
        sigma = amplitude / amplitude.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        sigma = sigma + float(cfg.uncertainty_floor)
        whitened = difference_spectrum.abs() / sigma
        clean_scale = clean.abs().mean().clamp_min(1e-6)
        terms.append(torch.sqrt(whitened.square().mean() + 1e-12) / clean_scale)
    return torch.stack(terms).mean()


def feature_phase_disruption(adv: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Amplitude-weighted cross-power phase disruption of spatial features."""
    adv_spectrum = torch.fft.fft2(adv.float(), norm="ortho")
    clean_spectrum = torch.fft.fft2(clean.detach().float(), norm="ortho")
    adv_amplitude = adv_spectrum.abs()
    clean_amplitude = clean_spectrum.abs()
    cross_real = (clean_spectrum.conj() * adv_spectrum).real
    cosine = cross_real / (clean_amplitude * adv_amplitude).clamp_min(1e-8)
    weight = torch.sqrt(clean_amplitude * adv_amplitude).detach()
    weight = weight / weight.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return (weight * (1.0 - cosine.clamp(-1.0, 1.0))).sum() / weight.sum().clamp_min(1e-8)


def spatial_quadrature_target(features: torch.Tensor) -> torch.Tensor:
    """Return a real feature map with preserved amplitude and quadrature phase.

    The transform is a channelwise 1-D Hilbert rotation along width. Positive
    and negative frequencies receive opposite 90-degree shifts so Hermitian
    symmetry, and therefore a real spatial target, is preserved. DC and the
    Nyquist frequency receive a 180-degree phase flip.
    """
    if features.ndim != 4:
        raise ValueError(f"expected BCHW features, got {tuple(features.shape)}")
    spectrum = torch.fft.fft(features.detach().float(), dim=-1, norm="ortho")
    width = features.shape[-1]
    frequency = torch.fft.fftfreq(width, device=features.device)
    multiplier = torch.empty(width, device=features.device, dtype=spectrum.dtype)
    multiplier[frequency > 0] = -1j
    multiplier[frequency < 0] = 1j
    multiplier[frequency == 0] = -1.0
    if width % 2 == 0:
        multiplier[width // 2] = -1.0
    target = torch.fft.ifft(
        spectrum * multiplier.view(1, 1, 1, width), dim=-1, norm="ortho"
    ).real
    return target.to(features.dtype)


def quadrature_collision_objective(adv: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Pull features toward an amplitude-preserving orthogonal-phase target."""
    target = spatial_quadrature_target(clean)
    scale = clean.detach().float().abs().mean().clamp_min(1e-8)
    return -(adv.float() - target.float()).abs().mean() / scale


def spatial_translation_target(
    features: torch.Tensor, shift: Tuple[int, int]
) -> torch.Tensor:
    """Translate BCHW features with an exact amplitude-preserving phase ramp."""
    if features.ndim != 4:
        raise ValueError(f"expected BCHW features, got {tuple(features.shape)}")
    if len(shift) != 2:
        raise ValueError(f"expected a two-axis shift, got {shift}")
    return torch.roll(
        features.detach(), shifts=(int(shift[0]), int(shift[1])), dims=(-2, -1)
    )


def translation_collision_objective(
    adv: torch.Tensor,
    clean: torch.Tensor,
    shift: Tuple[int, int],
) -> torch.Tensor:
    """Pull features toward a spatially misregistered, isospectral target."""
    target = spatial_translation_target(clean, shift)
    scale = clean.detach().float().abs().mean().clamp_min(1e-8)
    return -(adv.float() - target.float()).abs().mean() / scale


def spatial_feature_polar_disruption_bhwc(
    adv: torch.Tensor, clean: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return dimensionless spatial phase and amplitude displacement for BHWC tokens."""
    if adv.ndim != 4 or adv.shape != clean.shape:
        raise ValueError(f"expected matching BHWC features, got {adv.shape} and {clean.shape}")
    adv_bchw = adv.float().permute(0, 3, 1, 2)
    clean_bchw = clean.detach().float().permute(0, 3, 1, 2)
    adv_spectrum = torch.fft.fft2(adv_bchw, dim=(-2, -1), norm="ortho")
    clean_spectrum = torch.fft.fft2(clean_bchw, dim=(-2, -1), norm="ortho")
    adv_amplitude = adv_spectrum.abs()
    clean_amplitude = clean_spectrum.abs()

    cross_real = (clean_spectrum.conj() * adv_spectrum).real
    phase_cosine = cross_real / (clean_amplitude * adv_amplitude).clamp_min(1e-8)
    energy_weight = torch.sqrt(clean_amplitude * adv_amplitude).detach()
    phase = (
        energy_weight * (1.0 - phase_cosine.clamp(-1.0, 1.0))
    ).sum() / energy_weight.sum().clamp_min(1e-8)

    floor = 0.1 * clean_amplitude.mean(dim=(-2, -1), keepdim=True)
    relative_amplitude = (
        (adv_amplitude - clean_amplitude).abs()
        / (clean_amplitude + floor).clamp_min(1e-8)
    )
    amplitude_weight = torch.sqrt(clean_amplitude + floor).detach()
    amplitude = (
        amplitude_weight * relative_amplitude
    ).sum() / amplitude_weight.sum().clamp_min(1e-8)
    return phase.to(adv.dtype), amplitude.to(adv.dtype)


def depth_robust_disruption(
    layers_adv: Sequence[torch.Tensor],
    layers_clean: Sequence[torch.Tensor],
    count: int = 6,
    beta: float = 8.0,
) -> torch.Tensor:
    """Smooth lower bound on normalized feature displacement across depth."""
    available = min(len(layers_adv), len(layers_clean))
    if available == 0:
        raise ValueError("depth robust disruption requires captured layers")
    count = min(max(1, int(count)), available)
    indices = torch.linspace(0, available - 1, count).round().long().tolist()
    values = []
    for index in indices:
        clean = layers_clean[index].detach()
        scale = clean.abs().mean().clamp_min(1e-8)
        values.append(F.l1_loss(layers_adv[index] * 0.9999, clean) / scale)
    return smooth_minimum(values, beta)


def spatial_graph_collapse(
    layers_adv: Sequence[torch.Tensor],
    layers_clean: Sequence[torch.Tensor],
    count: int = 1,
) -> torch.Tensor:
    """Reduce spatial Dirichlet energy along the shared encoder path.

    Captured SAM block features are BHWC tensors.  For each selected depth we
    compute the L1 graph total variation over the 2-D token lattice and divide
    it by the clean energy.  The returned negative ratio is maximized by the
    attack, so a value approaching zero means that adjacent token features
    have collapsed toward a spatially constant representation.
    """
    available = min(len(layers_adv), len(layers_clean))
    if available == 0:
        raise ValueError("spatial graph collapse requires captured layers")
    count = min(max(1, int(count)), available)
    indices = torch.linspace(0, available - 1, count).round().long().tolist()

    ratios = []
    for index in indices:
        adv = layers_adv[index].float()
        clean = layers_clean[index].detach().float()
        if adv.ndim != 4 or adv.shape != clean.shape:
            raise ValueError(
                f"expected matching BHWC features, got {adv.shape} and {clean.shape}"
            )
        adv_energy = 0.5 * (
            (adv[:, 1:, :, :] - adv[:, :-1, :, :]).abs().mean()
            + (adv[:, :, 1:, :] - adv[:, :, :-1, :]).abs().mean()
        )
        clean_energy = 0.5 * (
            (clean[:, 1:, :, :] - clean[:, :-1, :, :]).abs().mean()
            + (clean[:, :, 1:, :] - clean[:, :, :-1, :]).abs().mean()
        )
        ratios.append(adv_energy / clean_energy.clamp_min(1e-8))
    return -torch.stack(ratios).mean()


def phase_scrambled_target(image: torch.Tensor, seed: int) -> torch.Tensor:
    """Construct a deterministic phase-incompatible target with source amplitude."""
    generator = torch.Generator(device=image.device).manual_seed(int(seed))
    source_spectrum = torch.fft.rfft2(image.float(), dim=(-2, -1), norm="ortho")
    noise = torch.randn(image.shape, device=image.device, dtype=image.dtype, generator=generator)
    noise_phase = torch.fft.rfft2(noise.float(), dim=(-2, -1), norm="ortho")
    unit_phase = noise_phase / noise_phase.abs().clamp_min(1e-8)
    target = torch.fft.irfft2(
        source_spectrum.abs() * unit_phase,
        s=image.shape[-2:], dim=(-2, -1), norm="ortho",
    )
    spatial_dims = (-2, -1)
    target = target - target.mean(spatial_dims, keepdim=True)
    target = target / target.std(spatial_dims, keepdim=True, unbiased=False).clamp_min(1e-6)
    target = target * image.std(spatial_dims, keepdim=True, unbiased=False)
    target = target + image.mean(spatial_dims, keepdim=True)
    return target.clamp(0.0, 1.0).to(image.dtype)


def relative_feature_distance(observed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.detach()
    scale = target.abs().mean().clamp_min(1e-8)
    relative_l1 = (observed - target).abs().mean() / scale
    angular = 1.0 - F.cosine_similarity(
        observed.flatten(1), target.flatten(1), dim=1, eps=1e-8
    ).mean()
    return relative_l1 + 0.5 * angular


def input_phase_disruption(
    adv: torch.Tensor,
    clean: torch.Tensor,
    low: float = 0.0,
    high: float = 1.0,
) -> torch.Tensor:
    """Amplitude-weighted input phase disruption inside a radial FFT band."""
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"expected 0 <= low < high <= 1, got {low}, {high}")
    adv_spectrum = torch.fft.fft2(adv.float(), norm="ortho")
    clean_spectrum = torch.fft.fft2(clean.detach().float(), norm="ortho")
    adv_amplitude = adv_spectrum.abs()
    clean_amplitude = clean_spectrum.abs()
    cross_real = (clean_spectrum.conj() * adv_spectrum).real
    cosine = cross_real / (clean_amplitude * adv_amplitude).clamp_min(1e-8)

    height, width = adv.shape[-2:]
    fy = torch.fft.fftfreq(height, device=adv.device, dtype=adv_spectrum.real.dtype)
    fx = torch.fft.fftfreq(width, device=adv.device, dtype=adv_spectrum.real.dtype)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square()) / math.sqrt(0.5)
    band = ((radius >= low) & (radius < high)).to(adv_spectrum.real.dtype)
    if high == 1.0:
        band = ((radius >= low) & (radius <= high + 1e-6)).to(adv_spectrum.real.dtype)

    weight = torch.sqrt(clean_amplitude * adv_amplitude).detach() * band
    weight = weight / weight.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return (weight * (1.0 - cosine.clamp(-1.0, 1.0))).sum() / weight.sum().clamp_min(1e-8)


def input_amplitude_disruption(adv: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Energy-aware relative radial displacement of the input spectrum."""
    adv_amplitude = torch.fft.fft2(adv.float(), norm="ortho").abs()
    clean_amplitude = torch.fft.fft2(clean.detach().float(), norm="ortho").abs()
    floor = 0.1 * clean_amplitude.mean(dim=(-2, -1), keepdim=True)
    relative = (adv_amplitude - clean_amplitude).abs() / (clean_amplitude + floor).clamp_min(1e-8)
    weight = torch.sqrt(clean_amplitude + floor).detach()
    return (weight * relative).sum() / weight.sum().clamp_min(1e-8)


def polar_amplitude_gradient(
    gradient: torch.Tensor,
    image: torch.Tensor,
    low: float = 0.0,
    high: float = 0.2,
) -> torch.Tensor:
    """Project an image gradient onto radial Fourier amplitude tangents.

    At each Fourier coefficient, the radial component is the orthogonal
    projection of the complex gradient onto the current image spectrum.  The
    inverse transform is real, so the result remains a valid image direction.
    """
    spectrum = torch.fft.fft2(image.detach().float(), norm="ortho")
    gradient_spectrum = torch.fft.fft2(gradient.float(), norm="ortho")
    energy = spectrum.abs().square()
    radial_coefficient = (gradient_spectrum.conj() * spectrum).real / energy.clamp_min(1e-8)
    radial = radial_coefficient * spectrum

    height, width = image.shape[-2:]
    fy = torch.fft.fftfreq(height, device=image.device, dtype=radial.real.dtype)
    fx = torch.fft.fftfreq(width, device=image.device, dtype=radial.real.dtype)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square()) / math.sqrt(0.5)
    if high >= 1.0:
        band = (radius >= low) & (radius <= high + 1e-6)
    else:
        band = (radius >= low) & (radius < high)
    radial = radial * band[None, None]
    return torch.fft.ifft2(radial, norm="ortho").real.to(gradient.dtype)


def polar_tangent_gradient(
    gradient: torch.Tensor,
    image: torch.Tensor,
    coordinate: str,
    low: float,
    high: float,
) -> torch.Tensor:
    """Project a real image gradient onto one polar FFT tangent subspace.

    The radial component changes Fourier amplitude to first order while fixing
    phase.  Its orthogonal complement is the tangential component, which changes
    phase to first order while fixing amplitude.  Conjugate symmetry of a real
    gradient makes the inverse transform real up to numerical precision.
    """
    if coordinate not in {"amplitude", "phase"}:
        raise ValueError(f"unknown polar coordinate: {coordinate}")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"expected 0 <= low < high <= 1, got {low}, {high}")
    spectrum = torch.fft.fft2(image.detach().float(), norm="ortho")
    gradient_spectrum = torch.fft.fft2(gradient.float(), norm="ortho")
    energy = spectrum.abs().square().clamp_min(1e-8)
    radial = (gradient_spectrum.conj() * spectrum).real / energy * spectrum
    selected = radial if coordinate == "amplitude" else gradient_spectrum - radial
    height, width = image.shape[-2:]
    fy = torch.fft.fftfreq(height, device=image.device, dtype=selected.real.dtype)
    fx = torch.fft.fftfreq(width, device=image.device, dtype=selected.real.dtype)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square()) / math.sqrt(0.5)
    band = (radius >= low) & (
        radius < high if high < 1.0 else radius <= high + 1e-6
    )
    selected = selected * band[None, None]
    return torch.fft.ifft2(selected, norm="ortho").real.to(gradient.dtype)


def amplitude_spectrum_view(image: torch.Tensor, log_multiplier: torch.Tensor) -> torch.Tensor:
    """Apply a differentiable amplitude-only view while preserving rFFT phase."""
    spectrum = torch.fft.rfft2(image.float(), norm="ortho")
    multiplier = log_multiplier.exp().to(spectrum.real.dtype)
    viewed = torch.fft.irfft2(
        spectrum * multiplier, s=image.shape[-2:], norm="ortho"
    )
    return viewed.to(image.dtype)


def block_shuffle_view(image: torch.Tensor, grid: int, seed: int) -> torch.Tensor:
    """Apply a deterministic permutation of equal spatial blocks."""
    grid = max(1, int(grid))
    height, width = image.shape[-2:]
    if height % grid or width % grid:
        raise ValueError(f"image size {(height, width)} must be divisible by block grid {grid}")
    batch, channels = image.shape[:2]
    tile_h, tile_w = height // grid, width // grid
    tiles = image.reshape(batch, channels, grid, tile_h, grid, tile_w)
    tiles = tiles.permute(0, 2, 4, 1, 3, 5).reshape(batch, grid * grid, channels, tile_h, tile_w)
    generator = torch.Generator(device=image.device).manual_seed(int(seed))
    order = torch.randperm(grid * grid, generator=generator, device=image.device)
    shuffled = tiles[:, order]
    return shuffled.reshape(batch, grid, grid, channels, tile_h, tile_w).permute(
        0, 3, 1, 4, 2, 5
    ).reshape(batch, channels, height, width)


def bsr_shuffle_rotate_view(image: torch.Tensor, seed: int) -> torch.Tensor:
    """Repository BSR transform with deterministic unequal 2x2 blocks."""
    height, width = image.shape[-2:]
    generator = torch.Generator(device=image.device).manual_seed(int(seed))
    cut_h = int(torch.randint(max(1, height // 4), max(2, 3 * height // 4), (1,), generator=generator, device=image.device))
    cut_w = int(torch.randint(max(1, width // 4), max(2, 3 * width // 4), (1,), generator=generator, device=image.device))
    row_order = torch.randperm(2, generator=generator, device=image.device).tolist()
    column_order = torch.randperm(2, generator=generator, device=image.device).tolist()
    angle = int(torch.round(torch.randn((), generator=generator, device=image.device) * 2.0).item())
    rows = image.split((cut_h, height - cut_h), dim=2)
    output_rows = []
    for row_index in row_order:
        blocks = rows[row_index].split((cut_w, width - cut_w), dim=3)
        output_rows.append(torch.cat([
            TF.rotate(blocks[column_index], angle, interpolation=InterpolationMode.BILINEAR)
            for column_index in column_order
        ], dim=3))
    return torch.cat(output_rows, dim=2)


def build_amplitude_eot_multipliers(image: torch.Tensor, cfg: AttackConfig) -> List[torch.Tensor]:
    """Create fixed smooth log-amplitude fields for reproducible EOT views."""
    count = max(0, int(cfg.amplitude_eot_samples))
    if count == 0:
        return []
    height, half_width = image.shape[-2], image.shape[-1] // 2 + 1
    generator = torch.Generator(device=image.device).manual_seed(cfg.seed + 41000)
    fields = []
    for _ in range(count):
        field = torch.randn(
            (1, 1, height, half_width), device=image.device,
            dtype=image.dtype, generator=generator,
        )
        field = F.avg_pool2d(field, kernel_size=7, stride=1, padding=3)
        field = field / field.std(unbiased=False).clamp_min(1e-6)
        fields.append(field * float(cfg.amplitude_eot_scale))
    return fields


def monotone_contrast_view(image: torch.Tensor, center: torch.Tensor,
                           scale: float) -> torch.Tensor:
    """Change CT window width while preserving the order of unsaturated voxels."""
    if scale <= 0:
        raise ValueError("contrast scale must be positive")
    return (center + float(scale) * (image - center)).clamp(0.0, 1.0)


def patch_embedding_disruption(adv: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Normalized displacement in the patch-token space shared by PEFT descendants."""
    scale = clean.detach().abs().mean().clamp_min(1e-8)
    relative_l1 = (adv - clean.detach()).abs().mean() / scale
    cosine = 1.0 - F.cosine_similarity(adv.flatten(1), clean.detach().flatten(1), dim=1).mean()
    return relative_l1 + cosine


def patch_boundary_disruption(adv: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Disrupt spatial adjacency in the frozen patch embedding shared by PEFT.

    SAM patch embeddings are BHWC. The angular term increases adjacent-token
    disagreement, while the high-pass displacement term rewards spatially
    alternating changes instead of a removable common feature shift.
    """
    if adv.ndim != 4 or adv.shape[-1] != clean.shape[-1]:
        raise ValueError(f"expected matching BHWC patch embeddings, got {adv.shape}, {clean.shape}")
    adv_normalized = F.normalize(adv.float(), dim=-1, eps=1e-8)
    horizontal = (adv_normalized[:, :, 1:] * adv_normalized[:, :, :-1]).sum(-1)
    vertical = (adv_normalized[:, 1:, :] * adv_normalized[:, :-1, :]).sum(-1)
    angular_boundary = 1.0 - 0.5 * (horizontal.mean() + vertical.mean())
    displacement = adv.float() - clean.detach().float()
    scale = clean.detach().float().abs().mean().clamp_min(1e-8)
    horizontal_change = (displacement[:, :, 1:] - displacement[:, :, :-1]).abs().mean()
    vertical_change = (displacement[:, 1:, :] - displacement[:, :-1, :]).abs().mean()
    high_pass = 0.5 * (horizontal_change + vertical_change) / scale
    return angular_boundary + 0.5 * high_pass


def token_collapse_objective(adv: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Minimize spatial token variance without using a prompt or decoder."""
    spatial_dims = tuple(range(2, adv.ndim))
    adv_centered = adv - adv.mean(dim=spatial_dims, keepdim=True)
    clean_centered = clean.detach() - clean.detach().mean(dim=spatial_dims, keepdim=True)
    adv_scale = torch.sqrt(adv_centered.square().mean() + 1e-12)
    clean_scale = torch.sqrt(clean_centered.square().mean() + 1e-12)
    return -adv_scale / clean_scale.clamp_min(1e-8)


def dc_feature_shift(adv: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Displace the spatial DC amplitude of the final image embedding.

    The spatial mean is exactly the zero-frequency Fourier coefficient up to
    FFT normalization. Channelwise normalization prevents a few high-energy
    channels from dominating the shared context shift.
    """
    if adv.ndim != 4 or adv.shape != clean.shape:
        raise ValueError(f"expected matching BCHW features, got {adv.shape}, {clean.shape}")
    adv_dc = adv.float().mean(dim=(-2, -1))
    clean_dc = clean.detach().float().mean(dim=(-2, -1))
    scale = clean.detach().float().square().mean(dim=(-2, -1)).sqrt().clamp_min(1e-6)
    return ((adv_dc - clean_dc) / scale).square().mean().sqrt()


def low_level_activation_lift(
    layers_adv: Sequence[torch.Tensor],
    layers_clean: Sequence[torch.Tensor],
    count: int = 2,
) -> torch.Tensor:
    """Lift energy in early representations that are least changed by fine tuning.

    Parseval makes this exactly an amplitude-spectrum energy objective. Normalizing
    by clean energy prevents a single layer's scale from determining the update.
    """
    count = min(max(int(count), 1), len(layers_adv), len(layers_clean))
    if count == 0:
        raise ValueError("low-level lifting requires intermediate encoder features")
    values = []
    for adv, clean in zip(layers_adv[:count], layers_clean[:count]):
        clean_rms = torch.sqrt(clean.detach().float().square().mean() + 1e-12)
        adv_rms = torch.sqrt(adv.float().square().mean() + 1e-12)
        values.append(adv_rms / clean_rms.clamp_min(1e-8))
    return torch.stack(values).mean()


def phase_amplitude_stochastic_disruption(
    layers_adv: Sequence[torch.Tensor],
    layers_clean: Sequence[torch.Tensor],
    cfg: AttackConfig,
) -> torch.Tensor:
    """Exact PASD probe loss used by the repository's prior strong experiments."""
    if not layers_adv or not layers_clean:
        raise ValueError("PASD requires intermediate encoder features")
    if cfg.pasd_region == "dual":
        late = phase_amplitude_stochastic_disruption(
            layers_adv, layers_clean, replace(cfg, pasd_region="late", pasd_early_weight=0.0)
        )
        early = phase_amplitude_stochastic_disruption(
            layers_adv, layers_clean,
            replace(cfg, pasd_region="early", pasd_layer_count=2, pasd_early_weight=0.0),
        )
        return late + float(cfg.pasd_early_weight) * early
    if cfg.pasd_region == "late_final":
        late = phase_amplitude_stochastic_disruption(
            layers_adv, layers_clean, replace(cfg, pasd_region="late", pasd_final_weight=0.0)
        )
        final = phase_amplitude_stochastic_disruption(
            layers_adv, layers_clean,
            replace(cfg, pasd_region="final", pasd_layer_count=2, pasd_final_weight=0.0),
        )
        return late + float(cfg.pasd_final_weight) * final
    pre_layer = int(len(layers_adv) * 0.9)
    count = min(max(1, int(cfg.pasd_layer_count)), pre_layer)
    if cfg.pasd_region == "late":
        indices = list(range(pre_layer - count, pre_layer))
    elif cfg.pasd_region == "early":
        indices = list(range(count))
    elif cfg.pasd_region == "mixed":
        early_count = count // 2
        late_count = count - early_count
        indices = list(range(early_count)) + list(range(pre_layer - late_count, pre_layer))
    elif cfg.pasd_region == "final":
        count = min(max(1, int(cfg.pasd_layer_count)), len(layers_adv))
        indices = list(range(len(layers_adv) - count, len(layers_adv)))
    else:
        raise ValueError(f"unknown PASD layer region {cfg.pasd_region!r}")
    sample_losses = []
    for _ in range(max(1, int(cfg.pasd_samples))):
        amplitude_grid = (
            tuple(float(alpha) for alpha in cfg.pasd_amplitude_grid)
            if cfg.pasd_mode == "shuffled_amplitude_hull"
            else (None,)
        )
        if not amplitude_grid:
            raise ValueError("PASD amplitude hull requires at least one grid point")
        if any(alpha is not None and not 0.0 <= alpha <= 1.0 for alpha in amplitude_grid):
            raise ValueError("PASD amplitude grid points must lie in [0,1]")
        grid_residuals = [[] for _ in amplitude_grid]
        clean_values = []
        for index in indices:
            clean = layers_clean[index].detach()
            adv = layers_adv[index]
            if cfg.pasd_mode.startswith("spatial_"):
                probes = [spatial_fourier_probe_bhwc(clean, cfg.pasd_mode)]
            else:
                spectrum = torch.fft.fftn(clean, dim=(-2, -1), norm="ortho")
                amplitude = spectrum.abs()
                phase = spectrum.angle()
                if cfg.pasd_mode.startswith("shuffled"):
                    order = torch.randperm(phase.numel(), device=phase.device)
                    phase = phase.reshape(-1)[order].reshape_as(phase)
                elif cfg.pasd_mode != "clean_amplitude":
                    raise ValueError(f"unknown PASD mode {cfg.pasd_mode!r}")
                gaussian = torch.randn_like(amplitude)
                probes = []
                for alpha in amplitude_grid:
                    if alpha is None:
                        probe_amplitude = (
                            torch.ones_like(amplitude)
                            if cfg.pasd_mode == "shuffled_white" else amplitude
                        )
                    else:
                        probe_amplitude = torch.lerp(
                            torch.ones_like(amplitude), amplitude, alpha
                        )
                    probe_spectrum = gaussian * probe_amplitude * torch.exp(1j * phase)
                    probes.append(torch.fft.ifftn(
                        probe_spectrum, dim=(-2, -1), norm="ortho"
                    ).real)
            difference = adv * 0.999 - clean
            for residuals, probe in zip(grid_residuals, probes):
                residuals.append(difference - float(cfg.pasd_probe_scale) * probe)
            clean_values.append(clean)
        clean_stack = torch.stack(clean_values)
        scales = clean_stack.abs().flatten(1).mean(1)
        grid_losses = []
        for residuals in grid_residuals:
            residual = torch.stack(residuals)
            view_shape = (len(indices),) + (1,) * (residual.ndim - 1)
            normalized = residual / scales.view(view_shape).clamp_min(1e-6)
            grid_losses.append(normalized.mean(0).abs().mean())
        if len(grid_losses) == 1:
            sample_losses.append(grid_losses[0])
        else:
            sample_losses.append(smooth_minimum(grid_losses, cfg.pasd_softmin_beta))
    sample_losses = torch.stack(sample_losses)
    if cfg.pasd_reduce == "mean":
        return sample_losses.mean()
    if cfg.pasd_reduce == "softmin":
        beta = max(float(cfg.pasd_softmin_beta), 1e-6)
        return -torch.logsumexp(-beta * sample_losses, dim=0) / beta + math.log(len(sample_losses)) / beta
    raise ValueError(f"unknown PASD sample reduction {cfg.pasd_reduce!r}")


def smooth_minimum(values: Sequence[torch.Tensor], beta: float) -> torch.Tensor:
    """A normalized differentiable minimum that preserves equal inputs."""
    if not values:
        raise ValueError("smooth minimum requires at least one value")
    stacked = torch.stack(tuple(values))
    beta = max(float(beta), 1e-6)
    return -torch.logsumexp(-beta * stacked, dim=0) / beta + math.log(len(values)) / beta


def spatial_fourier_probe_bhwc(clean: torch.Tensor, mode: str) -> torch.Tensor:
    """Sample a channelwise probe in the spatial Fourier basis of BHWC features.

    ``spatial_uniform_white`` has equal expected power at every spatial
    frequency and independent random phase. ``spatial_shuffled_white`` keeps
    the empirical phase marginal of each channel while removing its spatial
    placement. Both avoid the legacy W-by-C transform, which mixes semantic
    channels into the Fourier axes.
    """
    if clean.ndim != 4:
        raise ValueError(f"spatial PASD expects BHWC features, got {tuple(clean.shape)}")
    height, width = clean.shape[1:3]
    spectrum = torch.fft.rfft2(
        clean.detach().permute(0, 3, 1, 2), dim=(-2, -1), norm="ortho"
    )
    phase = spectrum.angle()
    if mode == "spatial_uniform_white":
        phase = (2.0 * torch.rand_like(phase) - 1.0) * math.pi
    elif mode == "spatial_shuffled_white":
        flat = phase.flatten(-2)
        orders = torch.rand(flat.shape, device=flat.device).argsort(dim=-1)
        phase = torch.gather(flat, -1, orders).reshape_as(phase)
    else:
        raise ValueError(f"unknown spatial PASD mode {mode!r}")
    coefficients = torch.randn_like(phase) * torch.exp(1j * phase)
    probe = torch.fft.irfft2(
        coefficients, s=(height, width), dim=(-2, -1), norm="ortho"
    )
    return probe.permute(0, 2, 3, 1)


def average_token_angular_disruption(
    layers_adv: Sequence[torch.Tensor],
    layers_clean: Sequence[torch.Tensor],
    region: str = "late",
    count: int = 6,
) -> torch.Tensor:
    """Negative average token cosine similarity on selected ViT blocks."""
    if not layers_adv or not layers_clean:
        raise ValueError("ATCS requires intermediate encoder features")
    pre_layer = int(len(layers_adv) * 0.9)
    count = min(max(1, int(count)), pre_layer)
    if region == "late":
        indices = list(range(pre_layer - count, pre_layer))
    elif region == "early":
        indices = list(range(count))
    elif region == "mixed":
        early_count = count // 2
        indices = list(range(early_count)) + list(range(pre_layer - (count - early_count), pre_layer))
    elif region == "final":
        count = min(max(1, int(count)), len(layers_adv))
        indices = list(range(len(layers_adv) - count, len(layers_adv)))
    else:
        raise ValueError(f"unknown ATCS layer region {region!r}")
    values = []
    for index in indices:
        adv = layers_adv[index].float()
        clean = layers_clean[index].detach().float()
        values.append(-F.cosine_similarity(adv, clean, dim=-1, eps=1e-8).mean())
    return torch.stack(values).mean()


def normalized_token_gram_disruption(
    layers_adv: Sequence[torch.Tensor],
    layers_clean: Sequence[torch.Tensor],
    region: str = "late",
    count: int = 6,
    pool_size: int = 16,
) -> torch.Tensor:
    """Displace centered spatial token correlations across selected layers.

    SAM block captures are BHWC.  Pooling bounds memory, after which each
    layer is represented by the Frobenius-normalized token Gram matrix
    ``ZZ^T``.  This representation is invariant to a shared orthogonal change
    of channel basis and depends on spatial cross-correlation rather than an
    arbitrarily selected Fourier band.
    """
    if not layers_adv or not layers_clean:
        raise ValueError("token Gram disruption requires intermediate features")
    pre_layer = int(len(layers_adv) * 0.9)
    count = min(max(1, int(count)), pre_layer)
    if region == "late":
        indices = list(range(pre_layer - count, pre_layer))
    elif region == "early":
        indices = list(range(count))
    elif region == "mixed":
        early_count = count // 2
        indices = list(range(early_count)) + list(range(pre_layer - (count - early_count), pre_layer))
    elif region == "final":
        count = min(max(1, int(count)), len(layers_adv))
        indices = list(range(len(layers_adv) - count, len(layers_adv)))
    else:
        raise ValueError(f"unknown token Gram layer region {region!r}")

    values = []
    size = max(1, int(pool_size))
    for index in indices:
        adv = layers_adv[index].float().permute(0, 3, 1, 2)
        clean = layers_clean[index].detach().float().permute(0, 3, 1, 2)
        target_size = (min(size, adv.shape[-2]), min(size, adv.shape[-1]))
        adv = F.adaptive_avg_pool2d(adv, target_size).flatten(2).transpose(1, 2)
        clean = F.adaptive_avg_pool2d(clean, target_size).flatten(2).transpose(1, 2)
        adv = adv - adv.mean(dim=1, keepdim=True)
        clean = clean - clean.mean(dim=1, keepdim=True)
        adv_gram = adv @ adv.transpose(1, 2)
        clean_gram = clean @ clean.transpose(1, 2)
        adv_gram = adv_gram / adv_gram.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
        clean_gram = clean_gram / clean_gram.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
        squared = (adv_gram - clean_gram).square().mean()
        values.append(torch.sqrt(squared + 1e-12) - 1e-6)
    return torch.stack(values).mean()


def gradient_consensus(
    gradients: Sequence[torch.Tensor], floor: float = 0.25, weights: Optional[Sequence[float]] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not gradients:
        raise ValueError("at least one gradient is required")
    normalized = [g / g.abs().mean().clamp_min(1e-12) for g in gradients]
    stack = torch.stack(normalized)
    if weights is None:
        weight_tensor = torch.ones(len(normalized), device=stack.device, dtype=stack.dtype)
    else:
        if len(weights) != len(normalized):
            raise ValueError("gradient and weight counts differ")
        weight_tensor = torch.as_tensor(weights, device=stack.device, dtype=stack.dtype)
    weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-12)
    view_shape = (len(normalized),) + (1,) * gradients[0].ndim
    weighted = weight_tensor.view(view_shape)
    mean = (stack * weighted).sum(0)
    agreement = (stack.sign() * weighted).sum(0).abs()
    weight = float(floor) + (1.0 - float(floor)) * agreement
    consensus = mean * weight
    diagnostics = {
        "sign_agreement": float(agreement.mean().detach()),
        "gradient_std": float(stack.std(0, unbiased=False).mean().detach()),
    }
    return consensus, diagnostics


def _radial_band_mask(height: int, width: int, low: float, high: float, device) -> torch.Tensor:
    fy = torch.fft.fftfreq(height, device=device).view(height, 1)
    fx = torch.fft.rfftfreq(width, device=device).view(1, width // 2 + 1)
    radius = torch.sqrt(fx.square() + fy.square()) / math.sqrt(0.5**2 + 0.5**2)
    ramp = 0.025
    lo = torch.sigmoid((radius - low) / ramp)
    hi = torch.sigmoid((high - radius) / ramp)
    return lo * hi


def spectral_consensus(gradient: torch.Tensor, cfg: AttackConfig) -> torch.Tensor:
    spectrum = torch.fft.rfft2(gradient, norm="ortho")
    mask = _radial_band_mask(
        gradient.shape[-2], gradient.shape[-1], cfg.spectral_low, cfg.spectral_high, gradient.device
    )
    band = torch.fft.irfft2(spectrum * mask, s=gradient.shape[-2:], norm="ortho")
    mixed = (1.0 - cfg.spectral_weight) * gradient + cfg.spectral_weight * band
    if cfg.ti_kernel > 1:
        k = int(cfg.ti_kernel) | 1
        coords = torch.arange(k, device=gradient.device, dtype=gradient.dtype) - k // 2
        sigma = max(k / 6.0, 0.5)
        one_d = torch.exp(-coords.square() / (2.0 * sigma * sigma))
        kernel = torch.outer(one_d, one_d)
        kernel = (kernel / kernel.sum()).view(1, 1, k, k).repeat(gradient.shape[1], 1, 1, 1)
        mixed = F.conv2d(mixed, kernel, padding=k // 2, groups=gradient.shape[1])
    return mixed


def cross_spectral_anchor(
    anchor: torch.Tensor,
    auxiliary: torch.Tensor,
    floor: float = 0.5,
    auxiliary_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Fuse an auxiliary gradient without erasing the anchor cross-spectrum.

    For every Fourier coefficient, the returned direction ``G`` satisfies
    ``Re(conj(F) * G) >= floor * |F|^2``, where ``F`` is the normalized anchor.
    This retains a prescribed fraction of the anchor's first-order gain while
    admitting phase-compatible and bounded phase-conflicting ADS components.
    """
    floor = min(max(float(floor), 0.0), 1.0)
    weight = max(float(auxiliary_weight), 0.0)
    anchor = anchor / anchor.abs().mean().clamp_min(1e-12)
    auxiliary = auxiliary / auxiliary.abs().mean().clamp_min(1e-12)
    anchor_fft = torch.fft.fft2(anchor, norm="ortho")
    auxiliary_fft = torch.fft.fft2(auxiliary, norm="ortho")
    cross = (anchor_fft.conj() * auxiliary_fft).real
    scale = torch.ones_like(cross)
    negative = cross < 0
    if weight > 0:
        cap = (1.0 - floor) * anchor_fft.abs().square()
        cap = cap / (-weight * cross).clamp_min(1e-12)
        scale[negative] = cap[negative].clamp(max=1.0)
    fused_fft = anchor_fft + weight * scale * auxiliary_fft
    fused = torch.fft.ifft2(fused_fft, norm="ortho").real
    achieved = (anchor_fft.conj() * fused_fft).real
    anchor_energy = anchor_fft.abs().square()
    ratio = achieved / anchor_energy.clamp_min(1e-12)
    supported = anchor_energy > 1e-12
    minimum_ratio = ratio[supported].min() if supported.any() else ratio.new_tensor(1.0)
    diagnostics = {
        "phase_conflict_fraction": float(negative.float().mean().detach()),
        "clipped_fraction": float((scale < 1.0).float().mean().detach()),
        "minimum_anchor_ratio": float(minimum_ratio.detach()),
    }
    return fused, diagnostics


def phase_interval_robust_direction(
    gradients: Sequence[torch.Tensor],
    beta: float = 8.0,
    reliability_floor: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Build a deterministic Fourier direction supported by all paths."""
    if not gradients:
        raise ValueError("at least one structural gradient is required")
    normalized = [value / value.abs().mean().clamp_min(1e-12) for value in gradients]
    stack = torch.stack(normalized)
    initial = stack.mean(0)
    flat_initial = initial.flatten(1)
    gains = torch.stack([
        F.cosine_similarity(value.flatten(1), flat_initial, dim=1).mean()
        for value in normalized
    ])
    weights = torch.softmax(-max(float(beta), 0.0) * gains, dim=0)
    spectra = torch.fft.fft2(stack, norm="ortho")
    view_shape = (len(normalized),) + (1,) * (spectra.ndim - 1)
    expanded_weights = weights.view(view_shape)
    center = (expanded_weights * spectra).sum(0)
    center_amplitude = center.abs()
    projections = (spectra.conj() * center).real
    projection_ratio = projections / (
        spectra.abs() * center_amplitude
    ).clamp_min(1e-12)
    supported = spectra.abs() > 1e-8
    projection_ratio = torch.where(supported, projection_ratio, torch.ones_like(projection_ratio))
    weakest_phase_gain = projection_ratio.amin(0).clamp(min=0.0, max=1.0)
    floor = min(max(float(reliability_floor), 0.0), 1.0)
    reliability = floor + (1.0 - floor) * weakest_phase_gain
    robust_spectrum = center * reliability
    robust = torch.fft.ifft2(robust_spectrum, norm="ortho").real
    robust = robust / robust.abs().mean().clamp_min(1e-12)
    robust_gains = torch.stack([
        F.cosine_similarity(value.flatten(1), robust.flatten(1), dim=1).mean()
        for value in normalized
    ])
    concentration = center_amplitude / (
        (expanded_weights * spectra.abs()).sum(0).clamp_min(1e-12)
    )
    diagnostics = {
        "phase_concentration": float(concentration.mean().detach()),
        "phase_rejected_fraction": float((weakest_phase_gain <= 0).float().mean().detach()),
        "min_path_cosine": float(robust_gains.min().detach()),
        "softmin_weight_max": float(weights.max().detach()),
    }
    return robust, diagnostics


def _edge_map(image: torch.Tensor) -> torch.Tensor:
    dx = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs()
    dy = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs()
    edge = F.pad(dx, (0, 1, 0, 0)) + F.pad(dy, (0, 0, 0, 1))
    minimum = edge.flatten(1).min(1).values.view(-1, 1, 1, 1)
    maximum = edge.flatten(1).max(1).values.view(-1, 1, 1, 1)
    return (edge - minimum) / (maximum - minimum + 1e-6)


def _asymmetric_step(momentum: torch.Tensor, step_size: float) -> torch.Tensor:
    update = momentum * (step_size / momentum.abs().mean().clamp_min(1e-12))
    positive = (momentum > 0).to(momentum.dtype)
    negative_step = torch.clamp(update, -step_size, -0.25 * step_size)
    positive_step = torch.clamp(update, 0.25 * step_size, step_size)
    return negative_step * (1.0 - positive) + positive_step * positive


def translation_orbit_average(gradient: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Average inverse-shifted gradients over a square translation orbit."""
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return gradient
    if kernel_size % 2 == 0:
        raise ValueError("translation orbit kernel must be odd")
    channels = gradient.shape[1]
    kernel = torch.full(
        (channels, 1, kernel_size, kernel_size),
        1.0 / (kernel_size * kernel_size),
        device=gradient.device, dtype=gradient.dtype,
    )
    return F.conv2d(
        gradient, kernel, padding=kernel_size // 2, groups=channels
    )


def build_virtual_views(encoder: nn.Module, cfg: AttackConfig) -> List[VirtualPEFTView]:
    if cfg.adapter_view_count > 0:
        return [VirtualPEFTView(encoder, "source", 1, 0.0, cfg.seed)] + [
            VirtualPEFTView(
                encoder, "adapter_exact", cfg.adapter_view_rank,
                cfg.adapter_view_scale, cfg.seed + 30000 + index,
            )
            for index in range(cfg.adapter_view_count)
        ]
    ranks = tuple(cfg.ranks) or (4,)
    views = [VirtualPEFTView(encoder, "source", ranks[0], 0.0, cfg.seed)]
    for index, rank in enumerate(ranks):
        views.append(VirtualPEFTView(encoder, "lora", rank, cfg.virtual_scale, cfg.seed + 10000 + index))
        views.append(VirtualPEFTView(encoder, "adapter", rank, cfg.virtual_scale, cfg.seed + 20000 + index))
    return views


def attack_image(
    encoder: nn.Module,
    image: torch.Tensor,
    cfg: Optional[AttackConfig] = None,
    preprocess=None,
    initial_momentum: Optional[torch.Tensor] = None,
    initial_delta: Optional[torch.Tensor] = None,
    first_step_multiplier: float = 2.5,
    gradient_sink: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """Attack one ``[1,C,H,W]`` image using the source encoder only."""
    cfg = cfg or AttackConfig()
    if cfg.use_dual_path:
        if len(cfg.skip_gammas) != len(cfg.skip_weights) or len(cfg.skip_gammas) != 2:
            raise ValueError("dual-path attack requires exactly two gammas and weights")
        path_outputs, path_traces, path_sinks = [], [], []
        for gamma in cfg.skip_gammas:
            path_cfg = replace(
                cfg, use_dual_path=False, use_skip_ensemble=False,
                use_skip_gradient=True, skip_gamma=float(gamma),
            )
            path_sink: Dict[str, torch.Tensor] = {}
            path_adv, path_trace = attack_image(
                encoder, image, path_cfg, preprocess=preprocess,
                initial_momentum=initial_momentum, initial_delta=initial_delta,
                first_step_multiplier=first_step_multiplier, gradient_sink=path_sink,
            )
            path_outputs.append(path_adv)
            path_traces.append(path_trace)
            path_sinks.append(path_sink)
        weights = torch.as_tensor(cfg.skip_weights, device=image.device, dtype=image.dtype)
        weights = weights / weights.sum().clamp_min(1e-12)
        directions = [
            (value - image) / (value - image).abs().mean().clamp_min(1e-12)
            for value in path_outputs
        ]
        fused_direction = sum(weight * value for weight, value in zip(weights, directions))
        fused = (image + cfg.epsilon * fused_direction.sign()).clamp(0.0, 1.0)
        if gradient_sink is not None:
            for key in ("source", "final"):
                values = [sink[key].to(image.device) for sink in path_sinks]
                normalized = [value / value.abs().mean().clamp_min(1e-12) for value in values]
                gradient_sink[key] = sum(
                    weight * value for weight, value in zip(weights, normalized)
                ).cpu()
        trace = path_traces[-1]
        trace["dual_path_delta_cosine"] = [float(F.cosine_similarity(
            directions[0].flatten(1), directions[1].flatten(1), dim=1
        ).mean().detach())]
        return fused.detach(), trace
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"image must have shape [1,C,H,W], got {tuple(image.shape)}")
    preprocess = preprocess or (lambda value: value)
    if cfg.use_dual_preprocess:
        primary_preprocess = preprocess
        sam_mean = image.new_tensor((123.675, 116.28, 103.53)).view(1, 3, 1, 1)
        sam_std = image.new_tensor((58.395, 57.12, 57.375)).view(1, 3, 1, 1)

        def preprocess(value):
            primary = primary_preprocess(value)
            sam_coordinates = (value * 255.0 - sam_mean) / sam_std
            return torch.cat((primary, sam_coordinates), dim=0)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    if image.is_cuda:
        torch.cuda.manual_seed_all(cfg.seed)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()

    if cfg.use_peft_consensus:
        views = build_virtual_views(encoder, cfg)
    elif cfg.use_amplitude_eot or cfg.use_block_shuffle_eot or cfg.use_contrast_eot:
        sample_count = (
            int(cfg.amplitude_eot_samples) if cfg.use_amplitude_eot
            else int(cfg.block_shuffle_samples) if cfg.use_block_shuffle_eot
            else len(cfg.contrast_eot_scales)
        )
        views = [
            VirtualPEFTView(encoder, "source", (cfg.ranks or (4,))[0], 0.0, cfg.seed + index)
            for index in range(max(1, sample_count + 1))
        ]
    else:
        views = [VirtualPEFTView(encoder, "source", (cfg.ranks or (4,))[0], 0.0, cfg.seed)]
    amplitude_fields = build_amplitude_eot_multipliers(image, cfg) if cfg.use_amplitude_eot else []
    contrast_center = image.detach().median() if cfg.use_contrast_eot else None
    current_patchout_mask: Optional[torch.Tensor] = None

    def input_view(value: torch.Tensor, view_index: int) -> torch.Tensor:
        viewed = value
        if current_patchout_mask is not None:
            viewed = image + current_patchout_mask * (viewed - image)
        if cfg.use_amplitude_eot and view_index > 0:
            viewed = amplitude_spectrum_view(viewed, amplitude_fields[view_index - 1])
        if cfg.use_contrast_eot and view_index > 0:
            viewed = monotone_contrast_view(
                viewed, contrast_center, cfg.contrast_eot_scales[view_index - 1]
            )
        if cfg.use_block_shuffle_eot and view_index > 0:
            view_seed = cfg.seed + 50000 + view_index
            if cfg.block_shuffle_mode == "grid":
                viewed = block_shuffle_view(viewed, cfg.block_shuffle_grid, view_seed)
            elif cfg.block_shuffle_mode == "bsr":
                viewed = bsr_shuffle_rotate_view(viewed, view_seed)
            else:
                raise ValueError(f"unknown block shuffle mode {cfg.block_shuffle_mode!r}")
        return viewed
    clean_refs = []
    collision_refs = []
    ilpd_clean_refs = []
    clean_patch = None
    clean_adapter_sites: Optional[Dict[Tuple[int, str], torch.Tensor]] = None
    clean_qk_site: Optional[torch.Tensor] = None
    uses_site_capture = cfg.use_adapter_tail or cfg.use_frequency_rank_tail
    adapter_layers = _adapter_layer_indices(encoder, cfg) if uses_site_capture else ()
    with torch.no_grad():
        collision_target = (
            phase_scrambled_target(image, cfg.seed + 61000)
            if cfg.use_phase_collision else None
        )
        if cfg.use_patch_loss or cfg.use_patch_boundary:
            clean_patch = _tensor_output(encoder.patch_embed(preprocess(image))).detach()
        for view_index, view in enumerate(views):
            with view.active():
                captured_ilpd = []
                ilpd_handle = None
                if cfg.use_ilpd:
                    if not 0 <= int(cfg.ilpd_layer) < len(encoder.blocks):
                        raise ValueError(f"ILPD layer {cfg.ilpd_layer} outside encoder depth")
                    ilpd_handle = encoder.blocks[int(cfg.ilpd_layer)].register_forward_hook(
                        lambda _module, _inputs, output: captured_ilpd.append(_tensor_output(output).detach())
                    )
                adapter_capture = None
                qk_capture = None
                with contextlib.ExitStack() as stack:
                    if view_index == 0 and uses_site_capture:
                        adapter_capture = AdapterSiteCapture(
                            encoder, adapter_layers, cfg.adapter_tail_sites
                        )
                        stack.enter_context(adapter_capture.active())
                    if view_index == 0 and cfg.use_qk_topology:
                        qk_capture = QKSiteCapture(encoder, cfg.qk_topology_layer)
                        stack.enter_context(qk_capture.active())
                    clean_refs.append(_unpack_encoder(
                        encoder(preprocess(input_view(image, view_index)))
                    ))
                if adapter_capture is not None:
                    clean_adapter_sites = dict(adapter_capture.outputs)
                if qk_capture is not None:
                    if qk_capture.output is None:
                        raise RuntimeError("clean QK site was not captured")
                    clean_qk_site = qk_capture.output.detach()
                if collision_target is not None:
                    collision_refs.append(_unpack_encoder(
                        encoder(preprocess(input_view(collision_target, view_index)))
                    ))
                if ilpd_handle is not None:
                    ilpd_handle.remove()
                    ilpd_clean_refs.append(captured_ilpd[-1])
                else:
                    ilpd_clean_refs.append(None)
    clean_spatial_bases = (
        [spatial_principal_basis(final, cfg.spatial_principal_rank) for final, _ in clean_refs]
        if cfg.use_spatial_principal else []
    )
    clean_semantic_partitions = (
        [semantic_partition(final, cfg.semantic_partition_clusters) for final, _ in clean_refs]
        if cfg.use_semantic_partition else []
    )

    generator = torch.Generator(device=image.device).manual_seed(cfg.seed)
    if cfg.random_start:
        delta = torch.empty_like(image).uniform_(-cfg.epsilon, cfg.epsilon, generator=generator)
    else:
        delta = torch.zeros_like(image)
    if cfg.initial_jitter > 0:
        delta = delta + torch.randn_like(image) * cfg.initial_jitter
    adv = (image + delta).clamp(0.0, 1.0).detach()
    def prepare_prior(value: torch.Tensor) -> torch.Tensor:
        prior = value.to(device=image.device, dtype=image.dtype)
        if prior.shape[-2:] != image.shape[-2:]:
            prior = F.interpolate(prior, image.shape[-2:], mode="bilinear", align_corners=False)
        if prior.shape[1] == 1 and image.shape[1] == 3:
            prior = prior.repeat(1, 3, 1, 1)
        if prior.shape != image.shape:
            raise ValueError(f"prior shape {tuple(prior.shape)} does not match image {tuple(image.shape)}")
        if cfg.update_channel >= 0:
            if cfg.update_channel >= prior.shape[1]:
                raise ValueError(f"update channel {cfg.update_channel} outside {prior.shape[1]} channels")
            mask = torch.zeros_like(prior)
            mask[:, cfg.update_channel] = prior[:, cfg.update_channel]
            prior = mask
        return prior

    if initial_momentum is None:
        momentum = torch.zeros_like(image)
    else:
        prior = prepare_prior(initial_momentum)
        momentum = prior / prior.abs().mean().clamp_min(1e-12)
    delta_prior = prepare_prior(initial_delta).detach() if initial_delta is not None else None
    edge = _edge_map(image).detach()
    trace: Dict[str, List[float]] = {
        "loss": [], "min_view_loss": [], "source_gain": [], "readout_coverage": [],
        "sign_agreement": [], "gradient_std": [], "cos_source_final": [],
        "phase_conflict_fraction": [], "clipped_fraction": [],
        "minimum_anchor_ratio": [], "adapter_tail_loss": [],
        "adapter_tail_ratio": [], "skip_path_cosine": [],
        "spectral_uncertainty": [], "phase_concentration": [],
        "phase_rejected_fraction": [], "min_path_cosine": [],
        "softmin_weight_max": [],
        "feature_phase_loss": [],
        "input_phase_loss": [],
        "input_amplitude_loss": [],
        "grat_loss": [],
        "patch_loss": [],
        "token_collapse": [],
        "dc_shift": [],
        "low_level_lift": [],
        "pasd_loss": [],
        "atcs_loss": [],
        "cos_online_prior": [],
        "cos_guided_prior": [],
        "patchout_fraction": [],
        "depth_layer_loss": [],
        "depth_phase_loss": [],
        "depth_amplitude_loss": [],
        "depth_boundary_loss": [],
        "depth_robust_loss": [],
        "phase_collision_loss": [],
        "graph_collapse_loss": [],
        "qk_topology_loss": [],
        "quadrature_collision_loss": [],
        "translation_collision_loss": [],
        "input_translation_loss": [],
        "spatial_principal_loss": [],
        "semantic_partition_loss": [],
        "cancellation_gap_loss": [],
        "peft_weight_max": [],
    }

    has_prior = initial_momentum is not None or initial_delta is not None
    for step_index in range(cfg.steps):
        current_patchout_mask = (
            patchout_mask(
                image, cfg.patchout_keep_ratio, cfg.patchout_patch_size, generator
            )
            if cfg.use_patchout else None
        )
        schedule_switch = (
            cfg.steps // 2 if cfg.pasd_switch_step < 0 else int(cfg.pasd_switch_step)
        )
        adv.requires_grad_(True)
        forward_adv = adv
        if step_index == 0 and not has_prior and cfg.first_step_jitter > 0:
            forward_adv = adv + torch.randn_like(adv) * cfg.first_step_jitter
        gradients, losses, feature_gains, coverage_values = [], [], [], []
        tail_losses, tail_ratios = [], []
        uncertainty_values = []
        phase_loss_values = []
        input_phase_values = []
        input_amplitude_values = []
        grat_loss_values = []
        patch_loss_values = []
        patch_boundary_values = []
        collapse_values = []
        dc_shift_values = []
        lift_values = []
        pasd_values = []
        atcs_values = []
        depth_layer_values = []
        depth_phase_values = []
        depth_amplitude_values = []
        depth_boundary_values = []
        depth_robust_values = []
        phase_collision_values = []
        graph_collapse_values = []
        qk_topology_values = []
        quadrature_collision_values = []
        translation_collision_values = []
        input_translation_values = []
        spatial_principal_values = []
        semantic_partition_values = []
        cancellation_gap_values = []
        step_anchor_diagnostics = {
            "phase_conflict_fraction": 0.0,
            "clipped_fraction": 0.0,
            "minimum_anchor_ratio": 1.0,
        }
        step_path_cosine = 1.0
        step_phase_diagnostics = {
            "phase_concentration": 1.0,
            "phase_rejected_fraction": 0.0,
            "min_path_cosine": 1.0,
            "softmin_weight_max": 1.0,
        }
        for view_index, (view, (clean_final, clean_layers)) in enumerate(zip(views, clean_refs)):
            if view_index == 0 and cfg.use_skip_ensemble:
                if len(cfg.skip_gammas) != len(cfg.skip_weights) or not cfg.skip_gammas:
                    raise ValueError("skip-gamma and skip-weight counts must match and be non-empty")
                path_gradients = []
                anchor_ads_loss = None
                anchor_feature_gain = None
                anchor_loss = None
                for path_index, gamma in enumerate(cfg.skip_gammas):
                    with SkipGradientView(encoder, gamma).active():
                        path_final, path_layers = _unpack_encoder(encoder(preprocess(input_view(forward_adv, view_index))))
                    path_gain = F.l1_loss(path_final * 0.9999, clean_final.detach())
                    path_loss = cfg.output_weight * path_gain
                    path_phase_loss = path_gain.new_zeros(())
                    if cfg.use_feature_phase:
                        path_phase_loss = feature_phase_disruption(path_final, clean_final)
                        path_loss = path_loss + cfg.feature_phase_weight * path_phase_loss
                    path_grat_loss = path_gain.new_zeros(())
                    if cfg.use_grat:
                        path_grat_loss = gradient_robust_disruption(path_layers, clean_layers, cfg)
                        path_loss = path_loss + path_grat_loss
                    is_anchor_path = path_index == len(cfg.skip_gammas) - 1
                    if is_anchor_path and cfg.use_ads and cfg.ads_weight > 0:
                        anchor_ads_loss = cfg.ads_weight * carrier_stochastic_disruption(
                            path_layers, clean_layers, cfg
                        )
                    path_gradient = torch.autograd.grad(
                        path_loss, adv, retain_graph=is_anchor_path and anchor_ads_loss is not None
                    )[0]
                    path_gradients.append(path_gradient)
                    if is_anchor_path:
                        anchor_feature_gain = path_gain
                        anchor_loss = path_loss
                normalized_paths = [
                    value / value.abs().mean().clamp_min(1e-12) for value in path_gradients
                ]
                path_weights = torch.as_tensor(
                    cfg.skip_weights, device=adv.device, dtype=adv.dtype
                )
                path_weights = path_weights / path_weights.sum().clamp_min(1e-12)
                if cfg.use_phase_robust:
                    robust_anchor, step_phase_diagnostics = phase_interval_robust_direction(
                        normalized_paths, cfg.phase_softmin_beta, cfg.phase_reliability_floor
                    )
                else:
                    robust_anchor = sum(
                        weight * value for weight, value in zip(path_weights, normalized_paths)
                    )
                if len(normalized_paths) >= 2:
                    step_path_cosine = float(F.cosine_similarity(
                        normalized_paths[0].flatten(1), normalized_paths[-1].flatten(1), dim=1
                    ).mean().detach())
                if anchor_ads_loss is not None:
                    ads_gradient = torch.autograd.grad(anchor_ads_loss, adv)[0]
                    gradient, anchor_diagnostics = cross_spectral_anchor(
                        robust_anchor, ads_gradient, cfg.anchor_floor, cfg.auxiliary_weight
                    )
                    step_anchor_diagnostics = anchor_diagnostics
                    anchor_loss = anchor_loss + anchor_ads_loss
                else:
                    gradient = robust_anchor
                gradients.append(gradient.detach())
                losses.append(anchor_loss.detach())
                feature_gains.append(anchor_feature_gain.detach())
                coverage_values.append(clean_final.new_zeros(()))
                tail_losses.append(clean_final.new_zeros(()))
                tail_ratios.append(clean_final.new_zeros(()))
                uncertainty_values.append(clean_final.new_zeros(()))
                phase_loss_values.append(path_phase_loss.detach())
                grat_loss_values.append(path_grat_loss.detach())
                patch_loss_values.append(clean_final.new_zeros(()))
                patch_boundary_values.append(clean_final.new_zeros(()))
                collapse_values.append(clean_final.new_zeros(()))
                dc_shift_values.append(clean_final.new_zeros(()))
                lift_values.append(clean_final.new_zeros(()))
                pasd_values.append(clean_final.new_zeros(()))
                atcs_values.append(clean_final.new_zeros(()))
                continue
            with view.active():
                skip_context = (
                    SkipGradientView(encoder, cfg.skip_gamma).active()
                    if cfg.use_skip_gradient else contextlib.nullcontext()
                )
                qk_context = (
                    QKGradientBypassView(encoder, cfg.qk_gradient_scale).active()
                    if cfg.use_qk_bypass else contextlib.nullcontext()
                )
                with skip_context, qk_context:
                    with intermediate_perturbation_decay(
                        encoder, cfg.ilpd_layer, ilpd_clean_refs[view_index], cfg.ilpd_gamma
                    ):
                        qk_capture = None
                        if view_index == 0 and (uses_site_capture or cfg.use_qk_topology):
                            capture = (
                                AdapterSiteCapture(encoder, adapter_layers, cfg.adapter_tail_sites)
                                if uses_site_capture else None
                            )
                            qk_capture = (
                                QKSiteCapture(encoder, cfg.qk_topology_layer)
                                if cfg.use_qk_topology else None
                            )
                            with contextlib.ExitStack() as stack:
                                if capture is not None:
                                    stack.enter_context(capture.active())
                                if qk_capture is not None:
                                    stack.enter_context(qk_capture.active())
                                final, layers = _unpack_encoder(encoder(preprocess(input_view(forward_adv, view_index))))
                        else:
                            final, layers = _unpack_encoder(encoder(preprocess(input_view(forward_adv, view_index))))
                        if view_index == 0 and uses_site_capture:
                            if clean_adapter_sites is None:
                                raise RuntimeError("clean adapter sites were not captured")
                            if cfg.use_frequency_rank_tail:
                                frequency_terms, frequency_ratios = [], []
                                for key, site_adv in capture.outputs.items():
                                    term, ratio = frequency_balanced_irreducible_tail(
                                        site_adv, clean_adapter_sites[key], cfg.adapter_rank
                                    )
                                    frequency_terms.append(term)
                                    frequency_ratios.append(ratio)
                                final_term, final_ratio = frequency_balanced_irreducible_tail(
                                    final.permute(0, 2, 3, 1),
                                    clean_final.permute(0, 2, 3, 1),
                                    cfg.adapter_rank,
                                )
                                tail_loss = torch.stack([*frequency_terms, final_term]).mean()
                                tail_ratio = torch.stack([*frequency_ratios, final_ratio]).mean()
                            else:
                                tail_loss, tail_ratio = adapter_irreducible_tail(
                                    capture.outputs, clean_adapter_sites, cfg.adapter_rank
                                )
                        else:
                            tail_loss = final.new_zeros(())
                            tail_ratio = final.new_zeros(())
            feature_gain = F.l1_loss(final * 0.9999, clean_final.detach())
            if cfg.use_depth_layer and not 0 <= cfg.depth_layer_index < len(layers):
                raise IndexError(
                    f"depth layer {cfg.depth_layer_index} outside {len(layers)} captured layers"
                )
            early_continuation_phase = (
                cfg.use_depth_layer
                and cfg.depth_schedule in {"early_then_grat", "early_then_final"}
                and step_index < schedule_switch
            )
            base_active = not (
                cfg.use_depth_layer
                and (cfg.depth_schedule == "early_only" or early_continuation_phase)
            )
            loss = cfg.output_weight * feature_gain if base_active else feature_gain.new_zeros(())
            depth_layer_loss = final.new_zeros(())
            depth_phase_loss = final.new_zeros(())
            depth_amplitude_loss = final.new_zeros(())
            depth_boundary_loss = final.new_zeros(())
            depth_robust_loss = final.new_zeros(())
            phase_collision_loss = final.new_zeros(())
            graph_collapse_loss = final.new_zeros(())
            qk_topology_loss = final.new_zeros(())
            quadrature_collision_loss = final.new_zeros(())
            translation_collision_loss = final.new_zeros(())
            input_translation_loss = final.new_zeros(())
            spatial_principal_loss = final.new_zeros(())
            if cfg.use_spatial_principal:
                spatial_principal_loss = spatial_principal_disruption(
                    final, clean_final, clean_spatial_bases[view_index]
                )
                loss = loss + cfg.spatial_principal_weight * spatial_principal_loss
            semantic_partition_loss = final.new_zeros(())
            if cfg.use_semantic_partition:
                centroids, partition_labels = clean_semantic_partitions[view_index]
                semantic_partition_loss = semantic_partition_disruption(
                    final, centroids, partition_labels,
                    cfg.semantic_partition_temperature,
                )
                loss = loss + cfg.semantic_partition_weight * semantic_partition_loss
            cancellation_gap_loss = final.new_zeros(())
            if cfg.use_cancellation_gap:
                cancellation_gap_loss = cancellation_gap_disruption(
                    final, clean_final, layers, clean_layers,
                    cfg.cancellation_layer_count,
                    cfg.cancellation_final_penalty,
                )
                loss = loss + cfg.cancellation_weight * cancellation_gap_loss
            depth_active = cfg.use_depth_layer and (
                cfg.depth_schedule in {"early_only", "joint"}
                or early_continuation_phase
            ) and (not cfg.task_after_switch or step_index >= schedule_switch)
            if depth_active:
                depth_layer_loss = F.l1_loss(
                    layers[cfg.depth_layer_index] * 0.9999,
                    clean_layers[cfg.depth_layer_index].detach(),
                )
                loss = loss + cfg.depth_layer_weight * depth_layer_loss
                if cfg.use_depth_polar:
                    depth_phase_loss, depth_amplitude_loss = spatial_feature_polar_disruption_bhwc(
                        layers[cfg.depth_layer_index], clean_layers[cfg.depth_layer_index]
                    )
                    loss = (
                        loss
                        + cfg.depth_phase_weight * depth_phase_loss
                        + cfg.depth_amplitude_weight * depth_amplitude_loss
                    )
                if cfg.use_depth_boundary:
                    depth_boundary_loss = patch_boundary_disruption(
                        layers[cfg.depth_layer_index], clean_layers[cfg.depth_layer_index]
                    )
                    loss = loss + cfg.depth_boundary_weight * depth_boundary_loss
            if cfg.use_depth_robust:
                depth_robust_loss = depth_robust_disruption(
                    layers, clean_layers, cfg.depth_robust_count, cfg.depth_robust_beta
                )
                loss = loss + cfg.depth_robust_weight * depth_robust_loss
            if cfg.use_phase_collision:
                target_layers = collision_refs[view_index][1]
                index = int(cfg.phase_collision_layer)
                if not 0 <= index < min(len(layers), len(target_layers)):
                    raise IndexError(f"phase collision layer {index} is unavailable")
                phase_collision_loss = relative_feature_distance(layers[index], target_layers[index])
                loss = loss - cfg.phase_collision_weight * phase_collision_loss
            if cfg.use_graph_collapse:
                graph_collapse_loss = spatial_graph_collapse(
                    layers, clean_layers, cfg.graph_collapse_count
                )
                loss = loss + cfg.graph_collapse_weight * graph_collapse_loss
            if cfg.use_qk_topology and view_index == 0:
                if clean_qk_site is None or qk_capture is None or qk_capture.output is None:
                    raise RuntimeError("QK topology sites were not captured")
                heads = encoder.blocks[int(cfg.qk_topology_layer)].attn.num_heads
                qk_topology_loss = qk_topology_disruption(
                    qk_capture.output, clean_qk_site, heads
                )
                loss = loss + cfg.qk_topology_weight * qk_topology_loss
            quadrature_active = cfg.use_quadrature_collision and (
                (cfg.quadrature_schedule == "" )
                or (
                    cfg.quadrature_schedule == "quadrature_then_ads"
                    and step_index < schedule_switch
                )
                or (
                    cfg.quadrature_schedule == "after_switch"
                    and step_index >= schedule_switch
                )
            )
            if quadrature_active:
                quadrature_collision_loss = quadrature_collision_objective(
                    final, clean_final
                )
                loss = loss + (
                    cfg.quadrature_collision_weight * quadrature_collision_loss
                )
            if cfg.use_translation_collision:
                translation_collision_loss = translation_collision_objective(
                    final, clean_final, cfg.translation_collision_shift
                )
                loss = loss + (
                    cfg.translation_collision_weight * translation_collision_loss
                )
            if cfg.use_input_translation_collision and view_index == 0:
                input_translation_loss = translation_collision_objective(
                    forward_adv, image, cfg.input_translation_shift
                )
                loss = loss + cfg.input_translation_weight * input_translation_loss
            patch_loss = final.new_zeros(())
            patch_boundary_loss = final.new_zeros(())
            if cfg.use_patch_loss or cfg.use_patch_boundary:
                if clean_patch is None:
                    raise RuntimeError("clean patch embedding was not captured")
                patch_adv = _tensor_output(encoder.patch_embed(preprocess(input_view(forward_adv, view_index))))
            if cfg.use_patch_loss:
                patch_loss = patch_embedding_disruption(patch_adv, clean_patch)
                loss = loss + cfg.patch_weight * patch_loss
            if cfg.use_patch_boundary:
                patch_boundary_loss = patch_boundary_disruption(patch_adv, clean_patch)
                loss = loss + cfg.patch_boundary_weight * patch_boundary_loss
            collapse_loss = final.new_zeros(())
            if cfg.use_token_collapse:
                collapse_loss = token_collapse_objective(final, clean_final)
                loss = loss + cfg.collapse_weight * collapse_loss
            dc_shift_loss = final.new_zeros(())
            if cfg.use_dc_shift:
                dc_shift_loss = dc_feature_shift(final, clean_final)
                loss = loss + cfg.dc_shift_weight * dc_shift_loss
            lift_loss = final.new_zeros(())
            if cfg.use_low_level_lift:
                lift_loss = low_level_activation_lift(layers, clean_layers, cfg.lift_layers)
                loss = loss + cfg.lift_weight * lift_loss
            pasd_loss = final.new_zeros(())
            if cfg.use_pasd:
                pasd_cfg = cfg
                run_pasd = not (
                    cfg.pasd_schedule == "final_then_ads" and step_index >= schedule_switch
                )
                if cfg.pasd_schedule == "grat_then_pasd":
                    run_pasd = step_index >= schedule_switch
                elif cfg.pasd_schedule == "pasd_then_grat":
                    run_pasd = step_index < schedule_switch
                elif cfg.pasd_schedule == "grat_then_joint":
                    run_pasd = step_index >= schedule_switch
                elif cfg.pasd_schedule and cfg.pasd_schedule != "final_then_ads":
                    halfway = schedule_switch
                    if cfg.pasd_schedule == "late_then_final":
                        region = "late" if step_index < halfway else "final"
                    elif cfg.pasd_schedule == "final_then_late":
                        region = "final" if step_index < halfway else "late"
                    elif cfg.pasd_schedule == "final_then_mixed":
                        region = "final" if step_index < halfway else "mixed"
                    elif cfg.pasd_schedule == "final_then_early":
                        region = "final" if step_index < halfway else "early"
                    elif cfg.pasd_schedule == "alternate":
                        region = "late" if step_index % 2 == 0 else "final"
                    else:
                        raise ValueError(f"unknown PASD schedule {cfg.pasd_schedule!r}")
                    pasd_cfg = replace(
                        cfg, pasd_region=region,
                        pasd_layer_count={"late": 6, "mixed": 6, "early": 4, "final": 2}[region],
                    )
                if run_pasd and cfg.pasd_depth_robust:
                    final_cfg = replace(
                        cfg, pasd_depth_robust=False, pasd_schedule="",
                        pasd_region="final", pasd_layer_count=2,
                    )
                    late_cfg = replace(
                        cfg, pasd_depth_robust=False, pasd_schedule="",
                        pasd_region="late", pasd_layer_count=6,
                    )
                    pasd_loss = smooth_minimum(
                        [
                            phase_amplitude_stochastic_disruption(
                                layers, clean_layers, final_cfg
                            ),
                            phase_amplitude_stochastic_disruption(
                                layers, clean_layers, late_cfg
                            ),
                        ],
                        cfg.pasd_depth_beta,
                    )
                elif run_pasd:
                    pasd_loss = phase_amplitude_stochastic_disruption(
                        layers, clean_layers, pasd_cfg
                    )
                if run_pasd:
                    loss = loss + cfg.pasd_weight * pasd_loss
            atcs_loss = final.new_zeros(())
            if cfg.use_atcs:
                atcs_loss = average_token_angular_disruption(
                    layers, clean_layers, cfg.pasd_region, cfg.pasd_layer_count
                )
                loss = loss + cfg.atcs_weight * atcs_loss
            if cfg.use_token_gram:
                token_gram_loss = normalized_token_gram_disruption(
                    layers, clean_layers, cfg.token_gram_region,
                    cfg.token_gram_layer_count, cfg.token_gram_pool,
                )
                loss = loss + cfg.token_gram_weight * token_gram_loss
            phase_loss = final.new_zeros(())
            if cfg.use_feature_phase:
                phase_loss = feature_phase_disruption(final, clean_final)
                loss = loss + cfg.feature_phase_weight * phase_loss
            input_phase_loss = final.new_zeros(())
            if view_index == 0 and cfg.use_input_phase:
                input_phase_loss = input_phase_disruption(
                    forward_adv, image, cfg.input_phase_low, cfg.input_phase_high
                )
                loss = loss + cfg.input_phase_weight * input_phase_loss
            input_amplitude_loss = final.new_zeros(())
            if view_index == 0 and cfg.use_input_amplitude:
                input_amplitude_loss = input_amplitude_disruption(forward_adv, image)
                loss = loss + cfg.input_amplitude_weight * input_amplitude_loss
            grat_loss = final.new_zeros(())
            grat_active = cfg.use_grat and base_active and (
                cfg.pasd_schedule not in {"grat_then_pasd", "pasd_then_grat"}
                or (cfg.pasd_schedule == "grat_then_pasd" and step_index < schedule_switch)
                or (cfg.pasd_schedule == "pasd_then_grat" and step_index >= schedule_switch)
            ) and (not cfg.task_after_switch or step_index >= schedule_switch)
            if grat_active:
                grat_loss = gradient_robust_disruption(layers, clean_layers, cfg)
                loss = loss + cfg.grat_weight * grat_loss
            uncertainty = final.new_zeros(())
            if view_index == 0 and cfg.use_spectral_uncertainty:
                uncertainty = spectral_uncertainty_separation(layers, clean_layers, cfg)
                loss = loss + cfg.uncertainty_weight * uncertainty
            if view_index == 0 and cfg.use_adapter_tail:
                loss = loss + cfg.adapter_tail_weight * tail_loss
            if view_index == 0 and cfg.use_frequency_rank_tail:
                loss = loss + cfg.frequency_tail_weight * tail_loss
            if cfg.use_multilevel:
                loss = loss + cfg.layer_weight * intermediate_disruption(layers, clean_layers, cfg)
            ads_loss = None
            ads_active = cfg.use_ads and (
                cfg.pasd_schedule != "final_then_ads" or step_index >= schedule_switch
            )
            if (view_index == 0 or cfg.ads_all_views) and ads_active and cfg.ads_weight > 0:
                ads_loss = cfg.ads_weight * carrier_stochastic_disruption(layers, clean_layers, cfg)
            coverage = final.new_zeros(())
            if view_index == 0 and cfg.use_readout_coverage:
                coverage = readout_coverage_lower_bound(final, clean_final, cfg)
                loss = loss + cfg.coverage_weight * coverage
            if view_index == 0 and cfg.use_logdet_coverage:
                logdet_coverage = logdet_readout_coverage(
                    final, clean_final, cfg.logdet_coverage_temperature
                )
                loss = loss + cfg.logdet_coverage_weight * logdet_coverage
            anchor_diagnostics = {
                "phase_conflict_fraction": 0.0,
                "clipped_fraction": 0.0,
                "minimum_anchor_ratio": 1.0,
            }
            if view_index == 0 and cfg.use_final_ads_anchor:
                if ads_loss is None or not cfg.use_pasd:
                    raise RuntimeError("final/ADS anchoring requires both losses")
                anchor_loss = cfg.output_weight * feature_gain + ads_loss
                anchor_gradient = torch.autograd.grad(
                    anchor_loss, adv, retain_graph=True
                )[0]
                auxiliary_gradient = torch.autograd.grad(pasd_loss, adv)[0]
                gradient, anchor_diagnostics = cross_spectral_anchor(
                    anchor_gradient, auxiliary_gradient,
                    cfg.anchor_floor, cfg.auxiliary_weight,
                )
                step_anchor_diagnostics = anchor_diagnostics
                loss = anchor_loss + pasd_loss
            elif view_index == 0 and cfg.use_cross_spectral_anchor and ads_loss is not None:
                anchor_gradient = torch.autograd.grad(loss, adv, retain_graph=True)[0]
                auxiliary_gradient = torch.autograd.grad(ads_loss, adv)[0]
                gradient, anchor_diagnostics = cross_spectral_anchor(
                    anchor_gradient, auxiliary_gradient, cfg.anchor_floor, cfg.auxiliary_weight
                )
                step_anchor_diagnostics = anchor_diagnostics
                loss = loss + ads_loss
            else:
                if ads_loss is not None:
                    loss = loss + ads_loss
                gradient = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
            gradients.append(gradient.detach())
            losses.append(loss.detach())
            feature_gains.append(feature_gain.detach())
            coverage_values.append(coverage.detach())
            tail_losses.append(tail_loss.detach())
            tail_ratios.append(tail_ratio.detach())
            uncertainty_values.append(uncertainty.detach())
            phase_loss_values.append(phase_loss.detach())
            input_phase_values.append(input_phase_loss.detach())
            input_amplitude_values.append(input_amplitude_loss.detach())
            grat_loss_values.append(grat_loss.detach())
            patch_loss_values.append(patch_loss.detach())
            patch_boundary_values.append(patch_boundary_loss.detach())
            collapse_values.append(collapse_loss.detach())
            dc_shift_values.append(dc_shift_loss.detach())
            lift_values.append(lift_loss.detach())
            pasd_values.append(pasd_loss.detach())
            atcs_values.append(atcs_loss.detach())
            depth_layer_values.append(depth_layer_loss.detach())
            depth_phase_values.append(depth_phase_loss.detach())
            depth_amplitude_values.append(depth_amplitude_loss.detach())
            depth_boundary_values.append(depth_boundary_loss.detach())
            depth_robust_values.append(depth_robust_loss.detach())
            phase_collision_values.append(phase_collision_loss.detach())
            graph_collapse_values.append(graph_collapse_loss.detach())
            qk_topology_values.append(qk_topology_loss.detach())
            quadrature_collision_values.append(quadrature_collision_loss.detach())
            translation_collision_values.append(translation_collision_loss.detach())
            input_translation_values.append(input_translation_loss.detach())
            spatial_principal_values.append(spatial_principal_loss.detach())
            semantic_partition_values.append(semantic_partition_loss.detach())
            cancellation_gap_values.append(cancellation_gap_loss.detach())

        source = gradients[0] / gradients[0].abs().mean().clamp_min(1e-12)
        peft_weight_max = 1.0
        if cfg.use_peft_consensus and cfg.peft_reduce in {"mean", "softmin"}:
            normalized_views = [
                value / value.abs().mean().clamp_min(1e-12) for value in gradients
            ]
            stack = torch.stack(normalized_views)
            if cfg.peft_reduce == "softmin":
                view_losses = torch.stack(losses).float()
                weights = torch.softmax(
                    -float(cfg.peft_softmin_beta) * view_losses, dim=0
                ).to(stack.dtype)
            else:
                weights = torch.full(
                    (len(normalized_views),), 1.0 / len(normalized_views),
                    device=stack.device, dtype=stack.dtype,
                )
            gradient = (weights.view(-1, 1, 1, 1, 1) * stack).sum(0)
            signs = torch.sign(stack)
            diagnostics = {
                "sign_agreement": float(signs.mean(0).abs().mean().detach()),
                "gradient_std": float(stack.std(0, unbiased=False).mean().detach()),
            }
            peft_weight_max = float(weights.max().detach())
        elif cfg.use_peft_consensus:
            consensus, diagnostics = gradient_consensus(gradients[1:], cfg.consensus_floor)
            correction = spectral_consensus(consensus, cfg) if cfg.use_spectral else consensus
            correction = correction / correction.abs().mean().clamp_min(1e-12)
            gradient = source + cfg.robust_weight * correction
        elif cfg.use_amplitude_eot or cfg.use_block_shuffle_eot or cfg.use_contrast_eot:
            normalized_views = [
                value / value.abs().mean().clamp_min(1e-12) for value in gradients
            ]
            stack = torch.stack(normalized_views)
            gradient = stack.mean(0)
            signs = torch.sign(stack)
            diagnostics = {
                "sign_agreement": float(signs.mean(0).abs().mean().detach()),
                "gradient_std": float(stack.std(0, unbiased=False).mean().detach()),
            }
        else:
            diagnostics = {"sign_agreement": 1.0, "gradient_std": 0.0}
            gradient = source
        if cfg.update_channel >= 0:
            if cfg.update_channel >= gradient.shape[1]:
                raise ValueError(f"update channel {cfg.update_channel} outside {gradient.shape[1]} channels")
            channel_gradient = torch.zeros_like(gradient)
            channel_gradient[:, cfg.update_channel] = gradient[:, cfg.update_channel]
            gradient = channel_gradient
        if cfg.post_ti_kernel > 1:
            gradient = translation_orbit_average(gradient, cfg.post_ti_kernel)
        if cfg.use_edge:
            gradient = gradient * (1.0 + cfg.edge_gamma * edge)
        if cfg.use_polar_amplitude_gradient:
            radial_gradient = polar_amplitude_gradient(
                gradient, adv, cfg.polar_amplitude_low, cfg.polar_amplitude_high
            )
            radial_gradient = radial_gradient / radial_gradient.abs().mean().clamp_min(1e-12)
            gradient = gradient + cfg.polar_amplitude_weight * radial_gradient
        if cfg.polar_projection_coordinate:
            projected_gradient = polar_tangent_gradient(
                gradient,
                adv,
                cfg.polar_projection_coordinate,
                cfg.polar_projection_low,
                cfg.polar_projection_high,
            )
            projected_gradient = (
                projected_gradient
                / projected_gradient.abs().mean().clamp_min(1e-12)
            )
            if cfg.polar_projection_replace:
                gradient = projected_gradient
            else:
                gradient = gradient + cfg.polar_projection_weight * projected_gradient
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        if delta_prior is None:
            cos_online_prior = 0.0
            cos_guided_prior = 0.0
        else:
            prior_direction = delta_prior / delta_prior.abs().mean().clamp_min(1e-12)
            cos_online_prior = float(F.cosine_similarity(
                gradient.flatten(1), prior_direction.flatten(1), dim=1
            ).mean().detach())
            if cfg.prior_spectral_floor >= 0:
                gradient, step_anchor_diagnostics = cross_spectral_anchor(
                    gradient, prior_direction, cfg.prior_spectral_floor,
                    cfg.prior_spectral_weight,
                )
                gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
            elif cfg.prior_guidance_weight > 0:
                progress = step_index / max(cfg.steps - 1, 1)
                if cfg.prior_guidance_schedule == "constant":
                    schedule_scale = 1.0
                elif cfg.prior_guidance_schedule == "linear_decay":
                    schedule_scale = 1.0 - progress
                elif cfg.prior_guidance_schedule == "linear_ramp":
                    schedule_scale = progress
                elif cfg.prior_guidance_schedule == "early":
                    schedule_scale = float(step_index < max(cfg.steps // 2, 1))
                elif cfg.prior_guidance_schedule == "late":
                    schedule_scale = float(step_index >= cfg.steps // 2)
                else:
                    raise ValueError(
                        f"unknown prior guidance schedule {cfg.prior_guidance_schedule!r}"
                    )
                gradient = gradient + (
                    cfg.prior_guidance_weight * schedule_scale * prior_direction
                )
                gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
            cos_guided_prior = float(F.cosine_similarity(
                gradient.flatten(1), prior_direction.flatten(1), dim=1
            ).mean().detach())
        cosine = F.cosine_similarity(source.flatten(1), gradient.flatten(1), dim=1).mean()
        momentum = cfg.momentum * momentum + gradient
        update = _asymmetric_step(momentum, cfg.step_size) if cfg.asymmetric_step else cfg.step_size * momentum.sign()
        if step_index == 0 and delta_prior is not None:
            update = float(first_step_multiplier) * update + delta_prior
        if cfg.update_channel >= 0:
            channel_update = torch.zeros_like(update)
            channel_update[:, cfg.update_channel] = update[:, cfg.update_channel]
            update = channel_update
        adv = adv.detach() + update
        delta = (adv - image).clamp(-cfg.epsilon, cfg.epsilon)
        adv = (image + delta).clamp(0.0, 1.0).detach()

        trace["loss"].append(float(torch.stack(losses).mean()))
        trace["min_view_loss"].append(float(torch.stack(losses).min()))
        trace["source_gain"].append(float(feature_gains[0]))
        trace["readout_coverage"].append(float(coverage_values[0]))
        trace["sign_agreement"].append(diagnostics["sign_agreement"])
        trace["gradient_std"].append(diagnostics["gradient_std"])
        trace["cos_source_final"].append(float(cosine.detach()))
        trace["phase_conflict_fraction"].append(step_anchor_diagnostics["phase_conflict_fraction"])
        trace["clipped_fraction"].append(step_anchor_diagnostics["clipped_fraction"])
        trace["minimum_anchor_ratio"].append(step_anchor_diagnostics["minimum_anchor_ratio"])
        trace["adapter_tail_loss"].append(float(tail_losses[0]))
        trace["adapter_tail_ratio"].append(float(tail_ratios[0]))
        trace["skip_path_cosine"].append(step_path_cosine)
        trace["spectral_uncertainty"].append(float(uncertainty_values[0]))
        trace["feature_phase_loss"].append(float(phase_loss_values[0]))
        trace["input_phase_loss"].append(float(input_phase_values[0]))
        trace["input_amplitude_loss"].append(float(input_amplitude_values[0]))
        trace["grat_loss"].append(float(grat_loss_values[0]))
        trace["patch_loss"].append(float(patch_loss_values[0]))
        trace.setdefault("patch_boundary_loss", []).append(float(patch_boundary_values[0]))
        trace["token_collapse"].append(float(collapse_values[0]))
        trace["dc_shift"].append(float(dc_shift_values[0]))
        trace["low_level_lift"].append(float(lift_values[0]))
        trace["pasd_loss"].append(float(pasd_values[0]))
        trace["depth_layer_loss"].append(float(depth_layer_values[0]))
        trace["depth_phase_loss"].append(float(depth_phase_values[0]))
        trace["depth_amplitude_loss"].append(float(depth_amplitude_values[0]))
        trace["depth_boundary_loss"].append(float(depth_boundary_values[0]))
        trace["depth_robust_loss"].append(float(depth_robust_values[0]))
        trace["phase_collision_loss"].append(float(phase_collision_values[0]))
        trace["graph_collapse_loss"].append(float(graph_collapse_values[0]))
        trace["qk_topology_loss"].append(float(qk_topology_values[0]))
        trace["quadrature_collision_loss"].append(
            float(quadrature_collision_values[0])
        )
        trace["translation_collision_loss"].append(
            float(translation_collision_values[0])
        )
        trace["input_translation_loss"].append(float(input_translation_values[0]))
        trace["spatial_principal_loss"].append(float(spatial_principal_values[0]))
        trace["semantic_partition_loss"].append(float(semantic_partition_values[0]))
        trace["cancellation_gap_loss"].append(float(cancellation_gap_values[0]))
        trace["peft_weight_max"].append(peft_weight_max)
        trace["atcs_loss"].append(float(atcs_values[0]))
        trace["cos_online_prior"].append(cos_online_prior)
        trace["cos_guided_prior"].append(cos_guided_prior)
        trace["patchout_fraction"].append(
            float(current_patchout_mask.mean()) if current_patchout_mask is not None else 1.0
        )
        for key, value in step_phase_diagnostics.items():
            trace[key].append(value)

        if gradient_sink is not None:
            gradient_sink["source"] = source.detach().cpu()
            gradient_sink["final"] = gradient.detach().cpu()

    return adv, trace
