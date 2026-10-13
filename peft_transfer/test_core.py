import torch
import torch.nn as nn

from peft_transfer.core import (
    adapter_irreducible_tail,
    average_token_angular_disruption,
    AttackConfig,
    ablation_config,
    amplitude_spectrum_view,
    attack_image,
    bsr_shuffle_rotate_view,
    block_shuffle_view,
    cross_spectral_anchor,
    dc_feature_shift,
    depth_robust_disruption,
    frequency_balanced_irreducible_tail,
    feature_phase_disruption,
    input_phase_disruption,
    input_amplitude_disruption,
    gradient_consensus,
    gradient_robust_disruption,
    low_level_activation_lift,
    logdet_readout_coverage,
    normalized_token_gram_disruption,
    phase_amplitude_stochastic_disruption,
    phase_interval_robust_direction,
    polar_amplitude_gradient,
    patchout_mask,
    phase_scrambled_target,
    QKGradientBypassView,
    qk_topology_disruption,
    spatial_fourier_probe_bhwc,
    smooth_minimum,
    spatial_graph_collapse,
    spatial_quadrature_target,
    spatial_translation_target,
    spatial_principal_basis,
    spatial_principal_disruption,
    semantic_partition,
    semantic_partition_disruption,
    cancellation_gap_disruption,
    translation_collision_objective,
    spatial_feature_polar_disruption_bhwc,
    translation_orbit_average,
    patch_embedding_disruption,
    token_collapse_objective,
)


class ToyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.attn.qkv = nn.Linear(dim, dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.attn(x) + self.mlp(x)


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 1)
        self.blocks = nn.ModuleList([ToyBlock(8), ToyBlock(8)])

    def forward(self, x):
        x = self.stem(x).permute(0, 2, 3, 1)
        layers = []
        for block in self.blocks:
            x = block(x)
            layers.append(x.permute(0, 3, 1, 2))
        return layers[-1], layers


def test_gradient_consensus_cancels_opposition():
    positive = torch.ones(1, 1, 2, 2)
    negative = -positive
    result, diagnostics = gradient_consensus([positive, negative])
    assert torch.allclose(result, torch.zeros_like(result))
    assert diagnostics["sign_agreement"] == 0.0


def test_attack_respects_linf_and_changes_image():
    torch.manual_seed(0)
    encoder = ToyEncoder()
    image = torch.rand(1, 3, 8, 8)
    cfg = AttackConfig(
        epsilon=4 / 255, step_size=2 / 255, steps=2, ranks=(2,),
        selected_layers=2, ti_kernel=3, seed=7,
    )
    adv, trace = attack_image(encoder, image, cfg)
    assert (adv - image).abs().max() <= cfg.epsilon + 1e-6
    assert not torch.allclose(adv, image)
    assert len(trace["loss"]) == cfg.steps


def test_channel_constrained_attack_changes_only_selected_channel():
    torch.manual_seed(2)
    encoder = ToyEncoder()
    image = torch.rand(1, 3, 8, 8)
    cfg = ablation_config("FO", steps=2, epsilon=4 / 255, step_size=2 / 255)
    adv, _ = attack_image(encoder, image, cfg)
    delta = adv - image
    assert delta[:, 2].abs().sum() > 0
    assert torch.equal(delta[:, :2], torch.zeros_like(delta[:, :2]))
    assert delta.abs().max() <= 4 / 255 + 1e-7


def test_early_to_grat_continuation_switches_objectives():
    torch.manual_seed(3)
    encoder = ToyEncoder()
    image = torch.rand(1, 3, 8, 8)
    cfg = ablation_config(
        "FT", steps=4, pasd_switch_step=2,
        epsilon=4 / 255, step_size=1 / 255,
    )
    adv, trace = attack_image(encoder, image, cfg)
    assert all(value > 0 for value in trace["depth_layer_loss"][:2])
    assert trace["depth_layer_loss"][2:] == [0.0, 0.0]
    assert trace["grat_loss"][:2] == [0.0, 0.0]
    assert all(value > 0 for value in trace["grat_loss"][2:])
    assert (adv - image).abs().max() <= cfg.epsilon + 1e-7


def test_dual_preprocess_attack_combines_both_input_coordinates():
    torch.manual_seed(4)
    encoder = ToyEncoder()
    image = torch.rand(1, 3, 8, 8)
    cfg = ablation_config("FV", steps=2, epsilon=4 / 255, step_size=1 / 255)
    adv, trace = attack_image(
        encoder, image, cfg,
        preprocess=lambda value: (value - 0.5) / 0.25,
    )
    assert cfg.use_dual_preprocess and cfg.depth_schedule == "early_only"
    assert len(trace["depth_layer_loss"]) == 2
    assert all(value > 0 for value in trace["depth_layer_loss"])
    assert torch.isfinite(adv).all()
    assert (adv - image).abs().max() <= cfg.epsilon + 1e-7


def test_spatial_feature_polar_loss_uses_bhwc_spatial_axes():
    torch.manual_seed(5)
    clean = torch.rand(1, 6, 8, 4)
    same_phase, same_amplitude = spatial_feature_polar_disruption_bhwc(clean, clean)
    shifted = torch.roll(clean, shifts=1, dims=2).requires_grad_(True)
    phase, amplitude = spatial_feature_polar_disruption_bhwc(shifted, clean)
    gradient = torch.autograd.grad(phase + amplitude, shifted)[0]
    assert same_phase.abs() < 1e-6 and same_amplitude.abs() < 1e-6
    assert phase > same_phase
    assert torch.isfinite(amplitude) and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_early_polar_variant_activates_both_terms():
    torch.manual_seed(6)
    encoder = ToyEncoder()
    image = torch.rand(1, 3, 8, 8)
    prior = torch.randn_like(image) * (1 / 255)
    cfg = ablation_config("FX", steps=2, epsilon=4 / 255, step_size=1 / 255)
    _adv, trace = attack_image(encoder, image, cfg, initial_delta=prior)
    assert trace["depth_phase_loss"][1] > 0
    assert trace["depth_amplitude_loss"][1] > 0


def test_depth_robust_loss_is_bounded_by_sampled_layer_mean():
    torch.manual_seed(7)
    clean = [torch.rand(1, 4, 4, 3) for _ in range(6)]
    adv = [(value + 0.1 * torch.randn_like(value)).requires_grad_(True) for value in clean]
    per_layer = [
        torch.nn.functional.l1_loss(a * 0.9999, c) / c.abs().mean().clamp_min(1e-8)
        for a, c in zip(adv, clean)
    ]
    robust = depth_robust_disruption(adv, clean, count=6, beta=8.0)
    gradient = torch.autograd.grad(robust, adv)
    assert robust <= torch.stack(per_layer).mean() + 1e-6
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradient)


def test_phase_scrambled_target_is_deterministic_and_preserves_channel_statistics():
    torch.manual_seed(8)
    image = 0.2 + 0.6 * torch.rand(1, 3, 16, 16)
    first = phase_scrambled_target(image, 17)
    second = phase_scrambled_target(image, 17)
    assert torch.equal(first, second) and not torch.equal(first, image)
    assert first.min() >= 0 and first.max() <= 1
    assert torch.allclose(first.mean((-2, -1)), image.mean((-2, -1)), atol=2e-2)


def test_block_shuffle_is_deterministic_and_preserves_values():
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    first = block_shuffle_view(image, grid=4, seed=11)
    second = block_shuffle_view(image, grid=4, seed=11)
    other = block_shuffle_view(image, grid=4, seed=12)
    assert torch.equal(first, second) and not torch.equal(first, other)
    assert torch.equal(first.flatten().sort().values, image.flatten().sort().values)


def test_bsr_view_is_deterministic_and_shape_preserving():
    image = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)
    first = bsr_shuffle_rotate_view(image, seed=21)
    second = bsr_shuffle_rotate_view(image, seed=21)
    assert first.shape == image.shape and torch.equal(first, second)
    assert torch.isfinite(first).all() and not torch.equal(first, image)


def test_persistent_prior_guidance_increases_gradient_alignment():
    torch.manual_seed(1)
    encoder = ToyEncoder()
    image = torch.rand(1, 3, 8, 8)
    prior = torch.randn_like(image) * (2 / 255)
    cfg = AttackConfig(
        epsilon=4 / 255, step_size=1 / 255, steps=2, ranks=(2,),
        selected_layers=2, ti_kernel=3, seed=9, prior_guidance_weight=1.0,
    )
    _adv, trace = attack_image(encoder, image, cfg, initial_delta=prior)
    assert len(trace["cos_online_prior"]) == cfg.steps
    for before, after in zip(trace["cos_online_prior"], trace["cos_guided_prior"]):
        assert after >= before - 1e-6


def test_spectral_prior_fusion_preserves_online_frequency_gain():
    torch.manual_seed(2)
    encoder = ToyEncoder()
    image = torch.rand(1, 3, 8, 8)
    prior = torch.randn_like(image) * (2 / 255)
    cfg = AttackConfig(
        epsilon=4 / 255, step_size=1 / 255, steps=2, ranks=(2,),
        selected_layers=2, ti_kernel=3, seed=10,
        prior_spectral_floor=0.75, prior_spectral_weight=1.0,
    )
    _adv, trace = attack_image(encoder, image, cfg, initial_delta=prior)
    assert min(trace["minimum_anchor_ratio"]) >= 0.75 - 1e-5
    assert all(0 <= value <= 1 for value in trace["phase_conflict_fraction"])


def test_ablation_chain_adds_one_component_at_a_time():
    configs = [ablation_config(name) for name in "ABCDEF"]
    assert not configs[0].use_ads
    assert configs[1].use_ads and not configs[1].use_multilevel
    assert configs[2].use_multilevel and not configs[2].use_peft_consensus
    assert configs[3].use_peft_consensus and not configs[3].use_spectral
    assert configs[4].use_spectral and not configs[4].use_edge
    assert configs[5].use_edge
    candidate = ablation_config("G")
    assert candidate.use_ads and candidate.use_readout_coverage
    assert not candidate.use_peft_consensus and not candidate.use_spectral
    accepted = ablation_config("H")
    assert accepted.use_ads and accepted.use_cross_spectral_anchor
    assert not accepted.use_peft_consensus and not accepted.use_spectral
    rank_robust = ablation_config("I")
    assert rank_robust.use_adapter_tail and rank_robust.adapter_rank == 192
    frequency_robust = ablation_config("J")
    assert frequency_robust.use_frequency_rank_tail


def test_depth_continuation_variants_share_the_attack_protocol():
    mixed = ablation_config("CT")
    early = ablation_config("CV")
    assert mixed.pasd_schedule == "final_then_mixed"
    assert early.pasd_schedule == "final_then_early"
    assert mixed.pasd_switch_step == early.pasd_switch_step == 5
    assert mixed.use_pasd and early.use_pasd
    assert mixed.steps == early.steps == 10
    assert mixed.epsilon == early.epsilon == 10.0 / 255.0


def test_robust_phase_expectation_is_a_lower_bound_on_mean():
    torch.manual_seed(11)
    clean = [torch.randn(1, 4, 4, 4) for _ in range(2)]
    adv = [value + 0.1 * torch.randn_like(value) for value in clean]
    mean_cfg = AttackConfig(
        pasd_region="final", pasd_layer_count=2, pasd_samples=8,
        pasd_reduce="mean", seed=3,
    )
    robust_cfg = AttackConfig(
        pasd_region="final", pasd_layer_count=2, pasd_samples=8,
        pasd_reduce="softmin", pasd_softmin_beta=8.0, seed=3,
    )
    torch.manual_seed(21)
    mean_value = phase_amplitude_stochastic_disruption(adv, clean, mean_cfg)
    torch.manual_seed(21)
    robust_value = phase_amplitude_stochastic_disruption(adv, clean, robust_cfg)
    assert robust_value <= mean_value + 1e-6


def test_amplitude_hull_contains_white_and_empirical_endpoints():
    torch.manual_seed(12)
    clean = [torch.randn(1, 4, 4, 4) for _ in range(3)]
    adv = [value + 0.1 * torch.randn_like(value) for value in clean]
    base = dict(pasd_region="final", pasd_layer_count=3, pasd_samples=1)
    for alpha, endpoint in ((0.0, "shuffled_white"), (1.0, "shuffled_amplitude")):
        hull_cfg = AttackConfig(
            **base, pasd_mode="shuffled_amplitude_hull",
            pasd_amplitude_grid=(alpha,),
        )
        endpoint_cfg = AttackConfig(**base, pasd_mode=endpoint)
        torch.manual_seed(25)
        hull = phase_amplitude_stochastic_disruption(adv, clean, hull_cfg)
        torch.manual_seed(25)
        expected = phase_amplitude_stochastic_disruption(adv, clean, endpoint_cfg)
        assert torch.equal(hull, expected)


def test_amplitude_hull_has_finite_gradient():
    torch.manual_seed(14)
    clean = [torch.randn(1, 5, 4, 6) for _ in range(8)]
    adv = [(value + 1e-3 * torch.randn_like(value)).requires_grad_(True) for value in clean]
    cfg = ablation_config("FL", pasd_samples=1, pasd_layer_count=4)
    loss = phase_amplitude_stochastic_disruption(adv, clean, cfg)
    gradients = torch.autograd.grad(loss, adv, allow_unused=True)
    used = [value for value in gradients if value is not None]
    assert cfg.pasd_amplitude_grid == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert torch.isfinite(loss) and len(used) == 4
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in used)


def test_input_phase_objective_is_zero_only_for_matching_phase():
    torch.manual_seed(29)
    clean = torch.rand(1, 3, 8, 8)
    same = feature_phase_disruption(clean, clean)
    shifted = feature_phase_disruption(torch.roll(clean, shifts=1, dims=-1), clean)
    assert same.abs() < 1e-6
    assert shifted > same


def test_input_phase_radial_bands_partition_the_full_objective():
    torch.manual_seed(30)
    clean = torch.rand(1, 3, 16, 16)
    adv = torch.roll(clean, shifts=1, dims=-1).requires_grad_(True)
    low = input_phase_disruption(adv, clean, 0.0, 0.2)
    middle = input_phase_disruption(adv, clean, 0.05, 0.4)
    high = input_phase_disruption(adv, clean, 0.2, 1.0)
    assert all(torch.isfinite(value) for value in (low, middle, high))
    assert all(value > 0 for value in (low, middle, high))
    gradient = torch.autograd.grad(low + middle + high, adv)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_polar_amplitude_gradient_is_the_radial_projection():
    torch.manual_seed(32)
    image = torch.rand(1, 3, 16, 16) + 0.1
    gradient = torch.randn_like(image)
    projected = polar_amplitude_gradient(gradient, image, 0.0, 1.0)
    spectrum = torch.fft.fft2(image, norm="ortho")
    residual = torch.fft.fft2(gradient - projected, norm="ortho")
    radial_residual = (residual.conj() * spectrum).real
    energetic = spectrum.abs().square() > 1e-6
    assert radial_residual.masked_select(energetic).abs().max() < 1e-4
    assert torch.isfinite(projected).all()


def test_logdet_coverage_rewards_independent_channel_displacement():
    clean = torch.ones(1, 4, 2, 2)
    rank_one = clean.clone()
    rank_one[:, 0] += torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    full_rank = clean.clone()
    full_rank[0, 0] += torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    full_rank[0, 1] += torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    full_rank[0, 2] += torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    full_rank[0, 3] += torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    rank_one.requires_grad_(True)
    full_rank.requires_grad_(True)
    low = logdet_readout_coverage(rank_one, clean)
    high = logdet_readout_coverage(full_rank, clean)
    assert high > low > 0
    gradient = torch.autograd.grad(high, full_rank)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_amplitude_spectrum_view_preserves_fourier_phase():
    torch.manual_seed(33)
    image = torch.rand(1, 3, 16, 16, requires_grad=True)
    log_multiplier = 0.2 * torch.randn(1, 1, 16, 9)
    viewed = amplitude_spectrum_view(image, log_multiplier)
    original_spectrum = torch.fft.rfft2(image.detach(), norm="ortho")
    viewed_spectrum = torch.fft.rfft2(viewed.detach(), norm="ortho")
    supported = original_spectrum.abs() > 1e-5
    cross = original_spectrum.conj() * viewed_spectrum
    assert cross.imag.masked_select(supported).abs().max() < 1e-4
    assert cross.real.masked_select(supported).min() > 0
    torch.autograd.grad(viewed.square().mean(), image)


def test_patchout_mask_uses_whole_patches_and_requested_fraction():
    image = torch.zeros(1, 3, 32, 32)
    generator = torch.Generator().manual_seed(5)
    mask = patchout_mask(image, 0.5, 16, generator)
    assert mask.shape == (1, 1, 32, 32)
    assert mask.mean() == 0.5
    for row in range(2):
        for col in range(2):
            patch = mask[..., row * 16:(row + 1) * 16, col * 16:(col + 1) * 16]
            assert patch.unique().numel() == 1


def test_qk_bypass_preserves_forward_and_value_gradient():
    class ToyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(2, 6, bias=False)

        def forward(self, value):
            q, k, v = self.qkv(value).chunk(3, dim=-1)
            return q + 2 * k + 3 * v

    class ToyQKBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = ToyAttention()
            self.mlp = nn.Identity()

    class ToyQKEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([ToyQKBlock()])

    encoder = ToyQKEncoder()
    value = torch.randn(1, 2, requires_grad=True)
    reference = encoder.blocks[0].attn(value)
    reference_gradient = torch.autograd.grad(reference.sum(), value, retain_graph=True)[0]
    with QKGradientBypassView(encoder, 0.0).active():
        bypassed = encoder.blocks[0].attn(value)
    bypass_gradient = torch.autograd.grad(bypassed.sum(), value)[0]
    expected_v_gradient = 3 * encoder.blocks[0].attn.qkv.weight[4:].sum(0, keepdim=True)
    assert torch.allclose(reference, bypassed)
    assert torch.allclose(bypass_gradient, expected_v_gradient)
    assert not torch.allclose(reference_gradient, bypass_gradient)


def test_adapter_invariant_variants_isolate_qk_and_patchout():
    assert ablation_config("FA").use_qk_bypass
    assert not ablation_config("FA").use_patchout
    assert ablation_config("FC").use_patchout
    assert not ablation_config("FC").use_qk_bypass
    assert ablation_config("FD").use_qk_bypass and ablation_config("FD").use_patchout
    tail = ablation_config("FG")
    assert tail.use_adapter_tail and tail.adapter_rank == 192
    assert tail.pasd_mode == "shuffled_amplitude" and tail.use_grat


def test_dc_feature_shift_is_zero_frequency_displacement():
    clean = torch.randn(1, 4, 8, 8)
    same = dc_feature_shift(clean, clean)
    shifted = (clean + 0.2).requires_grad_(True)
    displaced = dc_feature_shift(shifted, clean)
    assert same == 0 and displaced > 0
    gradient = torch.autograd.grad(displaced, shifted)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert ablation_config("FJ").use_dc_shift


def test_lpdc_alias_is_the_selected_low_phase_configuration():
    selected = ablation_config("DL")
    named = ablation_config("LPDC")
    assert named.use_pasd and named.use_input_phase
    assert named.input_phase_low == selected.input_phase_low == 0.0
    assert named.input_phase_high == selected.input_phase_high == 0.2
    assert named.input_phase_weight == selected.input_phase_weight == 0.05
    assert named.pasd_schedule == selected.pasd_schedule == "final_then_late"


def test_depth_robust_softmin_is_bounded_by_the_mean_and_has_gradients():
    left = torch.tensor(0.7, requires_grad=True)
    right = torch.tensor(1.1, requires_grad=True)
    robust = smooth_minimum([left, right], beta=8.0)
    assert min(left, right) <= robust <= 0.5 * (left + right)
    robust.backward()
    assert left.grad > right.grad > 0
    assert ablation_config("DS").pasd_depth_robust
    assert ablation_config("DT").use_input_phase


def test_grat_pasd_continuations_have_complementary_schedules():
    forward = ablation_config("DU")
    reverse = ablation_config("DV")
    joint = ablation_config("DW")
    assert forward.use_grat and forward.use_pasd
    assert forward.pasd_schedule == "grat_then_pasd"
    assert reverse.pasd_schedule == "pasd_then_grat"
    assert joint.pasd_schedule == "" and joint.use_grat and joint.use_pasd
    assert [ablation_config(name).pasd_weight for name in ("DX", "DY", "DZ")] == [0.25, 0.5, 0.75]
    assert [ablation_config(name).pasd_weight for name in ("EA", "EB", "EC")] == [0.05, 0.1, 0.15]
    assert [ablation_config(name).pasd_switch_step for name in ("ED", "EE", "EF")] == [9, 8, 7]
    assert [(ablation_config(name).input_phase_low, ablation_config(name).input_phase_high) for name in ("EG", "EH", "EI")] == [(0.0, 0.2), (0.05, 0.4), (0.2, 1.0)]
    assert ablation_config("EJ").pasd_mode == "shuffled_amplitude"
    assert ablation_config("EK").pasd_mode == "clean_amplitude"
    assert ablation_config("EJ").pasd_weight == ablation_config("EK").pasd_weight == 0.25
    assert [ablation_config(name).pasd_early_weight for name in ("EL", "EM", "EN")] == [0.1, 0.25, 0.5]
    assert all(ablation_config(name).pasd_region == "dual" for name in ("EL", "EM", "EN"))
    assert [ablation_config(name).polar_amplitude_weight for name in ("EO", "EP", "EQ")] == [0.1, 0.25, 0.5]
    assert all(ablation_config(name).use_polar_amplitude_gradient for name in ("EO", "EP", "EQ"))
    assert [ablation_config(name).logdet_coverage_weight for name in ("ER", "ES", "ET")] == [0.05, 0.1, 0.25]
    assert all(ablation_config(name).use_logdet_coverage for name in ("ER", "ES", "ET"))
    assert [ablation_config(name).collapse_weight for name in ("EU", "EV", "EW")] == [0.05, 0.1, 0.25]
    assert all(ablation_config(name).use_token_collapse for name in ("EU", "EV", "EW"))
    assert [ablation_config(name).amplitude_eot_scale for name in ("EX", "EY", "EZ")] == [0.1, 0.2, 0.3]
    assert all(ablation_config(name).use_amplitude_eot for name in ("EX", "EY", "EZ"))


def test_input_amplitude_objective_is_zero_only_for_matching_amplitude():
    torch.manual_seed(31)
    clean = torch.rand(1, 3, 8, 8)
    same = input_amplitude_disruption(clean, clean)
    scaled = input_amplitude_disruption(0.9 * clean, clean)
    assert same.abs() < 1e-6
    assert scaled > same
    skip_robust = ablation_config("K")
    assert skip_robust.use_skip_gradient and skip_robust.skip_gamma == 0.5
    path_robust = ablation_config("L")
    assert path_robust.use_skip_ensemble and sum(path_robust.skip_weights) == 1.0
    dual_path = ablation_config("M")
    assert dual_path.use_dual_path and len(dual_path.skip_gammas) == 2
    spectral_robust = ablation_config("N")
    assert spectral_robust.use_spectral_uncertainty and spectral_robust.skip_gamma == 0.0
    phase_robust = ablation_config("O")
    assert phase_robust.use_skip_ensemble and phase_robust.use_phase_robust
    assert len(phase_robust.skip_gammas) == 3
    assert ablation_config("P").use_feature_phase and ablation_config("P").skip_gamma == 0.0
    assert ablation_config("Q").use_feature_phase and ablation_config("Q").skip_gamma == 0.5
    assert ablation_config("R").use_grat and not ablation_config("R").use_skip_gradient
    assert ablation_config("S").use_grat and ablation_config("S").skip_gamma == 0.0
    assert ablation_config("T").use_grat and ablation_config("T").skip_gamma == 0.5
    robust_fusion = ablation_config("U")
    assert robust_fusion.use_grat and robust_fusion.use_ads
    assert robust_fusion.use_cross_spectral_anchor and robust_fusion.skip_gamma == 0.5
    assert ablation_config("V").auxiliary_weight == 2.0
    assert ablation_config("W").auxiliary_weight == 4.0
    assert ablation_config("X").use_patch_loss and ablation_config("X").output_weight == 0.0
    assert ablation_config("Y").use_patch_loss and ablation_config("Y").skip_gamma == 0.0
    assert ablation_config("Z").use_patch_loss and ablation_config("Z").skip_gamma == 0.5
    assert ablation_config("AA").use_token_collapse and ablation_config("AA").skip_gamma == 0.0
    assert ablation_config("AB").use_token_collapse and ablation_config("AB").skip_gamma == 0.5
    assert ablation_config("AC").use_low_level_lift and not ablation_config("AC").use_skip_gradient
    assert ablation_config("AD").use_low_level_lift and ablation_config("AD").skip_gamma == 0.0
    assert ablation_config("AE").use_low_level_lift and ablation_config("AE").skip_gamma == 0.5
    assert ablation_config("AF").use_pasd and ablation_config("AF").pasd_mode == "shuffled_amplitude"
    assert ablation_config("AG").use_pasd and ablation_config("AG").pasd_mode == "shuffled_white"
    assert ablation_config("AH").use_pasd and ablation_config("AH").pasd_mode == "clean_amplitude"
    assert ablation_config("AI").use_pasd and ablation_config("AI").skip_gamma == 0.0
    assert ablation_config("AJ").use_pasd and ablation_config("AJ").skip_gamma == 0.5
    assert ablation_config("AK").pasd_mode == "shuffled_white" and ablation_config("AK").skip_gamma == 0.0
    assert ablation_config("AL").pasd_mode == "shuffled_white" and ablation_config("AL").skip_gamma == 0.5
    assert ablation_config("AM").pasd_region == "early" and ablation_config("AM").pasd_layer_count == 2
    assert ablation_config("AN").pasd_region == "early" and ablation_config("AN").pasd_layer_count == 4
    assert ablation_config("AO").pasd_region == "mixed" and ablation_config("AO").pasd_layer_count == 4
    assert ablation_config("AP").pasd_region == "mixed" and ablation_config("AP").pasd_layer_count == 6
    assert ablation_config("AQ").use_atcs and not ablation_config("AQ").use_pasd
    assert ablation_config("AT").use_atcs and ablation_config("AT").use_pasd
    assert ablation_config("AU").pasd_region == "final" and ablation_config("AU").pasd_layer_count == 2
    assert ablation_config("AX").pasd_region == "final" and ablation_config("AX").pasd_layer_count == 12
    assert not ablation_config("AY").use_edge and ablation_config("AY").pasd_region == "late"
    assert ablation_config("AZ").output_weight == 0.0
    assert ablation_config("BA").pasd_layer_count == 1 and ablation_config("BA").use_edge
    assert ablation_config("BB").pasd_layer_count == 1 and not ablation_config("BB").use_edge
    assert ablation_config("BC").pasd_layer_count == 2 and not ablation_config("BC").use_edge
    assert ablation_config("BD").pasd_layer_count == 2 and ablation_config("BD").output_weight == 0.0
    assert ablation_config("BE").pasd_mode == "spatial_shuffled_white"
    assert ablation_config("BF").pasd_mode == "spatial_uniform_white"
    assert ablation_config("BG").pasd_region == "final" and ablation_config("BG").pasd_layer_count == 2
    assert ablation_config("BH").pasd_mode == "spatial_uniform_white"
    assert ablation_config("BI").pasd_region == "dual" and ablation_config("BI").pasd_early_weight == 0.25
    assert ablation_config("BK").pasd_early_weight == 1.0
    assert ablation_config("BL").pasd_mode == "spatial_uniform_white"
    assert ablation_config("BM").pasd_early_weight == 1.0
    assert ablation_config("BN").pasd_region == "late_final" and ablation_config("BN").pasd_final_weight == 0.25
    assert ablation_config("BQ").pasd_final_weight == 2.0
    assert ablation_config("BR").use_edge and ablation_config("BS").pasd_final_weight == 1.0
    assert [ablation_config(name).post_ti_kernel for name in ("BT", "BU", "BV", "BW")] == [3, 5, 9, 15]
    assert ablation_config("BX").pasd_schedule == "late_then_final"
    assert ablation_config("BY").pasd_schedule == "final_then_late"
    assert ablation_config("BZ").pasd_schedule == "alternate"
    assert ablation_config("CA").use_edge
    assert ablation_config("CB").use_ads and not ablation_config("CB").use_pasd
    assert ablation_config("CC").pasd_schedule == "final_then_ads" and not ablation_config("CC").use_edge
    assert ablation_config("CD").use_edge and ablation_config("CD").pasd_schedule == "final_then_ads"
    assert [ablation_config(name).pasd_switch_step for name in ("CE", "CF", "CG")] == [3, 4, 6]
    assert ablation_config("CH").use_edge and ablation_config("CH").pasd_switch_step == 4
    assert ablation_config("CI").pasd_schedule == "final_then_ads" and ablation_config("CI").first_step_jitter == 0
    assert not ablation_config("CJ").use_edge
    assert ablation_config("CK").use_ads and not ablation_config("CK").use_pasd
    assert ablation_config("CL").use_final_ads_anchor and ablation_config("CL").auxiliary_weight == 0.25
    assert ablation_config("CN").anchor_floor == 1.0
    assert not ablation_config("CO").use_edge
    assert ablation_config("CP").adapter_view_rank == 192
    assert ablation_config("CQ").adapter_view_count == 4 and ablation_config("CQ").ads_all_views
    assert ablation_config("CR").robust_weight == 0.5


def test_spatial_fourier_probe_uses_bhwc_spatial_axes():
    torch.manual_seed(37)
    clean = torch.randn(2, 7, 9, 5)
    for mode in ("spatial_shuffled_white", "spatial_uniform_white"):
        probe = spatial_fourier_probe_bhwc(clean, mode)
        assert probe.shape == clean.shape
        assert torch.isfinite(probe).all()
        assert probe.square().mean() > 0


def test_spatial_pasd_has_finite_gradient():
    torch.manual_seed(41)
    clean = [torch.randn(1, 7, 9, 8) for _ in range(8)]
    adv = [(value + 1e-3 * torch.randn_like(value)).requires_grad_(True) for value in clean]
    cfg = ablation_config("BF", pasd_samples=1, pasd_layer_count=4)
    loss = phase_amplitude_stochastic_disruption(adv, clean, cfg)
    gradients = torch.autograd.grad(loss, adv, allow_unused=True)
    used = [value for value in gradients if value is not None]
    assert torch.isfinite(loss) and len(used) == 4
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in used)


def test_dual_depth_pasd_keeps_early_and_late_gradients():
    torch.manual_seed(43)
    clean = [torch.randn(1, 7, 9, 8) for _ in range(12)]
    adv = [(value + 1e-3 * torch.randn_like(value)).requires_grad_(True) for value in clean]
    loss = phase_amplitude_stochastic_disruption(
        adv, clean, ablation_config("BJ", pasd_samples=1)
    )
    gradients = torch.autograd.grad(loss, adv, allow_unused=True)
    used = [index for index, value in enumerate(gradients) if value is not None and value.abs().sum() > 0]
    assert used == [0, 1, 4, 5, 6, 7, 8, 9]


def test_late_final_pasd_keeps_both_depth_groups():
    torch.manual_seed(47)
    clean = [torch.randn(1, 7, 9, 8) for _ in range(12)]
    adv = [(value + 1e-3 * torch.randn_like(value)).requires_grad_(True) for value in clean]
    loss = phase_amplitude_stochastic_disruption(
        adv, clean, ablation_config("BP", pasd_samples=1)
    )
    gradients = torch.autograd.grad(loss, adv, allow_unused=True)
    used = [index for index, value in enumerate(gradients) if value is not None and value.abs().sum() > 0]
    assert used == list(range(4, 12))


def test_translation_orbit_average_preserves_constant_gradient():
    gradient = torch.ones(1, 3, 17, 17)
    averaged = translation_orbit_average(gradient, 5)
    assert torch.allclose(averaged[:, :, 2:-2, 2:-2], torch.ones(1, 3, 13, 13))
    assert averaged.shape == gradient.shape and torch.isfinite(averaged).all()


def test_average_token_angular_disruption_has_gradient():
    torch.manual_seed(31)
    clean = [torch.randn(1, 3, 3, 8) for _ in range(8)]
    adv = [(value + 1e-3 * torch.randn_like(value)).requires_grad_(True) for value in clean]
    loss = average_token_angular_disruption(adv, clean, count=4)
    gradients = torch.autograd.grad(loss, adv, allow_unused=True)
    used = [value for value in gradients if value is not None]
    assert torch.isfinite(loss) and len(used) == 4
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in used)


def test_token_gram_is_channel_rotation_invariant_and_has_gradient():
    torch.manual_seed(37)
    clean = [torch.randn(1, 4, 4, 8) for _ in range(8)]
    rotation, _ = torch.linalg.qr(torch.randn(8, 8))
    rotated = [value @ rotation for value in clean]
    invariant = normalized_token_gram_disruption(rotated, clean, count=4, pool_size=4)
    assert invariant < 1e-6

    adv = [(value + 1e-2 * torch.randn_like(value)).requires_grad_(True) for value in clean]
    loss = normalized_token_gram_disruption(adv, clean, count=4, pool_size=4)
    gradients = torch.autograd.grad(loss, adv, allow_unused=True)
    used = [value for value in gradients if value is not None]
    assert torch.isfinite(loss) and loss > 0 and len(used) == 4
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in used)

    identical = [value.detach().clone().requires_grad_(True) for value in clean]
    zero = normalized_token_gram_disruption(identical, clean, count=4, pool_size=4)
    zero_gradients = torch.autograd.grad(zero, identical, allow_unused=True)
    assert torch.isfinite(zero) and abs(float(zero)) < 1e-8
    assert all(value is None or torch.isfinite(value).all() for value in zero_gradients)


def test_low_level_lift_is_normalized_and_has_gradient():
    clean = [torch.ones(1, 4, 3, 3), 2 * torch.ones(1, 4, 3, 3)]
    adv = [(1.5 * value).requires_grad_(True) for value in clean]
    objective = low_level_activation_lift(adv, clean, count=2)
    gradients = torch.autograd.grad(objective, adv)
    assert torch.allclose(objective, torch.tensor(1.5))
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients)


def test_adapter_tail_matches_eckart_young_residual():
    singular_values = torch.arange(8.0, 0.0, -1.0)
    clean = torch.ones(1, 8, 1, 8)
    displacement = torch.diag(singular_values).reshape_as(clean)
    _, ratio = adapter_irreducible_tail(
        {(0, "attn"): clean + displacement}, {(0, "attn"): clean}, rank=3
    )
    expected = singular_values[3:].square().sum() / singular_values.square().sum()
    assert torch.allclose(ratio, expected)


def test_frequency_tail_has_finite_gradient():
    torch.manual_seed(4)
    clean = torch.randn(1, 8, 8, 8)
    adv = (clean + 0.1 * torch.randn_like(clean)).requires_grad_(True)
    objective, ratio = frequency_balanced_irreducible_tail(adv, clean, rank=3)
    gradient = torch.autograd.grad(objective, adv)[0]
    assert torch.isfinite(objective) and torch.isfinite(ratio)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_cross_spectral_anchor_preserves_anchor_floor():
    torch.manual_seed(3)
    anchor = torch.randn(1, 3, 9, 11)
    auxiliary = torch.randn_like(anchor)
    fused, diagnostics = cross_spectral_anchor(anchor, auxiliary, floor=0.5)
    normalized_anchor = anchor / anchor.abs().mean()
    assert torch.sum(normalized_anchor * fused) >= 0.5 * torch.sum(normalized_anchor.square()) - 1e-4
    assert diagnostics["minimum_anchor_ratio"] >= 0.5 - 1e-5


def test_phase_interval_direction_is_deterministic_and_serves_all_paths():
    torch.manual_seed(11)
    common = torch.randn(1, 3, 9, 11)
    gradients = [common + 0.15 * torch.randn_like(common) for _ in range(3)]
    first, diagnostics = phase_interval_robust_direction(gradients)
    second, repeated = phase_interval_robust_direction(gradients)
    assert torch.equal(first, second)
    assert diagnostics == repeated
    assert diagnostics["min_path_cosine"] > 0
    assert 0 <= diagnostics["phase_rejected_fraction"] <= 1


def test_feature_phase_disruption_has_finite_gradient():
    torch.manual_seed(13)
    clean = torch.randn(1, 8, 7, 9)
    adv = (clean + 1e-3 * torch.randn_like(clean)).requires_grad_(True)
    loss = feature_phase_disruption(adv, clean)
    gradient = torch.autograd.grad(loss, adv)[0]
    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_gradient_robust_disruption_has_finite_gradient():
    torch.manual_seed(17)
    random_state = __import__("random")
    random_state.seed(17)
    clean = [torch.randn(1, 8, 7, 9) for _ in range(8)]
    adv = [(value + 1e-3 * torch.randn_like(value)).requires_grad_(True) for value in clean]
    loss = gradient_robust_disruption(adv, clean, AttackConfig())
    gradients = torch.autograd.grad(loss, adv, allow_unused=True)
    assert torch.isfinite(loss)
    used = [value for value in gradients if value is not None]
    assert len(used) == 4
    assert all(torch.isfinite(value).all() for value in used)


def test_patch_embedding_disruption_has_finite_gradient():
    torch.manual_seed(19)
    clean = torch.randn(1, 7, 9, 8)
    adv = (clean + 1e-3 * torch.randn_like(clean)).requires_grad_(True)
    loss = patch_embedding_disruption(adv, clean)
    gradient = torch.autograd.grad(loss, adv)[0]
    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_token_collapse_has_finite_descent_gradient():
    torch.manual_seed(23)
    clean = torch.randn(1, 8, 7, 9)
    adv = clean.clone().requires_grad_(True)
    loss = token_collapse_objective(adv, clean)
    gradient = torch.autograd.grad(loss, adv)[0]
    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_spatial_graph_collapse_reduces_normalized_token_variation():
    torch.manual_seed(41)
    clean = [torch.randn(1, 7, 9, 8) for _ in range(3)]
    adv = [value.clone().requires_grad_(True) for value in clean]
    objective = spatial_graph_collapse(adv, clean, count=3)
    gradients = torch.autograd.grad(objective, adv)
    stepped = [value + 1e-2 * gradient for value, gradient in zip(adv, gradients)]
    improved = spatial_graph_collapse(stepped, clean, count=3)
    assert torch.isfinite(objective) and torch.isfinite(improved)
    assert improved > objective
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients)


def test_qk_topology_disruption_is_scale_invariant_and_differentiable():
    torch.manual_seed(43)
    clean = torch.randn(2, 4, 4, 3 * 8)
    scaled = 3.0 * clean
    assert qk_topology_disruption(scaled, clean, num_heads=2) < 1e-5
    adv = (clean + 1e-2 * torch.randn_like(clean)).requires_grad_(True)
    loss = qk_topology_disruption(adv, clean, num_heads=2)
    gradient = torch.autograd.grad(loss, adv)[0]
    assert torch.isfinite(loss) and loss > 0
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_spatial_quadrature_target_preserves_fourier_amplitude():
    torch.manual_seed(47)
    features = torch.randn(2, 5, 7, 10)
    target = spatial_quadrature_target(features)
    source_amplitude = torch.fft.fft(features, dim=-1, norm="ortho").abs()
    target_amplitude = torch.fft.fft(target, dim=-1, norm="ortho").abs()
    assert torch.allclose(source_amplitude, target_amplitude, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(target).all()


def test_spatial_translation_target_is_isospectral_and_collision_has_gradient():
    torch.manual_seed(53)
    features = torch.randn(2, 5, 8, 10)
    target = spatial_translation_target(features, (3, -2))
    source_amplitude = torch.fft.fft2(features, dim=(-2, -1), norm="ortho").abs()
    target_amplitude = torch.fft.fft2(target, dim=(-2, -1), norm="ortho").abs()
    assert torch.allclose(source_amplitude, target_amplitude, atol=2e-5, rtol=2e-5)
    adv = (features + 1e-3 * torch.randn_like(features)).requires_grad_(True)
    loss = translation_collision_objective(adv, features, (3, -2))
    gradient = torch.autograd.grad(loss, adv)[0]
    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_spatial_principal_disruption_targets_clean_modes():
    torch.manual_seed(59)
    clean = torch.randn(1, 12, 6, 7)
    basis = spatial_principal_basis(clean, rank=4)
    adv = (clean + 1e-3 * torch.randn_like(clean)).requires_grad_(True)
    loss = spatial_principal_disruption(adv, clean, basis)
    gradient = torch.autograd.grad(loss, adv)[0]
    assert basis.shape == (42, 4)
    assert torch.isfinite(loss) and loss > 0
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_semantic_partition_is_deterministic_and_differentiable():
    torch.manual_seed(61)
    clean = torch.randn(1, 16, 6, 7)
    centroids, labels = semantic_partition(clean, clusters=5)
    repeated_centroids, repeated_labels = semantic_partition(clean, clusters=5)
    assert torch.equal(labels, repeated_labels)
    assert torch.allclose(centroids, repeated_centroids)
    adv = (clean + 1e-3 * torch.randn_like(clean)).requires_grad_(True)
    loss = semantic_partition_disruption(adv, centroids, labels, 0.1)
    gradient = torch.autograd.grad(loss, adv)[0]
    assert torch.isfinite(loss) and loss > 0
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_cancellation_gap_penalizes_final_displacement():
    torch.manual_seed(67)
    clean_final = torch.randn(1, 8, 5, 6)
    clean_layers = [torch.randn(1, 5, 6, 8) for _ in range(4)]
    final = (clean_final + 1e-3 * torch.randn_like(clean_final)).requires_grad_(True)
    layers = [(value + 1e-3 * torch.randn_like(value)).requires_grad_(True) for value in clean_layers]
    no_penalty = cancellation_gap_disruption(final, clean_final, layers, clean_layers, 4, 0.0)
    penalized = cancellation_gap_disruption(final, clean_final, layers, clean_layers, 4, 0.5)
    gradients = torch.autograd.grad(penalized, [final, *layers])
    assert penalized < no_penalty
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients)
