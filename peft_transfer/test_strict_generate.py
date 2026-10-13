from peft_transfer.generate import parse_args
from peft_transfer.core import ablation_config, monotone_contrast_view
import torch


BASE = ["--dataset", "Synapse", "--vit", "vit_b", "--out", "testset/strict_test"]


def test_strict_base_is_default_and_uses_no_prior():
    args = parse_args(BASE)
    assert args.threat_model == "strict-base"
    assert args.checkpoint is None
    assert args.umi_prior is None


def test_strict_base_rejects_external_components_or_unlabelled_grat():
    for extra in (
        ["--checkpoint", "other.pth"],
        ["--umi-prior", "prior.pth"],
        ["--prior-guidance-weight", "0.1"],
        ["--variant", "R"],
    ):
        try:
            parse_args(BASE + extra)
        except SystemExit as error:
            assert error.code != 0
        else:
            raise AssertionError(f"strict-base accepted {extra}")


def test_legacy_requires_explicit_opt_in():
    args = parse_args(BASE + ["--threat-model", "legacy", "--variant", "R"])
    assert args.threat_model == "legacy"


def test_strict_base_allows_grat_as_external_baseline():
    args = parse_args(BASE + ["--variant", "R", "--role", "external-baseline"])
    assert args.threat_model == "strict-base"
    assert args.role == "external-baseline"
    assert ablation_config(args.variant).use_grat


def test_ct_contrast_eot_is_training_free_and_order_preserving():
    args = parse_args(BASE + ["--variant", "CTEOT"])
    config = ablation_config(args.variant)
    assert args.threat_model == "strict-base"
    assert config.use_contrast_eot and not config.use_grat
    assert not config.use_peft_consensus
    image = torch.tensor([[[[0.2, 0.35, 0.5, 0.65, 0.8]]]])
    center = image.median()
    for scale in config.contrast_eot_scales:
        view = monotone_contrast_view(image, center, scale)
        assert torch.all(view[..., 1:] > view[..., :-1])
        assert torch.allclose(view[..., 2], image[..., 2])


def test_ct_early_uses_only_frozen_source_features():
    args = parse_args(BASE + ["--variant", "CTEARLY"])
    config = ablation_config(args.variant)
    assert args.threat_model == "strict-base"
    assert config.use_depth_layer and config.depth_layer_index == 0
    assert not config.use_grat and not config.use_peft_consensus


def test_ct_early_weight_variants_change_one_variable():
    tenth = ablation_config("CTEARLY010")
    quarter = ablation_config("CTEARLY025")
    half = ablation_config("CTEARLY050")
    full = ablation_config("CTEARLY")
    assert quarter.depth_layer_weight == 0.25
    assert half.depth_layer_weight == 0.50
    assert full.depth_layer_weight == 1.0
    for config in (tenth, quarter, half, full):
        assert config.use_depth_layer and config.depth_layer_index == 0
        assert not config.use_grat and not config.use_peft_consensus


def test_ct_early_stage_is_source_only_sequential_depth():
    config = ablation_config("CTEARLYSTAGE")
    assert config.use_depth_layer and config.depth_layer_index == 0
    assert config.depth_schedule == "early_then_final"
    assert not config.use_grat and not config.use_peft_consensus
    assert tenth.depth_layer_weight == 0.10
