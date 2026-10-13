import torch

from peft_transfer.generate_synapse_prototype_attack import (
    balanced_prototype_loss,
    balanced_prototype_margin,
    background_swap_classes,
    class_prototypes,
    class_prototype_bank,
    class_prototype_bank_from_features,
    coverage_margin,
    balanced_lda_loss,
    diagonal_lda_bank_from_features,
    diagonal_lda_scores,
    feature_bchw,
    lda_coverage_margin,
    nearest_wrong_classes,
    prototype_attack_loss,
    spatial_absent_class_targets,
    spatial_collision_image,
    spatial_collision_loss,
    smooth_deformation_image,
    deformation_fidelity_loss,
)


def test_smooth_deformation_is_nonperiodic_and_shape_preserving():
    image = torch.zeros(1, 1, 17, 17)
    image[..., 4:13, 4:13] = 1
    deformed = smooth_deformation_image(image, 0.2)
    assert deformed.shape == image.shape
    assert float(deformed.min()) >= 0
    assert float(deformed.max()) <= 1
    assert not torch.equal(deformed, image)
    assert torch.equal(deformed[..., 0, 0], image[..., 0, 0])


def test_deformation_fidelity_prefers_deformed_features():
    clean = [torch.tensor([[[[1.0]], [[0.0]]]])]
    deformed = [torch.tensor([[[[0.0]], [[1.0]]]])]
    clean_score = deformation_fidelity_loss(clean, clean, deformed)
    deformed_score = deformation_fidelity_loss(deformed, clean, deformed)
    assert deformed_score > clean_score


def test_prototype_loss_prefers_clean_class_assignment():
    feature = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]],
                             [[0.0, 0.0], [1.0, 1.0]]]])
    label = torch.tensor([[[[0, 0], [1, 1]]]])
    prototypes, targets, classes = class_prototypes(feature, label)
    clean_loss = balanced_prototype_loss(feature, prototypes, targets)
    swapped_loss = balanced_prototype_loss(feature.flip(-2), prototypes, targets)
    assert classes.tolist() == [0, 1]
    assert swapped_loss > clean_loss


def test_margin_prefers_semantic_class_swap():
    prototypes = torch.eye(3)
    targets = torch.tensor([[[0, 1], [2, 0]]])
    confusers = nearest_wrong_classes(prototypes)
    clean = prototypes[targets].permute(0, 3, 1, 2).contiguous()
    swapped = prototypes[confusers[targets]].permute(0, 3, 1, 2).contiguous()
    clean_loss = balanced_prototype_margin(clean, prototypes, targets, confusers)
    swapped_loss = balanced_prototype_margin(swapped, prototypes, targets, confusers)
    assert swapped_loss > clean_loss


def test_prototype_attack_loss_rejects_unknown_objective():
    feature = torch.randn(1, 3, 2, 2)
    prototypes = torch.eye(3)
    targets = torch.zeros(1, 2, 2, dtype=torch.long)
    try:
        prototype_attack_loss(feature, prototypes, targets, "unknown", 0.07)
    except ValueError as error:
        assert "unknown prototype objective" in str(error)
    else:
        raise AssertionError("unknown objective must fail")


def test_background_swap_maps_organs_to_background():
    prototypes = torch.eye(3)
    class_ids = torch.tensor([0, 3, 7])
    confusers = background_swap_classes(prototypes, class_ids)
    assert confusers[1:].tolist() == [0, 0]
    assert int(confusers[0]) in (1, 2)


def test_prototype_bank_adds_reference_only_class():
    current_feature = torch.tensor([[[[1.0, 1.0]], [[0.0, 0.0]], [[0.0, 0.0]]]])
    reference_feature = torch.tensor([[[[0.0, 0.0]], [[0.0, 0.0]], [[1.0, 1.0]]]])
    current_label = torch.zeros(1, 1, 1, 2, dtype=torch.long)
    reference_label = torch.full((1, 1, 1, 2), 8, dtype=torch.long)
    features = iter((current_feature, reference_feature))
    prototypes, targets, classes = class_prototype_bank(
        lambda _image: next(features),
        [(current_feature, current_label), (reference_feature, reference_label)],
    )
    assert classes.tolist() == [0, 8]
    assert prototypes.shape == (2, 3)
    assert torch.equal(targets, torch.zeros_like(targets))


def test_ce_single_class_has_zero_gradient_for_base_only_fallback():
    feature = torch.randn(1, 3, 2, 2, requires_grad=True)
    label = torch.zeros(1, 1, 2, 2, dtype=torch.long)
    prototypes, targets, classes = class_prototypes(feature.detach(), label)
    loss = prototype_attack_loss(feature, prototypes, targets, "ce", 0.07, classes)
    gradient = torch.autograd.grad(loss, feature)[0]
    assert classes.tolist() == [0]
    assert torch.count_nonzero(gradient) == 0


def test_margin_rejects_single_class():
    feature = torch.randn(1, 3, 2, 2)
    label = torch.zeros(1, 1, 2, 2, dtype=torch.long)
    prototypes, targets, classes = class_prototypes(feature, label)
    try:
        prototype_attack_loss(feature, prototypes, targets, "margin", 0.07, classes)
    except ValueError as error:
        assert "at least two classes" in str(error)
    else:
        raise AssertionError("single-class margin objective must fail")


def test_base_only_schedule_is_exposed_in_attack_signature():
    import inspect
    from peft_transfer.generate_synapse_prototype_attack import attack

    parameter = inspect.signature(attack).parameters["base_only_steps"]
    assert parameter.default == 0


def test_bhwc_block_feature_matches_bchw_layout():
    bchw = torch.randn(1, 8, 2, 2)
    bhwc = bchw.permute(0, 2, 3, 1)
    assert torch.equal(feature_bchw(bhwc), bchw)
    assert torch.equal(feature_bchw(bchw), bchw)


def test_cached_prototype_bank_matches_function_bank():
    feature_a = torch.randn(1, 4, 2, 2)
    feature_b = torch.randn(1, 4, 2, 2)
    label_a = torch.zeros(1, 1, 2, 2, dtype=torch.long)
    label_b = torch.ones(1, 1, 2, 2, dtype=torch.long)
    features = iter((feature_a, feature_b))
    expected = class_prototype_bank(
        lambda _image: next(features),
        [(feature_a, label_a), (feature_b, label_b)],
    )
    actual = class_prototype_bank_from_features([
        (feature_a, label_a), (feature_b, label_b),
    ])
    for left, right in zip(expected, actual):
        assert torch.equal(left, right)


def test_coverage_margin_prefers_absent_class_stripes():
    prototypes = torch.eye(3)
    class_ids = torch.tensor([0, 1, 2])
    targets = torch.zeros(1, 2, 2, dtype=torch.long)
    clean = prototypes[targets].permute(0, 3, 1, 2).contiguous()
    covered = torch.stack((
        prototypes[1], prototypes[2], prototypes[1], prototypes[2]
    )).reshape(1, 2, 2, 3).permute(0, 3, 1, 2).contiguous()
    assert coverage_margin(covered, prototypes, targets, class_ids) > coverage_margin(
        clean, prototypes, targets, class_ids
    )


def test_diagonal_lda_scores_are_invariant_to_channel_rescaling():
    feature = torch.tensor([[[[1.0, 2.0]], [[3.0, 5.0]]]])
    label = torch.tensor([[[[0, 1]]]])
    reference_feature = torch.tensor([[[[1.5, 2.5]], [[4.0, 6.0]]]])
    means, precision, targets, classes = diagonal_lda_bank_from_features([
        (feature, label), (reference_feature, label),
    ])
    scores = diagonal_lda_scores(feature, means, precision)
    scale = torch.tensor([2.0, 0.5]).view(1, 2, 1, 1)
    scaled_means, scaled_precision, _, _ = diagonal_lda_bank_from_features([
        (feature * scale, label), (reference_feature * scale, label),
    ])
    scaled_scores = diagonal_lda_scores(feature * scale, scaled_means, scaled_precision)
    assert classes.tolist() == [0, 1]
    assert targets.tolist() == [[[0, 1]]]
    assert torch.allclose(scores, scaled_scores, atol=1e-4, rtol=1e-4)


def test_lda_loss_increases_after_class_swap():
    feature = torch.tensor([[[[0.0, 2.0]], [[0.0, 2.0]]]])
    label = torch.tensor([[[[0, 1]]]])
    reference = feature + torch.tensor([[[[0.1, -0.1]], [[-0.1, 0.1]]]])
    means, precision, targets, _ = diagonal_lda_bank_from_features([
        (feature, label), (reference, label),
    ])
    assert balanced_lda_loss(feature.flip(-1), means, precision, targets) > balanced_lda_loss(
        feature, means, precision, targets
    )


def test_lda_coverage_prefers_absent_classes_in_distinct_regions():
    means = torch.eye(3)
    precision = torch.ones(3)
    classes = torch.tensor([0, 1, 2])
    targets = torch.zeros(1, 2, 2, dtype=torch.long)
    clean = means[targets].permute(0, 3, 1, 2).contiguous()
    covered = torch.stack((means[1], means[2], means[1], means[2]))
    covered = covered.reshape(1, 2, 2, 3).permute(0, 3, 1, 2).contiguous()
    assert lda_coverage_margin(covered, means, precision, targets, classes) > (
        lda_coverage_margin(clean, means, precision, targets, classes)
    )


def test_absent_class_targets_exclude_present_classes():
    label = torch.tensor([[[[0, 1], [1, 0]]]])
    targets = spatial_absent_class_targets(label, num_classes=4)
    assert set(torch.unique(targets).tolist()) == {2, 3}


def test_absent_class_targets_return_none_when_all_foreground_present():
    label = torch.tensor([[[[0, 1], [2, 3]]]])
    assert spatial_absent_class_targets(label, num_classes=4) is None


def test_spatial_collision_loss_prefers_translated_reference():
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    translated = spatial_collision_image(image, 0.25)
    assert spatial_collision_loss([translated], [translated]) > spatial_collision_loss(
        [image], [translated]
    )
