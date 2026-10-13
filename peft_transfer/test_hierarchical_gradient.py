import torch

from peft_transfer.generate_2d_dual_peft import (
    antithetic_views,
    feature_loss,
    gradient_from_polar_components,
    hierarchical_gradient,
    hierarchical_polar_gradient,
    normalized_gradient,
    polar_gradient_components,
    project_grayscale_gradient,
    scheduled_proxy_indices,
    spatial_collision_target,
)


def test_hierarchical_mean_respects_seed_groups():
    gradients = [
        torch.tensor([1.0, 3.0]),
        torch.tensor([3.0, 1.0]),
        torch.tensor([5.0, 7.0]),
        torch.tensor([7.0, 5.0]),
    ]
    value, retained = hierarchical_gradient(gradients, group_size=2)
    assert torch.equal(value, torch.tensor([4.0, 4.0]))
    assert retained == 1.0


def test_confidence_penalties_only_shrink_coordinates():
    gradients = [
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0, -1.0]),
        torch.tensor([3.0, 1.0]),
        torch.tensor([3.0, -1.0]),
    ]
    mean, _ = hierarchical_gradient(gradients, group_size=2)
    robust, retained = hierarchical_gradient(
        gradients, group_size=2, seed_lambda=0.5, trajectory_lambda=0.5
    )
    assert torch.all(robust.abs() <= mean.abs())
    assert 0.0 <= retained <= 1.0


def test_invalid_proxy_grouping_is_rejected():
    try:
        hierarchical_gradient([torch.ones(1), torch.ones(1), torch.ones(1)], group_size=2)
    except ValueError as error:
        assert "cannot be divided" in str(error)
    else:
        raise AssertionError("invalid grouping was accepted")


def test_polar_gradient_round_trip():
    generator = torch.Generator().manual_seed(19)
    image = torch.randn(1, 3, 16, 16, generator=generator)
    gradient = torch.randn(1, 3, 16, 16, generator=generator)
    amplitude, phase, spectrum = polar_gradient_components(gradient, image)
    recovered = gradient_from_polar_components(amplitude, phase, spectrum)
    assert torch.allclose(recovered, gradient, atol=2e-5, rtol=2e-5)


def test_unpenalized_polar_ensemble_equals_pixel_mean():
    generator = torch.Generator().manual_seed(23)
    image = torch.randn(1, 3, 16, 16, generator=generator)
    gradients = [torch.randn(1, 3, 16, 16, generator=generator) for _ in range(4)]
    polar, retained = hierarchical_polar_gradient(gradients, image, group_size=2)
    assert torch.allclose(polar, torch.stack(gradients).mean(0), atol=2e-5, rtol=2e-5)
    assert retained == 1.0


def test_training_time_schedule_is_seed_major():
    assert scheduled_proxy_indices(6, 3, 0, "forward") == [0, 3]
    assert scheduled_proxy_indices(6, 3, 4, "forward") == [1, 4]
    assert scheduled_proxy_indices(6, 3, 9, "forward") == [2, 5]
    assert scheduled_proxy_indices(6, 3, 0, "reverse") == [2, 5]
    assert scheduled_proxy_indices(6, 3, 9, "reverse") == [0, 3]


def test_antithetic_views_are_centered_and_deterministic():
    image = torch.full((1, 3, 8, 8), 0.5)
    first = antithetic_views(image, 1.0 / 255.0, seed=7)
    second = antithetic_views(image, 1.0 / 255.0, seed=7)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.allclose((first[0] + first[1]) / 2.0, image)


def test_centered_antithetic_views_remain_centered_at_pixel_bounds():
    image = torch.tensor([[[[0.0, 0.2], [0.8, 1.0]]]])
    views = antithetic_views(image, 1.0 / 255.0, seed=7, boundary="centered")
    assert torch.allclose((views[0] + views[1]) / 2.0, image)
    assert min(float(view.min()) for view in views) >= 0.0
    assert max(float(view.max()) for view in views) <= 1.0


def test_shared_antithetic_graph_supports_proxy_gradients():
    image = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
    views = antithetic_views(image, 1.0 / 255.0, seed=3)
    losses = [torch.stack([(view * scale).square().mean() for view in views]).mean()
              for scale in (1.0, 2.0)]
    gradients = [
        normalized_gradient(loss, image, retain_graph=index + 1 < len(losses))
        for index, loss in enumerate(losses)
    ]
    assert len(gradients) == 2
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_collision_objective_rewards_translated_feature_match():
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    target = spatial_collision_target(image, 0.25)
    assert torch.equal(target, torch.roll(image, shifts=(1, 1), dims=(-2, -1)))
    matching = feature_loss([target], [target], objective="collision")
    original = feature_loss([image], [target], objective="collision")
    assert matching > original


def test_grayscale_projection_repeats_channel_mean():
    gradient = torch.tensor([[[[1.0]], [[2.0]], [[6.0]]]])
    projected = project_grayscale_gradient(gradient)
    assert torch.equal(projected, torch.full_like(gradient, 3.0))
