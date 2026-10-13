import torch

from peft_transfer.task_basin import (
    apply_delta, basin_linear_lower_bound, damped_precision,
    diagonal_empirical_fisher, peft_parameters, restore_parameters,
    sample_delta, split_names,
)
from peft_transfer.generate_2d_dual_peft import (
    constrained_pareto_direction, constrained_pareto_multiobjective_direction,
    safeguarded_basin_gradient,
)


def test_split_is_deterministic_and_disjoint():
    names = [f"image_{index}" for index in range(31)]
    train, calibration = split_names(names, 0.2, 17)
    assert set(train).isdisjoint(calibration)
    assert set(train) | set(calibration) == set(names)
    assert (train, calibration) == split_names(names, 0.2, 17)


def test_damped_bound_and_sampling():
    model = torch.nn.Module()
    model.linear_a_q = torch.nn.Linear(2, 1, bias=False)
    parameters = peft_parameters(model, "lora")
    samples = [torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, -1.0]])]
    fisher = diagonal_empirical_fisher(
        model, samples, lambda net, sample: net.linear_a_q(sample).square().mean(),
        parameters,
    )
    precision, damping = damped_precision(fisher)
    assert damping > 0
    baseline = {name: value.detach().clone() for name, value in parameters.items()}
    for geometry in ("isotropic", "fisher"):
        delta = sample_delta(parameters, precision, 0.1, 11, geometry)
        assert torch.isclose(
            torch.stack([value.square().sum() for value in delta.values()]).sum().sqrt(),
            torch.stack([value.square().sum() for value in baseline.values()]).sum().sqrt() * 0.1,
            rtol=1e-5,
        )
        apply_delta(parameters, baseline, delta)
        restore_parameters(parameters, baseline)
        assert all(torch.equal(parameters[name], value) for name, value in baseline.items())
    gradient = {name: torch.ones_like(value) for name, value in baseline.items()}
    radius = 0.3
    bound = basin_linear_lower_bound(torch.tensor(1.0), gradient, precision, radius)
    expected = 1.0 - radius * sum(
        (gradient[name].square() / precision[name]).sum().item()
        for name in gradient
    ) ** 0.5
    assert abs(bound.item() - expected) < 1e-6


def test_basin_correction_respects_carrier_floor():
    carrier = torch.tensor([[[[1.0, 0.0]]]])
    opposing = torch.tensor([[[[-1.0, 2.0]]]])
    result, scale = safeguarded_basin_gradient(carrier, opposing, 10.0, 0.8)
    cosine = torch.nn.functional.cosine_similarity(result.flatten(1), carrier.flatten(1))
    assert 0 < scale < 10.0
    assert float(cosine) >= 0.8 - 1e-5
    unchanged, scale = safeguarded_basin_gradient(carrier, opposing, 0.0, 0.8)
    assert scale == 0 and torch.equal(unchanged, carrier)


def test_pareto_direction_enforces_actual_linf_margins():
    carrier = torch.tensor([[[[1.0, 1.0, 1.0, 1.0]]]])
    lora = torch.tensor([[[[1.0, 1.0, -1.0, -1.0]]]])
    adapter = torch.tensor([[[[-1.0, 1.0, -1.0, 1.0]]]])
    direction, diagnostics = constrained_pareto_direction(
        adapter, carrier, [lora], beta=0.8,
    )
    assert set(direction.unique().tolist()).issubset({-1.0, 0.0, 1.0})
    assert diagnostics["feasible"]
    assert all(
        margin + 5e-4 >= bound for margin, bound in zip(
            diagnostics["constraint_margins"], diagnostics["constraint_bounds"]
        )
    )


def test_pareto_direction_keeps_reference_feasible_for_conflicting_constraints():
    carrier = torch.tensor([[[[1.0, 1.0, 1.0, 1.0]]]])
    conflicting = torch.tensor([[[[-1.0, -1.0, -1.0, -1.0]]]])
    adapter = torch.tensor([[[[-1.0, 1.0, -1.0, 1.0]]]])
    _direction, diagnostics = constrained_pareto_direction(
        adapter, carrier, [conflicting], beta=0.8,
    )
    assert diagnostics["feasible"]
    assert diagnostics["fallback_to_reference"]
    assert diagnostics["requested_constraint_bounds"][1] == 0.0
    assert diagnostics["constraint_bounds"][1] == diagnostics["reference_margins"][1]


def test_pareto_direction_is_feasible_for_random_gradients():
    for seed in range(20):
        torch.manual_seed(seed)
        objective = torch.randn(1, 3, 4, 4)
        carrier = torch.randn_like(objective)
        constraints = [torch.randn_like(objective) for _ in range(4)]
        direction, diagnostics = constrained_pareto_direction(
            objective, carrier, constraints, beta=0.8,
        )
        assert diagnostics["feasible"]
        assert set(direction.unique().tolist()).issubset({-1.0, 0.0, 1.0})
        assert all(
            margin + 5e-4 >= bound for margin, bound in zip(
                diagnostics["constraint_margins"], diagnostics["constraint_bounds"]
            )
        )


def test_multiobjective_pareto_matches_small_bruteforce_optimum():
    objective_a = torch.tensor([[[[1.0, 1.0, -1.0, -1.0]]]])
    objective_b = torch.tensor([[[[1.0, -1.0, 1.0, -1.0]]]])
    carrier = torch.tensor([[[[1.0, 1.0, 1.0, 1.0]]]])
    direction, diagnostics = constrained_pareto_multiobjective_direction(
        [objective_a, objective_b], carrier, [], beta=0.5,
    )
    candidates = []
    for mask in range(16):
        value = torch.tensor([[[[
            1.0 if mask & (1 << index) else -1.0 for index in range(4)
        ]]]])
        carrier_margin = float((carrier * value).mean())
        if carrier_margin + 5e-4 >= 0.5:
            candidates.append(min(
                float((objective_a * value).mean()),
                float((objective_b * value).mean()),
            ))
    achieved = min(
        float((objective_a * direction).mean()),
        float((objective_b * direction).mean()),
    )
    assert diagnostics["feasible"]
    assert abs(achieved - max(candidates)) < 1e-6
