from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from autoresearch_med.diagnostics.mechanism import (
    discrete_path_statistics,
    path_response_statistics,
)
from autoresearch_med.diagnostics.statistics import h1_gate, relative_mae_improvement
from autoresearch_med.evaluation.budget_audit import audit_arrays
from autoresearch_med.evaluation import locked_runner
from autoresearch_med.evaluation.protocol_audit import audit


class MechanismTests(unittest.TestCase):
    def test_constant_response_is_consistent(self):
        responses = [torch.ones(2, 3) for _ in range(5)]
        stats = path_response_statistics(
            responses, [0.0, 0.25, 0.5, 0.75, 1.0], endpoint_response=torch.ones(2, 3)
        )
        self.assertAlmostEqual(stats["path_consistency"], 1.0, places=6)
        self.assertAlmostEqual(stats["quadrature_residual_ratio"], 0.0, places=6)

    def test_reversal_reduces_consistency(self):
        responses = [torch.ones(4), torch.ones(4), -torch.ones(4)]
        stats = path_response_statistics(responses, [0.0, 0.5, 1.0])
        self.assertLess(stats["path_consistency"], 1.0)
        self.assertGreater(stats["mean_sign_reversal_fraction"], 0.0)

    def test_linear_path_recovers_endpoint(self):
        samples = [0.0, 0.25, 0.5, 0.75, 1.0]
        response = torch.arange(1, 5, dtype=torch.float32)
        stats = path_response_statistics(
            [response for _ in samples], samples, endpoint_response=response
        )
        self.assertAlmostEqual(stats["quadrature_residual_ratio"], 0.0, places=6)

    def test_discrete_path_telescopes(self):
        features = [
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        ]
        stats = discrete_path_statistics(features, [0.0, 0.5, 1.0])
        self.assertAlmostEqual(stats["telescoping_residual_ratio"], 0.0, places=6)
        self.assertAlmostEqual(stats["discrete_path_consistency"], 2 ** 0.5 / 2, places=6)


class AuditTests(unittest.TestCase):
    def test_budget(self):
        clean = np.zeros((4, 4, 3), dtype=np.uint8)
        adversarial = np.full_like(clean, 10)
        result = audit_arrays(clean, adversarial, 10.0)
        self.assertTrue(result["budget_pass"])
        self.assertTrue(result["range_pass"])

    def test_gate(self):
        self.assertAlmostEqual(relative_mae_improvement(0.2, 0.18), 0.1)
        result = h1_gate(0.06, {"SAMed": 0.2, "MedSA": 0.1}, True)
        self.assertTrue(result["passed"])

    def test_protocol(self):
        result = audit()
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["rules"]["checkpoint_hashes_match"])
        self.assertTrue(result["rules"]["sample_identities_resolve"])

    def test_tier3_can_be_registered_as_confirmation_research(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            (workspace / "results/run_manifests").mkdir(parents=True)
            (workspace / "results/experiments.jsonl").touch()
            with patch.object(locked_runner, "WORKSPACE", workspace):
                output = locked_runner.register_run(
                    "confirmation-test",
                    3,
                    Path("autoresearch_med/configs/confirmation.yaml").resolve(),
                    Path("autoresearch_med/candidates/active.py").resolve(),
                    research_stage="confirmation",
                )
            payload = json.loads(output.read_text())
            self.assertEqual(payload["tier"], 3)
            self.assertEqual(payload["research_stage"], "confirmation")


if __name__ == "__main__":
    unittest.main()

