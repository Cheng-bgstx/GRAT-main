import numpy as np

from peft_transfer.blend_phase_endpoints import blend_quantized


def test_blend_stays_inside_exact_budget():
    clean = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    clean_255 = clean * 255
    left = np.clip(np.rint(clean_255 - 10), 0, 255).astype(np.uint8)
    right = np.clip(np.rint(clean_255 + 10), 0, 255).astype(np.uint8)
    result = blend_quantized(clean, left, right, 0.3)
    assert np.max(np.abs(result.astype(np.float32) - clean_255)) <= 10.0 + 1e-5
