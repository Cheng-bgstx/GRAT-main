#!/usr/bin/env python3
"""
Experiment 12: Computational Cost Benchmark.
Measures time and memory for SE-CGR vs baselines on Synapse SAMed vit_b.

Single 512x512 slice, 20 steps, average over 10 runs.
"""
import os, sys, time, gc, json
import numpy as np
import torch

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from attack_codd import AttackConfig, run_codd_attack
from evaluate import load_sam_encoder

DEVICE = torch.device("cuda:0")
IMAGE_SIZE = 512
VIT = "vit_b"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_WARMUP = 3
N_BENCH = 10


def benchmark_method(name, attack_fn, **kwargs):
    print(f"  Benchmarking {name}...")
    # Warmup
    for _ in range(N_WARMUP):
        attack_fn(**kwargs)
        gc.collect(); torch.cuda.empty_cache()

    torch.cuda.synchronize()
    times = []
    mems = []
    for _ in range(N_BENCH):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        attack_fn(**kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        mems.append(torch.cuda.max_memory_allocated() / 1024**3)
        gc.collect(); torch.cuda.empty_cache()

    return {
        "mean_time_s": float(np.mean(times)),
        "std_time_s": float(np.std(times)),
        "mean_mem_gb": float(np.mean(mems)),
        "peak_mem_gb": float(np.max(mems)),
    }


def main():
    print("Loading source SAM...")
    source_sam = load_sam_encoder(VIT, SAM_CKPT, IMAGE_SIZE, 8, DEVICE)

    # Create sample image
    img_t = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    config = AttackConfig(
        att_step_size=2.0, att_bound=10.0, att_numstep=20,
        direction_mode="radial", probe_amplification=2.0,
        edge_guidance=True, seed=42,
    )

    results = {}

    # SE-CGR
    results["se_cgr"] = benchmark_method(
        "SE-CGR", run_codd_attack,
        image_encoder=source_sam.image_encoder,
        input_image=img_t, config=config, return_trace=False,
    )

    # SE-CGR without edge guidance (faster variant)
    config_noedge = AttackConfig(
        att_step_size=2.0, att_bound=10.0, att_numstep=20,
        direction_mode="radial", probe_amplification=2.0,
        edge_guidance=False, seed=42,
    )
    results["se_cgr_no_edge"] = benchmark_method(
        "SE-CGR (no edge)", run_codd_attack,
        image_encoder=source_sam.image_encoder,
        input_image=img_t, config=config_noedge, return_trace=False,
    )

    # SE-CGR 10 steps (faster)
    config_10 = AttackConfig(
        att_step_size=2.0, att_bound=10.0, att_numstep=10,
        direction_mode="radial", probe_amplification=2.0,
        edge_guidance=True, seed=42,
    )
    results["se_cgr_10steps"] = benchmark_method(
        "SE-CGR (10 steps)", run_codd_attack,
        image_encoder=source_sam.image_encoder,
        input_image=img_t, config=config_10, return_trace=False,
    )

    print(f"\nResults:")
    for name, r in results.items():
        print(f"  {name}: {r['mean_time_s']:.3f}s ± {r['std_time_s']:.3f}s, "
              f"mem={r['mean_mem_gb']:.2f}GB (peak={r['peak_mem_gb']:.2f}GB)")

    output = {"config": {"vit": VIT, "image_size": IMAGE_SIZE,
             "warmup": N_WARMUP, "bench_runs": N_BENCH}, "results": results}
    out_path = os.path.join(OUTPUT_DIR, "exp12_cost.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
