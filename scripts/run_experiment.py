#!/usr/bin/env python3
"""
Unified experiment runner: Generate adversarial images → Evaluate → Compare
Handles both SE-CGR baseline and V2, all 3 datasets.
"""
import subprocess
import os
import sys
import time
import json
from datetime import datetime

ROOT = '/root/data2/fyc/fyc/GRAT-main'
os.chdir(ROOT)

# ========== CONFIG ==========
DATASETS = ['Synapse', 'ISIC', 'REFUGE']
METHODS = ['baseline', 'v2']
GPU_MAP = {
    ('baseline', 'Synapse'): 0,
    ('v2', 'Synapse'): 1,
    ('baseline', 'ISIC'): 2,
    ('v2', 'ISIC'): 3,
    ('baseline', 'REFUGE'): 4,
    ('v2', 'REFUGE'): 5,
}

TS = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_BASE = f'testset/exp_{TS}'
RESULTS_DIR = f'results/exp_{TS}'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset-specific configs
DS_CONFIG = {
    'Synapse': {
        'data_path': 'data/Synapse_preprocess/test_vol_h5',
        'list_dir': 'data/Synapse_preprocess/lists',
        'num_classes': 8,
        'lora_ckpt': 'checkpoints/SAMed_Synapse_epoch_159.pth',
        'msa_ckpt': 'checkpoints/Synapse_512_pretrain_vit_l_epo200_bs12_lr0.0001',
    },
    'ISIC': {
        'data_path': 'data/ISIC',
        'list_dir': 'data/ISIC',
        'num_classes': 1,
        'lora_ckpt': 'checkpoints/samed_isic/isic_512_pretrain_vit_b_epo200_bs12_lr0.0001/epoch_159.pth',
        'msa_ckpt': None,  # needs specific path
    },
    'REFUGE': {
        'data_path': 'data/REFUGE-Multirater',
        'list_dir': 'data/REFUGE-Multirater',
        'num_classes': 2,
        'lora_ckpt': 'checkpoints/samed_refuge/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.0001/epoch_159.pth',
        'msa_ckpt': None,
    },
}

ATTACK_STEPS = 5  # Quick comparison; increase to 10 for final


def run_cmd(cmd, gpu_id, log_file):
    """Run a command on a specific GPU and log output."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(cmd, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT)
        return proc


def attack_cmd(method, dataset, gpu, out_subdir):
    """Build attack command."""
    cfg = DS_CONFIG[dataset]
    out_path = f'{OUTPUT_BASE}/{out_subdir}'
    os.makedirs(out_path, exist_ok=True)
    
    if method == 'baseline':
        script = 'Sam_attack_SE-CGR.py'
        extra = '--SE 0 --CGR 1 --edge_guidance 1 --edge_gamma 3.0'
    else:
        script = 'Sam_attack_SE-CGR_v2.py'
        extra = ('--SE 0 --CGR 1 --edge_guidance 1 --edge_gamma 3.0 '
                 '--freq_bands 4 --phase_bound 0.15 --phase_smooth 3.0 '
                 '--spec_momentum 1 --layer_aggregation 1')
    
    cmd = (
        f'python3 {script} '
        f'--dataset {dataset} '
        f'--vit_name vit_b '
        f'--ckpt checkpoints/sam_vit_b_01ec64.pth '
        f'--img_size 512 '
        f'--volume_path {cfg["data_path"]} '
        f'--list_dir {cfg["list_dir"]} '
        f'--num_classes {cfg["num_classes"]} '
        f'--att_step_size 2.0 '
        f'--att_bound 10.0 '
        f'--att_numstep {ATTACK_STEPS} '
        f'--random_seed 42 '
        f'--out_path {out_path} '
        f'{extra}'
    )
    return cmd, out_path


def eval_samed_cmd(dataset, gpu, adv_path, method_label):
    """Build SAMed evaluation command."""
    cfg = DS_CONFIG[dataset]
    log_file = f'{RESULTS_DIR}/samed_{method_label}_{dataset}.log'
    
    cmd = (
        f'python3 test_samed.py '
        f'--dataset {dataset} '
        f'--vit_name vit_b '
        f'--ckpt checkpoints/sam_vit_b_01ec64.pth '
        f'--lora_ckpt {cfg["lora_ckpt"]} '
        f'--img_size 512 '
        f'--volume_path {cfg["data_path"]} '
        f'--list_dir {cfg["list_dir"]} '
        f'--num_classes {cfg["num_classes"]} '
        f'--out_path {adv_path} '
        f'--random_seed 42 '
    )
    return cmd, log_file


def extract_metrics(log_file):
    """Extract mDice and mHD95 from SAMed eval log."""
    try:
        with open(log_file) as f:
            content = f.read()
        # Look for "mean_dice : X.XXXX mean_hd95 : Y.YYYY"
        import re
        match = re.search(r'mean_dice\s*:\s*([\d.]+)\s*mean_hd95\s*:\s*([\d.]+)', content)
        if match:
            return float(match.group(1)), float(match.group(2))
    except Exception as e:
        print(f"  Error extracting metrics: {e}")
    return None, None


def main():
    log = open(f'{RESULTS_DIR}/pipeline.log', 'w')
    
    def log_print(msg):
        ts = datetime.now().strftime('%H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line)
        log.write(line + '\n')
        log.flush()
    
    log_print(f'=== Experiment Pipeline: {ATTACK_STEPS} steps ===')
    
    # ===== PHASE 1: Generate adversarial images (parallel) =====
    log_print('Phase 1: Generating adversarial images...')
    procs = {}
    out_paths = {}
    
    for method in METHODS:
        for dataset in DATASETS:
            gpu = GPU_MAP[(method, dataset)]
            label = f'{method}_{dataset}'
            out_subdir = f'{method}/{dataset}'
            cmd, out_path = attack_cmd(method, dataset, gpu, out_subdir)
            out_paths[(method, dataset)] = out_path
            log_file = f'{RESULTS_DIR}/attack_{label}.log'
            
            log_print(f'  Launching {label} on GPU {gpu}...')
            log_print(f'    CMD: {cmd[:120]}...')
            proc = run_cmd(cmd, gpu, log_file)
            procs[label] = proc
    
    # Wait for all attacks to complete
    log_print(f'  Waiting for {len(procs)} attack processes...')
    for label, proc in procs.items():
        ret = proc.wait()
        log_print(f'  [{label}] exited with code {ret}')
    
    log_print('Phase 1 complete.')
    
    # ===== PHASE 2: Evaluate on SAMed (sequential) =====
    log_print('Phase 2: Evaluating on SAMed...')
    results = {}
    
    for method in METHODS:
        for dataset in DATASETS:
            label = f'{method}_{dataset}'
            adv_path = out_paths[(method, dataset)]
            gpu = GPU_MAP[(method, dataset)]
            
            cmd, log_file = eval_samed_cmd(dataset, gpu, adv_path, label)
            log_print(f'  Evaluating {label} on GPU {gpu}...')
            proc = run_cmd(cmd, gpu, log_file)
            proc.wait()
            
            mDice, mHD95 = extract_metrics(log_file)
            results[label] = {'mDice': mDice, 'mHD95': mHD95}
            log_print(f'  [{label}] mDice={mDice}, mHD95={mHD95}')
    
    # ===== PHASE 3: Summary =====
    log_print('')
    log_print('=' * 60)
    log_print('RESULTS SUMMARY')
    log_print('=' * 60)
    
    for dataset in DATASETS:
        log_print(f'\n--- {dataset} ---')
        for method in METHODS:
            label = f'{method}_{dataset}'
            r = results.get(label, {})
            log_print(f'  {method:10s}: mDice={r.get("mDice","N/A")}, mHD95={r.get("mHD95","N/A")}')
        
        # Compare
        bl = results.get(f'baseline_{dataset}', {})
        v2 = results.get(f'v2_{dataset}', {})
        if bl.get('mDice') is not None and v2.get('mDice') is not None:
            dice_diff = bl['mDice'] - v2['mDice']
            if dice_diff > 0.02:
                log_print(f'  → V2 IMPROVES mDice by {dice_diff:.4f}!')
            elif dice_diff < -0.02:
                log_print(f'  → V2 DEGRADES mDice by {-dice_diff:.4f}')
            else:
                log_print(f'  → Similar mDice (diff={dice_diff:.4f})')
    
    # Save results
    with open(f'{RESULTS_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    log_print(f'\nFull results saved to {RESULTS_DIR}/')
    log.close()


if __name__ == '__main__':
    main()
