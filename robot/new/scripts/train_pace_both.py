"""
PACE Two-Pathway Training: Global FiLM (Pathway 1) + DSSA (Pathway 2).

Paper design:
  Pathway 1 (Global FiLM via time embedding): Coarse spatial awareness at every UNet layer
  Pathway 2 (DSSA): Fine-grained spatial cross-attention with zero-gated residual

Key fixes vs. previous failed training:
  1. fm_base_v2 pretrained backbone (FID=0.059, text quality preserved)
  2. Stage 1: FREEZE backbone, train ONLY goal modules (30K steps)
  3. Progressive FK loss: λ_fk ramps 0.05 → 1.0
  4. Text dropout 10% + goal dropout 15% (maintain text conditioning)
  5. Lower LR for goal modules (5e-5) vs backbone (1e-4)
  6. PACE augmented data (99K FK-consistent samples)

Usage:
    LD_LIBRARY_PATH="" PYTHONPATH="" .env/bin/torchrun --nproc_per_node=1 \
        new/scripts/train_pace_both.py --gpu 1
"""
import sys, os, math, argparse, time, json, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

SRC = '/mnt/workspace/Jensen/project/ECHO_Closd/ROBOTSHELL_GENERATOR/new'
ROOT = os.path.dirname(SRC)
# IMPORTANT: SRC must come BEFORE ROOT so new/models is used
sys.path.insert(0, SRC)
# But also need ROOT for checkpoints path - put it AFTER
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models import build_models
from models.robot_fk_layer import RobotFKLayer
from models.flow_matching import FlowMatchingSampler
from models.pace_flow_matching import flow_time_weight
from utils.robot_target_cond import (
    ROBOT_EXTENDED_GOAL_JOINT_NAMES,
    build_target_cond_from_goal,
    build_target_validity,
    build_empty_target_metadata,
    parse_target_joint_config,
)
from options.train_options import TrainOptions


# ============================================================
# PACE Dataset (loads augmented data)
# ============================================================

def validate_pace_data_dir(data_dir, allow_partial_data=False, allow_old_data=False):
    data_dir = os.path.abspath(data_dir)
    if os.path.basename(data_dir) == 'pace_data_v1' and not allow_old_data:
        raise RuntimeError('Refusing to train on pace_data_v1. Use --allow_old_data only for explicit debugging.')
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'PACE data_dir does not exist: {data_dir}')

    npz_files = glob.glob(os.path.join(data_dir, '*.npz'))
    if not npz_files:
        raise RuntimeError(f'No .npz files found in PACE data_dir: {data_dir}')

    manifest_path = os.path.join(data_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        if allow_partial_data:
            print(f'[PACE Data] WARNING: manifest.json missing, using partial data: {len(npz_files)} files')
            return None
        raise RuntimeError(f'manifest.json missing in {data_dir}; use --allow_partial_data only for smoke tests.')

    with open(manifest_path) as f:
        manifest = json.load(f)
    expected = int(manifest.get('n_output_files', -1))
    actual = len(npz_files)
    if expected != actual:
        if allow_partial_data:
            print(f'[PACE Data] WARNING: manifest count mismatch expected={expected} actual={actual}')
        else:
            raise RuntimeError(f'PACE manifest count mismatch: expected={expected}, actual={actual}')
    print(f'[PACE Data] {data_dir}: {actual} files, manifest={manifest}')
    return manifest


SLOT_NAMES = list(ROBOT_EXTENDED_GOAL_JOINT_NAMES)


def build_target_cond_from_pace_goal(goal_value, goal_dim_mask, goal_time_mask, goal_slot_mask):
    B, T = goal_value.shape[:2]
    device = goal_value.device
    target_cond = torch.zeros(B, 7, 3, device=device, dtype=goal_value.dtype)
    target_joint_names = []
    is_heading = torch.zeros(B, dtype=torch.bool, device=device)

    for b in range(B):
        names = []
        for s, name in enumerate(SLOT_NAMES):
            if not bool(goal_slot_mask[b, s].item()):
                continue
            t_active = torch.nonzero(goal_time_mask[b, :, s], as_tuple=False).flatten()
            if t_active.numel() == 0:
                continue
            t = int(t_active[-1].item())
            if s <= 4:
                if not bool(goal_dim_mask[b, t, s, :3].all().item()):
                    continue
                target_cond[b, s, :3] = goal_value[b, t, s, :3]
                names.append(name)
            elif s == 5:
                if not bool(goal_dim_mask[b, t, s, :2].all().item()):
                    continue
                target_cond[b, s, :2] = goal_value[b, t, s, :2]
                names.append(name)
            else:
                if not bool(goal_dim_mask[b, t, s, :2].all().item()):
                    continue
                target_cond[b, s, :2] = goal_value[b, t, s, :2]
                is_heading[b] = True
        target_joint_names.append(names)
    return target_cond, target_joint_names, is_heading


def build_identity_target_cond_from_fk(motion, goal_slot_mask, fk_layer, lengths):
    B = motion.shape[0]
    device = motion.device
    with torch.no_grad():
        fk_out = fk_layer(motion)
        kp = fk_out['keypoint_pos']
        rp = fk_out['root_pos']
        rr = fk_out['root_rot']

    target_cond = torch.zeros(B, 7, 3, device=device, dtype=motion.dtype)
    target_joint_names = []
    is_heading = torch.zeros(B, dtype=torch.bool, device=device)
    for b in range(B):
        last_t = max(0, min(int(lengths[b].item()) - 1, motion.shape[1] - 1))
        names = []
        for s, name in enumerate(SLOT_NAMES):
            if not bool(goal_slot_mask[b, s].item()):
                continue
            if s <= 4:
                target_cond[b, s, :3] = kp[b, last_t, s]
                names.append(name)
            elif s == 5:
                target_cond[b, s, :2] = rp[b, last_t, :2]
                names.append(name)
            else:
                yaw = torch.atan2(rr[b, last_t, 1, 0], rr[b, last_t, 0, 0])
                target_cond[b, s, 0] = torch.cos(yaw)
                target_cond[b, s, 1] = torch.sin(yaw)
                is_heading[b] = True
        target_joint_names.append(names)
    return target_cond, target_joint_names, is_heading

class PACEDataset(torch.utils.data.Dataset):
    """Load PACE augmented 38D data."""

    def __init__(self, data_dir, split='train', train_ratio=0.95, max_samples=None,
                 allow_partial_data=False, allow_old_data=False):
        self.data_dir = os.path.abspath(data_dir)
        validate_pace_data_dir(self.data_dir, allow_partial_data=allow_partial_data,
                               allow_old_data=allow_old_data)
        all_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])

        # Group by parent clip
        groups = {}
        for f in all_files:
            # Parse from filename: XXXXXX_aNN.npz or XXXXXX_bNN.npz
            parts = f.replace('.npz', '').rsplit('_', 1)
            if len(parts) != 2: continue
            pid = parts[0]
            pid_parts = pid.rsplit('_', 1)
            if len(pid_parts) == 2 and pid_parts[1][0] in 'ab':
                pid = pid_parts[0]

            if pid not in groups: groups[pid] = []
            groups[pid].append(f)

        self.group_ids = sorted(groups.keys())
        self.groups = groups

        # Split
        np.random.seed(42)
        idx = np.random.permutation(len(self.group_ids))
        n = int(len(self.group_ids) * train_ratio)
        if split == 'train':
            self.group_ids = [self.group_ids[i] for i in idx[:n]]
        else:
            self.group_ids = [self.group_ids[i] for i in idx[n:]]

        if max_samples:
            self.group_ids = self.group_ids[:max_samples]

        print(f'[PACE Dataset] {split}: {len(self.group_ids)} groups')

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, idx):
        gid = self.group_ids[idx]
        # Randomly pick a sample from this group
        f = np.random.choice(self.groups[gid])
        d = np.load(os.path.join(self.data_dir, f), allow_pickle=True)

        motion = torch.from_numpy(d['motion_38d'].copy()).float()
        goal_world = torch.from_numpy(d['goal_world'].copy()).float()
        goal_rel = torch.from_numpy(d['goal_rel'].copy()).float()
        goal_value = torch.from_numpy(d['goal_value'].copy()).float()
        goal_fk_target = torch.from_numpy(d['goal_fk_target'].copy()).float() if 'goal_fk_target' in d else goal_world.clone()
        goal_dim_mask = torch.from_numpy(d['goal_dim_mask'].copy()).bool()
        goal_time_mask = torch.from_numpy(d['goal_time_mask'].copy()).bool()
        goal_slot_mask = torch.from_numpy(d['goal_slot_mask'].copy()).bool()
        loss_slot_mask = torch.from_numpy(d['loss_slot_mask'].copy()).bool() if 'loss_slot_mask' in d else goal_slot_mask.clone()
        aux_slot_mask = torch.from_numpy(d['aux_slot_mask'].copy()).bool() if 'aux_slot_mask' in d else torch.zeros(7, dtype=torch.bool)
        goal_confidence = torch.from_numpy(d['goal_confidence'].copy()).float()
        loss_slot_weight = torch.from_numpy(d['loss_slot_weight'].copy()).float() if 'loss_slot_weight' in d else torch.ones(7)
        fk_weight = float(np.asarray(d['fk_weight']).reshape(-1)[0]) if 'fk_weight' in d else 1.0
        cfm_weight = float(np.asarray(d['cfm_weight']).reshape(-1)[0]) if 'cfm_weight' in d else 1.0
        stop_weight = float(np.asarray(d['stop_weight']).reshape(-1)[0]) if 'stop_weight' in d else 1.0
        schema_version = str(d['schema_version']) if 'schema_version' in d else 'legacy'
        teacher_type = str(d['teacher_type']) if 'teacher_type' in d else 'legacy'
        goal_type = str(d['goal_type'])
        text = str(d['text'])

        T = min(motion.shape[0], 300)
        motion = motion[:T]
        goal_value = goal_value[:T]
        goal_fk_target = goal_fk_target[:T]
        goal_dim_mask = goal_dim_mask[:T]
        goal_time_mask = goal_time_mask[:T]

        return {
            'motion': motion,
            'goal_value': goal_value,
            'goal_fk_target': goal_fk_target,
            'goal_dim_mask': goal_dim_mask,
            'goal_time_mask': goal_time_mask,
            'goal_slot_mask': goal_slot_mask,
            'loss_slot_mask': loss_slot_mask,
            'aux_slot_mask': aux_slot_mask,
            'loss_slot_weight': loss_slot_weight,
            'fk_weight': torch.tensor(fk_weight, dtype=torch.float32),
            'cfm_weight': torch.tensor(cfm_weight, dtype=torch.float32),
            'stop_weight': torch.tensor(stop_weight, dtype=torch.float32),
            'schema_version': schema_version,
            'teacher_type': teacher_type,
            'goal_confidence': goal_confidence,
            'goal_type': goal_type,
            'text': text,
            'length': T,
        }


def collate_pace(batch, max_T=300):
    B = len(batch)
    D = 38
    max_len = min(max_T, max(b['motion'].shape[0] for b in batch))

    motion = torch.zeros(B, max_len, D)
    gv = torch.zeros(B, max_len, 7, 4)
    gfk = torch.zeros(B, max_len, 7, 4)  # goal_fk_target (world-space)
    gdm = torch.zeros(B, max_len, 7, 4, dtype=torch.bool)
    gtm = torch.zeros(B, max_len, 7, dtype=torch.bool)
    gsm = torch.stack([b['goal_slot_mask'] for b in batch])
    lsm = torch.stack([b['loss_slot_mask'] for b in batch])
    aux = torch.stack([b['aux_slot_mask'] for b in batch])
    lsw = torch.stack([b['loss_slot_weight'] for b in batch])
    sample_fk_w = torch.stack([b['fk_weight'] for b in batch])
    sample_cfm_w = torch.stack([b['cfm_weight'] for b in batch])
    sample_stop_w = torch.stack([b['stop_weight'] for b in batch])
    gconf = torch.stack([b['goal_confidence'] for b in batch])
    lengths = torch.zeros(B, dtype=torch.long)
    texts = []

    for i, b in enumerate(batch):
        T = b['motion'].shape[0]
        L = min(T, max_len)
        motion[i, :L] = b['motion'][:L]
        gv[i, :L] = b['goal_value'][:L]
        gfk[i, :L] = b['goal_fk_target'][:L]
        gdm[i, :L] = b['goal_dim_mask'][:L]
        gtm[i, :L] = b['goal_time_mask'][:L]
        lengths[i] = T
        texts.append(b['text'])

    return {
        'motion': motion, 'goal_value': gv,
        'goal_fk_target': gfk,
        'goal_dim_mask': gdm, 'goal_time_mask': gtm,
        'goal_slot_mask': gsm, 'loss_slot_mask': lsm,
        'aux_slot_mask': aux, 'loss_slot_weight': lsw,
        'fk_weight': sample_fk_w, 'cfm_weight': sample_cfm_w,
        'stop_weight': sample_stop_w,
        'goal_confidence': gconf,
        'lengths': lengths, 'text': texts,
    }


# ============================================================
# FK Loss
# ============================================================

def compute_fk_loss(x_hat_1_denorm, goal_fk_target, goal_dim_mask, goal_time_mask,
                    loss_slot_mask, fk_layer, lengths, loss_slot_weight=None):
    """FK consistency loss. Uses loss_slot_mask (broad FK supervision, always incl. pelvis aux).
    goal_fk_target must be in WORLD frame (same as FK output)."""
    B, T, D = x_hat_1_denorm.shape
    device = x_hat_1_denorm.device

    # Clamp for FK stability
    joints = torch.clamp(x_hat_1_denorm[:, :, :29], -3.0, 3.0)
    phys = torch.cat([joints, x_hat_1_denorm[:, :, 29:]], dim=-1)

    try:
        fk_out = fk_layer(phys)
        kp = fk_out['keypoint_pos']
        rp = fk_out['root_pos']
        rr = fk_out['root_rot']

        if torch.isnan(kp).any() or torch.isnan(rp).any():
            return torch.tensor(0.0, device=device, requires_grad=True)
    except Exception:
        return torch.tensor(0.0, device=device, requires_grad=True)

    pred = torch.zeros(B, T, 7, 4, device=device)
    pred[:, :, 0, :3] = kp[:, :, 0]
    pred[:, :, 1, :3] = kp[:, :, 1]
    pred[:, :, 2, :3] = kp[:, :, 2]
    pred[:, :, 3, :3] = kp[:, :, 3]
    pred[:, :, 4, :3] = kp[:, :, 4]
    pred[:, :, 5, :2] = rp[:, :, :2]
    pred_yaw = torch.atan2(rr[:, :, 1, 0], rr[:, :, 0, 0])
    pred[:, :, 6, 0] = torch.cos(pred_yaw)
    pred[:, :, 6, 1] = torch.sin(pred_yaw)

    if loss_slot_weight is None:
        slot_w = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5], device=device).view(1, 1, 7, 1)
    else:
        slot_w = loss_slot_weight.to(device).view(B, 1, 7, 1)
    sq_err = (pred - goal_fk_target) ** 2
    mask = (goal_dim_mask.float() *
            goal_time_mask.float().unsqueeze(-1) *
            loss_slot_mask.float().view(B, 1, 7, 1) *
            slot_w)
    n_active = mask.sum().clamp(min=1)

    return (sq_err * mask).sum() / n_active


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device(f'cuda:{args.gpu}')
    print(f'[PACE] Device: {device}')

    # Data
    data_dir = os.path.abspath(args.data_dir)
    train_ds = PACEDataset(data_dir, split='train', max_samples=args.max_samples,
                           allow_partial_data=args.allow_partial_data,
                           allow_old_data=args.allow_old_data)
    val_ds = PACEDataset(data_dir, split='val',
                         max_samples=args.max_samples // 10 if args.max_samples else None,
                         allow_partial_data=args.allow_partial_data,
                         allow_old_data=args.allow_old_data)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pace, num_workers=args.n_workers,
        pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pace, num_workers=args.n_workers, pin_memory=True)

    # ── Model: unet_small with BOTH pathways ──
    from types import SimpleNamespace as SN
    opt = SN()
    opt.dim_pose = 38
    opt.model_type = 'unet_small'
    opt.base_dim = args.base_dim
    opt.time_dim = args.time_dim
    opt.dim_mults = [1, 2, 2, 2]
    opt.text_latent_dim = 256
    opt.no_adagn = False
    opt.no_eff = False
    opt.cond_mask_prob = args.text_dropout
    opt.enable_goal_cond = True
    opt.unet_goal_inject = 'both'  # BOTH pathways
    opt.goal_dim = 3
    opt.target_joint_names = 'ROBOT_FULL'
    opt.enable_length_head = False
    opt.use_goal_adapter = True
    opt.goal_adapter_encoder = 'multi'
    opt.goal_adapter_injection = 'film'  # FiLM provides scale+shift
    opt.goal_adapter_layers = 1
    opt.text_drop_keep_goal_prob = args.text_drop_goal
    opt.use_goal_spatial_attn = True  # DSSA (Pathway 2)
    opt.end_token_exp = 'none'
    opt.max_motion_length = 490

    model = build_models(opt).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[PACE] Model: {n_params:,} params ({n_params/1e6:.1f}M)')

    # ── Load pretrained fm_base_v2 backbone ──
    pretrained_ckpt = args.pretrained_ckpt
    if pretrained_ckpt and os.path.exists(pretrained_ckpt):
        if os.path.isdir(pretrained_ckpt):
            pretrained_ckpt = os.path.join(pretrained_ckpt, 'model', 'latest.tar')
        print(f'[PACE] Loading backbone from {pretrained_ckpt}')
        state = torch.load(pretrained_ckpt, map_location='cpu')
        if 'encoder' in state: state = state['encoder']
        elif 'model' in state: state = state['model']
        elif 'model_ema' in state: state = state['model_ema']
        state = {k.replace('module.', ''): v for k, v in state.items()}

        # Only load backbone params (not goal modules which are randomly initialized)
        backbone_state = {
            k: v for k, v in state.items()
            if not any(x in k for x in [
                'embed_target_cond', 'embed_goal', 'goal_adapter',
                'goal_spatial', 'goal_token', 'stop_head',
            ])
        }
        missing, unexpected = model.load_state_dict(backbone_state, strict=False)
        print(f'[PACE] Loaded {len(backbone_state)} backbone params')
        print(f'[PACE] New goal modules: {len(missing)} params')
    else:
        print(f'[PACE] WARNING: No pretrained backbone found at {pretrained_ckpt}')

    # ── FK layer ──
    fk_layer = RobotFKLayer().to(device)

    # ── Normalization stats ──
    stats_dir = os.path.join(ROOT, 'checkpoints', 'robotv2_world', 'fm_base_v2', 'meta')
    mean_np = np.load(os.path.join(stats_dir, 'mean.npy'))
    std_np = np.load(os.path.join(stats_dir, 'std.npy'))
    mean_t = torch.from_numpy(mean_np).float().to(device)
    std_t = torch.from_numpy(std_np).float().to(device)

    # ── Optimizer: separate LR for backbone vs goal modules ──
    goal_module_keys = ['embed_target_cond', 'embed_goal', 'goal_adapter',
                        'goal_spatial', 'goal_token', 'goal_spatial_downs',
                        'goal_spatial_mid', 'goal_spatial_ups']
    backbone_params = []
    goal_params = []
    for name, p in model.named_parameters():
        if any(k in name for k in goal_module_keys):
            goal_params.append(p)
        else:
            backbone_params.append(p)

    optimizer_phase1 = torch.optim.AdamW([
        {'params': goal_params, 'lr': args.lr * args.goal_lr_mult},
    ], weight_decay=args.wd)

    total_steps = args.phase1_steps + args.phase2_steps
    optimizer_phase2 = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr},
        {'params': goal_params, 'lr': args.lr * args.goal_lr_mult},
    ], weight_decay=args.wd)
    scheduler_phase1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase1, T_max=max(1, args.phase1_steps), eta_min=args.lr * args.goal_lr_mult * 0.1)
    scheduler_phase2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase2, T_max=max(1, args.phase2_steps), eta_min=args.lr * 0.1)
    optimizer = optimizer_phase1
    scheduler = scheduler_phase1
    phase2_started = False
    print(f'[PACE] Params: backbone={sum(p.numel() for p in backbone_params):,}, goal={sum(p.numel() for p in goal_params):,}')

    # ── Training state ──
    output_dir = os.path.join(SRC, 'checkpoints', 'pace_both', args.name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    global_step = 0
    best_val_loss = float('inf')
    pbar = tqdm(total=total_steps)

    # Flow matching
    from models.flow_matching import FlowMatchingSampler

    print(f'[PACE] Phase 1 (frozen backbone): {args.phase1_steps} steps')
    print(f'[PACE] Phase 2 (full training): {args.phase2_steps} steps')
    print(f'[PACE] λ_fk: {args.lambda_fk_min} → {args.lambda_fk_max}')
    print(f'[PACE] Text dropout: {args.text_dropout}, Goal dropout: {args.goal_dropout}')

    while global_step < total_steps:
        for batch in train_loader:
            if global_step >= total_steps:
                break

            is_phase1 = global_step < args.phase1_steps

            motion = batch['motion'].to(device)  # (B, T, 38)
            gv = batch['goal_value'].to(device)       # root-relative (model condition)
            gfk = batch['goal_fk_target'].to(device)   # world-space (FK loss target)
            gdm = batch['goal_dim_mask'].to(device)
            gtm = batch['goal_time_mask'].to(device)
            gsm = batch['goal_slot_mask'].to(device)     # condition mask (text-driven)
            lsm = batch['loss_slot_mask'].to(device)      # loss mask (incl. pelvis aux)
            gconf = batch['goal_confidence'].to(device)
            texts = batch['text']
            B, T, D = motion.shape

            # Phase 1: train goal modules only; Phase 2: unfreeze backbone
            if is_phase1:
                for p in backbone_params:
                    p.requires_grad = False
                for p in goal_params:
                    p.requires_grad = True
                optimizer = optimizer_phase1
                scheduler = scheduler_phase1
            else:
                if not phase2_started:
                    for p in backbone_params:
                        p.requires_grad = True
                    for p in goal_params:
                        p.requires_grad = True
                    optimizer = optimizer_phase2
                    scheduler = scheduler_phase2
                    phase2_started = True
                    print(f'\n[PACE] Switched to Phase 2 at step {global_step}')

            # Normalize motion for model
            motion_norm = (motion - mean_t.view(1, 1, -1)) / std_t.view(1, 1, -1)

            # Goal mode + dropout
            rand_mode = torch.rand(1).item()
            is_identity = rand_mode < 0.30
            if is_identity:
                gsm_dropped = gsm.clone()
            elif rand_mode < 0.60:
                if args.goal_dropout > 0:
                    drop_mask = torch.rand(B, 7, device=device) < args.goal_dropout
                    gsm_dropped = gsm & (~drop_mask)
                else:
                    gsm_dropped = gsm.clone()
            else:
                gsm_dropped = torch.zeros_like(gsm)

            if is_identity:
                target_cond, target_joint_names, is_heading = build_identity_target_cond_from_fk(
                    motion, gsm_dropped, fk_layer, batch['lengths'].to(device))
            else:
                target_cond, target_joint_names, is_heading = build_target_cond_from_pace_goal(
                    gv, gdm, gtm, gsm_dropped)

            # ── Flow matching training step ──
            t_flow = torch.rand(B, device=device)
            x_0 = torch.randn(B, T, D, device=device)
            t_t = t_flow.view(-1, 1, 1)
            x_t = (1 - t_t) * x_0 + t_t * motion_norm
            v_target = motion_norm - x_0
            t_int = (t_flow * 999).long()

            # Forward through model with goal conditioning
            v_pred = model(x_t, t_int, enc_text=None, text=texts,
                          target_cond=target_cond,
                          target_joint_names=target_joint_names,
                          is_heading=is_heading)

            if isinstance(v_pred, tuple):
                v_pred = v_pred[0]

            # ── Losses ──
            loss_mse = F.mse_loss(v_pred, v_target, reduction='none')
            sample_cfm_w = batch.get('cfm_weight', torch.ones(B)).to(device).view(B, 1, 1)
            loss_mse = (loss_mse * sample_cfm_w).mean()

            # Predict clean endpoint for FK loss
            x_hat_1 = x_t + (1.0 - t_t) * v_pred  # PACE Eq. 8
            x_hat_1_denorm = x_hat_1 * std_t.view(1, 1, -1) + mean_t.view(1, 1, -1)

            # FK loss (progressive ramp)
            ramp = min(1.0, global_step / max(1, args.fk_ramp_steps))
            lambda_fk = args.lambda_fk_min + (args.lambda_fk_max - args.lambda_fk_min) * ramp

            if is_identity:
                lambda_fk = lambda_fk * 0.1  # Identity: weak FK (preserve FID)
            if lambda_fk > 0.001:
                loss_fk = compute_fk_loss(
                    x_hat_1_denorm, gfk, gdm, gtm, lsm, fk_layer,
                    batch['lengths'].to(device), batch.get('loss_slot_weight', None).to(device) if 'loss_slot_weight' in batch else None)
            else:
                loss_fk = torch.tensor(0.0, device=device)

            # Flow-time weight: focus FK loss on endpoint
            fk_weight = flow_time_weight(t_flow, peak=1.0, sigma=0.15).mean()
            sample_fk_scale = batch.get('fk_weight', torch.ones(B)).to(device).mean()
            loss_fk = loss_fk * fk_weight * sample_fk_scale

            loss = loss_mse + lambda_fk * loss_fk

            if torch.isnan(loss):
                print(f'[WARN] NaN at step {global_step}')
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()

            if global_step % args.log_every == 0:
                phase = 1 if is_phase1 else 2
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mse': f'{loss_mse.item():.4f}',
                    'fk': f'{loss_fk.item():.4f}',
                    'λfk': f'{lambda_fk:.3f}',
                    'P': phase,
                })

            global_step += 1
            pbar.update(1)

            # Save
            if global_step % args.save_every == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vi, vb in enumerate(val_loader):
                        if vi >= 5: break
                        vm = vb['motion'].to(device)
                        vm_norm = (vm - mean_t.view(1, 1, -1)) / std_t.view(1, 1, -1)
                        vgv = vb['goal_value'].to(device)
                        vgfk = vb['goal_fk_target'].to(device) if 'goal_fk_target' in vb else vb['goal_world'].to(device)
                        vgdm = vb['goal_dim_mask'].to(device)
                        vgtm = vb['goal_time_mask'].to(device)
                        vgsm = vb['goal_slot_mask'].to(device)
                        vlsm = vb['loss_slot_mask'].to(device) if 'loss_slot_mask' in vb else vgsm.clone()
                        vtgt, vtjn, vih = build_target_cond_from_pace_goal(vgv, vgdm, vgtm, vgsm)

                        v_pred = model(vm_norm[:1], torch.zeros(1, dtype=torch.long, device=device),
                                      enc_text=None, text=[vb['text'][0]],
                                      target_cond=vtgt[:1],
                                      target_joint_names=vtjn[:1],
                                      is_heading=vih[:1])
                        if isinstance(v_pred, tuple): v_pred = v_pred[0]

                        x_hat = vm_norm[:1] + (1.0 - 0.9) * v_pred  # t≈1.0
                        x_hat_denorm = x_hat * std_t.view(1, 1, -1) + mean_t.view(1, 1, -1)
                        vfk = compute_fk_loss(x_hat_denorm, vgfk[:1],
                                             vb['goal_dim_mask'][:1].to(device),
                                             vb['goal_time_mask'][:1].to(device),
                                             vlsm[:1].to(device),
                                             fk_layer,
                                             torch.tensor([vb['lengths'][0]], device=device))
                        val_losses.append(vfk.item())

                avg_val = np.mean(val_losses) if val_losses else float('inf')
                print(f'\n[Step {global_step}] val_fk={avg_val:.6f} best={best_val_loss:.6f}')

                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'global_step': global_step,
                    'phase': 1 if global_step < args.phase1_steps else 2,
                }, os.path.join(output_dir, f'ckpt_{global_step}.tar'))

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    torch.save({
                        'model': model.state_dict(),
                        'global_step': global_step,
                    }, os.path.join(output_dir, 'best.tar'))
                    print('  -> Best!')

                model.train()

    torch.save({'model': model.state_dict(), 'global_step': global_step,
                'phase': 1 if global_step < args.phase1_steps else 2},
               os.path.join(output_dir, 'final.tar'))
    print(f'\n[Done] Best val_fk={best_val_loss:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PACE Two-Pathway Training')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--name', type=str, default='pace_both_v1')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--phase1_steps', type=int, default=30000,
                       help='Stage 1: freeze backbone, train goal modules')
    parser.add_argument('--phase2_steps', type=int, default=70000,
                       help='Stage 2: unfreeze backbone, full training')
    parser.add_argument('--fk_ramp_steps', type=int, default=10000,
                       help='Steps to ramp FK loss from min to max')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--goal_lr_mult', type=float, default=0.5,
                       help='Goal module LR multiplier vs backbone')
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--base_dim', type=int, default=128)
    parser.add_argument('--time_dim', type=int, default=256)
    parser.add_argument('--lambda_fk_min', type=float, default=0.05)
    parser.add_argument('--lambda_fk_max', type=float, default=1.0)
    parser.add_argument('--text_dropout', type=float, default=0.1)
    parser.add_argument('--goal_dropout', type=float, default=0.15)
    parser.add_argument('--text_drop_goal', type=float, default=0.1)
    parser.add_argument('--log_every', type=int, default=200)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--data_dir', type=str, default=os.path.join(SRC, 'pace_data_v2'),
                       help='PACE augmented data directory')
    parser.add_argument('--allow_partial_data', action='store_true',
                       help='Allow training/smoke tests before manifest.json exists or matches file count')
    parser.add_argument('--allow_old_data', action='store_true',
                       help='Allow pace_data_v1 explicitly for debugging only')
    parser.add_argument('--pretrained_ckpt', type=str,
                       default=os.path.join(ROOT, 'checkpoints/robotv2_world/fm_base_v2'))

    args = parser.parse_args()
    train(args)
