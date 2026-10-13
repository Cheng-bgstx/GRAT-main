"""
PACE++ Dual-Path Training: frozen UNet + trainable DSSA Residual Adapter.

Key features:
  - Slot dropout: random deactivation of goal slots during training
  - Non-active preservation: constrain inactive slots to stay close to base
  - Flow-time-weighted FK loss
  - Velocity preservation for motion dynamics
"""
import sys, os, math, argparse, json, time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

SRC = '/mnt/workspace/Jensen/project/ECHO_Closd/ROBOTSHELL_GENERATOR/new'
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
sys.path.insert(0, ROOT)

from models import build_models
from models.robot_fk_layer import RobotFKLayer
from models.flow_matching import ConditionalFlowMatcher

import importlib.util
spec_adapter = importlib.util.spec_from_file_location('dp_adapter', os.path.join(SRC, 'models/pace_dual_path.py'))
dp_adapter = importlib.util.module_from_spec(spec_adapter); spec_adapter.loader.exec_module(dp_adapter)
DSSAResidualAdapter = dp_adapter.DSSAResidualAdapter

SLOT_NAMES = ['pelvis', 'lf', 'rf', 'lw', 'rw', 'traj', 'heading']
SLOT_WEIGHTS = torch.tensor([1.0, 2.5, 2.5, 3.0, 3.0, 1.0, 1.5])

SLOT_DROP_PROBS = {'traj': 0.15, 'heading': 0.15, 'pelvis': 0.30,
                   'lf': 0.40, 'rf': 0.40, 'lw': 0.40, 'rw': 0.40}


def apply_slot_dropout(gsm, drop_probs, unconditional_prob=0.10):
    """Randomly drop goal slots. 10% chance of dropping ALL slots."""
    B = gsm.shape[0]
    gsm_new = gsm.clone()
    for b in range(B):
        if torch.rand(1).item() < unconditional_prob:
            gsm_new[b] = False
            continue
        for s in range(7):
            name = SLOT_NAMES[s]
            if gsm[b, s] and torch.rand(1).item() < drop_probs[name]:
                gsm_new[b, s] = False
    return gsm_new


class PacedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split='train', train_ratio=0.95, max_groups=None):
        self.data_dir = data_dir
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        groups = {}
        for f in all_files:
            parts = f.replace('.npz', '').rsplit('_', 1)
            if len(parts) != 2: continue
            pid = parts[0]
            pid_parts = pid.rsplit('_', 1)
            if len(pid_parts) == 2 and pid_parts[1][0] in 'ab':
                pid = pid_parts[0]
            if pid not in groups: groups[pid] = []
            groups[pid].append(f)
        self.groups = {k: v for k, v in groups.items() if len(v) > 0}
        self.group_ids = sorted(self.groups.keys())
        np.random.seed(42)
        idx = np.random.permutation(len(self.group_ids))
        n = int(len(self.group_ids) * train_ratio)
        if split == 'train':
            self.group_ids = [self.group_ids[i] for i in idx[:n]]
        else:
            self.group_ids = [self.group_ids[i] for i in idx[n:]]
        if max_groups: self.group_ids = self.group_ids[:max_groups]
        print(f'[Dataset] {split}: {len(self.group_ids)} groups')

    def __len__(self): return len(self.group_ids)

    def __getitem__(self, idx):
        gid = self.group_ids[idx]
        f = np.random.choice(self.groups[gid])
        d = np.load(os.path.join(self.data_dir, f), allow_pickle=True)
        motion = torch.from_numpy(d['motion_38d'].copy()).float()
        gv = torch.from_numpy(d['goal_value'].copy()).float()
        gfk = torch.from_numpy(d['goal_fk_target'].copy()).float() if 'goal_fk_target' in d else gv.clone()
        gdm = torch.from_numpy(d['goal_dim_mask'].copy()).bool()
        gtm = torch.from_numpy(d['goal_time_mask'].copy()).bool()
        gsm = torch.from_numpy(d['goal_slot_mask'].copy()).bool()
        lsm = torch.from_numpy(d['loss_slot_mask'].copy()).bool() if 'loss_slot_mask' in d else gsm.clone()
        aux = torch.from_numpy(d['aux_slot_mask'].copy()).bool() if 'aux_slot_mask' in d else torch.zeros(7, dtype=torch.bool)
        lsw = torch.from_numpy(d['loss_slot_weight'].copy()).float() if 'loss_slot_weight' in d else torch.ones(7)
        contact = torch.from_numpy(d['contact'].copy()).bool() if 'contact' in d else torch.zeros(motion.shape[0], 2, dtype=torch.bool)
        text = str(d.get('text', 'a person walks forward'))
        T = min(motion.shape[0], 300)
        return {'motion': motion[:T], 'goal_value': gv[:T], 'goal_fk_target': gfk[:T], 'goal_dim_mask': gdm[:T],
                'goal_time_mask': gtm[:T], 'goal_slot_mask': gsm, 'loss_slot_mask': lsm,
                'aux_slot_mask': aux, 'loss_slot_weight': lsw, 'contact': contact[:T],
                'length': T, 'text': text}


def collate(batch, max_T=300):
    B, D = len(batch), 38
    max_len = min(max_T, max(b['motion'].shape[0] for b in batch))
    motion = torch.zeros(B, max_len, D); gv = torch.zeros(B, max_len, 7, 4)
    gfk = torch.zeros(B, max_len, 7, 4)
    gdm = torch.zeros(B, max_len, 7, 4, dtype=torch.bool)
    gtm = torch.zeros(B, max_len, 7, dtype=torch.bool)
    gsm = torch.stack([b['goal_slot_mask'] for b in batch])
    lsm = torch.stack([b['loss_slot_mask'] for b in batch])
    lsw = torch.stack([b['loss_slot_weight'] for b in batch])
    contact = torch.zeros(B, max_len, 2); lengths = torch.zeros(B, dtype=torch.long)
    texts = []
    for i, b in enumerate(batch):
        T = b['motion'].shape[0]; L = min(T, max_len)
        motion[i,:L] = b['motion'][:L]; gv[i,:L] = b['goal_value'][:L]
        gfk[i,:L] = b['goal_fk_target'][:L]
        gdm[i,:L] = b['goal_dim_mask'][:L]; gtm[i,:L] = b['goal_time_mask'][:L]
        contact[i,:L] = b['contact'][:L]; lengths[i] = T
        texts.append(b.get('text', 'a person walks forward'))
    return {'motion': motion, 'goal_value': gv, 'goal_fk_target': gfk, 'goal_dim_mask': gdm,
            'goal_time_mask': gtm, 'goal_slot_mask': gsm, 'loss_slot_mask': lsm,
            'loss_slot_weight': lsw, 'contact': contact,
            'lengths': lengths, 'texts': texts}


def world_goal_to_relative(motion, goal_world, fk_layer):
    B, T = motion.shape[:2]
    device = motion.device
    with torch.no_grad():
        joints = torch.clamp(motion[:, :, :29], -3.0, 3.0)
        fk_out = fk_layer(torch.cat([joints, motion[:, :, 29:]], dim=-1))
        rp0 = fk_out['root_pos'][:, 0, :2]
        rr0 = fk_out['root_rot'][:, 0]
        yaw0 = torch.atan2(rr0[:, 1, 0], rr0[:, 0, 0])
    cy = torch.cos(-yaw0).view(B, 1)
    sy = torch.sin(-yaw0).view(B, 1)
    rel = goal_world.clone()
    for s in range(7):
        if s == 6:
            yaw = torch.atan2(goal_world[:, :, s, 1], goal_world[:, :, s, 0]) - yaw0.view(B, 1)
            rel[:, :, s, 0] = torch.cos(yaw)
            rel[:, :, s, 1] = torch.sin(yaw)
        else:
            xy = goal_world[:, :, s, :2] - rp0.view(B, 1, 2)
            x = xy[:, :, 0] * cy - xy[:, :, 1] * sy
            y = xy[:, :, 0] * sy + xy[:, :, 1] * cy
            rel[:, :, s, 0] = x
            rel[:, :, s, 1] = y
    return rel


def activate_random_slots(gsm, min_extra=2, max_extra=7):
    """Randomly activate additional slots for multi-slot training."""
    B = gsm.shape[0]
    gsm_new = gsm.clone()
    for b in range(B):
        inactive = (gsm[b] == 0).nonzero(as_tuple=True)[0]
        n_inactive = len(inactive)
        if n_inactive == 0: continue
        r = torch.rand(1).item()
        if r < 0.30 and n_inactive >= 5:
            n = n_inactive
        elif r < 0.70:
            lo, hi = min(min_extra, n_inactive), min(max_extra, n_inactive)
            n = np.random.randint(lo, hi + 1) if hi > lo else lo
        else:
            lo, hi = 1, min(3, n_inactive)
            n = np.random.randint(lo, hi + 1) if hi > lo else lo
        n = max(1, min(n, n_inactive))
        chosen = inactive[torch.randperm(n_inactive)[:n]]
        gsm_new[b, chosen] = True
    return gsm_new


def build_synthetic_goals(motion, gv, gfk, gdm, gtm, gsm, fk_layer, add_pert=True):
    """Build synthetic root-relative goals, using dataset FK targets for active slots."""
    B, T = motion.shape[:2]; device = motion.device
    joints = torch.clamp(motion[:,:,:29], -3.0, 3.0)
    phys = torch.cat([joints, motion[:,:,29:]], dim=-1)
    try:
        fk_out = fk_layer(phys)
        kp = fk_out['keypoint_pos']; rp = fk_out['root_pos']; rr = fk_out['root_rot']
    except Exception:
        return gv, gdm, gtm, gsm

    gv_new, gdm_new, gtm_new = gv.clone(), gdm.clone(), gtm.clone()
    if gfk is not None:
        gfk_rel = world_goal_to_relative(motion, gfk.to(device), fk_layer)
        active_mask = gsm.view(B, 1, 7, 1) & gdm
        gv_new = torch.where(active_mask, gfk_rel, gv_new)
    last_t = T - 1
    pert_mag = {'pelvis': 0.15, 'foot': 0.15, 'wrist': 0.25, 'traj': 0.8, 'heading': 0.3}

    for b in range(B):
        for s in range(7):
            if gsm[b, s]: continue
            if s < 5:
                base_val = kp[b, last_t, s].clone()
                if add_pert:
                    slot_type = 'pelvis' if s == 0 else ('foot' if s in (1,2) else 'wrist')
                    mag = pert_mag[slot_type]
                    base_val = base_val + (torch.rand(3, device=device) - 0.5) * 2 * mag
                gv_new[b, last_t, s, :3] = base_val; gdm_new[b, last_t, s, :3] = True
            elif s == 5:
                base_val = rp[b, last_t, :2].clone()
                if add_pert:
                    mag = pert_mag['traj']
                    offset = (torch.rand(1, device=device).item() - 0.5) * 2 * mag
                    angle = torch.rand(1, device=device).item() * 2 * math.pi
                    base_val[0] += offset * math.cos(angle)
                    base_val[1] += offset * math.sin(angle)
                gv_new[b, last_t, s, :2] = base_val; gdm_new[b, last_t, s, :2] = True
            elif s == 6:
                yaw = torch.atan2(rr[b, last_t, 1, 0], rr[b, last_t, 0, 0])
                if add_pert:
                    yaw = yaw + (torch.rand(1, device=device).item() - 0.5) * 2 * pert_mag['heading'] * math.pi
                gv_new[b, last_t, s, 0] = torch.cos(yaw)
                gv_new[b, last_t, s, 1] = torch.sin(yaw)
                gdm_new[b, last_t, s, :2] = True
            gtm_new[b, last_t, s] = True
    return gv_new, gdm_new, gtm_new


def perturb_goals(gv, gdm, gtm, gsm, pert_cfg):
    """Apply spatial perturbation to goal positions."""
    B, T = gv.shape[:2]
    gv_pert = gv.clone(); device = gv.device
    for b in range(B):
        active = gsm[b].nonzero(as_tuple=True)[0]
        for s_idx in active:
            s = s_idx.item()
            t_active = gtm[b, :, s].nonzero(as_tuple=True)[0]
            if len(t_active) == 0: continue
            if s == 5:
                mag = pert_cfg['traj']
                offset = (torch.rand(1, device=device).item() - 0.5) * 2 * mag
                angle = torch.rand(1, device=device).item() * 2 * math.pi
                for t in t_active:
                    gv_pert[b, t, s, 0] += offset * math.cos(angle)
                    gv_pert[b, t, s, 1] += offset * math.sin(angle)
            elif s == 6:
                angle = (torch.rand(1, device=device).item() - 0.5) * 2 * pert_cfg['heading']
                ca, sa = math.cos(angle), math.sin(angle)
                for t in t_active:
                    ch, sh = gv_pert[b, t, s, 0].item(), gv_pert[b, t, s, 1].item()
                    gv_pert[b, t, s, 0] = ch * ca - sh * sa
                    gv_pert[b, t, s, 1] = ch * sa + sh * ca
            else:
                mag = pert_cfg.get('wrist' if s in (3,4) else ('foot' if s in (1,2) else 'pelvis'), 0.1)
                offset = (torch.rand(3, device=device) - 0.5) * 2 * mag
                for t in t_active: gv_pert[b, t, s, :3] += offset
    return gv_pert


def compute_fk_loss(refined, goal_value, goal_dim_mask, goal_time_mask,
                    goal_slot_mask, fk_layer, loss_slot_weight=None, base_motion=None):
    """Multi-frame FK loss with per-slot weighting."""
    B, T, D = refined.shape; device = refined.device
    joints = torch.clamp(refined[:,:,:29], -3.0, 3.0)
    phys = torch.cat([joints, refined[:,:,29:]], dim=-1)
    try:
        fk_out = fk_layer(phys)
        kp = fk_out['keypoint_pos']; rp = fk_out['root_pos']; rr = fk_out['root_rot']
        if torch.isnan(kp).any() or torch.isnan(rp).any():
            return torch.tensor(0.0, device=device, requires_grad=True), {}
    except Exception:
        return torch.tensor(0.0, device=device, requires_grad=True), {}

    pred = torch.zeros(B, T, 7, 4, device=device)
    for s in range(5): pred[:,:,s,:3] = kp[:,:,s]
    pred[:,:,5,:2] = rp[:,:,:2]
    py = torch.atan2(rr[:,:,1,0], rr[:,:,0,0])
    pred[:,:,6,0] = torch.cos(py); pred[:,:,6,1] = torch.sin(py)

    slot_w = loss_slot_weight.to(device) if loss_slot_weight is not None else SLOT_WEIGHTS.to(device)
    loss_target = goal_value
    if base_motion is not None:
        loss_target = world_goal_to_relative(base_motion, goal_value, fk_layer)
    sq_err = (pred - loss_target) ** 2
    base_mask = (goal_dim_mask.float() * goal_time_mask.float().unsqueeze(-1)
                 * goal_slot_mask.float().view(B, 1, 7, 1)
                 * slot_w.view(B, 1, 7, 1))
    n_active = base_mask.sum().clamp(min=1)
    loss_fk = (sq_err * base_mask).sum() / n_active

    per_slot = {}
    for i, name in enumerate(SLOT_NAMES):
        n = (goal_slot_mask[:, i].float().unsqueeze(1) * goal_time_mask[:, :, i].float()).sum()
        if n > 0:
            err = (sq_err[:,:,i].sum(dim=-1) * goal_time_mask[:,:,i].float()
                   * goal_slot_mask[:,i:i+1].float()).sum() / n.clamp(min=1)
            per_slot[name] = err.item()
    return loss_fk, per_slot


def compute_nonactive_loss(refined, base_motion, goal_slot_mask, fk_layer):
    """Constrain inactive slots to stay close to base motion FK positions."""
    B, T, D = refined.shape; device = refined.device
    inactive = ~goal_slot_mask
    if not inactive.any(): return torch.tensor(0.0, device=device)

    # Compute FK for both
    joints_r = torch.clamp(refined[:,:,:29], -3.0, 3.0)
    joints_b = torch.clamp(base_motion[:,:,:29], -3.0, 3.0)
    fk_r = fk_layer(torch.cat([joints_r, refined[:,:,29:]], dim=-1))
    fk_b = fk_layer(torch.cat([joints_b, base_motion[:,:,29:]], dim=-1))

    kp_r = fk_r['keypoint_pos']; kp_b = fk_b['keypoint_pos']  # (B,T,5,3)
    rp_r = fk_r['root_pos']; rp_b = fk_b['root_pos']            # (B,T,3)
    rr_r = fk_r['root_rot']; rr_b = fk_b['root_rot']            # (B,T,3,3)

    loss = torch.tensor(0.0, device=device)
    n_inactive = 0
    for s in range(7):
        if inactive[:, s].any():
            n_inactive += 1
            if s < 5:
                loss += ((kp_r[:,:,s] - kp_b[:,:,s]) ** 2).mean()
            elif s == 5:
                loss += ((rp_r[:,:,:2] - rp_b[:,:,:2]) ** 2).mean()
            elif s == 6:
                yaw_r = torch.atan2(rr_r[:,:,1,0], rr_r[:,:,0,0])
                yaw_b = torch.atan2(rr_b[:,:,1,0], rr_b[:,:,0,0])
                loss += (1 - torch.cos(yaw_r - yaw_b)).mean()
    return loss / max(1, n_inactive)


def compute_contact_loss(refined, contact_label, fk_layer):
    """Penalize foot sliding during contact phases."""
    B, T, D = refined.shape; device = refined.device
    joints = torch.clamp(refined[:,:,:29], -3.0, 3.0)
    phys = torch.cat([joints, refined[:,:,29:]], dim=-1)
    try:
        fk_out = fk_layer(phys); kp = fk_out['keypoint_pos']
    except Exception:
        return torch.tensor(0.0, device=device)
    lf_vel = (kp[:,1:,1] - kp[:,:-1,1]).norm(dim=-1)
    rf_vel = (kp[:,1:,2] - kp[:,:-1,2]).norm(dim=-1)
    lf_c = contact_label[:,1:,0].float().to(device)
    rf_c = contact_label[:,1:,1].float().to(device)
    return (lf_vel * lf_c).sum() / lf_c.sum().clamp(min=1) + \
           (rf_vel * rf_c).sum() / rf_c.sum().clamp(min=1)


def train(args):
    device = torch.device(f'cuda:{args.gpu}')
    print(f'[PACE++] Device: {device}')

    # Data
    data_dir = args.data_dir or os.path.join(SRC, 'pace_data_v1')
    train_ds = PacedDataset(data_dir, split='train', max_groups=args.max_groups)
    val_ds = PacedDataset(data_dir, split='val',
                          max_groups=args.max_groups//10 if args.max_groups else None)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.n_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.n_workers, pin_memory=True)

    # Frozen UNet
    from types import SimpleNamespace as SN
    opt_unet = SN()
    opt_unet.dim_pose = 38; opt_unet.model_type = 'unet_small'
    opt_unet.base_dim = 128; opt_unet.time_dim = 256
    opt_unet.dim_mults = [1, 2, 2, 2]; opt_unet.text_latent_dim = 256
    opt_unet.no_adagn = False; opt_unet.no_eff = False
    opt_unet.cond_mask_prob = 0.1; opt_unet.enable_goal_cond = False
    opt_unet.unet_goal_inject = 'temb'; opt_unet.goal_dim = 3
    opt_unet.target_joint_names = 'traj,heading'
    opt_unet.enable_length_head = False; opt_unet.use_goal_adapter = False
    opt_unet.goal_adapter_encoder = 'multi'; opt_unet.goal_adapter_injection = 'add'
    opt_unet.goal_adapter_layers = 1; opt_unet.text_drop_keep_goal_prob = 0.0
    opt_unet.use_goal_spatial_attn = False
    opt_unet.end_token_exp = 'none'; opt_unet.max_motion_length = 490

    unet = build_models(opt_unet).to(device)
    ckpt_path = os.path.join(ROOT, 'checkpoints', 'robotv2_world', 'fm_base_v2', 'model', 'latest.tar')
    state = torch.load(ckpt_path, map_location='cpu')
    if 'encoder' in state: state = state['encoder']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    mapped = {k: v for k, v in state.items() if k.startswith('embed_text.') or k.startswith('unet.')}
    unet.load_state_dict(mapped, strict=False)
    unet.eval()
    for p in unet.parameters(): p.requires_grad = False
    print('[UNet] Frozen fm_base_v2 loaded')

    # Adapter
    adapter = DSSAResidualAdapter(
        motion_dim=38, hidden_dim=args.hidden_dim, text_dim=256,
        n_slots=7, slot_dim=4, n_blocks=args.n_blocks,
        kernel_size=args.kernel_size
    ).to(device)
    print(f'[Adapter] {sum(p.numel() for p in adapter.parameters()):,} params ({sum(p.numel() for p in adapter.parameters())/1e6:.1f}M)')

    fk_layer = RobotFKLayer().to(device)
    cfm = ConditionalFlowMatcher()

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps, eta_min=args.lr*0.1)

    output_dir = os.path.join(SRC, 'checkpoints', 'pace_dual_path', args.name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f: json.dump(vars(args), f, indent=2)

    def get_pert_config(progress):
        p = 1.0 / (1.0 + math.exp(-10*(progress - 0.3)))
        return {'traj': 0.3+p*1.7, 'heading': (0.05+p*0.25)*math.pi,
                'wrist': 0.08+p*0.42, 'foot': 0.05+p*0.25, 'pelvis': 0.05+p*0.25}

    print(f'[Train] {args.n_steps} steps, lr={args.lr}')
    print(f'[Train] λ_fk={args.lambda_fk}, λ_smooth={args.lambda_smooth}, λ_contact={args.lambda_contact}, λ_vel={args.lambda_vel}, λ_nonactive={args.lambda_nonactive}, λ_vgoal={args.lambda_vgoal}')

    global_step = 0; best_val_fk = float('inf')
    pbar = tqdm(total=args.n_steps)

    while global_step < args.n_steps:
        for batch in train_loader:
            if global_step >= args.n_steps: break

            motion = batch['motion'].to(device)
            gv = batch['goal_value'].to(device)
            gfk = batch['goal_fk_target'].to(device)
            gdm = batch['goal_dim_mask'].to(device)
            gtm = batch['goal_time_mask'].to(device)
            gsm = batch['goal_slot_mask'].to(device)
            contact = batch['contact'].to(device)

            progress = global_step / args.n_steps
            pert_cfg = get_pert_config(progress)

            # --- Multi-slot activation + perturbation ---
            gsm_ext = activate_random_slots(gsm, min_extra=2, max_extra=7)
            gv_ext, gdm_ext, gtm_ext = build_synthetic_goals(
                motion, gv, gfk, gdm, gtm, gsm_ext, fk_layer, add_pert=True)
            pert_gv = perturb_goals(gv_ext, gdm_ext, gtm_ext, gsm_ext, pert_cfg)

            # --- Slot dropout (critical for generalization) ---
            gsm_train = apply_slot_dropout(gsm_ext, SLOT_DROP_PROBS)

            # --- Flow matching sampling ---
            t = cfm.sample_time(motion.shape[0], device)
            x_0 = cfm.sample_noise(motion.shape, device).to(motion.dtype)
            x_t = cfm.interpolate(x_0, motion, t)

            # 1. UNet velocity prediction (frozen) — use REAL text
            t_int = (t * 999).long()
            texts = batch.get('texts', ['a person walks forward'] * motion.shape[0])
            with torch.no_grad():
                enc_text = unet.encode_text(texts, device)
                v_text = unet(x_t, t_int, enc_text=enc_text)
                if isinstance(v_text, tuple): v_text = v_text[0]

            # 2. Adapter velocity prediction (trainable)
            priority = adapter.get_default_priority(gsm_train, device)
            v_goal = adapter(x_t, enc_text, pert_gv, gdm_ext, gtm_ext, gsm_train, priority)

            # 3. Blend with per-sample α(t) gating
            alpha_val = torch.sigmoid((t - 0.25) / 0.05)  # per-sample (B,)
            v_pred = v_text + alpha_val.view(-1, 1, 1) * v_goal

            # 4. Predict clean motion: x̂₁ = x_t + (1-t) · v_pred
            x_hat_1 = x_t + (1.0 - t.view(-1, 1, 1)) * v_pred

            # --- Losses ---
            # FK loss
            loss_fk, per_slot = compute_fk_loss(
                x_hat_1, pert_gv, gdm_ext, gtm_ext, gsm_train, fk_layer,
                batch.get('loss_slot_weight', None).to(device) if 'loss_slot_weight' in batch else None)

            # Non-active preservation
            loss_nonactive = compute_nonactive_loss(
                x_hat_1, motion, gsm_train, fk_layer)

            # Contact
            loss_contact = compute_contact_loss(x_hat_1, contact, fk_layer)

            # Velocity preservation
            vel_r = x_hat_1[:, 1:] - x_hat_1[:, :-1]
            vel_b = motion[:, 1:] - motion[:, :-1]
            loss_vel = F.mse_loss(vel_r[:, :, :29], vel_b[:, :, :29])

            # Smoothness
            loss_smooth = vel_r.pow(2).mean()
            loss_joint = F.relu(x_hat_1[:,:,:29].abs() - 2.8).pow(2).mean()

            # v_goal regularization: penalize large corrections
            loss_vgoal = v_goal.pow(2).mean()

            loss = (args.lambda_fk * loss_fk +
                    args.lambda_nonactive * loss_nonactive +
                    args.lambda_vgoal * loss_vgoal +
                    args.lambda_smooth * loss_smooth +
                    args.lambda_contact * loss_contact +
                    args.lambda_vel * loss_vel + 0.05 * loss_joint)

            if torch.isnan(loss) or torch.isinf(loss): continue

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.clip_grad)
            optimizer.step(); scheduler.step()

            if global_step % args.log_every == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}', 'fk': f'{loss_fk.item():.4f}',
                    'nonact': f'{loss_nonactive.item():.4f}', 'vg': f'{loss_vgoal.item():.4f}',
                    'ct': f'{loss_contact.item():.4f}', 'vel': f'{loss_vel.item():.4f}',
                })
            global_step += 1; pbar.update(1)

            # Validation
            if global_step % args.save_every == 0:
                adapter.eval()
                val_fks = []
                with torch.no_grad():
                    for vi, vb in enumerate(val_loader):
                        if vi >= 2: break
                        vm = vb['motion'].to(device)
                        vgv = vb['goal_value'].to(device)
                        vgfk = vb['goal_fk_target'].to(device)
                        vgdm = vb['goal_dim_mask'].to(device)
                        vgtm = vb['goal_time_mask'].to(device)
                        vgsm = vb['goal_slot_mask'].to(device)

                        gsm_all7 = torch.ones_like(vgsm)
                        gv_all7, gdm_all7, gtm_all7 = build_synthetic_goals(
                            vm, vgv, vgfk, vgdm, vgtm, gsm_all7, fk_layer, add_pert=True)
                        pert_all7 = perturb_goals(gv_all7, gdm_all7, gtm_all7, gsm_all7, get_pert_config(0.5))

                        t_val = torch.zeros(vm.shape[0], device=device)
                        x_t_val = cfm.interpolate(cfm.sample_noise(vm.shape, device).to(vm.dtype), vm, t_val)
                        t_int_val = (t_val * 999).long()
                        val_texts = vb.get('texts', ['a person walks forward'] * vm.shape[0])
                        enc_v = unet.encode_text(val_texts, device)
                        v_t = unet(x_t_val, t_int_val, enc_text=enc_v)
                        if isinstance(v_t, tuple): v_t = v_t[0]
                        v_g = adapter(x_t_val, enc_v, pert_all7, gdm_all7, gtm_all7, gsm_all7)
                        v_pred_val = v_t + 1.0 * v_g
                        x_hat_val = x_t_val + (1.0 - t_val.view(-1, 1, 1)) * v_pred_val

                        vfk, _ = compute_fk_loss(x_hat_val, pert_all7, gdm_all7, gtm_all7, gsm_all7, fk_layer)
                        if not torch.isnan(vfk): val_fks.append(vfk.item())

                avg_fk = np.mean(val_fks) if val_fks else float('inf')
                print(f'\n[Step {global_step}] val_fk={avg_fk:.6f} best={best_val_fk:.6f}')

                torch.save({'adapter': adapter.state_dict(), 'optimizer': optimizer.state_dict(),
                           'global_step': global_step}, os.path.join(output_dir, f'ckpt_{global_step}.tar'))
                if avg_fk < best_val_fk:
                    best_val_fk = avg_fk
                    torch.save({'adapter': adapter.state_dict(), 'global_step': global_step},
                              os.path.join(output_dir, 'best.tar'))
                    print('  -> Best!')
                adapter.train()

    torch.save({'adapter': adapter.state_dict(), 'global_step': global_step},
              os.path.join(output_dir, 'final.tar'))
    print(f'\n[Done] Best val_fk={best_val_fk:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--name', type=str, default='pace_dual_v1')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--n_steps', type=int, default=60000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--lambda_fk', type=float, default=5.0)
    parser.add_argument('--lambda_smooth', type=float, default=1.5)
    parser.add_argument('--lambda_contact', type=float, default=0.5)
    parser.add_argument('--lambda_vel', type=float, default=0.3)
    parser.add_argument('--lambda_nonactive', type=float, default=1.0)
    parser.add_argument('--lambda_vgoal', type=float, default=0.3,
                       help='L2 penalty on |v_goal| to prevent adapter from dominating UNet')
    parser.add_argument('--log_every', type=int, default=200)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--max_groups', type=int, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()
    train(args)
