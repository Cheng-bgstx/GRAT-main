"""
PACE-v2 Data Generation: Semantic-Motion-Aware Arrival-Manifold Expansion.

Operator pool (not fixed per-clip pipeline):
  O0: raw_prior       — preserve original motion prior
  O1: SE2_arrival     — omnidirectional rotation
  O2: arrival_slice   — temporal stop-state recasting
  O3: traj_warp       — contact-aware root perturbation
  O4: terminal_EE_IK  — wrist/foot precision IK
  O5: posture_recast  — pelvis/posture arrival

Scheduler adaptively selects operators per clip based on semantic-motion analysis.
All samples rebuilt through FK for goal-motion consistency.

Usage: python scripts/generate_pace_data.py --n_workers 4
"""
import sys, os, torch, numpy as np, math, json, argparse, hashlib, traceback, time, random, subprocess
from collections import defaultdict, Counter
from multiprocessing import Pool
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
from robot_fk_layer import RobotFKLayer
from utils.robot_process import process_robot_npz
from utils.robot_target_cond import ROBOT_TARGET_NAME_TO_INDEX
from utils.stats_paths import resolve_stats_path
from utils.rotation_utils import quat_wxyz_to_yaw

# ============================================================
# Schema / constants
# ============================================================

PACE_SCHEMA_VERSION = 'pace_v2.0'
SLOT_NAMES = ['pelvis', 'left_foot', 'right_foot', 'left_wrist', 'right_wrist', 'traj', 'heading']
GOAL_DIMS = 4
FPS = 50
DEFAULT_TOLERANCE = np.array([0.08, 0.05, 0.05, 0.05, 0.05, 0.10, 0.17], dtype=np.float32)
TEACHER_WEIGHTS = {
    'raw_prior': (1.0, 0.3, 0.5),
    'se2_arrival': (1.0, 0.8, 0.8),
    'arrival_slice': (1.0, 1.0, 1.5),
    'traj_warp': (1.0, 1.2, 0.8),
    'ee_ik': (1.0, 1.4, 0.8),
    'posture_recast': (1.0, 1.0, 1.2),
}
BASE_SLOT_WEIGHTS = np.array([0.8, 1.2, 1.2, 1.5, 1.5, 1.2, 0.8], dtype=np.float32)
REQUIRED_SAMPLE_KEYS = [
    'motion_38d', 'goal_world', 'goal_rel', 'goal_value', 'goal_fk_target',
    'goal_dim_mask', 'goal_time_mask', 'goal_slot_mask', 'loss_slot_mask',
    'aux_slot_mask', 'contact', 'text', 'teacher_type', 'applied_ops',
    'requested_achieved_residual', 'requested_achieved_ok',
]
_FK_LAYER = None


# ============================================================
# Utils
# ============================================================

def stable_seed(base_seed, *parts):
    payload = '::'.join([str(base_seed)] + [str(p) for p in parts]).encode('utf-8')
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def get_fk_layer():
    global _FK_LAYER
    if _FK_LAYER is None:
        _FK_LAYER = RobotFKLayer()
    return _FK_LAYER


def init_worker(seed):
    global _FK_LAYER
    worker_seed = stable_seed(seed, os.getpid())
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32 - 1))
    torch.manual_seed(worker_seed)
    _FK_LAYER = RobotFKLayer()


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def text_has_explicit_metric(text):
    import re
    text_lower = text.lower()
    numeric = re.search(r'\b\d+\.?\d*\s*(m|meters?|metres?|feet|ft|seconds?|sec|s|degrees?|deg)\b', text_lower)
    words = re.search(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|half)\s+(meters?|metres?|feet|seconds?|degrees?)\b', text_lower)
    return bool(numeric or words)


def build_time_mask(T, loss_mask):
    time_mask = np.zeros((T, 7), dtype=bool)
    if T > 0:
        time_mask[T - 1, loss_mask] = True
    return time_mask


def compute_goal_rel_all(goal_world, root_start_xy, root_start_yaw):
    cos_a, sin_a = math.cos(-root_start_yaw), math.sin(-root_start_yaw)
    R2_inv = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    g_rel = goal_world.copy()
    for t in range(goal_world.shape[0]):
        for slot in range(7):
            if slot == 6:
                if np.any(goal_world[t, slot, :2]):
                    yaw = math.atan2(goal_world[t, slot, 1], goal_world[t, slot, 0]) - root_start_yaw
                    g_rel[t, slot, 0] = math.cos(yaw)
                    g_rel[t, slot, 1] = math.sin(yaw)
            else:
                if np.any(goal_world[t, slot, :3]):
                    rel_xy = goal_world[t, slot, :2] - root_start_xy
                    g_rel[t, slot, :2] = R2_inv @ rel_xy
    return g_rel


def mask_goal_arrays(requested, achieved, dim_mask, time_mask, condition_mask, loss_mask):
    requested = requested.copy()
    achieved = achieved.copy()
    dim_mask = dim_mask.copy()
    active_time = time_mask[:, :, None]
    cond = condition_mask[None, :, None]
    loss = loss_mask[None, :, None]
    requested_mask = active_time & cond & dim_mask
    achieved_mask = active_time & loss & dim_mask
    requested[~requested_mask] = 0.0
    achieved[~achieved_mask] = 0.0
    dim_mask &= active_time & (cond | loss)
    return requested, achieved, dim_mask


def fill_requested_goal_from_achieved(requested, achieved, dim_mask, time_mask, condition_mask):
    requested = requested.copy()
    cond_mask = time_mask[:, :, None] & condition_mask[None, :, None] & dim_mask
    missing = cond_mask & (np.abs(requested) <= 1e-8)
    requested[missing] = achieved[missing]
    return requested


def requested_achieved_residual(requested, achieved, dim_mask, time_mask, condition_mask, tolerance):
    residual = np.zeros(7, dtype=np.float32)
    ok = np.ones(7, dtype=bool)
    errors = []
    for s in range(7):
        frames = np.where(time_mask[:, s] & condition_mask[s])[0]
        if len(frames) == 0:
            continue
        slot_errs = []
        for t in frames:
            if s < 5:
                if dim_mask[t, s, :3].any():
                    slot_errs.append(float(np.linalg.norm(requested[t, s, :3] - achieved[t, s, :3])))
            elif s == 5:
                if dim_mask[t, s, :2].any():
                    slot_errs.append(float(np.linalg.norm(requested[t, s, :2] - achieved[t, s, :2])))
            else:
                if dim_mask[t, s, :2].any():
                    yaw_req = math.atan2(requested[t, s, 1], requested[t, s, 0])
                    yaw_ach = math.atan2(achieved[t, s, 1], achieved[t, s, 0])
                    err = abs(yaw_req - yaw_ach)
                    slot_errs.append(float(min(err, 2 * math.pi - err)))
        if not slot_errs:
            continue
        residual[s] = max(slot_errs)
        threshold = max(float(tolerance[s]) * 2.0, 0.04)
        if residual[s] > threshold:
            ok[s] = False
            errors.append(f'requested_achieved_s{s}:{residual[s]:.4f}')
    return residual, ok, errors


def slot_action_type(tags):
    if tags.get('locomotion', 0.0) > 0.5:
        return 'locomotion'
    if tags.get('arm', 0.0) > 0.5:
        return 'arm'
    if tags.get('leg', 0.0) > 0.5:
        return 'leg'
    if tags.get('turning', 0.0) > 0.5:
        return 'turning'
    return 'other'


def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), text=True).strip()
    except Exception:
        return 'unknown'


def schema_hash():
    payload = json.dumps({'version': PACE_SCHEMA_VERSION, 'slots': SLOT_NAMES, 'goal_dims': GOAL_DIMS, 'keys': REQUIRED_SAMPLE_KEYS}, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]


def compute_root_state(motion_38d, fps=50):
    """Recover root_xy, root_yaw, root_z from 38D motion."""
    T = motion_38d.shape[0]
    vel_xy = motion_38d[:, 29:31].copy()
    vel_xy[0] = 0  # first frame velocity is 0
    dt = 1.0 / fps
    root_xy = np.cumsum(vel_xy * dt, axis=0)  # (T,2)
    root_z = motion_38d[:, 31]  # (T,)
    # yaw from 6D rotation
    from utils.rotation_utils import rot6d_to_quat_wxyz
    quat = rot6d_to_quat_wxyz(motion_38d[:, 32:38])
    yaw = np.array([quat_wxyz_to_yaw(q) for q in quat])
    return root_xy, root_z, yaw

def detect_contact(motion_38d, fk_layer, fps=50):
    """Detect foot-ground contact using FK height + velocity."""
    import torch
    T = motion_38d.shape[0]
    with torch.no_grad():
        mt = torch.from_numpy(motion_38d).unsqueeze(0).float()
        fk_out = fk_layer(mt)
        kp = fk_out['keypoint_pos'][0].cpu().numpy()  # (T,5,3)
    lf_z = kp[:, 1, 2]
    rf_z = kp[:, 2, 2]
    lf_vel = np.zeros(T); rf_vel = np.zeros(T)
    if T > 1:
        lf_vel[1:] = np.linalg.norm(np.diff(kp[:, 1], axis=0), axis=1) * fps
        rf_vel[1:] = np.linalg.norm(np.diff(kp[:, 2], axis=0), axis=1) * fps
    lf_contact = (lf_z < 0.05) & (lf_vel < 0.05)
    rf_contact = (rf_z < 0.05) & (rf_vel < 0.05)
    return np.stack([lf_contact, rf_contact], axis=1), kp  # (T,2)

def compute_goal_world(kp, root_xy, root_z, yaw, T):
    """Extract 7-slot world-space goal from FK positions at last frame."""
    g = np.zeros((T, 7, 4), dtype=np.float32)
    dm = np.zeros((T, 7, 4), dtype=bool)
    # All slots at last frame
    t = T - 1
    # pelvis (3D)
    g[t, 0, :3] = kp[t, 0]
    dm[t, 0, :3] = True
    # left_foot (3D)
    g[t, 1, :3] = kp[t, 1]
    dm[t, 1, :3] = True
    # right_foot (3D)
    g[t, 2, :3] = kp[t, 2]
    dm[t, 2, :3] = True
    # left_wrist (3D)
    g[t, 3, :3] = kp[t, 3]
    dm[t, 3, :3] = True
    # right_wrist (3D)
    g[t, 4, :3] = kp[t, 4]
    dm[t, 4, :3] = True
    # traj (2D in world XY)
    g[t, 5, :2] = root_xy[t]
    dm[t, 5, :2] = True
    # heading [cos, sin]
    g[t, 6, 0] = math.cos(yaw[t])
    g[t, 6, 1] = math.sin(yaw[t])
    dm[t, 6, :2] = True
    return g, dm

def classify_semantic_tags(text, motion_38d, root_xy, kp_positions, fps=50):
    """Classify motion into semantic_tags with FK keypoint velocity detection."""
    T = motion_38d.shape[0]
    disp = np.linalg.norm(root_xy[-1] - root_xy[0])
    vel_xy = np.linalg.norm(motion_38d[:, 29:31], axis=1)
    avg_speed = vel_xy.mean()

    # Locomotion: significant displacement + speed
    loco_conf = min(1.0, disp / 0.5) * min(1.0, avg_speed / 0.1)

    # Turning: yaw change
    from utils.rotation_utils import rot6d_to_quat_wxyz
    quat = rot6d_to_quat_wxyz(motion_38d[:, 32:38])
    yaw_change = abs(quat_wxyz_to_yaw(quat[-1]) - quat_wxyz_to_yaw(quat[0]))
    yaw_change = min(yaw_change, 2*math.pi - yaw_change)
    turn_conf = min(1.0, yaw_change / 0.5)

    # Arm: FK wrist velocity (keypoints 3,4 = left_wrist, right_wrist)
    if kp_positions is not None and T > 1:
        lw_vel = np.linalg.norm(np.diff(kp_positions[:, 3], axis=0), axis=1) * fps
        rw_vel = np.linalg.norm(np.diff(kp_positions[:, 4], axis=0), axis=1) * fps
        wrist_vel_max = max(lw_vel.max(), rw_vel.max())
        wrist_vel_mean = max(lw_vel.mean(), rw_vel.mean())
        arm_conf = min(1.0, wrist_vel_max / 0.4) * min(1.0, wrist_vel_mean / 0.15)
    else:
        arm_conf = 0.0

    # Leg: FK foot velocity (keypoints 1,2 = left_foot, right_foot), subtract root speed for relative
    if kp_positions is not None and T > 1:
        lf_vel = np.linalg.norm(np.diff(kp_positions[:, 1], axis=0), axis=1) * fps
        rf_vel = np.linalg.norm(np.diff(kp_positions[:, 2], axis=0), axis=1) * fps
        foot_vel_max = max(lf_vel.max(), rf_vel.max())
        foot_vel_mean = max(lf_vel.mean(), rf_vel.mean())
        # Foot velocity includes root movement; subtract root speed for isolated leg motion
        leg_conf = min(1.0, max(0, foot_vel_max - 0.5) / 1.0) * min(1.0, max(0, foot_vel_mean - 0.2) / 0.5)
    else:
        leg_conf = 0.0

    # Text keyword boost
    text_lower = text.lower()
    walk_keywords = ['walk','run','jog','step','stride','march','pace','stroll','go ','move forward','goes forward']
    reach_keywords = ['reach','grab','pick','touch','hand','arm','wave','raise','lift','punch','throw','catch']
    kick_keywords = ['kick','stomp','foot','leg','soccer','step on']
    if any(kw in text_lower for kw in walk_keywords): loco_conf = max(loco_conf, 0.7)
    if any(kw in text_lower for kw in reach_keywords): arm_conf = max(arm_conf, 0.6)
    if any(kw in text_lower for kw in kick_keywords): leg_conf = max(leg_conf, 0.6)

    static_conf = 1.0 - max(loco_conf, arm_conf, leg_conf)

    return {
        "locomotion": float(np.clip(loco_conf, 0, 1)),
        "turning": float(np.clip(turn_conf, 0, 1)),
        "arm": float(np.clip(arm_conf, 0, 1)),
        "leg": float(np.clip(leg_conf, 0, 1)),
        "static": float(np.clip(static_conf, 0, 1)),
    }

def determine_active_slots(tags, text="", kp_positions=None, fps=50):
    """Three-layer slot mask (PACE-v2).
    Returns:
      condition_slot_mask: text-driven explicit user goals (model condition input)
      loss_slot_mask:      FK loss supervision targets (always includes pelvis aux)
      aux_slot_mask:       pelvis stability supervision (always on for FK, never for user)
    """
    condition_mask = np.zeros(7, dtype=bool)
    loss_mask = np.zeros(7, dtype=bool)
    aux_mask = np.zeros(7, dtype=bool)
    text_lower = text.lower()

    # ── Text-keyword tables ──
    walk_kw = ['walk','run','jog','step','stride','march','pace','stroll',
               'go ','move forward','goes forward','go straight']
    reach_kw = ['reach','grab','pick','touch','hand','arm','wave','raise','lift',
                'punch','throw','catch','clap','gesture','swing','cross','fold','stretch arm']
    kick_kw = ['kick','stomp','foot','leg','soccer','step on','tap foot','lift leg','raise leg']
    posture_kw = ['crouch','bow','squat','sit','bend','kneel']

    has_walk_text = any(kw in text_lower for kw in walk_kw)
    has_reach_text = any(kw in text_lower for kw in reach_kw)
    has_kick_text = any(kw in text_lower for kw in kick_kw)
    has_posture_text = any(kw in text_lower for kw in posture_kw)

    # ── Pelvis (slot 0): only activate on explicit posture or locomotion text ──
    if tags["locomotion"] > 0.3 or has_walk_text:
        condition_mask[0] = True  # pelvis
        condition_mask[5] = True  # traj
        condition_mask[6] = True  # heading
    if has_posture_text:
        condition_mask[0] = True  # pelvis for crouch/squat/bend

    # ── Wrist (slots 3,4): text gate. No-direction → pick higher FK velocity side ──
    if has_reach_text and tags["arm"] > 0.3:
        if 'left' in text_lower or 'l hand' in text_lower or 'l arm' in text_lower:
            condition_mask[3] = True
        elif 'right' in text_lower or 'r hand' in text_lower or 'r arm' in text_lower:
            condition_mask[4] = True
        elif 'both' in text_lower or 'hands' in text_lower:
            condition_mask[3] = True; condition_mask[4] = True
        else:
            # Pick side with higher FK wrist velocity
            if kp_positions is not None and kp_positions.shape[0] > 1:
                lw_vel = np.linalg.norm(np.diff(kp_positions[:, 3], axis=0), axis=1).mean()
                rw_vel = np.linalg.norm(np.diff(kp_positions[:, 4], axis=0), axis=1).mean()
                if lw_vel >= rw_vel: condition_mask[3] = True
                else: condition_mask[4] = True
            else:
                condition_mask[3] = True  # default left if no FK info

    # ── Foot (slots 1,2): text gate. No-direction → pick higher FK velocity side ──
    if has_kick_text and tags["leg"] > 0.3:
        if 'left' in text_lower or 'l foot' in text_lower or 'l leg' in text_lower:
            condition_mask[1] = True
        elif 'right' in text_lower or 'r foot' in text_lower or 'r leg' in text_lower:
            condition_mask[2] = True
        elif 'both' in text_lower or 'feet' in text_lower:
            condition_mask[1] = True; condition_mask[2] = True
        else:
            # Pick side with higher FK foot velocity
            if kp_positions is not None and kp_positions.shape[0] > 1:
                lf_vel = np.linalg.norm(np.diff(kp_positions[:, 1], axis=0), axis=1).mean()
                rf_vel = np.linalg.norm(np.diff(kp_positions[:, 2], axis=0), axis=1).mean()
                if lf_vel >= rf_vel: condition_mask[1] = True
                else: condition_mask[2] = True
            else:
                condition_mask[1] = True  # default left if no FK info

    # ── Loss mask: condition slots + pelvis aux (FK stability) ──
    loss_mask = condition_mask.copy()
    has_any_condition = condition_mask.any()
    if has_any_condition and not condition_mask[0]:
        aux_mask[0] = True
        loss_mask[0] = True

    return condition_mask, loss_mask, aux_mask


def detect_keyframe(kp_positions, slot_name, contact, fps=50):
    """
    Detect keyframe for EE target (not just last frame).
    For feet: foot contact frame or peak velocity frame.
    For wrists: peak height or peak velocity frame.
    """
    T = kp_positions.shape[0]
    slot_to_kp = {'left_wrist': 3, 'right_wrist': 4, 'left_foot': 1, 'right_foot': 2}
    kp_idx = slot_to_kp.get(slot_name)
    if kp_idx is None:
        return T - 1  # default: last frame

    if 'foot' in slot_name:
        # Foot: use contact frame or peak height frame
        foot_idx = 0 if 'left' in slot_name else 1
        z = kp_positions[:, kp_idx, 2]
        # Find frames where foot is near ground (contact)
        contact_frames = np.where(z < 0.05)[0]
        if len(contact_frames) > 0:
            # Use last contact frame before peak height
            peak_frame = np.argmax(z)
            contact_after_peak = contact_frames[contact_frames > peak_frame * 0.5]
            if len(contact_after_peak) > 0:
                return int(contact_after_peak[-1])
            return int(contact_frames[-1])
        # Fallback: peak velocity frame
        vel = np.linalg.norm(np.diff(kp_positions[:, kp_idx], axis=0, prepend=kp_positions[:1, kp_idx]), axis=1)
        return int(np.argmax(vel))

    elif 'wrist' in slot_name:
        # Wrist: peak height or peak velocity
        z = kp_positions[:, kp_idx, 2]
        vel = np.linalg.norm(np.diff(kp_positions[:, kp_idx], axis=0, prepend=kp_positions[:1, kp_idx]), axis=1)
        # Use frame with both high velocity and height
        score = vel * (z - z.min() + 0.1)
        return int(np.argmax(score))

    return T - 1

# ============================================================
# A类: SE(2) rotation in world space
# ============================================================

def se2_rotate_world(motion_38d, angle_rad):
    """Rotate 38D motion in world space: vel_xy + rot_6d."""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    R2 = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    motion_rot = motion_38d.copy()
    # Rotate root velocity XY (ch 29-30)
    motion_rot[:, 29:31] = motion_rot[:, 29:31] @ R2.T
    # Rotate root orientation 6D (ch 32-37)
    R3 = np.eye(3); R3[:2, :2] = R2
    for t in range(motion_rot.shape[0]):
        r6d = motion_rot[t, 32:38].reshape(2, 3)
        motion_rot[t, 32:38] = (r6d @ R3.T).flatten()
    return motion_rot

def se2_rotate_goal_world(goal_world, angle_rad):
    """Rotate goal_world positions in world space."""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    R2 = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    g_rot = goal_world.copy()
    for slot in range(7):
        if slot == 6:  # heading: add rotation angle
            yaw = math.atan2(g_rot[-1, slot, 1], g_rot[-1, slot, 0]) + angle_rad
            g_rot[-1, slot, 0] = math.cos(yaw)
            g_rot[-1, slot, 1] = math.sin(yaw)
        else:  # position: rotate XY
            g_rot[-1, slot, :2] = R2 @ g_rot[-1, slot, :2]
    return g_rot

def compute_goal_rel(goal_world, root_start_xy, root_start_yaw):
    """Convert world goal to root-relative."""
    cos_a, sin_a = math.cos(-root_start_yaw), math.sin(-root_start_yaw)
    R2_inv = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    g_rel = goal_world.copy()
    for slot in range(7):
        if slot == 6:  # heading
            yaw = math.atan2(goal_world[-1, slot, 1], goal_world[-1, slot, 0]) - root_start_yaw
            g_rel[-1, slot, 0] = math.cos(yaw)
            g_rel[-1, slot, 1] = math.sin(yaw)
        else:  # position
            rel_xy = goal_world[-1, slot, :2] - root_start_xy
            g_rel[-1, slot, :2] = R2_inv @ rel_xy
    return g_rel

# ============================================================
# B类: Trajectory warp for locomotion
# ============================================================
# Micro-IK: Windowed EE IK
# ============================================================

def windowed_ee_ik(motion, goal_world, slot_mask, time_mask, fk_layer, contact,
                   K=30, n_iters=30, lr=0.03):
    """
    Contact-aware q-only windowed IK (PACE-v2).
    - Goal perturbations are applied OUTSIDE this function (no internal perturbation).
    - Optimizes joint angles (ch 0:29) on last K frames.
    - Locks contact-foot world positions to prevent sliding.
    - Joint-limit soft barrier prevents hyperextension.
    - Smooth blend at IK boundary.
    Returns: motion_out (T, 38) numpy array.
    """
    import torch
    T = motion.shape[0]
    K = min(K, T // 3)
    ee_slots = [1, 2, 3, 4]  # lf, rf, lw, rw
    active_ee = [s for s in ee_slots if slot_mask[s]]
    if not active_ee:
        return motion

    mt = torch.from_numpy(motion).unsqueeze(0).float()
    mt_orig = mt.clone()

    # EE joint chains (38D indices)
    ee_chains = {1: [1,4,7,10,14,18], 2: [0,3,6,9,13,17],
                 3: [11,15,19,21], 4: [12,16,20,22]}

    # Contact detection in IK window
    contact_t = torch.from_numpy(contact[T-K:T].copy()).float()  # (K, 2)
    with torch.no_grad():
        fk_init = fk_layer(torch.cat([mt[:, :, :29], mt[:, :, 29:]], dim=-1))
        kp_init = fk_init['keypoint_pos']
    foot_world_init = {
        1: kp_init[0, T-K:, 1].clone(),
        2: kp_init[0, T-K:, 2].clone(),
    }

    # Optimize: joint angles only (ch 0:29) on last K frames
    opt_mask = torch.zeros(1, T, 38, dtype=torch.float32)
    for s in active_ee:
        for j in ee_chains[s]:
            opt_mask[:, -K:, j] = 1.0

    # Joint limits (G1 approximate, radians)
    joint_lo = torch.tensor([
        -0.5,-1.0,-2.0, -0.5,-1.0,-2.0,
        -2.5,-0.5,-0.5, -2.5,-0.5,-0.5,
        -2.0,-1.5,-2.0, -2.0,-1.5,-2.0,
        -2.0,-1.0,-1.5, -2.0,-1.0,-1.5,
        -0.5,-0.5,-0.5, -0.5,-0.5
    ], dtype=torch.float32).view(1, 1, 29)
    joint_hi = torch.tensor([
        0.5,2.0,1.0, 0.5,2.0,1.0,
        2.5,0.5,0.5, 2.5,0.5,0.5,
        2.0,2.0,1.5, 2.0,2.0,1.5,
        2.0,2.0,0.5, 2.0,2.0,0.5,
        0.5,0.5,0.5, 0.5,0.5
    ], dtype=torch.float32).view(1, 1, 29)

    pos = torch.linspace(-3, 3, K)
    sw = torch.sigmoid(pos).view(1, K, 1)

    # Target keyframe: terminal-only baseline
    for _ in range(n_iters):
        mt = mt.detach().clone().requires_grad_(True)
        joints = torch.clamp(mt[:, :, :29], -3.0, 3.0)
        fk_out = fk_layer(torch.cat([joints, mt[:, :, 29:]], dim=-1))
        kp = fk_out['keypoint_pos']

        loss = torch.tensor(0.0)

        # EE goal error (terminal frame, sigmoid-weighted window)
        for s in active_ee:
            kpi = min(s, 4)
            for ti in range(K):
                t = T - K + ti
                loss += sw[0, ti, 0] * ((kp[0, t, kpi] - torch.from_numpy(goal_world[T-1, s, :3]).float())**2).sum()

        # Contact foot-locking
        for fs in [1, 2]:
            foot_in_contact = contact_t[:, fs-1] > 0.5
            if foot_in_contact.any():
                foot_now = kp[0, T-K:, fs]
                foot_lock = foot_world_init[fs]
                contact_penalty = ((foot_now[foot_in_contact] - foot_lock[foot_in_contact])**2).sum()
                loss += 3.0 * contact_penalty

        # Joint limit barrier (soft)
        j_win = mt[:, -K:, :29]
        limit_lo = torch.clamp(joint_lo - j_win, min=0) ** 2
        limit_hi = torch.clamp(j_win - joint_hi, min=0) ** 2
        loss += 0.5 * (limit_lo.sum() + limit_hi.sum())

        # Regularization
        vel = mt[0, -K:, :29] - mt[0, -K-1:-1, :29]
        loss += 2.0 * vel.pow(2).mean() + 0.3 * ((mt[:, -K:, :29] - mt_orig[:, -K:, :29])**2).mean()

        loss.backward()
        with torch.no_grad():
            mt = mt - lr * mt.grad * opt_mask
            mt[:, :, :29] = torch.clamp(mt[:, :, :29], -3.0, 3.0)

    # Smooth blend at boundary
    motion_out = mt[0].detach().cpu().numpy()
    blend_start = T - K; blend_frames = min(15, K // 2)
    for i in range(blend_frames):
        a = (i + 1) / (blend_frames + 1)
        motion_out[blend_start + i, :29] = (1 - a) * motion[blend_start + i, :29] + a * motion_out[blend_start + i, :29]

    return motion_out


def trajectory_warp(motion_38d, goal_world_new, goal_world_old, contact, fk_layer=None, fps=50):
    """
    Contact-aware trajectory warp (PACE-v2).
    Warps root trajectory, then checks contact-foot velocity.
    Returns (motion_out, ok) — ok=False means warp caused excessive foot sliding.
    """
    T = motion_38d.shape[0]
    root_xy, root_z, yaw = compute_root_state(motion_38d, fps)

    delta = goal_world_new[-1, 5, :2] - goal_world_old[-1, 5, :2]
    if np.linalg.norm(delta) < 0.01:
        return motion_38d, True

    # Clamp delta: max 50% of original displacement
    orig_disp = np.linalg.norm(root_xy[-1] - root_xy[0])
    max_delta = max(orig_disp * 0.5, 0.05)
    delta_norm = np.linalg.norm(delta)
    if delta_norm > max_delta:
        delta = delta / delta_norm * max_delta

    # Smoothstep weight
    t = np.arange(T) / max(T - 1, 1)
    alpha = t ** 2 * (3 - 2 * t)
    root_xy_new = root_xy + alpha[:, None] * delta

    # Centered difference for smooth velocity
    motion_out = motion_38d.copy()
    dt = 1.0 / fps
    vel_xy_new = np.zeros_like(root_xy_new)
    vel_xy_new[1:-1] = (root_xy_new[2:] - root_xy_new[:-2]) / (2 * dt)
    vel_xy_new[0] = (root_xy_new[1] - root_xy_new[0]) / dt
    vel_xy_new[-1] = (root_xy_new[-1] - root_xy_new[-2]) / dt
    vel_norm = np.linalg.norm(vel_xy_new, axis=1)
    if vel_norm.max() > 2.5:
        vel_xy_new = vel_xy_new * 2.5 / vel_norm.max()
    motion_out[:, 29:31] = vel_xy_new.astype(np.float32)

    # ── Contact consistency check (per continuous segment) ──
    if fk_layer is not None:
        import torch
        with torch.no_grad():
            mt = torch.from_numpy(motion_out).unsqueeze(0).float()
            fk_out_w = fk_layer(mt)
            kp_w = fk_out_w['keypoint_pos'][0].cpu().numpy()  # (T, 5, 3)
        for fs, fi in [('left_foot', 1), ('right_foot', 2)]:
            foot_contact = contact[:, fi - 1]  # (T,)
            if not foot_contact.any():
                continue
            contact_frames = np.where(foot_contact)[0]
            # Split into continuous contact segments (gap > 1 frame = new segment)
            segments = np.split(contact_frames, np.where(np.diff(contact_frames) > 1)[0] + 1)
            for seg in segments:
                if len(seg) < 2:
                    continue
                foot_vel = np.linalg.norm(
                    np.diff(kp_w[seg, fi], axis=0), axis=1
                ) * fps
                if foot_vel.max() > 0.15:  # 15 cm/s threshold
                    return motion_out, False

    return motion_out, True

# ============================================================
# Operator Scheduler (PACE-v2)
# ============================================================

def _score_stop_state(root_xy, root_z, fps=50):
    """How stable the terminal state is (low speed = high score)."""
    vel_xy = np.linalg.norm(np.diff(root_xy[-10:], axis=0, prepend=root_xy[-11:-10]), axis=1)
    vel_z = np.abs(np.diff(root_z[-10:], prepend=root_z[-11:-10]))
    speed_score = 1.0 - min(1.0, vel_xy.mean() / 0.3)
    z_stable = 1.0 - min(1.0, vel_z.mean() / 0.05)
    return 0.5 * speed_score + 0.5 * z_stable


def _score_displacement(root_xy):
    """Normalized displacement (0~1), saturates at 2m."""
    disp = np.linalg.norm(root_xy[-1] - root_xy[0])
    return min(1.0, disp / 2.0)


def _score_contact_stability(contact):
    """Fraction of frames with at least one foot in contact."""
    any_contact = contact.any(axis=1)
    return float(any_contact.mean())


def _score_root_smoothness(motion_38d):
    """Inverse of max velocity jerk (smooth = high score)."""
    vel = np.linalg.norm(motion_38d[:, 29:31], axis=1)
    if len(vel) < 3: return 0.5
    acc = np.diff(vel[:-1])
    jerk = np.diff(acc) if len(acc) > 1 else np.array([0.0])
    return 1.0 - min(1.0, np.abs(jerk).mean() / 0.5)


def _score_heading_stability(yaw):
    """How stable heading is near the end."""
    if len(yaw) < 10: return 0.5
    yaw_diff = np.abs(np.diff(yaw[-10:]))
    yaw_diff = np.minimum(yaw_diff, 2*np.pi - yaw_diff)
    return 1.0 - min(1.0, yaw_diff.mean() / 0.3)


def _score_ee_activity(kp_positions, slot):
    """Normalized mean EE velocity for a given slot."""
    if kp_positions is None or kp_positions.shape[0] < 2:
        return 0.0
    kpi = min(slot, 4)
    vel = np.linalg.norm(np.diff(kp_positions[:, kpi], axis=0), axis=1)
    return min(1.0, vel.mean() / 0.5)


def _score_terminal_goalness(motion_38d, kp_positions, active_slots, fps=50):
    """How 'arrival-oriented' the last frames are (low joint vel + stable root)."""
    if len(motion_38d) < 10: return 0.5
    j_vel = np.abs(np.diff(motion_38d[-10:, :29], axis=0)).mean()
    r_vel = np.linalg.norm(motion_38d[-10:, 29:31], axis=1).mean()
    j_score = 1.0 - min(1.0, j_vel / 0.3)
    r_score = 1.0 - min(1.0, r_vel / 0.2)
    return 0.5 * j_score + 0.5 * r_score


def select_augmentation_ops(text, tags, motion_38d, root_xy, root_z, yaw, kp, contact,
                            n_aug=5, n_pert=2, phase='full'):
    """Adaptive operator scheduler per clip (PACE-v2).
    Returns (ops, diagnostics). ops values are {op_name: (enabled, n_variants)}.
    """
    text_lower = text.lower()
    posture_kw = ['crouch','bow','squat','sit','bend','kneel']
    cyclic_kw = ['wave', 'clap', 'dance', 'swing']
    has_posture_text = any(kw in text_lower for kw in posture_kw)
    has_cyclic_text = any(kw in text_lower for kw in cyclic_kw)
    explicit_metric = text_has_explicit_metric(text)

    motion_stats = {
        'stop_state_score': _score_stop_state(root_xy, root_z),
        'displacement_score': _score_displacement(root_xy),
        'contact_stability_score': _score_contact_stability(contact),
        'root_smoothness_score': _score_root_smoothness(motion_38d),
        'heading_stability_score': _score_heading_stability(yaw),
        'ee_activity_score': max(_score_ee_activity(kp, 1), _score_ee_activity(kp, 2),
                                  _score_ee_activity(kp, 3), _score_ee_activity(kp, 4)),
        'terminal_goalness_score': _score_terminal_goalness(motion_38d, kp, None),
        'pelvis_height_change_score': float(min(1.0, abs(root_z[-1] - root_z[0]) / 0.3)) if len(root_z) > 0 else 0.0,
        'posture_text_conf': 1.0 if has_posture_text else 0.0,
        'explicit_metric': 1.0 if explicit_metric else 0.0,
        'cyclic_text_conf': 1.0 if has_cyclic_text else 0.0,
    }

    s_slice = (0.45 * tags['locomotion'] + 0.25 * motion_stats['stop_state_score'] +
               0.20 * motion_stats['displacement_score'] + 0.10 * motion_stats['contact_stability_score'])
    s_traj_warp = (0.50 * tags['locomotion'] + 0.25 * motion_stats['contact_stability_score'] +
                   0.15 * motion_stats['root_smoothness_score'] + 0.10 * motion_stats['heading_stability_score'])
    s_ee_ik = (0.40 * max(tags['arm'], tags['leg']) + 0.25 * motion_stats['ee_activity_score'] +
               0.20 * motion_stats['terminal_goalness_score'] + 0.15 * motion_stats['contact_stability_score'])
    s_posture = (0.50 * motion_stats['posture_text_conf'] + 0.30 * motion_stats['pelvis_height_change_score'] +
                 0.20 * motion_stats['stop_state_score'])

    structural_allowed = phase == 'full' or tags['locomotion'] > 0.3
    ops = {
        'raw': (True, 1),
        'se2': (n_aug > 0, int(max(0, n_aug))),
        'slice': (structural_allowed and not explicit_metric and s_slice > 0.35 and tags['locomotion'] > 0.3, int(max(0, n_pert))),
        'traj_warp': (structural_allowed and s_traj_warp > 0.4 and tags['locomotion'] > 0.3, int(max(0, n_pert))),
        'ee_ik': (phase == 'full' and not has_cyclic_text and s_ee_ik > 0.4 and (tags['arm'] > 0.3 or tags['leg'] > 0.3), int(max(0, n_pert))),
        'posture_recast': (phase == 'full' and s_posture > 0.35 and has_posture_text, int(max(0, n_pert))),
    }
    is_dynamic_cyclic = (tags['static'] > 0.7 or has_cyclic_text or (tags['arm'] > 0.5 and tags['locomotion'] < 0.15))
    if is_dynamic_cyclic:
        ops['slice'] = (False, 0)
        ops['traj_warp'] = (False, 0)
        if has_cyclic_text:
            ops['ee_ik'] = (False, 0)

    scores = {
        'slice': float(s_slice),
        'traj_warp': float(s_traj_warp),
        'ee_ik': float(s_ee_ik),
        'posture_recast': float(s_posture),
    }
    diagnostics = {
        'scores': scores,
        'motion_stats': motion_stats,
        'tags': tags,
        'phase': phase,
        'explicit_metric_text': explicit_metric,
        'cyclic_text': has_cyclic_text,
        'ops': {k: {'enabled': bool(v[0]), 'n_variants': int(v[1])} for k, v in ops.items()},
    }
    return ops, diagnostics


def select_arrival_slice_frames(motion_38d, root_xy, root_z, yaw, contact, kp,
                                 candidate_ratios=(0.4, 0.55, 0.7, 0.85),
                                 min_len=40, min_disp=0.25, fps=50):
    """Find valid intermediate stop frames for temporal arrival slicing."""
    T = motion_38d.shape[0]
    vel_xy = np.linalg.norm(motion_38d[:, 29:31], axis=1)
    candidates = []
    for ratio in candidate_ratios:
        t = int(T * ratio)
        if t < min_len: continue
        disp = np.linalg.norm(root_xy[t] - root_xy[0])
        if disp < min_disp: continue
        rz = root_z[t]
        if rz < 0.2 or rz > 1.5: continue
        if not contact[t].any(): continue
        # Local speed low (not during acceleration peak)
        local_speed = vel_xy[max(0,t-5):min(T,t+5)].mean()
        if local_speed > 1.0: continue
        # Heading continuity
        if t > 0:
            yaw_diff = abs(yaw[t] - yaw[t-1])
            yaw_diff = min(yaw_diff, 2*np.pi - yaw_diff)
            if yaw_diff > 0.3: continue
        candidates.append(t)
    return candidates


def select_posture_arrival_frames(motion_38d, root_z, contact,
                                   candidate_ratios=(0.45, 0.6, 0.75, 0.9),
                                   min_len=40, fps=50):
    """Find valid posture-arrival frames (pelvis height changed, root stable)."""
    T = motion_38d.shape[0]
    vel_xy = np.linalg.norm(motion_38d[:, 29:31], axis=1)
    candidates = []
    for ratio in candidate_ratios:
        t = int(T * ratio)
        if t < min_len: continue
        rz = root_z[t]
        if rz < 0.2 or rz > 1.5: continue
        # Pelvis height must have changed notably
        dz = abs(rz - root_z[0])
        if dz < 0.05: continue
        # Root should be near-stationary
        local_speed = vel_xy[max(0,t-5):min(T,t+5)].mean()
        if local_speed > 0.3: continue
        if not contact[t].any(): continue
        # Joint velocity should be low
        j_vel = np.abs(np.diff(motion_38d[max(0,t-5):min(T,t+5), :29], axis=0)).mean()
        if j_vel > 0.2: continue
        candidates.append(t)
    return candidates


def normalize_text_for_slice(text):
    """Remove explicit distance/duration/angle numbers from text for sliced samples."""
    import re
    text = re.sub(r'\b\d+\.?\d*\s*(m|meters?|metres?|feet|ft|seconds?|sec|s|degrees?|deg)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|half)\s+(meters?|metres?|feet|seconds?|degrees?)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip(' ,.;')
    if len(text) < 10:
        text = 'a person moves and stops at the target'
    return text


def save_sample_impl(motion, text, condition_mask, loss_mask, aux_mask,
                     goal_dim_mask_base, time_mask, tolerance, confidence,
                     tags, contact, clip_id, aug_type, teacher_type, applied_ops,
                     source_t_start, source_t_end, out_dir, fk_layer,
                     requested_goal_world=None, scheduler_diag=None, operator_meta=None,
                     source_path='', source_text_index=0, phase='full', seed=0,
                     strict_validation=True):
    """Build, validate, and save one versioned PACE-v2 sample."""
    result = {'saved': 0, 'sample_id': f'{clip_id}_{aug_type}', 'errors': [], 'validation_errors': []}
    motion = motion.astype(np.float32, copy=False)
    T = motion.shape[0]
    contact = contact[:T].astype(bool)
    condition_mask = condition_mask.astype(bool)
    loss_mask = loss_mask.astype(bool)
    aux_mask = aux_mask.astype(bool)
    time_mask = time_mask[:T].astype(bool)
    tolerance = tolerance.astype(np.float32)
    confidence = confidence.astype(np.float32)

    root_xy, root_z, yaw = compute_root_state(motion)
    with torch.no_grad():
        mt = torch.from_numpy(motion).unsqueeze(0).float()
        fk_out = fk_layer(mt)
        kp = fk_out['keypoint_pos'][0].cpu().numpy()

    achieved_world, dim_mask = compute_goal_world(kp, root_xy, root_z, yaw, T)
    achieved_world, dim_mask = refresh_goal_from_motion(motion, achieved_world, dim_mask, loss_mask, time_mask, fk_layer)
    if requested_goal_world is None:
        requested_world = achieved_world.copy()
    else:
        requested_world = np.zeros_like(achieved_world)
        n_req = min(T, requested_goal_world.shape[0])
        requested_world[:n_req] = requested_goal_world[:n_req].astype(np.float32)
        requested_world = fill_requested_goal_from_achieved(requested_world, achieved_world, dim_mask, time_mask, condition_mask)

    requested_world, achieved_world, dim_mask = mask_goal_arrays(
        requested_world, achieved_world, dim_mask, time_mask, condition_mask, loss_mask)
    req_ach_residual, req_ach_ok, req_ach_errors = requested_achieved_residual(
        requested_world, achieved_world, dim_mask, time_mask, condition_mask, tolerance)
    requested_rel = compute_goal_rel_all(requested_world, root_xy[0], yaw[0])
    achieved_rel = compute_goal_rel_all(achieved_world, root_xy[0], yaw[0])
    requested_rel[~dim_mask] = 0.0
    achieved_rel[~dim_mask] = 0.0

    cfm_w, fk_w, stop_w = TEACHER_WEIGHTS.get(teacher_type, (1.0, 1.0, 1.0))
    loss_slot_weight = BASE_SLOT_WEIGHTS.copy()
    loss_slot_weight[~loss_mask] = 0.0
    if aux_mask.any():
        loss_slot_weight[aux_mask] *= 0.5

    sample_id = f'{clip_id}_{aug_type}'
    out_data = {
        'schema_version': np.array(PACE_SCHEMA_VERSION),
        'schema_hash': np.array(schema_hash()),
        'sample_id': np.array(sample_id),
        'motion_38d': motion.astype(np.float32),
        'goal_world': achieved_world.astype(np.float32),
        'goal_world_requested': requested_world.astype(np.float32),
        'goal_world_achieved': achieved_world.astype(np.float32),
        'goal_rel': requested_rel.astype(np.float32),
        'goal_value': requested_rel.astype(np.float32),
        'goal_fk_target': achieved_world.astype(np.float32),
        'goal_achieved_rel': achieved_rel.astype(np.float32),
        'goal_dim_mask': dim_mask.astype(bool),
        'goal_time_mask': time_mask.astype(bool),
        'goal_slot_mask': condition_mask.astype(bool),
        'loss_slot_mask': loss_mask.astype(bool),
        'aux_slot_mask': aux_mask.astype(bool),
        'goal_type': np.array('soft' if any(op in applied_ops for op in ['slice', 'posture_recast']) else 'hard'),
        'goal_tolerance': tolerance,
        'goal_confidence': confidence,
        'requested_achieved_residual': req_ach_residual.astype(np.float32),
        'requested_achieved_ok': req_ach_ok.astype(bool),
        'loss_slot_weight': loss_slot_weight.astype(np.float32),
        'cfm_weight': np.float32(cfm_w),
        'fk_weight': np.float32(fk_w),
        'stop_weight': np.float32(stop_w),
        'text': np.array(text),
        'action_type': np.array(slot_action_type(tags)),
        'semantic_tags': np.array(json.dumps(to_jsonable(tags), sort_keys=True)),
        'aug_type': np.array(aug_type),
        'teacher_type': np.array(teacher_type),
        'applied_ops': np.array(','.join(applied_ops)),
        'parent_clip_id': np.array(clip_id),
        'source_path': np.array(source_path),
        'source_text_index': np.int32(source_text_index),
        'source_t_start': np.int32(source_t_start),
        'source_t_end': np.int32(source_t_end),
        'phase': np.array(phase),
        'seed': np.uint32(seed % (2**32 - 1)),
        'scheduler_scores': np.array(json.dumps(to_jsonable((scheduler_diag or {}).get('scores', {})), sort_keys=True)),
        'scheduler_decision': np.array(json.dumps(to_jsonable(scheduler_diag or {}), sort_keys=True)),
        'operator_meta': np.array(json.dumps(to_jsonable(operator_meta or {}), sort_keys=True)),
        'contact': contact,
        'valid_len': np.int32(T),
    }

    ok_schema, schema_errors = validate_schema(out_data)
    ok_phys, phys_errors = validate_sample(
        motion, out_data['goal_fk_target'], loss_mask, time_mask, fk_layer,
        tolerance=tolerance, contact=contact)
    validation_errors = schema_errors + phys_errors + req_ach_errors
    out_data['validation_status'] = np.array('ok' if not validation_errors else 'failed')
    out_data['validation_errors'] = np.array(json.dumps(validation_errors))
    result['validation_errors'] = validation_errors
    if strict_validation and validation_errors:
        return result

    fname = f'{sample_id}.npz'
    np.savez_compressed(os.path.join(out_dir, fname), **out_data)
    result['saved'] = 1
    result['path'] = os.path.join(out_dir, fname)
    return result

# ============================================================
# Validation
# ============================================================

def validate_schema(sample):
    errors = []
    for key in REQUIRED_SAMPLE_KEYS:
        if key not in sample:
            errors.append(f'missing:{key}')
    if errors:
        return False, errors
    motion = sample['motion_38d']
    T = motion.shape[0]
    expected = {
        'motion_38d': (T, 38),
        'goal_world': (T, 7, GOAL_DIMS),
        'goal_world_requested': (T, 7, GOAL_DIMS),
        'goal_world_achieved': (T, 7, GOAL_DIMS),
        'goal_rel': (T, 7, GOAL_DIMS),
        'goal_value': (T, 7, GOAL_DIMS),
        'goal_fk_target': (T, 7, GOAL_DIMS),
        'requested_achieved_residual': (7,),
        'requested_achieved_ok': (7,),
        'goal_dim_mask': (T, 7, GOAL_DIMS),
        'goal_time_mask': (T, 7),
        'contact': (T, 2),
    }
    for key, shape in expected.items():
        if key in sample and tuple(sample[key].shape) != shape:
            errors.append(f'shape:{key}:{sample[key].shape}!={shape}')
    for key in ['motion_38d', 'goal_world', 'goal_world_requested', 'goal_world_achieved', 'goal_rel', 'goal_value', 'goal_fk_target']:
        if key in sample and not np.isfinite(sample[key]).all():
            errors.append(f'nonfinite:{key}')
    for key in ['goal_slot_mask', 'loss_slot_mask', 'aux_slot_mask', 'goal_confidence', 'goal_tolerance', 'loss_slot_weight']:
        if key in sample and tuple(sample[key].shape) != (7,):
            errors.append(f'shape:{key}:{sample[key].shape}!=(7,)')
    if 'goal_world' in sample and 'goal_world_achieved' in sample:
        if not np.allclose(sample['goal_world'], sample['goal_world_achieved']):
            errors.append('alias:goal_world')
    if 'goal_fk_target' in sample and 'goal_world_achieved' in sample:
        if not np.allclose(sample['goal_fk_target'], sample['goal_world_achieved']):
            errors.append('alias:goal_fk_target')
    if 'goal_dim_mask' in sample:
        dm = sample['goal_dim_mask']
        if np.any(np.abs(sample['goal_value'][~dm]) > 1e-6):
            errors.append('inactive_values:goal_value')
        if np.any(np.abs(sample['goal_fk_target'][~dm]) > 1e-6):
            errors.append('inactive_values:goal_fk_target')
    return len(errors) == 0, errors


def validate_sample(motion_38d, goal_world, slot_mask, time_mask, fk_layer, fps=50,
                    tolerance=None, contact=None):
    """Hard-filter: check physics validity against achieved world targets."""
    errors = []
    if motion_38d.shape[0] < 2 or motion_38d.shape[1] != 38:
        return False, ['bad_motion_shape']
    if not np.isfinite(motion_38d).all():
        return False, ['nonfinite_motion']

    joints = motion_38d[:, :29]
    if np.abs(joints).max() > 3.2:
        errors.append('joint_abs_range')
    j_vel = np.abs(np.diff(joints, axis=0)) * fps
    if j_vel.size and np.percentile(j_vel, 99) > 12.0:
        errors.append('joint_velocity')

    root_z = motion_38d[:, 31]
    if root_z.min() < 0.2 or root_z.max() > 1.5:
        errors.append('root_height')
    vel_xy = np.linalg.norm(motion_38d[:, 29:31], axis=1)
    if vel_xy.max() > 3.0:
        errors.append('max_velocity')

    with torch.no_grad():
        mt = torch.from_numpy(motion_38d).unsqueeze(0).float()
        fk_out = fk_layer(mt)
        kp = fk_out['keypoint_pos'][0].cpu().numpy()
    lf_z = kp[:, 1, 2]
    rf_z = kp[:, 2, 2]
    if lf_z.min() > 0.15 and rf_z.min() > 0.15:
        errors.append('foot_float')

    if contact is not None:
        for foot_idx, contact_idx in [(1, 0), (2, 1)]:
            frames = np.where(contact[:, contact_idx])[0]
            if len(frames) > 1:
                segments = np.split(frames, np.where(np.diff(frames) > 1)[0] + 1)
                for seg in segments:
                    if len(seg) < 2:
                        continue
                    slip = np.linalg.norm(np.diff(kp[seg, foot_idx], axis=0), axis=1).max() * fps
                    if slip > 0.20:
                        errors.append(f'contact_slip_s{foot_idx}')
                        break

    root_xy_val, _, yaw_val = compute_root_state(motion_38d, fps)
    tolerance = DEFAULT_TOLERANCE if tolerance is None else tolerance
    for t in range(motion_38d.shape[0]):
        if not time_mask[t].any():
            continue
        for s in range(7):
            if slot_mask[s] and time_mask[t, s]:
                g = goal_world[t, s]
                if s < 5:
                    err = np.linalg.norm(kp[t, s] - g[:3])
                elif s == 5:
                    err = np.linalg.norm(root_xy_val[t] - g[:2])
                else:
                    yaw_target = math.atan2(g[1], g[0])
                    err = abs(yaw_val[t] - yaw_target)
                    err = min(err, 2*math.pi - err)
                if err > max(float(tolerance[s]) * 2.5, 0.05):
                    errors.append(f'goal_error_s{s}:{err:.4f}')
    return len(errors) == 0, errors

def refresh_goal_from_motion(motion, goal_world, goal_dim_mask, loss_mask, time_mask, fk_layer):
    """After warp/IK, refresh ALL loss-mask goal slots from actual FK.
    Ensures goal_world = FK(motion) at all time_mask frames — no stale goals."""
    import torch
    root_xy, root_z, yaw = compute_root_state(motion)
    with torch.no_grad():
        mt = torch.from_numpy(motion).unsqueeze(0).float()
        fk_out = fk_layer(mt)
        kp = fk_out['keypoint_pos'][0].cpu().numpy()

    g_new = goal_world.copy()
    dm_new = goal_dim_mask.copy()

    for t in range(motion.shape[0]):
        for s in range(7):
            if loss_mask[s] and time_mask[t, s]:
                if s < 5:
                    g_new[t, s, :3] = kp[t, min(s, 4)]
                    dm_new[t, s, :3] = True
                elif s == 5:
                    g_new[t, 5, :2] = root_xy[t]
                    dm_new[t, 5, :2] = True
                elif s == 6:
                    g_new[t, 6, 0] = math.cos(yaw[t])
                    g_new[t, 6, 1] = math.sin(yaw[t])
                    dm_new[t, 6, :2] = True
    return g_new, dm_new


# ============================================================
# Main: process single clip
# ============================================================

def record_save(result, save_result, aug_type, teacher_type):
    if save_result.get('saved', 0):
        result['saved'] += 1
        result['sample_ids'].append(save_result['sample_id'])
        result['operator_counts'][aug_type] += 1
        result['teacher_counts'][teacher_type] += 1
    else:
        result['validation_failed'] += 1
        for err in save_result.get('validation_errors', []):
            result['validation_reasons'][err.split(':')[0]] += 1


def se2_angles(n_aug):
    if n_aug <= 0:
        return []
    return [2 * math.pi * i / n_aug for i in range(n_aug)]


def process_clip(args_tuple):
    (clip_id, npz_path, text, source_text_index, out_dir, n_aug, n_pert,
     phase, base_seed, strict_validation) = args_tuple
    fk_layer = get_fk_layer()
    result = {
        'clip_id': clip_id,
        'saved': 0,
        'sample_ids': [],
        'operator_counts': Counter(),
        'teacher_counts': Counter(),
        'validation_failed': 0,
        'validation_reasons': Counter(),
        'errors': [],
        'skipped': [],
        'scheduler': {},
    }

    try:
        npz_data = np.load(npz_path, allow_pickle=True)
        out = process_robot_npz(npz_data, root_idx=0, return_body_pos=False, root_vel_frame='world')
        motion_raw = out[0] if isinstance(out, tuple) else out
        motion_raw = motion_raw.astype(np.float32)
        T = motion_raw.shape[0]
        if T < 40:
            result['skipped'].append('too_short')
            return result

        root_xy, root_z, yaw = compute_root_state(motion_raw)
        contact, kp = detect_contact(motion_raw, fk_layer)
        tags = classify_semantic_tags(text, motion_raw, root_xy, kp)
        condition_mask, loss_mask, aux_mask = determine_active_slots(tags, text, kp_positions=kp)
        time_mask = build_time_mask(T, loss_mask)
        tolerance = DEFAULT_TOLERANCE.copy()
        confidence = np.zeros(7, dtype=np.float32)
        confidence[condition_mask] = 1.0
        confidence[aux_mask] = 0.3

        ops, scheduler_diag = select_augmentation_ops(
            text, tags, motion_raw, root_xy, root_z, yaw, kp, contact,
            n_aug=n_aug, n_pert=n_pert, phase=phase)
        result['scheduler'] = scheduler_diag
        angles = se2_angles(ops['se2'][1]) if ops['se2'][0] else []

        def save_variant(motion, variant_text, cond, loss, aux, tm, ct, aug_type,
                         teacher_type, applied_ops, t_start, t_end,
                         requested_goal=None, operator_meta=None):
            seed = stable_seed(base_seed, clip_id, aug_type)
            sr = save_sample_impl(
                motion, variant_text, cond, loss, aux, None, tm, tolerance,
                confidence.copy(), tags, ct, clip_id, aug_type, teacher_type,
                applied_ops, t_start, t_end, out_dir, fk_layer,
                requested_goal_world=requested_goal,
                scheduler_diag=scheduler_diag,
                operator_meta=operator_meta,
                source_path=npz_path,
                source_text_index=source_text_index,
                phase=phase,
                seed=seed,
                strict_validation=strict_validation)
            record_save(result, sr, aug_type.split('_')[0], teacher_type)

        save_variant(motion_raw, text, condition_mask, loss_mask, aux_mask, time_mask, contact,
                     'raw', 'raw_prior', ['raw'], 0, T)

        for i, angle in enumerate(angles):
            if abs(angle) < 1e-8:
                continue
            save_variant(se2_rotate_world(motion_raw, angle), text, condition_mask, loss_mask, aux_mask,
                         time_mask, contact, f'se2_{i:02d}', 'se2_arrival', ['se2'], 0, T,
                         operator_meta={'angle_rad': angle})

        if ops['slice'][0]:
            slice_frames = select_arrival_slice_frames(motion_raw, root_xy, root_z, yaw, contact, kp)
            base_text = normalize_text_for_slice(text)
            for ti, t_end in enumerate(slice_frames[:ops['slice'][1]]):
                m_sl = motion_raw[:t_end].copy()
                tm_sl = build_time_mask(t_end, loss_mask)
                ct_sl = contact[:t_end]
                save_variant(m_sl, base_text, condition_mask, loss_mask, aux_mask, tm_sl, ct_sl,
                             f'slice_{ti:02d}', 'arrival_slice', ['slice'], 0, t_end,
                             operator_meta={'t_end': t_end, 'normalized_text': base_text})
                for j, angle in enumerate(angles):
                    if abs(angle) < 1e-8:
                        continue
                    save_variant(se2_rotate_world(m_sl, angle), base_text, condition_mask, loss_mask, aux_mask,
                                 tm_sl, ct_sl, f'slice_{ti:02d}_se2_{j:02d}', 'se2_arrival',
                                 ['slice','se2'], 0, t_end, operator_meta={'t_end': t_end, 'angle_rad': angle})

        if ops['traj_warp'][0] and loss_mask[5]:
            g_ref, _ = compute_goal_world(kp, root_xy, root_z, yaw, T)
            orig_disp = max(float(np.linalg.norm(root_xy[-1] - root_xy[0])), 1e-6)
            for si in range(ops['traj_warp'][1]):
                rng = np.random.default_rng(stable_seed(base_seed, clip_id, 'warp', si))
                direction = rng.normal(size=2)
                norm = np.linalg.norm(direction)
                if norm < 1e-8:
                    continue
                mag = min(float(rng.uniform(0.10, 0.30)), 0.5 * orig_disp)
                delta = direction / norm * mag
                g_pert = g_ref.copy()
                g_pert[-1, 5, :2] += delta
                m_w, ok_w = trajectory_warp(motion_raw.copy(), g_pert, g_ref, contact, fk_layer)
                if not ok_w:
                    result['validation_failed'] += 1
                    result['validation_reasons']['traj_warp_contact_slip'] += 1
                    continue
                save_variant(m_w, text, condition_mask, loss_mask, aux_mask, time_mask, contact,
                             f'warp_{si:02d}', 'traj_warp', ['traj_warp'], 0, T,
                             requested_goal=g_pert, operator_meta={'delta_xy': delta.tolist()})
                for j, angle in enumerate(angles):
                    if abs(angle) < 1e-8:
                        continue
                    g_w_rot = se2_rotate_goal_world(g_pert, angle)
                    save_variant(se2_rotate_world(m_w, angle), text, condition_mask, loss_mask, aux_mask,
                                 time_mask, contact, f'warp_{si:02d}_se2_{j:02d}', 'se2_arrival',
                                 ['traj_warp','se2'], 0, T, requested_goal=g_w_rot,
                                 operator_meta={'delta_xy': delta.tolist(), 'angle_rad': angle})

        ee_slots = [si for si in [1,2,3,4] if loss_mask[si]]
        support_feet = {1 for v in [contact[-10:, 0].mean()] if v > 0.5} | {2 for v in [contact[-10:, 1].mean()] if v > 0.5}
        ee_slots = [s for s in ee_slots if s not in support_feet]
        if ops['ee_ik'][0] and ee_slots:
            g_base, _ = compute_goal_world(kp, root_xy, root_z, yaw, T)
            for si in range(ops['ee_ik'][1]):
                rng = np.random.default_rng(stable_seed(base_seed, clip_id, 'eeik', si))
                g_ik = g_base.copy()
                perturb = {}
                for es in ee_slots:
                    mag = np.array([0.08, 0.08, 0.06]) if es in (1, 2) else np.array([0.15, 0.15, 0.10])
                    delta = (rng.random(3) - 0.5) * 2.0 * mag
                    g_ik[T-1, es, :3] += delta
                    perturb[SLOT_NAMES[es]] = delta.tolist()
                m_ik = windowed_ee_ik(motion_raw.copy(), g_ik, loss_mask, time_mask, fk_layer, contact)
                save_variant(m_ik, text, condition_mask, loss_mask, aux_mask, time_mask, contact,
                             f'eeik_{si:02d}', 'ee_ik', ['ee_ik'], 0, T,
                             requested_goal=g_ik, operator_meta={'perturb_xyz': perturb})
                for j, angle in enumerate(angles):
                    if abs(angle) < 1e-8:
                        continue
                    g_ik_rot = se2_rotate_goal_world(g_ik, angle)
                    save_variant(se2_rotate_world(m_ik, angle), text, condition_mask, loss_mask, aux_mask,
                                 time_mask, contact, f'eeik_{si:02d}_se2_{j:02d}', 'se2_arrival',
                                 ['ee_ik','se2'], 0, T, requested_goal=g_ik_rot,
                                 operator_meta={'perturb_xyz': perturb, 'angle_rad': angle})

        if ops['posture_recast'][0]:
            post_frames = select_posture_arrival_frames(motion_raw, root_z, contact)
            p_cond = condition_mask.copy(); p_cond[0] = True
            p_loss = loss_mask.copy(); p_loss[0] = True
            p_aux = aux_mask.copy(); p_aux[0] = False
            p_conf = confidence.copy(); p_conf[0] = 1.0
            old_conf = confidence
            confidence = p_conf
            for ti, t_end in enumerate(post_frames[:ops['posture_recast'][1]]):
                m_pt = motion_raw[:t_end].copy()
                tm_pt = build_time_mask(t_end, p_loss)
                save_variant(m_pt, text, p_cond, p_loss, p_aux, tm_pt, contact[:t_end],
                             f'post_{ti:02d}', 'posture_recast', ['posture_recast'], 0, t_end,
                             operator_meta={'t_end': t_end})
            confidence = old_conf

        result['operator_counts'] = dict(result['operator_counts'])
        result['teacher_counts'] = dict(result['teacher_counts'])
        result['validation_reasons'] = dict(result['validation_reasons'])
        return result
    except Exception:
        result['errors'].append(traceback.format_exc())
        result['operator_counts'] = dict(result['operator_counts'])
        result['teacher_counts'] = dict(result['teacher_counts'])
        result['validation_reasons'] = dict(result['validation_reasons'])
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/root/data2/fyc/fyc/GRAT-main/robot/robot_humanml_data_v2')
    parser.add_argument('--out_dir', default='/root/data2/fyc/fyc/GRAT-main/robot/new/pace_data_v2')
    parser.add_argument('--n_aug', type=int, default=5, help='SE(2) variants per clip; angle 0 is skipped because raw already covers it')
    parser.add_argument('--n_pert', type=int, default=3, help='Structural perturbation variants per compatible operator')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--max_clips', type=int, default=0, help='Max clips to process (0=all)')
    parser.add_argument('--phase', default='full', choices=['locomotion','full'])
    parser.add_argument('--split_file', type=str, default='train_balanced.txt')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--allow_invalid', action='store_true', help='Debug only: write samples even if validation fails')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)

    mean = np.load(resolve_stats_path(args.data_root, 38, 'Mean', 'world')).flatten()[:38]
    std = np.load(resolve_stats_path(args.data_root, 38, 'Std', 'world')).flatten()[:38]

    npz_dir = os.path.join(args.data_root, 'npz')
    text_dir = os.path.join(args.data_root, 'texts')
    with open(os.path.join(args.data_root, args.split_file)) as f:
        train_ids = [l.strip() for l in f if l.strip()]
    if args.max_clips > 0:
        train_ids = train_ids[:args.max_clips]

    print(f'Processing {len(train_ids)} clips...')
    print(f'  Schema: {PACE_SCHEMA_VERSION} ({schema_hash()})')
    print(f'  Operators: raw + adaptive structural + optional SE2')
    print(f'  Workers: {args.n_workers}')
    print(f'  Seed: {args.seed}')
    print(f'  Output: {args.out_dir}')

    tasks = []
    missing_npz = []
    for tid in train_ids:
        npz_path = os.path.join(npz_dir, tid + '.npz')
        if not os.path.exists(npz_path):
            missing_npz.append(tid)
            continue
        txt_path = os.path.join(text_dir, tid + '.txt')
        text = 'a person performs an action'
        source_text_index = 0
        if os.path.exists(txt_path):
            import codecs as cs
            with cs.open(txt_path) as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split('#')
                    if parts and parts[0].strip():
                        text = parts[0].strip()
                        source_text_index = idx
                        break
        tasks.append((tid, npz_path, text, source_text_index, args.out_dir, args.n_aug,
                      args.n_pert, args.phase, args.seed, not args.allow_invalid))

    results = []
    if args.n_workers > 1:
        with Pool(args.n_workers, initializer=init_worker, initargs=(args.seed,)) as pool:
            for r in tqdm(pool.imap_unordered(process_clip, tasks), total=len(tasks)):
                results.append(r)
    else:
        init_worker(args.seed)
        for task in tqdm(tasks):
            results.append(process_clip(task))

    produced_ids = []
    operator_counts = Counter()
    teacher_counts = Counter()
    validation_reasons = Counter()
    action_counts = Counter()
    requested_residual_max = np.zeros(7, dtype=np.float32)
    requested_residual_sum = np.zeros(7, dtype=np.float64)
    requested_residual_count = np.zeros(7, dtype=np.int64)
    errors = []
    skipped = Counter()
    for r in results:
        produced_ids.extend(r.get('sample_ids', []))
        operator_counts.update(r.get('operator_counts', {}))
        teacher_counts.update(r.get('teacher_counts', {}))
        validation_reasons.update(r.get('validation_reasons', {}))
        errors.extend([{'clip_id': r.get('clip_id'), 'error': e} for e in r.get('errors', [])])
        skipped.update(r.get('skipped', []))
    for sid in produced_ids:
        path = os.path.join(args.out_dir, sid + '.npz')
        if os.path.exists(path):
            try:
                d = np.load(path, allow_pickle=True)
                action_counts[str(d.get('action_type', 'unknown'))] += 1
                if 'requested_achieved_residual' in d and 'goal_slot_mask' in d:
                    res = d['requested_achieved_residual'].astype(np.float32)
                    mask = d['goal_slot_mask'].astype(bool)
                    requested_residual_max[mask] = np.maximum(requested_residual_max[mask], res[mask])
                    requested_residual_sum[mask] += res[mask]
                    requested_residual_count[mask] += 1
            except Exception:
                action_counts['unknown'] += 1

    print(f'\nGenerated {len(produced_ids)} files from {len(tasks)} available clips')
    print(f'Validation rejects: {sum(validation_reasons.values())}; errors: {len(errors)}')

    manifest = {
        'version': PACE_SCHEMA_VERSION,
        'schema_hash': schema_hash(),
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'git_commit': get_git_commit(),
        'args': vars(args),
        'source': {
            'data_root': os.path.abspath(args.data_root),
            'split_file': args.split_file,
            'n_split_ids': len(train_ids),
            'n_available_tasks': len(tasks),
            'missing_npz': missing_npz[:200],
        },
        'stats': {
            'mean_shape': list(mean.shape),
            'std_shape': list(std.shape),
        },
        'n_input_clips': len(tasks),
        'n_output_files': len(produced_ids),
        'sample_ids': produced_ids,
        'operators': ['raw_prior','se2_arrival','arrival_slice','traj_warp','ee_ik','posture_recast'],
        'operator_counts': dict(operator_counts),
        'teacher_counts': dict(teacher_counts),
        'action_counts': dict(action_counts),
        'validation_reject_counts': dict(validation_reasons),
        'requested_achieved_residual': {
            'max': requested_residual_max.tolist(),
            'mean': np.divide(
                requested_residual_sum,
                np.maximum(requested_residual_count, 1),
                out=np.zeros_like(requested_residual_sum),
                where=requested_residual_count > 0,
            ).tolist(),
            'count': requested_residual_count.tolist(),
            'slot_names': SLOT_NAMES,
        },
        'skipped_counts': dict(skipped),
        'errors': errors[:200],
        'field_schema': {
            'motion_38d': '[T,38] float32 denormalized robot motion',
            'goal_value': '[T,7,4] float32 root-relative requested conditioning target',
            'goal_fk_target': '[T,7,4] float32 world-frame achieved FK/root/heading supervision target',
            'goal_world_requested': '[T,7,4] float32 operator-requested world target',
            'goal_world_achieved': '[T,7,4] float32 FK-rebuilt achieved world target',
            'goal_slot_mask': '[7] bool model conditioning slots',
            'loss_slot_mask': '[7] bool FK supervision slots',
            'aux_slot_mask': '[7] bool auxiliary stabilization slots',
        },
        'hard_tolerance_cm': {'traj':10,'wrist':5,'foot':5,'pelvis':8,'heading_deg':10},
        'compatibility': 'goal_world aliases goal_world_achieved; goal_value remains root-relative conditioning; goal_fk_target is world-frame supervision.',
    }
    with open(os.path.join(args.out_dir, 'manifest.json'), 'w') as f:
        json.dump(to_jsonable(manifest), f, indent=2)
    print('Manifest saved. Done.')

if __name__ == '__main__':
    main()
