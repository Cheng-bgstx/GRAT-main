"""
CLoSD-aligned target conditioning: 目标名称 + 目标位置 + 转向.

- 目标名称 (target_joint_names): which joints are constrained; drives validity mask.
- 目标位置 (target_cond): (B, n_slots, 3). Dual-format contract:
  训练时 (displacement): EE slots = (last-first)_xy + last_z [m]; traj/pelvis/heading = absolute.
  推理时 (absolute):     EE slots = absolute world xyz [m]; traj/pelvis/heading = absolute.
  推理管线维护两个张量:
    target_tensor       — 传入 denoiser 的模型条件（与训练语义一致，pelvis/traj/heading 不变）
    target_tensor_control — FK guidance / IK / terminal inpaint 使用的绝对世界坐标
  详见 build_target_cond_from_goal() 和 get_pred_target_from_fk_keypoints(ee_format=...).
- 转向 (is_heading): (B,) bool, whether heading is active.
"""
import torch
import numpy as np
from copy import copy
import random

from utils.rotation_utils_torch import (
    prediction_to_goal_xy_heading_torch,
    rot6d_to_quat_wxyz_torch,
    rot6d_to_rotation_matrix_torch,
    quat_wxyz_to_yaw_torch,
)

DEFAULT_TARGET_JOINT_CONFIG = 'ROBOT_AUTO'
DEFAULT_TRAIN_TARGET_JOINT_CONFIG = 'ROBOT_FULL'
# Head target slot is intentionally removed: pelvis + 4 EE + traj + heading.
ROBOT_EE_JOINT_NAMES = ['left_foot', 'right_foot', 'left_wrist', 'right_wrist']
ROBOT_ALL_GOAL_JOINT_NAMES = ['pelvis']
ROBOT_EXTENDED_GOAL_JOINT_NAMES = (
    ROBOT_ALL_GOAL_JOINT_NAMES + ROBOT_EE_JOINT_NAMES + ['traj', 'heading']
)
LEGACY_ROBOT_EXTENDED_GOAL_JOINT_NAMES = [
    'pelvis', 'left_foot', 'right_foot', 'left_wrist', 'right_wrist', 'head', 'traj', 'heading'
]
ROBOT_TARGET_NAME_TO_INDEX = {
    name: idx for idx, name in enumerate(ROBOT_EXTENDED_GOAL_JOINT_NAMES)
}
ROBOT_FK_KEYPOINT_NAMES = ['pelvis', 'left_foot', 'right_foot', 'left_wrist', 'right_wrist']

# 38D cannot recover EE; only these slots are valid for 38D goal loss / inference.
ROBOT_38D_ALLOWED_TARGET_NAMES = {'pelvis', 'traj', 'heading'}


def validate_target_joints_for_38d(target_joint_names_list, strict=True):
    """
    For 38D mode, EE targets (left_foot, right_foot, left_wrist, right_wrist) are invalid.
    - strict=True: raise ValueError if any sample requests EE.
    - strict=False: return filtered list of lists, keeping only pelvis/traj/heading per sample.
    """
    ee_names = set(ROBOT_EE_JOINT_NAMES)
    out = []
    for sample_names in target_joint_names_list:
        allowed = [n for n in sample_names if n in ROBOT_38D_ALLOWED_TARGET_NAMES]
        disallowed = [n for n in sample_names if n in ee_names]
        if strict and disallowed:
            raise ValueError(
                '38D mode does not support EE targets (left_foot, right_foot, left_wrist, right_wrist). '
                f'Requested: {disallowed}. Use 50D or remove EE from target_joint_names.'
            )
        out.append(allowed)
    return out

# Body index in body_pos (T, 30, 3). Skeleton-dependent; override if your URDF/30-body order differs.
# Prefer mappings verified against MuJoCo/runtime FK replay over heuristic semantic statistics.
ROBOT_BODY_IDX = {
    'pelvis': 0,
    'left_foot': 18,      # left_ankle_roll_link (verified from body_mapping_mujoco_100ids.json)
    'right_foot': 19,     # right_ankle_roll_link
    'left_wrist': 28,     # left_wrist_yaw_link
    'right_wrist': 29,    # right_wrist_yaw_link
}

# 50D extension (CLoSD-aligned, head removed): same order as robot_process.EE_NAMES_50; each EE 3d = (vel_x_local, vel_y_local, z_abs) in m/s
DIM_38 = 38
DIM_50 = 50
EE_SLICES_53 = {
    'left_foot': slice(38, 41),
    'right_foot': slice(41, 44),
    'left_wrist': slice(44, 47),
    'right_wrist': slice(47, 50),
}


def apply_goal_adapter(goal_emb, opt=None):
    """
    轻量目标适配层扩展点：在将 goal_emb 注入 temb 前可在此做变换。
    默认恒等返回，不改变 goal_emb。若 recon guidance 对 root 级控制仍不足，
    可在此接入小型可训模块（冻结主干、仅训此适配层），避免重污染扩散主干。
    goal_emb: (B, D) 或 (B, T, D)；opt: 可选训练/推理配置。
    """
    return goal_emb


def get_allowed_target_options(config_name):
    """CLoSD-aligned target options: traj, pelvis, EE, heading in any combination."""
    if config_name in (None, ''):
        config_name = DEFAULT_TARGET_JOINT_CONFIG
    if config_name in ('ROBOT_AUTO', 'ROBOT_CLOSD_RANDOM', 'ROBOT_CLOSD'):
        return []
    if config_name == 'ROBOT_ROOT':
        return [['traj', 'heading']]
    if config_name == 'ROBOT_MINIMAL':
        return [
            ['traj', 'heading'],
            ['pelvis', 'heading'],
            ['traj'],
            ['pelvis'],
            ['left_foot', 'right_foot'],
            [],
        ]
    if config_name == 'ROBOT_FULL':
        return [[
            'pelvis',
            'left_foot',
            'right_foot',
            'left_wrist',
            'right_wrist',
            'traj',
            'heading',
        ]]
    return [config_name.split(',')]


def sample_robot_goal(batch_size, device, force_joints=None):
    """
    Sample target joint config per sample (CLoSD-aligned).
    - force_joints is None / ROBOT_CLOSD_RANDOM:
      sample a random semantic subset from pelvis/traj/4EE, plus heading separately.
      Compared with the earlier <=2-slot version, this stronger variant covers
      multi-joint full-control cases more often while still keeping some no-goal samples.
    - force_joints is set: get options from get_allowed_target_options(force_joints), then random.choice per sample.
    Returns target_joint_names (list of list[str], heading removed from names), is_heading (BoolTensor[B]).
    """
    if force_joints in (None, '', 'ROBOT_CLOSD_RANDOM', 'ROBOT_CLOSD', 'ROBOT_AUTO'):
        semantic_names = np.array(['traj', 'pelvis'] + ROBOT_EE_JOINT_NAMES)
        # Keep some unconditional samples, but cover 2~4 active slots much more often.
        subset_size_probs = np.array([0.30, 0.14, 0.18, 0.16, 0.11, 0.07, 0.04], dtype=np.float64)
        subset_size_probs = subset_size_probs / subset_size_probs.sum()
        # Bias random subsets slightly toward root semantics so root arrival remains trained well.
        semantic_weights = np.array([2.5, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        semantic_weights = semantic_weights / semantic_weights.sum()
        target_joint_names = []
        is_heading = torch.zeros(batch_size, device=device, dtype=torch.bool)
        for sample_idx in range(batch_size):
            k = int(np.random.choice(np.arange(len(semantic_names) + 1), p=subset_size_probs))
            if k <= 0:
                names = []
            else:
                names = np.random.choice(
                    semantic_names,
                    size=k,
                    replace=False,
                    p=semantic_weights,
                ).tolist()
            target_joint_names.append(names)
            heading_prob = 0.50 if names else 0.12
            is_heading[sample_idx] = bool(torch.rand(1, device=device) < heading_prob)
        return target_joint_names, is_heading
    options = get_allowed_target_options(force_joints)
    names = [copy(random.choice(options)) for _ in range(batch_size)]
    is_heading = torch.zeros(batch_size, device=device, dtype=torch.bool)
    for i, n in enumerate(names):
        if 'heading' in n:
            is_heading[i] = True
            n.remove('heading')
    return names, is_heading


def infer_target_joint_metadata(goal_cond, batch_size, device):
    """
    Infer target_joint_names / is_heading from user-provided goal format.

    Rules:
    - (B,3): traj + heading
    - (B,4): pelvis + traj + heading
    - (B,7,3): treat as full target_cond and activate all semantic slots + heading
    """
    target_joint_names = []
    is_heading = torch.ones(batch_size, device=device, dtype=torch.bool)

    if goal_cond is None:
        return build_empty_target_metadata(batch_size, device)

    if not torch.is_tensor(goal_cond):
        goal_cond = torch.as_tensor(goal_cond, device=device)
    else:
        goal_cond = goal_cond.to(device=device)

    if goal_cond.dim() == 1:
        goal_cond = goal_cond.unsqueeze(0)
    elif goal_cond.dim() == 2 and goal_cond.shape[1] == 3 and goal_cond.shape[0] in (
        len(ROBOT_EXTENDED_GOAL_JOINT_NAMES),
        len(LEGACY_ROBOT_EXTENDED_GOAL_JOINT_NAMES),
    ):
        # Interpret (S,3) as one full-slot target_cond.
        goal_cond = goal_cond.unsqueeze(0)
    if goal_cond.dim() == 3 and goal_cond.shape[-2:] == (len(ROBOT_EXTENDED_GOAL_JOINT_NAMES), 3):
        full_names = [name for name in ROBOT_EXTENDED_GOAL_JOINT_NAMES if name != 'heading']
        target_joint_names = [list(full_names) for _ in range(batch_size)]
        return target_joint_names, is_heading

    if goal_cond.dim() != 2:
        raise ValueError(f'Unsupported goal_cond shape for metadata inference: {tuple(goal_cond.shape)}')

    if goal_cond.shape[1] >= 4:
        target_joint_names = [['pelvis', 'traj'] for _ in range(batch_size)]
    elif goal_cond.shape[1] >= 3:
        target_joint_names = [['traj'] for _ in range(batch_size)]
    else:
        target_joint_names = [[] for _ in range(batch_size)]
        is_heading = torch.zeros(batch_size, device=device, dtype=torch.bool)
    return target_joint_names, is_heading


def parse_target_joint_config(config_name, batch_size, device):
    """
    Convert a config string into the CLoSD-style pair:
    - target_joint_names: list[list[str]] without the special "heading" token
    - is_heading: BoolTensor[B]
    """
    if config_name in ('ROBOT_AUTO', None, ''):
        raise ValueError(
            'ROBOT_AUTO depends on the provided goal format. '
            'Use infer_target_joint_metadata(...) or pass explicit target_joint_names/is_heading.'
        )
    options = get_allowed_target_options(config_name)
    if len(options) != 1:
        raise ValueError(
            f'Expected a single fixed target option, got {config_name!r} -> {options}. '
            'For training/inference in this project, please choose one explicit option such as '
            '"traj,heading" or "pelvis,heading".'
        )

    target_joint_names = []
    is_heading = torch.zeros(batch_size, device=device, dtype=torch.bool)
    for sample_idx in range(batch_size):
        names = list(options[0])
        if 'head' in names:
            raise ValueError(
                "Target slot 'head' has been removed from this project. "
                "Please use pelvis/feet/wrists/traj/heading only."
            )
        if 'heading' in names:
            is_heading[sample_idx] = True
            names.remove('heading')
        target_joint_names.append(names)
    return target_joint_names, is_heading


def build_empty_target_metadata(batch_size, device):
    return [[] for _ in range(batch_size)], torch.zeros(batch_size, device=device, dtype=torch.bool)


def duplicate_target_metadata(target_joint_names, is_heading, copies=2):
    target_joint_names_dup = []
    for _ in range(copies):
        target_joint_names_dup.extend([list(names) for names in target_joint_names])
    is_heading_dup = torch.cat([is_heading] * copies, dim=0)
    return target_joint_names_dup, is_heading_dup


def apply_target_cond_dropout(target_cond, target_joint_names, is_heading, drop_mask):
    if drop_mask is None or not bool(drop_mask.any()):
        return target_cond, [list(names) for names in target_joint_names], is_heading

    # CRITICAL FIX: Zero target_cond when dropping goal to prevent NaN
    # Previous behavior kept real values but disabled validity, which created
    # inconsistent state that could cause NaN in training.
    target_cond = target_cond.clone()
    target_joint_names = [list(names) for names in target_joint_names]
    is_heading = is_heading.clone()
    for sample_idx in torch.nonzero(drop_mask, as_tuple=False).flatten().tolist():
        target_cond[sample_idx] = 0.0  # Zero the target values
        target_joint_names[sample_idx] = []
        is_heading[sample_idx] = False
    return target_cond, target_joint_names, is_heading


def build_target_validity(target_joint_names, is_heading, batch_size, device, dtype):
    validity = torch.zeros(
        batch_size,
        len(ROBOT_EXTENDED_GOAL_JOINT_NAMES),
        1,
        device=device,
        dtype=dtype,
    )
    for sample_idx, sample_joint_names in enumerate(target_joint_names):
        for joint_name in sample_joint_names:
            if joint_name == 'head':
                raise ValueError("Target slot 'head' is disabled and cannot be used.")
            if joint_name not in ROBOT_TARGET_NAME_TO_INDEX:
                raise ValueError(f"Unknown target joint name: {joint_name}")
            validity[sample_idx, ROBOT_TARGET_NAME_TO_INDEX[joint_name], 0] = 1.0
        if bool(is_heading[sample_idx]):
            validity[sample_idx, ROBOT_TARGET_NAME_TO_INDEX['heading'], 0] = 1.0
    return validity


def _heading_from_motion_38d_last(motion_38d, lengths):
    """Extract heading (yaw) at last valid frame from 38d motion. Returns (B,) in radians."""
    B, T, _ = motion_38d.shape
    device = motion_38d.device
    last_idx = (lengths - 1).clamp(0, T - 1)
    last_rot6d = motion_38d[torch.arange(B, device=device), last_idx, 32:38]
    quat = rot6d_to_quat_wxyz_torch(last_rot6d)
    return quat_wxyz_to_yaw_torch(quat)


def _heading_from_motion_38d_all(motion_38d):
    """Extract heading (yaw) for every frame from 38d motion. Returns (B, T) in radians."""
    B, T, _ = motion_38d.shape
    rot6d = motion_38d[..., 32:38].reshape(B * T, 6)
    quat = rot6d_to_quat_wxyz_torch(rot6d)
    yaw = quat_wxyz_to_yaw_torch(quat)
    return yaw.view(B, T)


def get_ref_target_from_body_pos(body_pos, lengths, motion_38d_denorm, goal_last_idx=None):
    """
    目标位置 (ref target_cond) 来自 GT body 位置；EE 为位移 xy + z（相对首帧位移 + 末帧绝对 Z）。
    pelvis/traj 取末帧位置；heading 来自 motion。

    Args:
        body_pos: (B, T, 30, 3) aligned world positions, same frame as 38d
        lengths: (B,) int
        motion_38d_denorm: (B, T, 38) denormalized motion for heading at last frame
        goal_last_idx: optional (B,) long tensor selecting which frame to treat as the goal frame.
            When None, uses the true last valid frame (length-1).
    Returns:
        ref_target: (B, len(ROBOT_EXTENDED_GOAL_JOINT_NAMES), 3) = (B, 7, 3)
    """
    B, T, Nb, _ = body_pos.shape
    device = body_pos.device
    dtype = body_pos.dtype
    if goal_last_idx is None:
        last_idx = (lengths - 1).clamp(0, T - 1)
    else:
        if not torch.is_tensor(goal_last_idx):
            goal_last_idx = torch.as_tensor(goal_last_idx, device=device)
        goal_last_idx = goal_last_idx.to(device=device, dtype=torch.long).view(-1)
        if goal_last_idx.shape[0] != B:
            raise ValueError(f'goal_last_idx length {goal_last_idx.shape[0]} != batch size {B}')
        last_idx = goal_last_idx.clamp(0, T - 1)

    ref_target = torch.zeros(
        B,
        len(ROBOT_EXTENDED_GOAL_JOINT_NAMES),
        3,
        device=device,
        dtype=dtype,
    )
    pelvis_idx = ROBOT_TARGET_NAME_TO_INDEX['pelvis']
    traj_idx = ROBOT_TARGET_NAME_TO_INDEX['traj']
    heading_idx = ROBOT_TARGET_NAME_TO_INDEX['heading']
    pid = ROBOT_BODY_IDX['pelvis']

    last_pelvis = body_pos[torch.arange(B, device=device), last_idx, pid, :]
    ref_target[:, pelvis_idx, :] = last_pelvis
    ref_target[:, traj_idx, :2] = last_pelvis[:, :2]
    ref_target[:, traj_idx, 2] = 0.0
    # EE: ref = (last - first)_xy, last_z (displacement + z_abs), CLoSD-aligned with vel_local+z representation
    first_idx = torch.zeros(B, dtype=torch.long, device=device)
    for ee_name in ROBOT_EE_JOINT_NAMES:
        if ee_name not in ROBOT_BODY_IDX:
            continue
        bid = ROBOT_BODY_IDX[ee_name]
        if bid >= Nb:
            continue
        slot_idx = ROBOT_TARGET_NAME_TO_INDEX[ee_name]
        last_pos = body_pos[torch.arange(B, device=device), last_idx, bid, :]
        first_pos = body_pos[torch.arange(B, device=device), first_idx, bid, :]
        ref_target[:, slot_idx, :2] = last_pos[:, :2] - first_pos[:, :2]
        ref_target[:, slot_idx, 2] = last_pos[:, 2]
    # Heading is always taken from the true last valid frame to keep semantics consistent
    # with inference-time "final heading". Goal frame sampling is only for position targets.
    ref_target[:, heading_idx, 0] = _heading_from_motion_38d_last(motion_38d_denorm, lengths)
    return ref_target


def build_full_sparse_target_labels(body_pos, lengths, motion_38d_denorm):
    """
    构造 (B, 7, 3) 完整稀疏目标标签，供外部位置预测器与 goal adapter 训练共用。
    语义与 get_ref_target_from_body_pos 一致：pelvis/traj/heading + 4EE（位移 xy + 末帧 z）。
    见 docs/CLOSD_FULL_TARGET_SEMANTICS_38D_EXT.md。
    """
    return get_ref_target_from_body_pos(body_pos, lengths, motion_38d_denorm)


def build_dense_sparse_target_labels(body_pos, lengths, motion_38d_denorm):
    """
    Build dense sparse semantic targets (B, T, 7, 3) for predictor-v2 and adapter regularization.

    Semantics follow the terminal target representation frame-by-frame:
    - pelvis: current world position
    - traj: current pelvis xy, z = 0
    - heading: current yaw in radians, stored in [:, :, heading_idx, 0]
    - EE: xy displacement relative to frame 0, plus current absolute z
    """
    B, T, Nb, _ = body_pos.shape
    device = body_pos.device
    dtype = body_pos.dtype

    dense_target = torch.zeros(
        B,
        T,
        len(ROBOT_EXTENDED_GOAL_JOINT_NAMES),
        3,
        device=device,
        dtype=dtype,
    )

    pelvis_idx = ROBOT_TARGET_NAME_TO_INDEX['pelvis']
    traj_idx = ROBOT_TARGET_NAME_TO_INDEX['traj']
    heading_idx = ROBOT_TARGET_NAME_TO_INDEX['heading']
    pid = ROBOT_BODY_IDX['pelvis']

    pelvis_pos = body_pos[:, :, pid, :]
    dense_target[:, :, pelvis_idx, :] = pelvis_pos
    dense_target[:, :, traj_idx, :2] = pelvis_pos[:, :, :2]
    dense_target[:, :, traj_idx, 2] = 0.0

    for ee_name in ROBOT_EE_JOINT_NAMES:
        if ee_name not in ROBOT_BODY_IDX:
            continue
        bid = ROBOT_BODY_IDX[ee_name]
        if bid >= Nb:
            continue
        slot_idx = ROBOT_TARGET_NAME_TO_INDEX[ee_name]
        ee_pos = body_pos[:, :, bid, :]
        ee_pos0 = ee_pos[:, :1, :]
        dense_target[:, :, slot_idx, :2] = ee_pos[:, :, :2] - ee_pos0[:, :, :2]
        dense_target[:, :, slot_idx, 2] = ee_pos[:, :, 2]

    heading_all = _heading_from_motion_38d_all(motion_38d_denorm)
    dense_target[:, :, heading_idx, 0] = heading_all.to(dtype=dtype)

    valid_mask = (
        torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    ).to(dtype=dtype)
    dense_target = dense_target * valid_mask[:, :, None, None]
    return dense_target


def build_fk_keypoint_labels(body_pos, lengths=None):
    """
    Build dense FK supervision keypoints (B, T, 5, 3) directly from GT body_pos.
    Keypoint order: pelvis, left_foot, right_foot, left_wrist, right_wrist.
    """
    Nb = body_pos.shape[2]
    keypoint_idx = [ROBOT_BODY_IDX[name] for name in ROBOT_FK_KEYPOINT_NAMES]
    # Safety: clamp indices to valid range to prevent CUDA index-out-of-bounds
    keypoint_idx = [min(idx, Nb - 1) for idx in keypoint_idx]
    keypoint_pos = body_pos[:, :, keypoint_idx, :]
    if lengths is not None:
        B, T = body_pos.shape[:2]
        valid_mask = (
            torch.arange(T, device=body_pos.device).unsqueeze(0) < lengths.unsqueeze(1)
        ).to(dtype=body_pos.dtype)
        keypoint_pos = keypoint_pos * valid_mask[:, :, None, None]
    return keypoint_pos


def get_pred_target_from_motion(motion, src_mask, lengths, root_vel_frame='world', time_weighted_gradient=False):
    """
    Recover predicted target from motion (38D or 50D). Dispatches by last feature dim.
    root_vel_frame: 'world' or 'body' (must match data/training).
    time_weighted_gradient: if True, use time-weighted cumsum backward (later frames get more gradient).
    """
    d = motion.shape[-1]
    if d >= DIM_50:
        return get_pred_target_from_motion_53d(
            motion[..., :DIM_50], src_mask, lengths,
            root_vel_frame=root_vel_frame,
            time_weighted_gradient=time_weighted_gradient,
        )
    return get_pred_target_from_motion_38d(
        motion, src_mask, lengths,
        root_vel_frame=root_vel_frame,
        time_weighted_gradient=time_weighted_gradient,
    )


def get_pred_target_from_motion_38d(motion_38d, src_mask, lengths, root_vel_frame='world', time_weighted_gradient=False):
    """
    Recover predicted target from 38d motion (e.g. model output).
    root_vel_frame: 'world' or 'body' (must match how root_vel_xy was stored).
    time_weighted_gradient: passed to prediction_to_goal_xy_heading_torch for recon guidance.
    """
    return build_target_cond_from_motion_38d(
        motion_38d, src_mask, lengths,
        root_vel_frame=root_vel_frame,
        time_weighted_gradient=time_weighted_gradient,
    )


# FPS for EE velocity→position integration (m/s * dt = displacement). Match robot_process.ROBOT_FPS.
ROBOT_FPS = 50


def get_pred_target_from_motion_53d(motion_53d, src_mask, lengths, root_vel_frame='world', time_weighted_gradient=False):
    """
    Recover predicted target from 53d motion (38D + 5 EE as [vel_local_xy, z_abs] per EE, CLoSD-aligned).
    pelvis/traj/heading from 38d (root_vel_frame: world or body); EE: vel*dt cumsum -> last + z.
    """
    batch_size, seq_len, _ = motion_53d.shape
    device = motion_53d.device
    dtype = motion_53d.dtype
    motion_38d = motion_53d[..., :DIM_38]
    target_cond = build_target_cond_from_motion_38d(
        motion_38d, src_mask, lengths,
        root_vel_frame=root_vel_frame,
        time_weighted_gradient=time_weighted_gradient,
    )
    last_idx = (lengths - 1).clamp(0, seq_len - 1)
    root_rot6d = motion_53d[:, :, 32:38]
    B, T = batch_size, seq_len
    R = rot6d_to_rotation_matrix_torch(root_rot6d.reshape(-1, 6)).reshape(B, T, 3, 3)
    mask = (torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1).float()
    dt = 1.0 / ROBOT_FPS
    for name, sl in EE_SLICES_53.items():
        idx = ROBOT_TARGET_NAME_TO_INDEX.get(name)
        if idx is None:
            continue
        v_local = motion_53d[:, :, sl].clone()  # (B, T, 3) = (vel_x_local, vel_y_local, z_abs) m/s
        v_local[..., 2] = 0.0
        v_world = torch.einsum('btij,btj->bti', R, v_local)[..., :2]  # (B, T, 2) m/s
        v_world = v_world * mask
        vel_step = v_world * dt  # displacement per frame
        vel_step = vel_step.clone()
        vel_step[:, 0, :] = 0.0  # frame 0 no prior → P[0]=0 (CLoSD convention)
        disp_xy = vel_step.cumsum(dim=1)
        target_cond[:, idx, :2] = disp_xy[torch.arange(B, device=device), last_idx, :]
        target_cond[:, idx, 2] = motion_53d[torch.arange(B, device=device), last_idx, sl.start + 2]
    return target_cond


def get_pred_target_from_fk_keypoints(keypoint_pos, motion_38d, lengths, ee_format='displacement'):
    """
    Build target_cond (B, 7, 3) from FK keypoint output and 38D motion.
    keypoint_pos: (B, T, 5, 3) from RobotFKLayer - order: pelvis, left_foot, right_foot, left_wrist, right_wrist.
    motion_38d: (B, T, 38) denormalized for heading at last frame.
    lengths: (B,) int.
    ee_format: 'displacement' = (last-first)_xy + last_z (training, matches get_ref_target_from_body_pos)
               'absolute'     = absolute world xyz at last frame (inference recon guidance)

    NOTE: Uses functional ops (no in-place assignment) to preserve autograd graph for recon guidance.
    """
    B, T, _, _ = keypoint_pos.shape
    device = keypoint_pos.device
    dtype = keypoint_pos.dtype
    last_idx = (lengths - 1).clamp(0, T - 1)
    first_idx = torch.zeros(B, dtype=torch.long, device=device)
    batch_arange = torch.arange(B, device=device)

    n_slots = len(ROBOT_EXTENDED_GOAL_JOINT_NAMES)
    # Build each slot as a (B, 3) tensor, then stack → (B, 7, 3)
    slots = [torch.zeros(B, 3, device=device, dtype=dtype) for _ in range(n_slots)]

    pelvis_idx = ROBOT_TARGET_NAME_TO_INDEX['pelvis']
    traj_idx = ROBOT_TARGET_NAME_TO_INDEX['traj']
    heading_idx = ROBOT_TARGET_NAME_TO_INDEX['heading']

    # pelvis: absolute last-frame FK position
    last_pelvis = keypoint_pos[batch_arange, last_idx, 0, :]  # (B, 3)
    slots[pelvis_idx] = last_pelvis

    # traj: pelvis XY with Z=0
    traj_z = torch.zeros(B, 1, device=device, dtype=dtype)
    slots[traj_idx] = torch.cat([last_pelvis[:, :2], traj_z], dim=1)

    # EE joints
    ee_names = ['left_foot', 'right_foot', 'left_wrist', 'right_wrist']
    for i, name in enumerate(ee_names):
        slot_idx = ROBOT_TARGET_NAME_TO_INDEX[name]
        kp_idx = i + 1
        last_pos = keypoint_pos[batch_arange, last_idx, kp_idx, :]  # (B, 3)
        if ee_format == 'absolute':
            slots[slot_idx] = last_pos
        else:
            first_pos = keypoint_pos[batch_arange, first_idx, kp_idx, :]
            disp_xy = last_pos[:, :2] - first_pos[:, :2]
            slots[slot_idx] = torch.cat([disp_xy, last_pos[:, 2:3]], dim=1)

    # heading
    heading_val = _heading_from_motion_38d_last(motion_38d, lengths)  # (B,)
    heading_zeros = torch.zeros(B, 2, device=device, dtype=dtype)
    slots[heading_idx] = torch.cat([heading_val.unsqueeze(1), heading_zeros], dim=1)

    target_cond = torch.stack(slots, dim=1)  # (B, 7, 3)
    return target_cond


def build_target_cond_from_motion_38d(motion_38d, src_mask, lengths, root_vel_frame='world', time_weighted_gradient=False):
    """
    Robot version of CLoSD get_target_location().
    Returns last-frame target_cond; 38D has no EE so only pelvis/traj/heading filled.
    root_vel_frame: 'world' or 'body' for root velocity integration.
    time_weighted_gradient: if True, later frames get larger gradient in cumsum (for recon guidance).

    NOTE: Uses functional ops (no in-place assignment) to preserve autograd graph for recon guidance.
    """
    batch_size, seq_len, _ = motion_38d.shape
    device = motion_38d.device
    dtype = motion_38d.dtype

    goal = prediction_to_goal_xy_heading_torch(
        motion_38d, src_mask, lengths,
        root_vel_frame=root_vel_frame,
        time_weighted_gradient=time_weighted_gradient,
    )
    last_idx = (lengths - 1).clamp(0, seq_len - 1)
    root_z = motion_38d[torch.arange(batch_size, device=device), last_idx, 31]

    n_slots = len(ROBOT_EXTENDED_GOAL_JOINT_NAMES)
    pelvis_idx = ROBOT_TARGET_NAME_TO_INDEX['pelvis']
    traj_idx = ROBOT_TARGET_NAME_TO_INDEX['traj']
    heading_idx = ROBOT_TARGET_NAME_TO_INDEX['heading']

    # Build each slot functionally
    slots = [torch.zeros(batch_size, 3, device=device, dtype=dtype) for _ in range(n_slots)]
    slots[pelvis_idx] = torch.cat([goal[:, :2], root_z.unsqueeze(1)], dim=1)
    slots[traj_idx] = torch.cat([goal[:, :2], torch.zeros(batch_size, 1, device=device, dtype=dtype)], dim=1)
    slots[heading_idx] = torch.cat([goal[:, 2:3], torch.zeros(batch_size, 2, device=device, dtype=dtype)], dim=1)

    target_cond = torch.stack(slots, dim=1)  # (B, 7, 3)
    return target_cond


# Default pelvis height (standing height for robot)
DEFAULT_PELVIS_HEIGHT = 0.95  # meters

# Default EE positions relative to pelvis (approximate standing pose)
# Format: (x_offset, y_offset, z_offset) relative to pelvis
DEFAULT_EE_OFFSETS = {
    'left_foot': (0.0, 0.1, -0.95),   # Below pelvis, slightly left
    'right_foot': (0.0, -0.1, -0.95),  # Below pelvis, slightly right
    'left_wrist': (0.2, 0.3, 0.0),     # In front, left side
    'right_wrist': (0.2, -0.3, 0.0),   # In front, right side
}


def build_target_cond_from_goal(goal_cond, device=None, dtype=None):
    """
    Build 目标位置 (target_cond) from user goal. Input is always position [m] and heading [rad].

    EE TARGET FORMAT CONTRACT (inference-time):
    - For full (B,7,3) goals, EE slots (left_foot, right_foot, left_wrist, right_wrist)
      are expected in ABSOLUTE WORLD POSITIONS (x, y, z) at the terminal frame.
    - This matches the inference-time FK prediction format (ee_format='absolute')
      used by _predict_sparse_targets and _terminal_inpaint_ee_fk.
    - pelvis: absolute world position (x, y, z)
    - traj: root XY position (x, y, 0)
    - heading: yaw angle in radians (stored in [:, heading_idx, 0])

    CLoSD-aligned input formats:
    - (B, 3) or (3,): [x, y, heading] -> traj/pelvis end position (x,y), heading; pelvis z set to DEFAULT_PELVIS_HEIGHT
    - (B, 4) or (4,): [x, y, z, heading] -> pelvis (x,y,z), traj (x,y,0), heading
    - (B, 7, 3) or (7, 3): full target_cond — EE in absolute world positions, use as-is
    Legacy compatibility:
    - (B, 8, 3) / (8, 3): old format with head slot will be accepted and head is dropped.
    """
    if goal_cond is None:
        return None
    if not torch.is_tensor(goal_cond):
        goal_cond = torch.tensor(goal_cond, device=device, dtype=dtype or torch.float32)
    else:
        goal_cond = goal_cond.to(device=device or goal_cond.device, dtype=dtype or goal_cond.dtype)
    if goal_cond.dim() == 1:
        goal_cond = goal_cond.unsqueeze(0)
    elif goal_cond.dim() == 2 and goal_cond.shape[1] == 3 and goal_cond.shape[0] in (
        len(ROBOT_EXTENDED_GOAL_JOINT_NAMES),
        len(LEGACY_ROBOT_EXTENDED_GOAL_JOINT_NAMES),
    ):
        # Interpret (S,3) as one full-slot target_cond.
        goal_cond = goal_cond.unsqueeze(0)

    batch_size = goal_cond.shape[0]
    n_slots = len(ROBOT_EXTENDED_GOAL_JOINT_NAMES)
    pelvis_idx = ROBOT_TARGET_NAME_TO_INDEX['pelvis']
    traj_idx = ROBOT_TARGET_NAME_TO_INDEX['traj']
    heading_idx = ROBOT_TARGET_NAME_TO_INDEX['heading']

    if goal_cond.shape[-2:] == (n_slots, 3):
        target_cond = goal_cond.clone()
        if target_cond.shape[0] == 1 and batch_size > 1:
            target_cond = target_cond.expand(batch_size, -1, -1)
        return target_cond
    if goal_cond.shape[-2:] == (len(LEGACY_ROBOT_EXTENDED_GOAL_JOINT_NAMES), 3):
        # Legacy migration: old 8-slot target_cond included a head slot.
        # Drop head and remap by semantic names.
        target_cond = torch.zeros(
            batch_size,
            n_slots,
            3,
            device=goal_cond.device,
            dtype=goal_cond.dtype,
        )
        for name, new_idx in ROBOT_TARGET_NAME_TO_INDEX.items():
            old_idx = LEGACY_ROBOT_EXTENDED_GOAL_JOINT_NAMES.index(name)
            target_cond[:, new_idx, :] = goal_cond[:, old_idx, :]
        if target_cond.shape[0] == 1 and batch_size > 1:
            target_cond = target_cond.expand(batch_size, -1, -1)
        return target_cond

    target_cond = torch.zeros(
        batch_size,
        n_slots,
        3,
        device=goal_cond.device,
        dtype=goal_cond.dtype,
    )

    if goal_cond.shape[1] >= 3:
        # Set pelvis XY and default Z height
        target_cond[:, pelvis_idx, :2] = goal_cond[:, :2]
        target_cond[:, pelvis_idx, 2] = DEFAULT_PELVIS_HEIGHT
        # Set traj XY, Z=0
        target_cond[:, traj_idx, :2] = goal_cond[:, :2]
        target_cond[:, traj_idx, 2] = 0.0
        # Set heading
        target_cond[:, heading_idx, 0] = goal_cond[:, 2]

        # Set default EE positions based on goal XY and heading
        # This provides reasonable targets when user doesn't specify EE explicitly
        heading_rad = goal_cond[:, 2]
        cos_h = torch.cos(heading_rad)
        sin_h = torch.sin(heading_rad)

        for ee_name, (ox, oy, oz) in DEFAULT_EE_OFFSETS.items():
            if ee_name not in ROBOT_TARGET_NAME_TO_INDEX:
                continue
            slot_idx = ROBOT_TARGET_NAME_TO_INDEX[ee_name]
            # Rotate offset by heading and add to pelvis position
            rotated_x = ox * cos_h - oy * sin_h
            rotated_y = ox * sin_h + oy * cos_h
            target_cond[:, slot_idx, 0] = goal_cond[:, 0] + rotated_x
            target_cond[:, slot_idx, 1] = goal_cond[:, 1] + rotated_y
            target_cond[:, slot_idx, 2] = DEFAULT_PELVIS_HEIGHT + oz

    if goal_cond.shape[1] >= 4:
        # Override pelvis Z with provided value
        target_cond[:, pelvis_idx, 2] = goal_cond[:, 2]
        target_cond[:, heading_idx, 0] = goal_cond[:, 3]

    return target_cond


def compute_target_cond_loss(pred_target, ref_target, target_joint_names, is_heading):
    """
    CLoSD-style masked end-target loss:
    - location loss on requested semantic joints
    - periodic heading loss on the heading slot
    - pelvis and traj both constrain root xy; only count root xy once per sample to avoid double weight
    """
    batch_size = pred_target.shape[0]
    device = pred_target.device
    loc_mask = torch.zeros_like(pred_target[:, :-1], dtype=torch.bool, device=device)
    pelvis_idx = ROBOT_TARGET_NAME_TO_INDEX['pelvis']
    traj_idx = ROBOT_TARGET_NAME_TO_INDEX['traj']

    for sample_idx, sample_joint_names in enumerate(target_joint_names):
        has_pelvis = 'pelvis' in sample_joint_names
        has_traj = 'traj' in sample_joint_names
        for joint_name in sample_joint_names:
            if joint_name == 'heading':
                continue
            joint_idx = ROBOT_TARGET_NAME_TO_INDEX[joint_name]
            if joint_idx < len(ROBOT_EXTENDED_GOAL_JOINT_NAMES) - 1:
                loc_mask[sample_idx, joint_idx] = True
        # NOTE: Keep traj XY active even when pelvis is active — see note in
        # compute_target_cond_loss_components above.

    # traj only constrains horizontal position (z never used)
    loc_mask[:, traj_idx, 2] = False
    loc_mask_f = loc_mask.float()
    loc_sq = (pred_target[:, :-1] - ref_target[:, :-1]) ** 2
    loc_denom = loc_mask_f.sum(dim=(1, 2)).clamp(min=1.0)
    loc_loss = (loc_sq * loc_mask_f).sum(dim=(1, 2)) / loc_denom

    heading_diff = pred_target[:, -1, 0] - ref_target[:, -1, 0]
    # 2π periodicity: facing direction matters, not just axis alignment
    heading_diff = torch.remainder(heading_diff + torch.pi, 2 * torch.pi) - torch.pi
    heading_loss = (heading_diff ** 2) * is_heading.float()

    return loc_loss.mean() + heading_loss.mean()


def compute_target_cond_loss_components(pred_target, ref_target, target_joint_names, is_heading):
    """Same as compute_target_cond_loss but returns (total, loc_mean, heading_mean, per_slot_dict) for logging."""
    batch_size = pred_target.shape[0]
    device = pred_target.device
    loc_mask = torch.zeros_like(pred_target[:, :-1], dtype=torch.bool, device=device)
    traj_idx = ROBOT_TARGET_NAME_TO_INDEX['traj']
    for sample_idx, sample_joint_names in enumerate(target_joint_names):
        has_pelvis = 'pelvis' in sample_joint_names
        has_traj = 'traj' in sample_joint_names
        for joint_name in sample_joint_names:
            if joint_name == 'heading':
                continue
            joint_idx = ROBOT_TARGET_NAME_TO_INDEX[joint_name]
            if joint_idx < len(ROBOT_EXTENDED_GOAL_JOINT_NAMES) - 1:
                loc_mask[sample_idx, joint_idx] = True
        # NOTE: Previously, traj XY was masked when pelvis was also active to avoid
        # "double-counting". However, pelvis and traj measure DIFFERENT things when
        # using FK-based goal loss: pelvis = FK chain position, traj = integrated
        # root velocity position. Keeping both provides a useful consistency signal
        # and ensures direct trajectory arrival supervision.
        # Only traj Z is always masked (root Z is in pelvis slot, not traj).
    loc_mask[:, traj_idx, 2] = False
    loc_mask_f = loc_mask.float()
    loc_sq = (pred_target[:, :-1] - ref_target[:, :-1]) ** 2
    loc_denom = loc_mask_f.sum(dim=(1, 2)).clamp(min=1.0)
    loc_loss = (loc_sq * loc_mask_f).sum(dim=(1, 2)) / loc_denom
    heading_diff = pred_target[:, -1, 0] - ref_target[:, -1, 0]
    # 2π periodicity: facing direction matters, not just axis alignment
    heading_diff = torch.remainder(heading_diff + torch.pi, 2 * torch.pi) - torch.pi
    heading_loss = (heading_diff ** 2) * is_heading.float()
    loc_mean = loc_loss.mean()
    heading_mean = heading_loss.mean()
    # Per-slot loss breakdown for monitoring
    per_slot = {}
    for name in ROBOT_EXTENDED_GOAL_JOINT_NAMES[:-1]:  # exclude heading (handled separately)
        idx = ROBOT_TARGET_NAME_TO_INDEX[name]
        slot_mask = loc_mask_f[:, idx]  # (B, 3)
        slot_sq = loc_sq[:, idx]  # (B, 3)
        slot_denom = slot_mask.sum(dim=1).clamp(min=1.0)  # (B,)
        slot_loss = (slot_sq * slot_mask).sum(dim=1) / slot_denom  # (B,)
        n_active = (slot_mask.sum(dim=1) > 0).float().sum()
        per_slot[name] = slot_loss.sum() / n_active.clamp(min=1.0)
    per_slot['heading'] = heading_mean
    return loc_mean + heading_mean, loc_mean, heading_mean, per_slot


def compute_target_consistency_38d_50d(pred_target_38, pred_target_50):
    """
    Consistency loss between target_cond recovered from 38D vs from 50D (same batch).
    Ensures the same prediction does not imply different pelvis/traj/heading.
    pred_target_38, pred_target_50: (B, n_slots, 3) from build_target_cond_from_motion_38d and get_pred_target_from_motion_53d.
    Compares pelvis, traj (xy and z where used), and heading (with periodicity).
    """
    pelvis_idx = ROBOT_TARGET_NAME_TO_INDEX['pelvis']
    traj_idx = ROBOT_TARGET_NAME_TO_INDEX['traj']
    heading_idx = ROBOT_TARGET_NAME_TO_INDEX['heading']
    loc_loss = (pred_target_38[:, pelvis_idx, :] - pred_target_50[:, pelvis_idx, :]) ** 2
    loc_loss = loc_loss + (pred_target_38[:, traj_idx, :] - pred_target_50[:, traj_idx, :]) ** 2
    heading_diff = pred_target_38[:, heading_idx, 0] - pred_target_50[:, heading_idx, 0]
    heading_diff = torch.remainder(heading_diff + torch.pi, 2 * torch.pi) - torch.pi
    return loc_loss.mean() + (heading_diff ** 2).mean()
