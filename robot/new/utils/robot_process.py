"""
Robot motion data processing utilities.
Adapted from HumanML3D's motion_process.py for Z-up coordinate system.

Key differences from HumanML3D:
- Z-up coordinate system (X/Y ground plane, Z is height)
- Face direction unified to +X (forward)
- Pure numpy implementation (no torch dependency for fast multiprocessing)

50D extension (CLoSD-aligned, head removed): when return_53d=True and full format, appends 12 dims as
[vel_x_local, vel_y_local, z_abs] per EE (4 EE) — body-local XY velocity (m/s) + absolute Z.
Same units as root_vel_xy: velocity (m/s). Recovery: integrate with ×dt (dt=1/fps).
CLoSD: vel[t] = (pos[t]-pos[t-1])*FPS; we match that. Reduces over-determination vs world xyz.
"""
import numpy as np
from utils.rotation_utils import quat_wxyz_to_6d

# 50D layout: 38D + [38:41] left_foot, [41:44] right_foot, [44:47] left_wrist, [47:50] right_wrist
DIM_38 = 38
DIM_50 = 50
EE_NAMES_50 = ['left_foot', 'right_foot', 'left_wrist', 'right_wrist']
# Backward-compatible aliases (historical naming).
DIM_53 = DIM_50
EE_NAMES_53 = EE_NAMES_50


# ---- Pure numpy quaternion ops (avoids importing torch in worker processes) ----

def _qrot_np(q, v):
    """Rotate vector(s) v by quaternion(s) q. Pure numpy, no torch.
    Args: q (*, 4) wxyz, v (*, 3). Returns (*, 3)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    qvec = q[..., 1:]                          # (*, 3)
    uv = np.cross(qvec, v)                     # (*, 3)
    uuv = np.cross(qvec, uv)                   # (*, 3)
    return v + 2.0 * (q[..., :1] * uv + uuv)  # (*, 3)


def _qmul_np(q, r):
    """Multiply quaternion(s) q * r. Pure numpy, no torch.
    Args: q (*, 4) wxyz, r (*, 4) wxyz. Returns (*, 4) wxyz."""
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    w = q[..., 0]*r[..., 0] - q[..., 1]*r[..., 1] - q[..., 2]*r[..., 2] - q[..., 3]*r[..., 3]
    x = q[..., 0]*r[..., 1] + q[..., 1]*r[..., 0] + q[..., 2]*r[..., 3] - q[..., 3]*r[..., 2]
    y = q[..., 0]*r[..., 2] - q[..., 1]*r[..., 3] + q[..., 2]*r[..., 0] + q[..., 3]*r[..., 1]
    z = q[..., 0]*r[..., 3] + q[..., 1]*r[..., 2] - q[..., 2]*r[..., 1] + q[..., 3]*r[..., 0]
    return np.stack([w, x, y, z], axis=-1)


def _qconj_np(q):
    """Conjugate quaternion (inverse for unit quat). q (*, 4) wxyz -> (*, 4) wxyz."""
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def _world_to_local_xy_np(root_quat_wxyz, vec_xy):
    """Rotate world-frame XY vector(s) to body-local XY. root_quat (*,4) wxyz, vec_xy (*,2). Returns (*,2)."""
    v3 = np.zeros(vec_xy.shape[:-1] + (3,), dtype=np.float64)
    v3[..., :2] = vec_xy
    q_inv = _qconj_np(root_quat_wxyz)
    out = _qrot_np(q_inv, v3)
    return out[..., :2].astype(np.float32)


# Default FPS for velocity scale (m/s). Must match opt.fps (50) used in recovery.
ROBOT_FPS = 50


def process_robot_npz(npz_data, root_idx=0, return_body_pos=False, return_53d=False, fps=None, root_vel_frame='world'):
    """
    Process robot NPZ data with root alignment and facing normalization.
    
    Supports two data formats (auto-detected):
    - Full format: body_pos_w (T,30,3), body_quat_w (T,30,4), joint_pos (T,29)
    - Simple format: root_pos (T,3), root_rot (T,4), joint_pos (T,29)
    
    Pipeline:
    1. Put on floor: min(Z) -> 0 (full format only; simple format ground is already Z=0)
    2. Root XY at origin: root_pos[0, XY] -> (0, 0)
    3. Face +X direction: extract forward from root quaternion, rotate to +X
    4. Extract velocity (m/s): (pos[t]-pos[t-1])*fps; root_vel_frame 'world'=aligned world XY, 'body'=body-local (CLoSD); EE always body-local.
    
    Args:
        npz_data: dict - loaded NPZ file
        root_idx: int - index of root body (default: 0, used in full format)
        return_body_pos: bool - if True and full format, also return aligned body_pos_w (T,30,3)
        return_53d: bool - if True and full format, append 4 EE [vel_local_xy, z] (T,12) → (T,50)
        fps: int - frame rate for velocity scale (default: ROBOT_FPS=50). Stored values = velocity in m/s.
        root_vel_frame: str - 'world' (aligned world XY) or 'body' (body-local, CLoSD). Root velocity only.
    
    Returns:
        features_38d or features_50d: (T, 38) or (T, 50) numpy array
        If return_body_pos and full format: (features, body_pos_w_aligned).
    """
    fps = int(fps) if fps is not None else ROBOT_FPS
    root_vel_frame = (root_vel_frame or 'world').lower()
    if root_vel_frame not in ('world', 'body'):
        root_vel_frame = 'world'
    joint_pos = npz_data['joint_pos']  # (T, 29)
    
    # --- Auto-detect data format ---
    has_full_body = 'body_pos_w' in npz_data and 'body_quat_w' in npz_data
    
    if has_full_body:
        body_pos_w = npz_data['body_pos_w'].copy()   # (T, 30, 3)
        body_quat_w = npz_data['body_quat_w'].copy()  # (T, 30, 4) wxyz
        T = body_pos_w.shape[0]
    else:
        root_pos = npz_data['root_pos'].copy()  # (T, 3)
        root_rot = npz_data['root_rot'].copy()  # (T, 4) wxyz
        T = root_pos.shape[0]
    
    # --- Step 1: Floor normalization (Z-up) ---
    if has_full_body:
        # Full format: subtract min Z across all bodies (feet touch ground ≈ 0)
        floor_height = body_pos_w[:, :, 2].min()
        body_pos_w[:, :, 2] -= floor_height
    # Simple format: simulation ground already at Z=0; subtracting root Z min
    # would destroy absolute hip height, so we skip this step.
    
    # --- Step 2: Root XY at origin ---
    if has_full_body:
        root_xy_init = body_pos_w[0, root_idx, :2].copy()
        body_pos_w[:, :, 0] -= root_xy_init[0]
        body_pos_w[:, :, 1] -= root_xy_init[1]
    else:
        root_xy_init = root_pos[0, :2].copy()
        root_pos[:, 0] -= root_xy_init[0]
        root_pos[:, 1] -= root_xy_init[1]
    
    # --- Step 3: Align facing direction to +X ---
    # Extract initial forward from root quaternion (local +X → world frame)
    root_quat_0 = (body_quat_w[0, root_idx, :] if has_full_body
                    else root_rot[0, :])  # (4,) wxyz
    
    local_forward = np.array([1.0, 0.0, 0.0])
    forward_world = _qrot_np(
        root_quat_0[np.newaxis, :], local_forward[np.newaxis, :]
    )[0]  # (3,)
    
    # Project to XY plane (only correct yaw, ignore pitch/roll)
    forward_xy = forward_world[:2].copy()
    forward_xy /= (np.linalg.norm(forward_xy) + 1e-8)
    
    # Compute Z-axis rotation angle: from current forward to +X
    target_xy = np.array([1.0, 0.0])
    cos_angle = np.dot(forward_xy, target_xy)
    # 2D cross: forward × target  (sin of angle from forward to target)
    sin_angle = forward_xy[0] * target_xy[1] - forward_xy[1] * target_xy[0]
    angle = np.arctan2(sin_angle, cos_angle)
    
    # Build pure Z-axis rotation quaternion: [cos(θ/2), 0, 0, sin(θ/2)]
    half = angle / 2.0
    align_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])  # (4,)
    align_quat_T = np.repeat(align_quat[np.newaxis, :], T, axis=0)  # (T, 4)
    
    # Apply alignment rotation
    if has_full_body:
        # Rotate all body positions  (T*30, 3)
        n_bodies = body_pos_w.shape[1]
        pos_flat = body_pos_w.reshape(-1, 3)                          # (T*n_bodies, 3)
        q_flat = np.repeat(align_quat_T, n_bodies, axis=0)            # (T*n_bodies, 4)
        pos_flat = _qrot_np(q_flat, pos_flat)
        body_pos_w = pos_flat.reshape(T, n_bodies, 3)
        
        # Rotate root quaternions
        root_quat_orig = body_quat_w[:, root_idx, :]                  # (T, 4)
        root_quat_aligned = _qmul_np(align_quat_T, root_quat_orig)    # (T, 4)
        
        # Aligned root position
        root_pos_aligned = body_pos_w[:, root_idx, :]                 # (T, 3)
    else:
        # Rotate root positions
        root_pos = _qrot_np(align_quat_T, root_pos)                    # (T, 3)
        
        # Rotate root quaternions
        root_quat_aligned = _qmul_np(align_quat_T, root_rot)           # (T, 4)
        
        root_pos_aligned = root_pos
    
    # --- Step 4: Root velocity (m/s). World: aligned XY; body: body-local XY (CLoSD). ---
    root_vel_xy = np.zeros((T, 2), dtype=np.float32)
    if T > 1:
        root_vel_xy[1:] = (root_pos_aligned[1:, :2] - root_pos_aligned[:-1, :2]) * fps
    if root_vel_frame == 'body':
        root_vel_xy = _world_to_local_xy_np(root_quat_aligned, root_vel_xy)
    root_z = root_pos_aligned[:, 2:3]  # (T, 1)
    
    # Convert root quaternion to 6D representation
    root_rot_6d = quat_wxyz_to_6d(root_quat_aligned)  # (T, 6)
    
    # --- Concatenate to 38D features ---
    features_38d = np.concatenate([
        joint_pos,      # (T, 29)
        root_vel_xy,    # (T, 2)
        root_z,         # (T, 1)
        root_rot_6d     # (T, 6)
    ], axis=1)  # (T, 38)
    
    if return_53d and has_full_body:
        from utils.robot_target_cond import ROBOT_BODY_IDX
        nb = body_pos_w.shape[1]
        root_quat = root_quat_aligned  # (T, 4) wxyz
        ee_15 = []
        for name in EE_NAMES_50:
            bid = ROBOT_BODY_IDX.get(name)
            if bid is None or bid >= nb:
                ee_15.append(np.zeros((T, 3), dtype=np.float32))
            else:
                ee_pos = body_pos_w[:, bid, :]  # (T, 3) world xyz
                ee_vel_world_xy = np.zeros((T, 2), dtype=np.float32)
                if T > 1:
                    ee_vel_world_xy[1:] = (ee_pos[1:, :2] - ee_pos[:-1, :2]) * fps
                ee_vel_local_xy = _world_to_local_xy_np(root_quat, ee_vel_world_xy)  # (T, 2) m/s
                ee_z_abs = ee_pos[:, 2:3].astype(np.float32)  # (T, 1)
                ee_feat = np.concatenate([ee_vel_local_xy, ee_z_abs], axis=1)  # (T, 3)
                ee_15.append(ee_feat)
        ee_12d = np.concatenate(ee_15, axis=1)  # (T, 12)
        features_53d = np.concatenate([features_38d, ee_12d], axis=1).astype(np.float32)  # (T, 50)
        if return_body_pos:
            return features_53d, body_pos_w.astype(np.float32)
        return features_53d
    
    if return_body_pos and has_full_body:
        return features_38d, body_pos_w.astype(np.float32)
    return features_38d


# Keep backward-compatible alias (deprecated)
process_robot_npz_simple = process_robot_npz


def _local_to_world_xy_np(root_quat_wxyz, vec_xy):
    """Rotate body-local XY vector(s) to world XY. root_quat (*,4) wxyz, vec_xy (*,2). Returns (*,2)."""
    v3 = np.zeros(vec_xy.shape[:-1] + (3,), dtype=np.float64)
    v3[..., :2] = vec_xy
    out = _qrot_np(root_quat_wxyz, v3)
    return out[..., :2].astype(np.float32)


def recover_root_xy_from_velocity(root_vel_xy, root_rot_quat, fps=None, vel_frame='world'):
    """
    Recover global XY position from velocity (m/s): P[t] = sum_s vel_world[s]*dt, dt=1/fps.
    
    Args:
        root_vel_xy: (T, 2) - velocity (m/s), in aligned world XY if vel_frame='world', else body-local.
        root_rot_quat: (T, 4) - root rotation quaternions (wxyz). Used when vel_frame='body' to rotate to world.
        fps: int - frame rate (default: ROBOT_FPS=50). dt = 1/fps.
        vel_frame: str - 'world' (root_vel_xy already in aligned world) or 'body' (rotate to world each frame).
    
    Returns:
        root_xy_pos: (T, 2) - global XY positions (starts at origin)
    """
    fps = int(fps) if fps is not None else ROBOT_FPS
    dt = 1.0 / fps
    T = root_vel_xy.shape[0]
    if (vel_frame or 'world').lower() == 'body':
        vel_world = _local_to_world_xy_np(root_rot_quat, root_vel_xy)
    else:
        vel_world = root_vel_xy
    root_xy_pos = np.zeros((T, 2), dtype=np.float32)
    root_xy_pos[0] = np.array([0, 0], dtype=np.float32)
    for t in range(1, T):
        root_xy_pos[t] = root_xy_pos[t - 1] + vel_world[t] * dt
    return root_xy_pos

# def recover_root_xy_from_velocity(root_vel_xy, root_rot_quat):
if __name__ == "__main__":
    # Test the processing pipeline
    print("Testing robot data processing...")
    
    # Load a sample NPZ file
    import sys
    sys.path.insert(0, '..')
    
    npz_path = '../robot_humanml_data/npz/000000.npz'
    npz_data = np.load(npz_path)
    
    print(f"Loaded NPZ: {npz_path}")
    print(f"  joint_pos: {npz_data['joint_pos'].shape}")
    print(f"  body_pos_w: {npz_data['body_pos_w'].shape}")
    print(f"  body_quat_w: {npz_data['body_quat_w'].shape}")
    print(f"  body_lin_vel_w: {npz_data['body_lin_vel_w'].shape}")
    
    # Process the data
    features_38d = process_robot_npz(npz_data)
    
    print(f"\nProcessed features shape: {features_38d.shape}")
    print(f"Expected shape: (T, 38)")
    
    # Test velocity integration
    root_vel_xy = features_38d[:, 29:31]
    root_rot_6d = features_38d[:, 32:38]
    
    from utils.rotation_utils import rot6d_to_quat_wxyz
    root_rot_quat = rot6d_to_quat_wxyz(root_rot_6d)
    
    root_xy_recovered = recover_root_xy_from_velocity(root_vel_xy, root_rot_quat)
    
    print(f"\nRecovered root XY positions shape: {root_xy_recovered.shape}")
    print(f"Start position: {root_xy_recovered[0]}")
    print(f"End position: {root_xy_recovered[-1]}")
    
    print("\n✓ Test completed!")
    