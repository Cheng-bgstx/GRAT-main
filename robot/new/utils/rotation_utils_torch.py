"""Differentiable PyTorch rotation helpers for goal loss (rot6d -> quat -> yaw)."""
import torch
import torch.nn as nn


def rot6d_to_rotation_matrix_torch(rot6d):
    """(B, 6) -> (B, 3, 3), Gram-Schmidt."""
    a1, a2 = rot6d[..., :3], rot6d[..., 3:6]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rotation_matrix_to_quat_wxyz_torch(R):
    """(B, 3, 3) -> (B, 4) wxyz, differentiable."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    w = (1 + trace).clamp(min=1e-8).sqrt() * 0.5
    x = (R[..., 2, 1] - R[..., 1, 2]) / (4 * w + 1e-8)
    y = (R[..., 0, 2] - R[..., 2, 0]) / (4 * w + 1e-8)
    z = (R[..., 1, 0] - R[..., 0, 1]) / (4 * w + 1e-8)
    quat = torch.stack([w, x, y, z], dim=-1)
    return quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)


def rot6d_to_quat_wxyz_torch(rot6d):
    """(B, 6) -> (B, 4) wxyz."""
    R = rot6d_to_rotation_matrix_torch(rot6d)
    return rotation_matrix_to_quat_wxyz_torch(R)


def quat_wxyz_to_yaw_torch(quat):
    """(B, 4) wxyz -> (B,) yaw in radians, Z-up."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


# FPS for velocity→position integration (m/s * dt). Match robot_process.ROBOT_FPS.
ROBOT_FPS = 50


class _TimeWeightedCumsum(torch.autograd.Function):
    """
    Cumsum with time-weighted backward: dL/d(vel_t) scaled by (1 + t/T).
    Forward identical to cumsum; backward gives later frames larger gradient.
    """

    @staticmethod
    def forward(ctx, x):
        return x.cumsum(dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        _, T, _ = grad_output.shape
        device = grad_output.device
        # Standard cumsum backward: grad_input = flip(cumsum(flip(grad_output)))
        grad_rev = torch.flip(grad_output, dims=[1])
        grad_cumrev = grad_rev.cumsum(dim=1)
        grad_standard = torch.flip(grad_cumrev, dims=[1])
        # Time weight: (1 + t/T) for t in [0, T-1]
        t_idx = torch.arange(T, device=device, dtype=grad_output.dtype).view(1, T, 1)
        weight = 1.0 + t_idx / max(T, 1)
        return grad_standard * weight


def prediction_to_goal_xy_heading_torch(
    prediction,
    src_mask,
    cur_len,
    root_vel_frame='world',
    time_weighted_gradient=False,
):
    """
    Differentiable: (B, T, 38) + (B, T) mask + (B,) lengths -> (B, 3) [root_x, root_y, heading].
    root_vel_xy is velocity (m/s). world: vel*dt cumsum; body: R@vel then dt cumsum (CLoSD).
    time_weighted_gradient: if True, scale dL/d(vel_t) by (1+t/T) so later frames get more
        gradient; mitigates cumsum gradient smearing for recon guidance.
    """
    B, T, _ = prediction.shape
    device = prediction.device
    dt = 1.0 / ROBOT_FPS
    vel_xy = prediction[:, :, 29:31] * src_mask.unsqueeze(-1)
    vel_xy = vel_xy.clone()
    vel_xy[:, 0, :] = 0.0
    if (root_vel_frame or 'world').lower() == 'body':
        root_rot6d = prediction[:, :, 32:38]
        R = rot6d_to_rotation_matrix_torch(root_rot6d.reshape(B * T, 6)).reshape(B, T, 3, 3)
        vel_3d = torch.zeros(B, T, 3, device=device, dtype=vel_xy.dtype)
        vel_3d[:, :, :2] = vel_xy
        vel_world = torch.einsum('btij,btj->bti', R, vel_3d)[:, :, :2]
        vel_xy = vel_world
    vel_xy = vel_xy * dt
    if time_weighted_gradient:
        root_xy = _TimeWeightedCumsum.apply(vel_xy)
    else:
        root_xy = vel_xy.cumsum(dim=1)
    last_idx = (cur_len - 1).clamp(0, T - 1)
    pred_root_xy = root_xy[torch.arange(B, device=device), last_idx, :]
    last_rot6d = prediction[torch.arange(B, device=device), last_idx, 32:38]
    quat = rot6d_to_quat_wxyz_torch(last_rot6d)
    heading = quat_wxyz_to_yaw_torch(quat)
    return torch.cat([pred_root_xy, heading.unsqueeze(-1)], dim=-1)
