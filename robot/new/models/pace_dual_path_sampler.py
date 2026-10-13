"""
PACE++ Dual-Path Flow Matching Sampler.

ODE integration where each step combines:
  v_text = frozen UNet(x_t, t, text)
  v_goal = DSSA Residual Adapter(x_t, t, text_emb, goals)
  v_pred = v_text + α(t) · v_goal

With optional 7-slot geometric guidance for hard constraint satisfaction.
"""
import torch
import torch.nn.functional as F
import math
import numpy as np


def compute_alpha(t_frac, tau=0.25, sigma=0.05):
    """Time-gating: α(t) = sigmoid((t - τ) / σ)"""
    return 1.0 / (1.0 + math.exp(-(t_frac - tau) / sigma))


class DualPathFlowSampler:
    """
    PACE++ dual-path ODE sampler.

    At each step:
      1. Get v_text from frozen UNet
      2. Get v_goal from DSSA Residual Adapter
      3. Blend: v = v_text + α(t) · v_goal
      4. Optional: geometric guidance for active slots
      5. ODE step: x_{t+dt} = x_t + dt · v
    """

    def __init__(self, num_steps=20, method='midpoint',
                 guidance_scale=150.0, guidance_start=0.5,
                 alpha_tau=0.25, alpha_sigma=0.05):
        self.num_steps = num_steps
        self.method = method
        self.guidance_scale = guidance_scale
        self.guidance_start = guidance_start
        self.alpha_tau = alpha_tau
        self.alpha_sigma = alpha_sigma

    @torch.no_grad()
    def sample(self, unet_frozen, adapter, fk_layer, shape, device,
               dtype=torch.float32, text=None, enc_text=None,
               goal_value=None, goal_dim_mask=None, goal_time_mask=None,
               goal_slot_mask=None, priority=None,
               norm_mean=None, norm_std=None, return_trajectory=False):
        """
        Args:
            unet_frozen: frozen fm_base_v2 UNet (text path)
            adapter: DSSA Residual Adapter (goal path, trainable)
            fk_layer: differentiable FK for geometric guidance
            shape: (B, T, 38) output shape
            text: raw text string (for UNet text encoding)
            enc_text: pre-encoded text embedding (B, 77, 256)
            goal_*: 7-slot goal tensors
            priority: (B, 7) optional slot priorities
            norm_mean, norm_std: normalization stats for denorm (guidance)
            return_trajectory: if True, return full trajectory + final

        Returns:
            x: (B, T, 38) generated motion (normalized)
        """
        B, T, D = shape
        dt = 1.0 / self.num_steps
        x = torch.randn(shape, device=device, dtype=dtype)

        # Pre-encode text for UNet (done once)
        if enc_text is None and text is not None:
            enc_text = unet_frozen.encode_text(text, device)

        has_goals = goal_value is not None and goal_slot_mask is not None and goal_slot_mask.any()
        active_slots = goal_slot_mask[0].nonzero(as_tuple=True)[0].tolist() if has_goals else []

        for i in range(self.num_steps):
            t_frac = i * dt
            t_tensor = torch.full((B,), t_frac, device=device, dtype=dtype)
            t_int = (t_tensor * 999).long()

            # 1. Text path: frozen UNet velocity prediction
            v_text = unet_frozen(x, t_int, enc_text=enc_text)
            if isinstance(v_text, tuple):
                v_text = v_text[0]

            # 2. Goal path: DSSA Residual Adapter
            alpha = compute_alpha(t_frac, self.alpha_tau, self.alpha_sigma)
            if has_goals and alpha > 0.01:
                v_goal = adapter(x, enc_text, goal_value, goal_dim_mask,
                                goal_time_mask, goal_slot_mask, priority)
                v_pred = v_text + alpha * v_goal
            else:
                v_pred = v_text

            # 3. Geometric guidance (7-slot FK correction)
            if has_goals and self.guidance_scale > 0 and t_frac >= self.guidance_start:
                v_pred = self._apply_guidance(
                    x, t_frac, v_pred, goal_value, goal_dim_mask,
                    goal_time_mask, goal_slot_mask, fk_layer,
                    norm_mean, norm_std, active_slots)

            # 4. ODE step
            if self.method == 'midpoint':
                x_mid = x + 0.5 * dt * v_pred
                t_mid = t_frac + 0.5 * dt
                t_mid_int = (torch.full((B,), t_mid, device=device, dtype=dtype) * 999).long()

                v_text2 = unet_frozen(x_mid, t_mid_int, enc_text=enc_text)
                if isinstance(v_text2, tuple): v_text2 = v_text2[0]

                alpha_mid = compute_alpha(t_mid, self.alpha_tau, self.alpha_sigma)
                if has_goals and alpha_mid > 0.01:
                    v_goal2 = adapter(x_mid, enc_text, goal_value, goal_dim_mask,
                                     goal_time_mask, goal_slot_mask, priority)
                    v_pred2 = v_text2 + alpha_mid * v_goal2
                else:
                    v_pred2 = v_text2

                if has_goals and self.guidance_scale > 0 and t_mid >= self.guidance_start:
                    v_pred2 = self._apply_guidance(
                        x_mid, t_mid, v_pred2, goal_value, goal_dim_mask,
                        goal_time_mask, goal_slot_mask, fk_layer,
                        norm_mean, norm_std, active_slots)

                x = x + dt * v_pred2
            else:
                x = x + dt * v_pred

        return x

    def _apply_guidance(self, x_t, t_frac, v_pred, goal_value, goal_dim_mask,
                        goal_time_mask, goal_slot_mask, fk_layer,
                        norm_mean, norm_std, active_slots):
        """Apply 7-slot FK geometric guidance with quality gate."""
        if norm_mean is None or norm_std is None:
            return v_pred

        B, T, D = x_t.shape
        device = x_t.device
        dtype = x_t.dtype

        with torch.enable_grad():
            x_work = x_t.detach().clone().requires_grad_(True)

            # Predict clean endpoint
            t_flow = torch.full((B,), t_frac, device=device, dtype=dtype)
            t_expanded = t_flow.view(B, 1, 1)
            x_hat_1 = x_work + (1.0 - t_expanded) * v_pred.detach()

            # Denormalize for FK
            x_hat_denorm = x_hat_1 * norm_std + norm_mean

            # FK computation
            joints = torch.clamp(x_hat_denorm[:, :, :29], -3.0, 3.0)
            phys = torch.cat([joints, x_hat_denorm[:, :, 29:]], dim=-1)
            fk_out = fk_layer(phys)
            kp = fk_out['keypoint_pos']  # (B, T, 5, 3)
            rp = fk_out['root_pos']       # (B, T, 3)
            rr = fk_out['root_rot']       # (B, T, 3, 3)

            # Build FK predictions for all slots
            pred = torch.zeros(B, T, 7, 4, device=device, dtype=dtype)
            for slot_idx in range(5):  # pelvis, lf, rf, lw, rw
                pred[:, :, slot_idx, :3] = kp[:, :, slot_idx]
            pred[:, :, 5, :2] = rp[:, :, :2]  # traj XY
            pred[:, :, 6, 0] = torch.cos(torch.atan2(rr[:, :, 1, 0], rr[:, :, 0, 0]))
            pred[:, :, 6, 1] = torch.sin(torch.atan2(rr[:, :, 1, 0], rr[:, :, 0, 0]))

            # Per-slot FK loss at last frame
            last_idx = T - 1
            batch_idx = torch.arange(B, device=device)
            loss = torch.tensor(0.0, device=device)
            n_active = 0

            for s in active_slots:
                if goal_slot_mask[0, s]:
                    last_pred = pred[batch_idx, last_idx, s]
                    last_goal = goal_value[batch_idx, last_idx, s]
                    last_mask = goal_dim_mask[batch_idx, last_idx, s].float()
                    err = ((last_pred - last_goal) * last_mask).pow(2).sum(dim=1)
                    loss = loss + err.mean()
                    n_active += 1

            if n_active == 0:
                return v_pred

            loss = loss / n_active

            # Quality gate: check jerk before applying
            jerk_before = (x_hat_denorm[:, 2:, :29] - 2 * x_hat_denorm[:, 1:-1, :29]
                          + x_hat_denorm[:, :-2, :29]).pow(2).mean()

            # Gradient
            grad = torch.autograd.grad(loss, x_work)[0]

            # Channel-grouped weighting
            ch_w = torch.ones(D, device=device)
            ch_w[0:29] = 0.12    # joints
            ch_w[29:31] = 1.5    # root vel
            ch_w[31:32] = 0.4    # root Z
            ch_w[32:38] = 1.5    # root rot
            grad = grad * ch_w.view(1, 1, -1)

            # Normalize gradient
            grad_flat = grad.reshape(B, -1)
            grad_norms = grad_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
            grad = (grad_flat / grad_norms).view_as(grad)

            # Apply guidance with time ramp
            tau_ramp = math.sqrt(t_frac)
            scale = self.guidance_scale * tau_ramp * 0.01
            x_guided = x_work - scale * grad
            v_correction = (x_guided - x_work) / (1.0 / self.num_steps)

            # Quality gate: check jerk after guidance
            x_guided_denorm = x_guided * norm_std + norm_mean
            jerk_after = (x_guided_denorm[:, 2:, :29] - 2 * x_guided_denorm[:, 1:-1, :29]
                         + x_guided_denorm[:, :-2, :29]).pow(2).mean()

            if jerk_after > jerk_before * 1.5:
                v_correction = v_correction * 0.3  # Reduce if jerk increases too much

            return v_pred + v_correction
