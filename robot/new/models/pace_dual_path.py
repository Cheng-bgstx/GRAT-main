"""
PACE++ Dual-Path Flow Generator: Frozen UNet + DSSA Residual Adapter.

Architecture:
  x_t ──┬──→ [fm_base_v2 UNet, FROZEN] ──→ v_text (text semantics)
        └──→ [DSSA Residual Adapter, TRAINABLE] ──→ v_goal (spatial control)

  v_pred = v_text + α(τ) · m_active · v_goal

  Key properties:
  - DSSA adapter is zero-initialized → training starts at pure fm_base_v2 quality
  - Slot-gated DSSA: only active slots influence motion features
  - α(τ) time gating: early denoising → pure text, late denoising → spatial control
  - Non-active body preservation: inactive slots' FK positions are constrained
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def zero_init_(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class WeightedSlotSum(nn.Module):
    """Learned weighted sum over active goal slots."""
    def __init__(self, n_slots=7):
        super().__init__()
        self.weight_logits = nn.Parameter(torch.zeros(n_slots))

    def forward(self, slot_features, slot_mask):
        B, S, D = slot_features.shape
        logits = self.weight_logits.view(1, S).expand(B, S)
        logits = logits.masked_fill(~slot_mask, -1e4)
        weights = F.softmax(logits, dim=-1)
        weights = weights * slot_mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.einsum('bs,bsd->bd', weights, slot_features)


class GoalEncoder(nn.Module):
    """Per-slot MLP + WeightedSum + FiLM + Slot tokens."""
    def __init__(self, n_slots=7, slot_dim=4, latent_dim=256):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.latent_dim = latent_dim

        slot_feat_dim = max(1, latent_dim // n_slots)
        self.slot_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(slot_dim, slot_feat_dim), nn.SiLU(),
                         nn.Linear(slot_feat_dim, slot_feat_dim))
            for _ in range(n_slots)
        ])
        self.slot_pool = WeightedSlotSum(n_slots)
        self.aggregate_proj = nn.Linear(slot_feat_dim, latent_dim)
        nn.init.normal_(self.aggregate_proj.weight, std=0.02)
        nn.init.zeros_(self.aggregate_proj.bias)

        self.film_proj = nn.Linear(latent_dim, latent_dim * 2)
        zero_init_(self.film_proj)

        self.slot_token_proj = nn.Linear(slot_feat_dim, latent_dim)
        nn.init.normal_(self.slot_token_proj.weight, std=0.02)
        nn.init.zeros_(self.slot_token_proj.bias)

    def forward(self, goal_value, goal_dim_mask, goal_time_mask, goal_slot_mask):
        B, T = goal_value.shape[:2]
        S, Ld = self.n_slots, self.latent_dim
        D = self.slot_dim

        # Find last active frame per slot
        t_idx = torch.arange(T, device=goal_value.device).view(1, T, 1).expand(B, T, S)
        t_idx = t_idx * goal_time_mask.float()
        t_idx = t_idx + (~goal_time_mask).float() * (-1)
        last_t = t_idx.max(dim=1).values.clamp(min=0).long()

        # Gather goal at last_t
        goal_flat = goal_value.permute(0, 2, 1, 3).reshape(B * S, T, D)
        last_t_flat = last_t.reshape(B * S, 1, 1).expand(B * S, 1, D)
        last_gv_flat = torch.gather(goal_flat, 1, last_t_flat)
        last_gv = last_gv_flat.squeeze(1).view(B, S, D)
        last_gv = last_gv * goal_slot_mask.float().unsqueeze(-1)

        # Per-slot encoding
        feat_dim = max(1, Ld // S)
        slot_feats = torch.zeros(B, S, feat_dim, device=goal_value.device, dtype=goal_value.dtype)
        for s in range(S):
            active = goal_slot_mask[:, s]
            if active.any():
                slot_feats[active, s] = self.slot_mlps[s](last_gv[active, s])

        pooled = self.slot_pool(slot_feats, goal_slot_mask)
        pooled = self.aggregate_proj(pooled)

        film = self.film_proj(pooled)
        scale, shift = film.chunk(2, dim=-1)
        scale = scale.unsqueeze(1); shift = shift.unsqueeze(1)

        slot_tokens = torch.zeros(B, S, Ld, device=goal_value.device, dtype=goal_value.dtype)
        for s in range(S):
            active = goal_slot_mask[:, s]
            if active.any():
                slot_tokens[active, s] = self.slot_token_proj(slot_feats[active, s])

        return scale, shift, slot_tokens, goal_slot_mask


class SlotGatedCrossAttention(nn.Module):
    """DSSA: per-slot cross-attention gated by active_mask * priority."""
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = dim // n_heads
        self.norm_q = nn.LayerNorm(dim, eps=1e-4)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-4)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.zeros(1))  # zero-gate
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, slot_tokens, slot_mask, priority=None):
        B, D, T_x = x.shape
        S = slot_tokens.shape[1]
        H = self.n_heads

        x_t = x.permute(0, 2, 1)
        Q = self.q_proj(self.norm_q(x_t)).view(B, T_x, H, self.d_k).transpose(1, 2)
        K = self.k_proj(self.norm_kv(slot_tokens)).view(B, S, H, self.d_k).transpose(1, 2)
        V = self.v_proj(self.norm_kv(slot_tokens)).view(B, S, H, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = torch.clamp(scores, min=-50.0, max=50.0)

        inactive = ~slot_mask
        if inactive.any():
            scores = scores.masked_fill(inactive.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Priority weighting on attention scores: broadcast over H and T
        if priority is not None:
            pri_w = priority.float().view(B, 1, 1, S)
            scores = scores + torch.log(pri_w.clamp(min=1e-8))

        attn = F.softmax(scores, dim=-1)
        attn = attn.nan_to_num(nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T_x, D)
        out = self.out_proj(out)
        x_t = x_t + self.alpha * out
        return x_t.permute(0, 2, 1)


class FiLMConv1DBlock(nn.Module):
    """Conv1D → FiLM(goal) → GroupNorm → SiLU → Conv1D → GroupNorm + residual."""
    def __init__(self, dim, kernel_size=5, dilation=1, dropout=0.05):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, dim), dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size,
                               padding=(kernel_size // 2) * dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(min(8, dim), dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.conv1.weight, std=0.02); nn.init.zeros_(self.conv1.bias)
        nn.init.normal_(self.conv2.weight, std=0.02); nn.init.zeros_(self.conv2.bias)

    def forward(self, x, scale, shift):
        r = x
        h = self.conv1(F.silu(self.norm1(x)))
        h = h * (1.0 + scale.transpose(1, 2)) + shift.transpose(1, 2)
        h = self.dropout(h)
        h = self.conv2(F.silu(self.norm2(h)))
        return r + h


class TextCrossAttention(nn.Module):
    """Cross-attention from motion features to text embeddings."""
    def __init__(self, motion_dim, text_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = motion_dim // n_heads
        self.norm_q = nn.LayerNorm(motion_dim, eps=1e-4)
        self.norm_kv = nn.LayerNorm(text_dim, eps=1e-4)
        self.q_proj = nn.Linear(motion_dim, motion_dim)
        self.k_proj = nn.Linear(text_dim, motion_dim)
        self.v_proj = nn.Linear(text_dim, motion_dim)
        self.out_proj = nn.Linear(motion_dim, motion_dim)
        self.dropout = nn.Dropout(dropout)
        self.alpha_txt = nn.Parameter(torch.zeros(1))  # zero-gate
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, text_emb):
        B, D, T = x.shape
        H = self.n_heads
        x_t = x.permute(0, 2, 1)
        Q = self.q_proj(self.norm_q(x_t)).view(B, T, H, self.d_k).transpose(1, 2)
        K = self.k_proj(self.norm_kv(text_emb)).view(B, text_emb.shape[1], H, self.d_k).transpose(1, 2)
        V = self.v_proj(self.norm_kv(text_emb)).view(B, text_emb.shape[1], H, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        x_t = x_t + self.alpha_txt * out
        return x_t.permute(0, 2, 1)


class DSSAResidualAdapter(nn.Module):
    """
    PACE++ DSSA Residual Adapter.

    Predicts velocity correction v_goal from noisy state x_t, text features,
    and 7-slot goal targets. Operates INSIDE the ODE loop.

    Zero-initialized: training starts at pure fm_base_v2 quality.
    """
    def __init__(self, motion_dim=38, hidden_dim=256, text_dim=256,
                 n_slots=7, slot_dim=4, n_blocks=4, kernel_size=5, dropout=0.05):
        super().__init__()
        self.n_slots = n_slots

        self.goal_encoder = GoalEncoder(n_slots=n_slots, slot_dim=slot_dim, latent_dim=hidden_dim)
        self.motion_proj = nn.Conv1d(motion_dim, hidden_dim, 1)
        nn.init.normal_(self.motion_proj.weight, std=0.02)
        nn.init.zeros_(self.motion_proj.bias)

        self.text_attn = TextCrossAttention(hidden_dim, text_dim, n_heads=4, dropout=dropout)

        dilations = [1, 1, 2, 2][:n_blocks]
        self.blocks = nn.ModuleList([
            FiLMConv1DBlock(hidden_dim, kernel_size=kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])
        self.dssa_layers = nn.ModuleList([
            SlotGatedCrossAttention(hidden_dim, n_heads=4, dropout=dropout)
            for _ in range(n_blocks)
        ])

        self.out_norm = nn.GroupNorm(min(8, hidden_dim), hidden_dim)
        self.out_proj = nn.Conv1d(hidden_dim, motion_dim, 1)
        zero_init_(self.out_proj)

        self.hidden_dim = hidden_dim

    def forward(self, x_t, text_emb, goal_value, goal_dim_mask,
                goal_time_mask, goal_slot_mask, priority=None):
        """
        Args:
            x_t:             (B, T, 38) noisy state
            text_emb:        (B, 77, text_dim) frozen text features
            goal_value:      (B, T, 7, 4) goal values
            goal_dim_mask:   (B, T, 7, 4) active dimensions
            goal_time_mask:  (B, T, 7) active timesteps
            goal_slot_mask:  (B, 7) active slots
            priority:        (B, 7) optional per-slot priority weights

        Returns:
            v_goal: (B, T, 38) velocity correction
        """
        B, T, D = x_t.shape

        scale, shift, slot_tokens, slot_mask = self.goal_encoder(
            goal_value, goal_dim_mask, goal_time_mask, goal_slot_mask)

        x = x_t.permute(0, 2, 1)
        x = self.motion_proj(x)

        # Text context injection
        x = self.text_attn(x, text_emb)

        # FiLM + Slot-gated DSSA at each block
        for blk, dssa in zip(self.blocks, self.dssa_layers):
            x = blk(x, scale, shift)
            x = dssa(x, slot_tokens, slot_mask, priority)

        x = self.out_norm(x)
        v_goal = self.out_proj(x)  # (B, D, T)
        v_goal = v_goal.permute(0, 2, 1)  # (B, T, D)

        return v_goal

    def get_default_priority(self, slot_mask, device='cpu'):
        """Default priority: traj=1.0, heading=0.8, pelvis=0.7, EE=0.5"""
        B = slot_mask.shape[0]
        priority = torch.zeros(B, self.n_slots, device=device)
        default = torch.tensor([0.7, 0.5, 0.5, 0.5, 0.5, 1.0, 0.8], device=device)
        priority[:] = default.view(1, -1)
        return priority * slot_mask.float()
