"""D2TA / AD-CODD -- Domain-Task Drift-Decoupled Transfer Attack (shared utilities).

Theory
------
We do **not** propose a new noise or a new frequency modulation.  We take a
standard transferable optimizer (MI-FGSM over the image-encoder feature
disruption objective) as the *base update* and only add a mechanism on top of
it.  That mechanism, not the base optimizer, is the contribution.  The single
claim is:

    The victim model is  f_m = f_{theta0 + dTheta_d + dTheta_t}  where
    dTheta_d / dTheta_t are the (unknown) domain- and task-adaptation drifts.
    A first-order Taylor expansion of the attack gain gives

        G_m(delta) = G_0(delta) + < grad_theta G_0, dTheta_d + dTheta_t > + ...

    so the transfer gap is dominated by the component of the update direction
    that is *aligned with the adaptation drift*.  Transfer therefore improves
    if we (a) mine an offline "adaptation-drift basis" B_drift from proxy
    medical data, and (b) softly decouple that drift-aligned component from the
    base update while keeping the base direction consistent.

Everything in this file is shared by the offline (prior mining) and online
(instance attack) stages.  The two driver scripts are:

    ddta/offline_drift_prior.py      -- learns B_drift (+ w_share, w_drift)
    ddta/online_ddta_attack.py       -- runs the drift-decoupled attack
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# make `segment_anything` / `datasets` importable regardless of the cwd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "newest")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Small tensor helpers
# ---------------------------------------------------------------------------

def normalize_l1(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L1-normalize a [1, C, H, W] direction (returns same shape)."""
    return x / (x.abs().sum() + eps)


def _flatten(x: torch.Tensor) -> torch.Tensor:
    """[1, C, H, W] -> [1, C*H*W]."""
    return x.reshape(1, -1)


# ---------------------------------------------------------------------------
# Adaptation drift prior bundle
# ---------------------------------------------------------------------------

@dataclass
class DriftBundle:
    """Offline-mined adaptation-drift prior.

    basis    : [K, D]  orthonormal rows spanning the adaptation-drift subspace.
    shape    : [1, C, H, W]  gradient shape the basis is defined on.
    w_share  : scalar, E ||g_bar||      (how aligned the views are).
    w_drift  : scalar, E_v ||r_v||      (how large the drift residuals are).
    """
    basis: torch.Tensor
    shape: Tuple[int, int, int, int]
    w_share: float
    w_drift: float
    singular_values: torch.Tensor
    meta: Dict = field(default_factory=dict)

    def to(self, device: torch.device) -> "DriftBundle":
        self.basis = self.basis.to(device)
        self.singular_values = self.singular_values.to(device)
        return self

    def save(self, path: str) -> None:
        torch.save({
            "basis": self.basis.detach().cpu(),
            "shape": list(self.shape),
            "w_share": float(self.w_share),
            "w_drift": float(self.w_drift),
            "singular_values": self.singular_values.detach().cpu(),
            "meta": self.meta,
        }, path)

    @staticmethod
    def load(path: str, device: Optional[torch.device] = None) -> "DriftBundle":
        d = torch.load(path, map_location="cpu")
        bundle = DriftBundle(
            basis=d["basis"],
            shape=tuple(d["shape"]),
            w_share=float(d.get("w_share", 0.0)),
            w_drift=float(d.get("w_drift", 0.0)),
            singular_values=d.get("singular_values", torch.tensor([])),
            meta=d.get("meta", {}),
        )
        if device is not None:
            bundle = bundle.to(device)
        return bundle


def fit_drift_basis(
    residuals: List[torch.Tensor],
    rank: int,
    shape: Tuple[int, int, int, int],
    w_share: float = 0.0,
    w_drift: float = 0.0,
    meta: Optional[Dict] = None,
) -> DriftBundle:
    """TopK-SVD over the per-view drift residuals {r_v}.

    residuals : list of [1,C,H,W] residual directions  r_v = g_v/||g_v||_1 - g_bar.
    """
    N = len(residuals)
    X = torch.stack([_flatten(r).squeeze(0) for r in residuals], dim=0).float()  # [N, D]
    X = X / (X.abs().sum(dim=1, keepdim=True) + 1e-12)

    q = max(1, min(rank, N))
    # center=True is a no-op for the drift residuals (per-image residuals sum to
    # zero), but kept for numerical correctness of the SVD.
    _, S, V = torch.pca_lowrank(X, q=q, center=True, niter=2)
    basis = V[:, :q].T.contiguous()  # [q, D]
    # orthonormalize rows (pca_lowrank rows are already unit norm, keep it explicit)
    basis = basis / (basis.norm(dim=1, keepdim=True) + 1e-12)

    explained = float((S[:q].square().sum() / (X.square().sum() + 1e-12)).item())
    return DriftBundle(
        basis=basis,
        shape=shape,
        w_share=w_share,
        w_drift=w_drift,
        singular_values=S[:q],
        meta=dict(meta or {}, num_residuals=N, rank=q, explained_variance=explained),
    )


# ---------------------------------------------------------------------------
# Online drift-decoupling operators
# ---------------------------------------------------------------------------

def project_to_drift(g: torch.Tensor, bundle: DriftBundle) -> torch.Tensor:
    """B_drift B_drift^T g  (the drift-aligned component of g)."""
    B = bundle.basis.to(g.device)          # [K, D]
    gflat = _flatten(g)                     # [1, D]
    coeff = gflat @ B.t()                   # [1, K]
    return (coeff @ B).reshape(bundle.shape)


def drift_correlation(g: torch.Tensor, bundle: DriftBundle) -> torch.Tensor:
    """||B_drift^T g||_2 -- how much of the update lies in the drift subspace."""
    return project_to_drift(g, bundle).norm()


def make_random_basis(rank: int, shape: Tuple[int, int, int, int],
                      device: torch.device, seed: int = 0) -> torch.Tensor:
    """Random (near-)orthonormal basis [rank, D] for the 'random basis' control.

    Controls whether the *learned* drift basis captures meaningful adaptation
    drift, or whether any random low-rank direction gives the same effect.
    """
    D = 1
    for s in shape:
        D *= s
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(rank, D, generator=g)  # CPU (generator is CPU-bound)
    B = B / (B.norm(dim=1, keepdim=True) + 1e-12)
    return B.to(device)


def soft_decouple_offline(g_base: torch.Tensor, bundle: DriftBundle, alpha: float) -> torch.Tensor:
    """Soft removal of the offline drift component:  g_off = g_base - alpha * g_drift."""
    g_drift = project_to_drift(g_base, bundle)
    return g_base - alpha * g_drift


def residual_decouple_online(
    g_off: torch.Tensor,
    residuals: List[torch.Tensor],
    beta: float,
    eta: float = 1e-6,
) -> torch.Tensor:
    """Instance-level residual decoupling.

    g_on = g_off - beta * sum_v  <g_off, r_v> / (||r_v||^2 + eta) * r_v
    """
    if not residuals:
        return g_off
    g_off_flat = _flatten(g_off)            # [1, D]
    out = g_off
    for r in residuals:
        r_flat = _flatten(r)                # [1, D]
        denom = (r_flat.square().sum() + eta)
        proj = (g_off_flat @ r_flat.t()) / denom  # [1,1]
        out = out - beta * proj * r
    return out


def consistency_gate(
    g_base: torch.Tensor,
    g_on: torch.Tensor,
    rho: float,
    tau: float,
    rho_min: float = 0.0,
    rho_decay: float = 0.5,
) -> Tuple[torch.Tensor, float]:
    """Backbone-consistent drift decoupling (residual-correction form).

    g_final = g_base + rho * (||g_base||_1 / (||g_on - g_base||_1 + eta)) * (g_on - g_base)

    i.e. we ADD the drift-decoupled *correction* (g_on - g_base) back onto the
    base direction, reweighted so the final direction stays aligned with the
    base.  If the cosine constraint is violated we back off rho (line search)
    instead of corrupting the base transferable direction.
    """
    gb = _flatten(g_base)
    correction = g_on - g_base
    scale = g_base.abs().sum() / (correction.abs().sum() + 1e-12)
    rho_cur = rho
    while rho_cur >= rho_min:
        g_final = g_base + rho_cur * scale * correction
        gf = _flatten(g_final)
        cos = float((gf @ gb.t()).squeeze(0) / (gf.norm() * gb.norm() + 1e-12))
        if cos >= tau:
            return g_final, rho_cur
        if rho_cur <= 0.0:
            break
        rho_cur *= rho_decay
    # fallback: pure base direction (always consistent with itself)
    return g_base, 0.0


# ---------------------------------------------------------------------------
# Adaptation views  (domain-drift = input transform, task-drift = model perturb)
# ---------------------------------------------------------------------------

def _gaussian_kernel(k: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return (g.view(1, 1, k, 1) * g.view(1, 1, 1, k))  # [1,1,k,k]


def gaussian_blur(x: torch.Tensor, k: int = 5, sigma: float = 1.0) -> torch.Tensor:
    kernel = _gaussian_kernel(k, sigma).to(x.device).repeat(x.shape[1], 1, 1, 1)
    pad = k // 2
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel, groups=x.shape[1])


# Fixed, deterministic strengths -> each view is a well-defined "adaptation mode".
_DOMAIN_PARAMS = {
    "intensity": dict(a=1.15, b=-0.03),     # CT window / intensity shift
    "gamma":     dict(gamma=1.4),           # gamma shift
    "contrast":  dict(alpha=1.3),           # contrast stretch around mean
    "blur":      dict(k=5, sigma=1.0),      # blur
    "sharpen":   dict(alpha=0.5),           # unsharp mask
    "stats":     dict(a=1.2, b=0.02),       # per-channel feature-statistics shift
}


def apply_domain_view(name: str, x: torch.Tensor) -> torch.Tensor:
    """Apply a differentiable domain-drift transform to a 0-1 image tensor."""
    # clamp first so fractional powers (gamma) / stats never see out-of-range values
    # (the online perturbed image can dip slightly below 0 between steps)
    x = torch.clamp(x, 0.0, 1.0)
    p = _DOMAIN_PARAMS[name]
    if name == "intensity":
        return torch.clamp(x * p["a"] + p["b"], 0.0, 1.0)
    if name == "gamma":
        return torch.clamp(x.pow(p["gamma"]), 0.0, 1.0)
    if name == "contrast":
        mu = x.mean(dim=(2, 3), keepdim=True)
        return torch.clamp((x - mu) * p["alpha"] + mu, 0.0, 1.0)
    if name == "blur":
        return torch.clamp(gaussian_blur(x, k=p["k"], sigma=p["sigma"]), 0.0, 1.0)
    if name == "sharpen":
        blur = gaussian_blur(x, k=5, sigma=1.0)
        return torch.clamp(x + p["alpha"] * (x - blur), 0.0, 1.0)
    if name == "stats":
        # channel-wise affine (mean / std shift of the feature statistics)
        mu = x.mean(dim=(2, 3), keepdim=True)
        sd = x.std(dim=(2, 3), keepdim=True) + 1e-6
        return torch.clamp((x - mu) / sd * p["a"] + mu + p["b"], 0.0, 1.0)
    raise ValueError(f"unknown domain view: {name}")


def make_views() -> List[Dict]:
    """Full view set used by the offline stage (1 source + 6 domain + 2 task)."""
    return [
        {"name": "source", "type": "source"},
        {"name": "intensity", "type": "domain"},
        {"name": "gamma", "type": "domain"},
        {"name": "contrast", "type": "domain"},
        {"name": "blur", "type": "domain"},
        {"name": "sharpen", "type": "domain"},
        {"name": "stats", "type": "domain"},
        {"name": "task_lora", "type": "task", "kind": "lora"},
        {"name": "task_adapter", "type": "task", "kind": "adapter"},
    ]


class _ResidualPerturb(nn.Module):
    """Low-rank hidden residual  U * phi(V * h)  used to simulate LoRA/adapter drift."""

    def __init__(self, dim: int, rank: int, scale: float):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.scale = scale
        for p in self.parameters():
            p.requires_grad = False

    def _init_(self, seed: int):
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.down.weight.copy_(torch.randn(self.down.weight.shape, generator=g) * (1.0 / self.down.in_features ** 0.5))
            self.up.weight.copy_(torch.randn(self.up.weight.shape, generator=g) * (1.0 / self.up.in_features ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(x))) * self.scale


class TaskViewInjector:
    """Context manager that injects a low-rank residual into the encoder.

    kind == 'lora'    -> inject on each block's attention output (simulates W+BA).
    kind == 'adapter' -> inject on each block's MLP output (simulates h + U phi(Vh)).
    """

    def __init__(self, image_encoder: nn.Module, kind: str, scale: float, rank: int, seed: int):
        self.enc = image_encoder
        self.kind = kind
        self.scale = scale
        self.rank = rank
        self.seed = seed
        self._hooks: List = []
        self._mods: List[_ResidualPerturb] = []

    def __enter__(self) -> "TaskViewInjector":
        dim = self.enc.blocks[0].attn.qkv.in_features
        dev = self.enc.blocks[0].attn.qkv.weight.device
        for i, blk in enumerate(self.enc.blocks):
            target = blk.attn if self.kind == "lora" else blk.mlp
            perturb = _ResidualPerturb(dim, self.rank, self.scale).to(dev)
            perturb._init_(self.seed + i)
            self._mods.append(perturb)

            def _hook(_module, inp, out, _p=perturb):
                return out + _p(inp[0])

            self._hooks.append(target.register_forward_hook(_hook))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._mods = []


# ---------------------------------------------------------------------------
# Encoder forward helpers (reproduce the existing attack's input convention)
# ---------------------------------------------------------------------------

def encoder_forward(
    image_encoder: nn.Module,
    x: torch.Tensor,
    is_synapse: bool,
    norm_transform: Optional[Callable] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Feed a 0-1 image tensor through the image encoder.

    Synapse slices are single-channel: we re-extract channel 0 and repeat x3
    (identical to the original SE-CGR / DIIA-CODD attack).
    ISIC / REFUGE are RGB: ImageNet-normalize before the encoder.
    """
    if is_synapse:
        x_in = x[:, 0:1].repeat(1, 3, 1, 1)
    else:
        x_in = norm_transform(x) if norm_transform is not None else x
    return image_encoder(x_in)


def forward_view(
    image_encoder: nn.Module,
    x: torch.Tensor,
    view: Dict,
    is_synapse: bool,
    norm_transform: Optional[Callable] = None,
    task_scale: float = 0.1,
    task_rank: int = 8,
    task_seed: int = 0,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Run x through `image_encoder` under a single adaptation view."""
    vtype = view["type"]
    if vtype == "source":
        return encoder_forward(image_encoder, x, is_synapse, norm_transform)
    if vtype == "domain":
        return encoder_forward(image_encoder, apply_domain_view(view["name"], x), is_synapse, norm_transform)
    if vtype == "task":
        # task-view residual must be the SAME between the clean reference and the
        # perturbed pass; the seed therefore lives in the view dict.
        seed = view.get("seed", task_seed)
        with TaskViewInjector(image_encoder, view["kind"], task_scale, task_rank, seed):
            return encoder_forward(image_encoder, x, is_synapse, norm_transform)
    raise ValueError(f"unknown view type: {vtype}")


# ---------------------------------------------------------------------------
# Attack objectives
# ---------------------------------------------------------------------------

def core_disruption_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    features_layer: Optional[List[torch.Tensor]],
    targets_layer: Optional[List[torch.Tensor]],
    lam: float = 1.0,
    num_layers: int = 4,
) -> torch.Tensor:
    """Core encoder-disruption objective shared by offline view gradients and
    online drift-residual computation.

        L_core = L1(0.9999 F, F*) + lam * mean_l L1(F_l, F_l*)

    The layer term uses a *deterministic* subset of the last `num_layers` layers
    so the offline drift basis and the online residual computation are aligned
    (no stochastic layer sampling).
    """
    loss = F.l1_loss(features * 0.9999, targets)
    if lam > 0 and features_layer is not None and targets_layer is not None and len(features_layer) > 0:
        n = min(num_layers, len(features_layer))
        idxs = list(range(len(features_layer) - n, len(features_layer)))
        layer_loss = 0.0
        for i in idxs:
            layer_loss = layer_loss + F.l1_loss(features_layer[i] * 0.999, targets_layer[i])
        loss = loss + lam * (layer_loss / n)
    return loss


def base_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    features_layer: List[torch.Tensor],
    targets_layer: List[torch.Tensor],
    enable_ads: bool = True,
    ads_rho: float = 1.0,
    ads_gamma: float = 1e-4,
    ads_tangential_beta: float = 0.0,
    criterion: nn.Module = None,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Feature-disruption objective of the standard transferable base optimizer.

        L = L1(0.9999 * F_adv, F_clean) + [optional stochastic robust term]

    This is only the base update produced by a standard transferable
    encoder-disruption optimizer.  The method's contribution is the
    adaptation-drift decoupling applied on top of it.
    """
    crit = criterion if criterion is not None else nn.L1Loss()
    loss_org = crit(features * 0.9999, targets)

    pre_layer = int(len(features_layer) * 0.9)
    num_layers = 6
    if pre_layer < num_layers:
        num_layers = max(1, pre_layer)
    rng = rng if rng is not None else random
    indices = rng.sample(range(num_layers), max(1, num_layers - 2))
    sel_f = torch.stack([features_layer[i + pre_layer - num_layers] for i in indices])
    sel_t = torch.stack([targets_layer[i + pre_layer - num_layers] for i in indices])

    if enable_ads:
        diff_list = []
        for i in range(sel_f.shape[0]):
            F_clean = sel_t[i].unsqueeze(0)
            F_adv = sel_f[i].unsqueeze(0)
            F_freq = torch.fft.fftn(F_clean, dim=(-2, -1), norm="ortho")
            A = torch.abs(F_freq)
            phase_clean = F_freq.angle()
            diff_energy = torch.mean(torch.abs(F_adv * 0.999 - F_clean))
            spectrum_energy = torch.mean(A)
            lam = diff_energy / (spectrum_energy + 1e-6)
            sigma_freq = lam * (A + 1e-12).pow(ads_rho) + ads_gamma
            n_white = torch.randn_like(A)
            fft_c = (n_white * sigma_freq) * torch.exp(1j * phase_clean)
            if ads_tangential_beta > 0:
                n_tan = torch.randn_like(A)
                fft_c = fft_c + ads_tangential_beta * (n_tan * sigma_freq) * (1j * torch.exp(1j * phase_clean))
            xi_struct = torch.fft.ifftn(fft_c, dim=(-2, -1), norm="ortho").real
            diff_list.append((F_adv * 0.999 - F_clean - xi_struct).squeeze(0))

        diff = torch.stack(diff_list)
        # per-sample mean normalization (mirror return_mean_value)
        mean_abs = torch.stack([sel_t[j].abs().mean() for j in range(sel_t.shape[0])]).view(-1, 1, 1, 1)
        if mean_abs.shape[0] != diff.shape[0]:
            mean_abs = mean_abs.view(diff.shape[0], 1, 1, 1)
        diff = diff / (mean_abs + 1e-6)
        l1_ads = (torch.abs(torch.mean(diff, dim=0))).mean()
        return loss_org + 1.0 * l1_ads
    else:
        return loss_org


def return_eta(momentum: torch.Tensor, step_size: float) -> torch.Tensor:
    """Convert momentum to the asymmetric update step of the strong backbone."""
    upd = momentum * (step_size / (torch.mean(torch.abs(momentum)) + 1e-12))
    pos = (momentum > 0).float()
    eta = torch.clamp(upd, -step_size, -0.25 * step_size) * (1 - pos) + \
          pos * torch.clamp(upd, 0.25 * step_size, step_size)
    return eta
