"""Utilities for rank-constrained PEFT adaptation sensitivity attacks.

Core model of an unknown adapted encoder:
    y_c' = y_c + Delta_c(x_c)
where each candidate adaptation channel c is represented by a low-rank linear
operator DeltaW_c with rank(DeltaW_c) <= r_c and ||DeltaW_c||_F <= rho_c.

Two channel families are implemented for SAM image encoders:
  * ``qkv``   : a low-rank update to block.attn.qkv.weight (LoRA-like, exact
                for a LoRA update merged into qkv.weight).
  * ``block`` : a low-rank residual applied to a transformer block output
                y <- y + Linear(y, DeltaW) (local adapter envelope).

The code deliberately does not sample random LoRA/Adapter weights.  Instead it
computes the first-order gain sensitivity matrix Q_c exactly from activations
and downstream gradients, then derives the worst rank-r Frobenius-bounded
operator from the truncated SVD of Q_c.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ATTACK_REPO_ROOT", os.path.dirname(_THIS_DIR))
for _p in (ROOT, os.path.join(ROOT, "newest")):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)


# -----------------------------------------------------------------------------
# Generic math / serialization
# -----------------------------------------------------------------------------


def normalize_l1(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.abs().sum() + eps)


def cosine_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    af, bf = a.reshape(1, -1), b.reshape(1, -1)
    return float((af @ bf.t()).item() / (af.norm().item() * bf.norm().item() + eps))


def return_eta(momentum: torch.Tensor, step_size: float, rule: str = "asymmetric") -> torch.Tensor:
    """Convert momentum into an image-space attack step.

    ``asymmetric`` preserves the update convention used by the user's existing
    attack code. ``sign`` gives standard MI-FGSM.
    """
    if rule == "sign":
        return step_size * torch.sign(momentum)
    if rule != "asymmetric":
        raise ValueError(f"unknown step rule: {rule}")
    upd = momentum * (step_size / (torch.mean(torch.abs(momentum)) + 1e-12))
    pos = (momentum > 0).to(momentum.dtype)
    return (
        torch.clamp(upd, -step_size, -0.25 * step_size) * (1.0 - pos)
        + pos * torch.clamp(upd, 0.25 * step_size, step_size)
    )


def _tensor_output(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)) and x and torch.is_tensor(x[0]):
        return x[0]
    raise TypeError(f"expected tensor module output, got {type(x)}")


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    """Flatten all leading axes, preserving the last (feature) dimension."""
    if x.ndim < 2:
        raise ValueError(f"feature tensor must have ndim>=2, got {tuple(x.shape)}")
    return x.reshape(-1, x.shape[-1])


def matrix_q_from_io(operator_input: torch.Tensor, output_grad: torch.Tensor) -> torch.Tensor:
    """Exact first-order sensitivity matrix for a linear additive operator.

    For an operator ``y <- y + F.linear(x, DeltaW)`` with DeltaW [out,in],

        dG = <Q, DeltaW>_F,   Q = G^T X,

    where rows of X/G are flattened token/batch/spatial positions.
    """
    x = _flatten_tokens(operator_input.detach()).float()
    g = _flatten_tokens(output_grad.detach()).float()
    if x.shape[0] != g.shape[0]:
        raise ValueError(f"token count mismatch: input {x.shape}, grad {g.shape}")
    return g.t().matmul(x)  # [out_dim, in_dim]


@dataclass
class SVDResult:
    U: torch.Tensor
    S: torch.Tensor
    Vh: torch.Tensor


def truncated_svd(
    qmat: torch.Tensor,
    rank: int,
    oversample: int = 8,
    niter: int = 4,
    exact_if_min_dim_le: int = 256,
) -> SVDResult:
    """Return top singular triplets of Q.

    Uses exact SVD for small matrices and randomized ``torch.svd_lowrank`` for
    large matrices. Frobenius norms used elsewhere remain exact.
    """
    if qmat.ndim != 2:
        raise ValueError("qmat must be a matrix")
    m, n = qmat.shape
    rmax = min(m, n)
    if rmax == 0:
        raise ValueError("empty qmat")
    rank = max(1, min(int(rank), rmax))

    q = qmat.float()
    if rmax <= exact_if_min_dim_le:
        U, S, Vh = torch.linalg.svd(q, full_matrices=False)
        return SVDResult(U[:, :rank], S[:rank], Vh[:rank, :])

    qdim = min(rmax, max(rank, rank + int(oversample)))
    U, S, V = torch.svd_lowrank(q, q=qdim, niter=int(niter))
    order = torch.argsort(S, descending=True)
    U, S, V = U[:, order], S[order], V[:, order]
    return SVDResult(U[:, :rank], S[:rank], V[:, :rank].t().contiguous())


def rank_sensitivity(
    qmat: torch.Tensor,
    rank: int,
    oversample: int = 8,
    niter: int = 4,
) -> Tuple[float, float]:
    """Return (S_r, kappa_r).

    S_r = sqrt(sum_{i<=r} sigma_i(Q)^2)
    kappa_r = S_r^2 / ||Q||_F^2.
    """
    svd = truncated_svd(qmat, rank, oversample=oversample, niter=niter)
    sr2 = float(svd.S.square().sum().item())
    frob2 = float(qmat.float().square().sum().item())
    sr = math.sqrt(max(sr2, 0.0))
    kappa = sr2 / (frob2 + 1e-12)
    return sr, kappa


def rank_metrics_multi(
    qmat: torch.Tensor,
    ranks: Sequence[int],
    oversample: int = 8,
    niter: int = 4,
) -> Tuple[float, Dict[int, float], Dict[int, float]]:
    """One SVD gives exact/approx top-r sensitivities for several ranks."""
    ranks = sorted({int(r) for r in ranks if int(r) > 0})
    if not ranks:
        raise ValueError("ranks cannot be empty")
    max_rank = min(max(ranks), min(qmat.shape))
    svd = truncated_svd(qmat, max_rank, oversample=oversample, niter=niter)
    s2_cum = torch.cumsum(svd.S.square(), dim=0)
    frob = float(qmat.float().norm().item())
    frob2 = frob * frob
    sr: Dict[int, float] = {}
    kp: Dict[int, float] = {}
    for r in ranks:
        rr = min(r, len(svd.S))
        val2 = float(s2_cum[rr - 1].item())
        sr[r] = math.sqrt(max(val2, 0.0))
        kp[r] = val2 / (frob2 + 1e-12)
    return frob, sr, kp


def worst_case_lowrank_update(
    qmat: torch.Tensor,
    rank: int,
    rho: float,
    oversample: int = 8,
    niter: int = 4,
) -> Tuple[torch.Tensor, float]:
    """First-order worst-case low-rank operator for *reducing* attack gain.

    Solves
        min_{rank(Delta)<=r, ||Delta||_F<=rho} <Q, Delta>_F

    with solution
        Delta* = -rho * Q_r / ||Q_r||_F,
    where Q_r is the best rank-r approximation of Q. Returns (Delta*, S_r).
    """
    if rho < 0:
        raise ValueError("rho must be nonnegative")
    if rho == 0:
        return torch.zeros_like(qmat), 0.0
    svd = truncated_svd(qmat, rank, oversample=oversample, niter=niter)
    q_r = (svd.U * svd.S.unsqueeze(0)).matmul(svd.Vh)
    sr = float(q_r.norm().item())
    if sr <= 1e-12:
        return torch.zeros_like(qmat), 0.0
    delta = (-float(rho) / sr) * q_r
    return delta.to(qmat.dtype), sr


# -----------------------------------------------------------------------------
# SAM adaptation channels
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelSpec:
    key: str
    block_idx: int
    kind: str  # qkv | block | proj | mlp1 | mlp2
    module: nn.Module


def build_candidate_channels(
    image_encoder: nn.Module,
    kinds: Sequence[str] = ("qkv", "block"),
    block_indices: Optional[Sequence[int]] = None,
) -> List[ChannelSpec]:
    """Build candidate PEFT channels from a SAM image encoder.

    Defaults intentionally cover the two target families in the user's setup:
    qkv (LoRA/SAMed-like) and block residual (Adapter/Med-SA-like).
    """
    if not hasattr(image_encoder, "blocks"):
        raise AttributeError("image_encoder has no .blocks; expected SAM ImageEncoderViT")
    kinds = tuple(kinds)
    valid = {"qkv", "block", "proj", "mlp1", "mlp2"}
    unknown = set(kinds) - valid
    if unknown:
        raise ValueError(f"unsupported channel kinds: {sorted(unknown)}")
    indices = list(range(len(image_encoder.blocks))) if block_indices is None else [int(i) for i in block_indices]
    out: List[ChannelSpec] = []
    for i in indices:
        blk = image_encoder.blocks[i]
        if "qkv" in kinds:
            mod = getattr(getattr(blk, "attn", None), "qkv", None)
            if isinstance(mod, nn.Linear):
                out.append(ChannelSpec(f"b{i:02d}.qkv", i, "qkv", mod))
        if "proj" in kinds:
            mod = getattr(getattr(blk, "attn", None), "proj", None)
            if isinstance(mod, nn.Linear):
                out.append(ChannelSpec(f"b{i:02d}.proj", i, "proj", mod))
        if "mlp1" in kinds:
            mod = getattr(getattr(blk, "mlp", None), "lin1", None)
            if isinstance(mod, nn.Linear):
                out.append(ChannelSpec(f"b{i:02d}.mlp1", i, "mlp1", mod))
        if "mlp2" in kinds:
            mod = getattr(getattr(blk, "mlp", None), "lin2", None)
            if isinstance(mod, nn.Linear):
                out.append(ChannelSpec(f"b{i:02d}.mlp2", i, "mlp2", mod))
        if "block" in kinds:
            out.append(ChannelSpec(f"b{i:02d}.block", i, "block", blk))
    return out


class ChannelCapture:
    """Capture operator inputs and outputs for a set of candidate channels."""

    def __init__(self, channels: Sequence[ChannelSpec]):
        self.channels = list(channels)
        self.handles: List = []
        self.tag: Optional[str] = None
        self.records: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}

    def set_tag(self, tag: str) -> None:
        self.tag = str(tag)
        self.records.setdefault(self.tag, {})

    def __enter__(self):
        for spec in self.channels:
            if spec.kind == "block":
                def hook(module, inp, out, _spec=spec):
                    if self.tag is None:
                        raise RuntimeError("capture tag is not set")
                    y = _tensor_output(out)
                    # Adapter envelope: y <- y + Linear(y, DeltaW)
                    self.records[self.tag][_spec.key] = (y, y)
                self.handles.append(spec.module.register_forward_hook(hook))
            else:
                def hook(module, inp, out, _spec=spec):
                    if self.tag is None:
                        raise RuntimeError("capture tag is not set")
                    x = inp[0]
                    y = _tensor_output(out)
                    self.records[self.tag][_spec.key] = (x, y)
                self.handles.append(spec.module.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class LowRankInjector:
    """Inject fixed low-rank operators into selected source-encoder channels.

    ``updates`` maps channel key -> DeltaW tensor with shape [out_dim, in_dim].
    DeltaW is treated as a constant during the outer image attack (stop-gradient
    inner minimization).
    """

    def __init__(self, channels: Sequence[ChannelSpec], updates: Mapping[str, torch.Tensor]):
        self.channels = {c.key: c for c in channels}
        self.updates = dict(updates)
        self.handles: List = []

    def __enter__(self):
        for key, delta_w in self.updates.items():
            if key not in self.channels:
                raise KeyError(f"profile channel {key} does not exist in encoder")
            spec = self.channels[key]
            dw = delta_w.detach()
            if spec.kind == "block":
                def hook(module, inp, out, _dw=dw):
                    y = _tensor_output(out)
                    return y + F.linear(y, _dw.to(device=y.device, dtype=y.dtype), None)
                self.handles.append(spec.module.register_forward_hook(hook))
            else:
                def hook(module, inp, out, _dw=dw):
                    y = _tensor_output(out)
                    x = inp[0]
                    return y + F.linear(x, _dw.to(device=y.device, dtype=y.dtype), None)
                self.handles.append(spec.module.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# -----------------------------------------------------------------------------
# Encoder I/O and gain
# -----------------------------------------------------------------------------


def encoder_forward(
    image_encoder: nn.Module,
    x: torch.Tensor,
    is_synapse: bool,
    norm_transform=None,
):
    """Preserve the input convention of the user's existing attack code."""
    if is_synapse:
        x_in = x[:, 0:1].repeat(1, 3, 1, 1)
    else:
        x_in = norm_transform(x) if norm_transform is not None else x
    out = image_encoder(x_in)
    if isinstance(out, tuple) and len(out) >= 2:
        return out[0], list(out[1])
    return out, []


def core_disruption_gain(
    adv_features: torch.Tensor,
    clean_features: torch.Tensor,
    adv_layers: Optional[Sequence[torch.Tensor]] = None,
    clean_layers: Optional[Sequence[torch.Tensor]] = None,
    layer_weight: float = 0.25,
    core_num_layers: int = 4,
) -> torch.Tensor:
    """Deterministic encoder-disruption gain shared by profiling and attack.

    The loss contains no frequency/noise term.  It is intentionally identical
    in offline profiling, online sensitivity estimation, and outer optimization.
    """
    gain = F.l1_loss(adv_features, clean_features)
    adv_layers = list(adv_layers or [])
    clean_layers = list(clean_layers or [])
    n = min(len(adv_layers), len(clean_layers), max(0, int(core_num_layers)))
    if n > 0 and layer_weight > 0:
        vals = [F.l1_loss(adv_layers[-i], clean_layers[-i]) for i in range(1, n + 1)]
        gain = gain + float(layer_weight) * torch.stack(vals).mean()
    return gain


def compute_gain(
    image_encoder: nn.Module,
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    is_synapse: bool,
    norm_transform=None,
    layer_weight: float = 0.25,
    core_num_layers: int = 4,
    injector: Optional[LowRankInjector] = None,
) -> torch.Tensor:
    """Gain with the same (possibly adapted) encoder applied to clean and adv."""
    ctx = injector if injector is not None else _NullContext()
    with ctx:
        with torch.no_grad():
            f_c, l_c = encoder_forward(image_encoder, x_clean, is_synapse, norm_transform)
            f_c = f_c.detach()
            l_c = [z.detach() for z in l_c]
        f_a, l_a = encoder_forward(image_encoder, x_adv, is_synapse, norm_transform)
        return core_disruption_gain(f_a, f_c, l_a, l_c, layer_weight, core_num_layers)


class _NullContext:
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


def estimate_channel_q(
    image_encoder: nn.Module,
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    channels: Sequence[ChannelSpec],
    is_synapse: bool,
    norm_transform=None,
    layer_weight: float = 0.25,
    core_num_layers: int = 4,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """Compute exact first-order Q_c matrices at the source encoder.

    Crucially, the same hypothetical adaptation channel acts on both clean and
    adversarial branches.  Therefore Q includes both branches' derivatives,
    matching d/dDelta of D(E_Delta(x_adv), E_Delta(x_clean)) at Delta=0.
    """
    if not channels:
        raise ValueError("channels cannot be empty")

    xc = x_clean.detach().clone().requires_grad_(True)
    xa = x_adv.detach().clone().requires_grad_(True)

    with ChannelCapture(channels) as cap:
        cap.set_tag("clean")
        f_c, l_c = encoder_forward(image_encoder, xc, is_synapse, norm_transform)
        cap.set_tag("adv")
        f_a, l_a = encoder_forward(image_encoder, xa, is_synapse, norm_transform)
        gain = core_disruption_gain(f_a, f_c, l_a, l_c, layer_weight, core_num_layers)

        outputs: List[torch.Tensor] = []
        refs: List[Tuple[str, str, torch.Tensor]] = []
        for tag in ("clean", "adv"):
            for spec in channels:
                if spec.key not in cap.records.get(tag, {}):
                    raise RuntimeError(f"channel {spec.key} was not captured during {tag} forward")
                op_in, out = cap.records[tag][spec.key]
                outputs.append(out)
                refs.append((tag, spec.key, op_in))

        grads = torch.autograd.grad(
            gain,
            outputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

    qdict: Dict[str, torch.Tensor] = {c.key: None for c in channels}  # type: ignore
    for (_, key, op_in), gy in zip(refs, grads):
        q = matrix_q_from_io(op_in, gy)
        qdict[key] = q if qdict[key] is None else qdict[key] + q

    return float(gain.detach().item()), {k: v.detach() for k, v in qdict.items()}


# -----------------------------------------------------------------------------
# Offline profile bundle
# -----------------------------------------------------------------------------


@dataclass
class SelectedChannel:
    key: str
    kind: str
    block_idx: int
    rank: int
    weight: float
    rho: float
    mean_kappa: float
    mean_sr: float
    mean_frob: float
    score: float


@dataclass
class RankProfileBundle:
    candidate_kinds: List[str]
    selected: List[SelectedChannel]
    tau_risk: float
    gain_floor: float
    ranks_tested: List[int]
    target_kappa: float
    min_kappa: float
    risk_quantile: float
    gate1_pass: bool
    meta: Dict

    def save(self, path: str) -> None:
        payload = {
            "candidate_kinds": self.candidate_kinds,
            "selected": [asdict(x) for x in self.selected],
            "tau_risk": float(self.tau_risk),
            "gain_floor": float(self.gain_floor),
            "ranks_tested": list(self.ranks_tested),
            "target_kappa": float(self.target_kappa),
            "min_kappa": float(self.min_kappa),
            "risk_quantile": float(self.risk_quantile),
            "gate1_pass": bool(self.gate1_pass),
            "meta": dict(self.meta),
        }
        torch.save(payload, path)
        json_path = os.path.splitext(path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "RankProfileBundle":
        d = torch.load(path, map_location="cpu")
        return RankProfileBundle(
            candidate_kinds=list(d["candidate_kinds"]),
            selected=[SelectedChannel(**x) for x in d["selected"]],
            tau_risk=float(d["tau_risk"]),
            gain_floor=float(d["gain_floor"]),
            ranks_tested=[int(x) for x in d["ranks_tested"]],
            target_kappa=float(d["target_kappa"]),
            min_kappa=float(d["min_kappa"]),
            risk_quantile=float(d["risk_quantile"]),
            gate1_pass=bool(d["gate1_pass"]),
            meta=dict(d.get("meta", {})),
        )


def resolve_profile_channels(
    image_encoder: nn.Module,
    profile: RankProfileBundle,
) -> Tuple[List[ChannelSpec], Dict[str, SelectedChannel]]:
    all_channels = build_candidate_channels(image_encoder, profile.candidate_kinds)
    by_key = {c.key: c for c in all_channels}
    selected_meta = {x.key: x for x in profile.selected}
    missing = [k for k in selected_meta if k not in by_key]
    if missing:
        raise KeyError(f"profile channels not found in this encoder: {missing}")
    return [by_key[k] for k in selected_meta], selected_meta


def risk_from_q(
    qdict: Mapping[str, torch.Tensor],
    selected_meta: Mapping[str, SelectedChannel],
    gain_value: float,
    gain_floor: float,
    oversample: int = 8,
    niter: int = 4,
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    """Return (bound, relative risk, per-channel diagnostics)."""
    bound = 0.0
    diag: Dict[str, Dict[str, float]] = {}
    for key, meta in selected_meta.items():
        if key not in qdict:
            raise KeyError(f"Q missing selected channel {key}")
        sr, kp = rank_sensitivity(qdict[key], meta.rank, oversample, niter)
        term = meta.weight * meta.rho * sr
        bound += term
        diag[key] = {"sr": sr, "kappa": kp, "bound_term": term}
    denom = max(abs(float(gain_value)), float(gain_floor), 1e-12)
    return bound, bound / denom, diag


def worst_updates_from_q(
    qdict: Mapping[str, torch.Tensor],
    selected_meta: Mapping[str, SelectedChannel],
    oversample: int = 8,
    niter: int = 4,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    updates: Dict[str, torch.Tensor] = {}
    sr_diag: Dict[str, float] = {}
    for key, meta in selected_meta.items():
        dw, sr = worst_case_lowrank_update(qdict[key], meta.rank, meta.rho, oversample, niter)
        updates[key] = dw
        sr_diag[key] = sr
    return updates, sr_diag
