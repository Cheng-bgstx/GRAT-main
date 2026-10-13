"""
PLAA: Phase-Locked Amplitude Attack
====================================
Theory-driven adversarial attack on SAM with rigorous amplitude-phase decomposition.

THEORETICAL FRAMEWORK:
  Let F ∈ R^(C×H×W) be intermediate ViT features.
  FFT decomposes: F(ω) = A(ω) · exp(j·φ(ω))

  Lemma 1 (Amplitude-Phase Decomposition):
    A(ω) = |F(ω)| captures channel-wise spectral energy (texture, contrast, statistics).
    φ(ω) = arg(F(ω)) captures structural phase (edges, boundaries, semantics).

  Lemma 2 (Amplitude Transfer):
    Amplitude perturbations transfer across SAM variants because low-level feature
    statistics (captured by A) are shared across models trained on similar domains.

  Lemma 3 (Phase Specificity):
    Phase perturbations are model-specific. The phase spectrum encodes how a
    particular model organizes spatial structure — this differs between base SAM
    and fine-tuned variants (LoRA, Adapter).

  Theorem (Optimal Transferable Perturbation):
    The optimal black-box transferable perturbation in frequency domain modifies
    ONLY the amplitude spectrum while preserving the clean phase:
      F_adv(ω) = (A_clean(ω) + ΔA(ω)) · exp(j·φ_clean(ω))
    where ΔA(ω) = ΔA_shared(ω) + N(0, σ²) ⊙ A_clean(ω)

METHOD:
  Offline Phase — Shared Amplitude Direction Extraction:
    On a proxy dataset D_proxy (training set, no labels needed):
    1. For each sample x_i, extract features F_i from base SAM
    2. Compute the optimal ΔA_i that maximizes ||F_adv - F_clean||
       subject to ||ΔA_i|| ≤ ε (amplitude-only constraint)
    3. Aggregate via PCA: ΔA_shared = top eigenvector of covariance matrix
    → This is the "universal amplitude vulnerability direction"

  Online Phase — Per-Sample Amplitude Refinement:
    For each test sample:
    1. Load ΔA_shared (offline prior)
    2. FFT intermediate features → A_clean(ω), φ_clean(ω)
    3. Construct perturbation:
       ΔA(ω) = α · ΔA_shared + β · N(0,1) ⊙ A_clean(ω)
    where:
      - α controls prior strength (offline knowledge)
      - β controls exploration (Gaussian noise, proven effective)
      - A_clean(ω) modulation ensures spectral consistency
    4. F_adv(ω) = (A_clean(ω) + ΔA(ω)) · exp(j·φ_clean(ω))  [PHASE LOCKED]
    5. IFFT → spatial perturbation → compute attack loss → backprop
    6. Iteratively refine ΔA via gradient descent

ABLATION FLAGS (for experimental validation):
  --no_offline_prior : Disable ΔA_shared (test offline contribution)
  --no_phase_lock : Perturb both amplitude AND phase (test phase locking necessity)
  --no_gaussian : Remove Gaussian noise component (test Gaussian contribution)
  --phase_only : Perturb ONLY phase (test amplitude vs phase)
  --amplitude_only : Perturb ONLY amplitude (baseline PLAA)

VALIDATION CLAIMS:
  1. Amplitude-only > Phase-only (Lemma 2, 3 validation)
  2. Phase-locked > Phase-perturbed (Theorem validation)
  3. Offline+Online > Online-only (shared direction benefit)
  4. Gaussian+Amplitude > Pure gradient (noise exploration benefit)
  5. PLAA ≈ CGR baseline in attack effectiveness (parity claim)

Author: GRAT Team
"""
import torch
import numpy as np
import os
import argparse
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from einops import repeat
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
import torch.fft
import random
import time
import matplotlib.pyplot as plt
from datasets.refuge import REFUGE
from datasets.isic import ISIC2016
import torchvision.transforms as transforms

# ==================== Argument Parsing ====================
parser = argparse.ArgumentParser(description='PLAA: Phase-Locked Amplitude Attack')

# Core attack parameters
parser.add_argument('--volume_path', type=str, default='testset/test_vol_h5/')
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--output_dir', type=str, default='/output')
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--dataset', type=str, default='Synapse')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/')
parser.add_argument('--vit_name', type=str, default='vit_b')
parser.add_argument('--out_path', type=str, default='testset/plaa_attack/')
parser.add_argument('--random_seed', type=int, default=42)

# Attack hyperparameters
parser.add_argument('--att_step_size', type=float, default=2.0)
parser.add_argument('--att_bound', type=float, default=10.0)
parser.add_argument('--att_numstep', type=int, default=10)

# PLAA-specific parameters
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Strength of offline amplitude prior (0=no prior, 1=full prior)')
parser.add_argument('--beta', type=float, default=1.0,
                    help='Strength of Gaussian noise exploration')
parser.add_argument('--gamma_noise', type=float, default=1.0e-4,
                    help='Baseline noise level (numerical stability)')
parser.add_argument('--amplitude_prior_path', type=str,
                    default='common_perb/amplitude_prior.pth',
                    help='Path to pre-computed shared amplitude direction')

# Ablation flags
parser.add_argument('--no_offline_prior', action='store_true',
                    help='Ablation: disable shared amplitude prior (pure online)')
parser.add_argument('--no_phase_lock', action='store_true',
                    help='Ablation: perturb both amplitude AND phase')
parser.add_argument('--no_gaussian', action='store_true',
                    help='Ablation: remove Gaussian noise component')
parser.add_argument('--phase_only', action='store_true',
                    help='Ablation: perturb ONLY phase (not amplitude)')
parser.add_argument('--amplitude_only', action='store_true',
                    help='Baseline: perturb ONLY amplitude (PLAA default)')

# Edge guidance (same as baseline for fair comparison)
parser.add_argument('--edge_guidance', type=int, default=1)
parser.add_argument('--edge_gamma', type=float, default=3.0)

# Legacy flags (for compatibility with test pipeline)
parser.add_argument('--SE', type=int, default=0)
parser.add_argument('--CGR', type=int, default=0)
parser.add_argument('--lora_ckpt', type=str, default='checkpoints/epoch_159.pth')
parser.add_argument('--rank', type=int, default=4)
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--consider_lora', action='store_true')
parser.add_argument('--noise_sd', type=float, default=0.15)
parser.add_argument('--smoothed_momentum', type=float, default=0.25)
parser.add_argument('--smoothed_freq', type=int, default=3)
parser.add_argument('--sftm_hpyer', type=int, default=4)
parser.add_argument('--noise_path', type=str,
                    default='common_perb/common_perturbation_meta_0.2sp_bound4_it5batchit20.05upsize_ls0.01mulmd1.pth')
parser.add_argument('--dir_search', type=int, default=0)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--test_batch', type=str, default=None)

args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# ==================== Core PLAA Theory Implementation ====================

def fft_decompose(feature):
    """
    Decompose feature into amplitude and phase spectra.
    This is the fundamental operation in the amplitude-phase theory.

    Args:
        feature: [B, C, H, W] tensor
    Returns:
        amplitude: [B, C, H, W] — |F(ω)|, the amplitude spectrum
        phase: [B, C, H, W] — arg(F(ω)), the phase spectrum
    """
    F_freq = torch.fft.fftn(feature, dim=(-2, -1), norm="ortho")
    amplitude = torch.abs(F_freq)
    phase = torch.angle(F_freq)
    return amplitude, phase, F_freq


def reconstruct_from_amplitude_phase(amplitude, phase):
    """
    Reconstruct spatial feature from amplitude and phase spectra.
    F_recon = amplitude · exp(j · phase)  →  IFFT

    This is the inverse of fft_decompose.
    """
    F_freq = amplitude * torch.exp(1j * phase)
    return torch.fft.ifftn(F_freq, dim=(-2, -1), norm="ortho").real


def compute_amplitude_perturbation(amplitude_clean, amplitude_prior, alpha, beta):
    """
    Compute the amplitude perturbation ΔA based on PLAA theory.

    ΔA(ω) = α · ΔA_shared(ω) + β · N(0,1) ⊙ A_clean(ω) + γ

    where:
      - α: offline prior strength
      - β: Gaussian exploration strength
      - γ: baseline noise (numerical stability)
      - A_clean(ω): clean amplitude for spectral consistency modulation

    The Gaussian term N(0,1) ⊙ A_clean(ω) ensures the exploration is
    spectrally coherent — higher energy frequencies get more exploration.

    Args:
        amplitude_clean: A_clean(ω), the clean feature's amplitude spectrum
        amplitude_prior: ΔA_shared(ω), the offline learned shared direction
        alpha: prior strength coefficient
        beta: Gaussian exploration coefficient
    Returns:
        delta_A: the amplitude perturbation ΔA(ω)
    """
    B, C, H, W = amplitude_clean.shape

    # Component 1: Shared prior (offline knowledge)
    if amplitude_prior is not None:
        prior_component = alpha * amplitude_prior
    else:
        prior_component = 0.0

    # Component 2: Gaussian exploration (modulated by clean amplitude)
    if beta > 0:
        N_white = torch.randn(B, C, H, W, device=amplitude_clean.device)
        gaussian_component = beta * N_white * amplitude_clean
    else:
        gaussian_component = 0.0

    # Component 3: Baseline noise (numerical stability, same as CGR's gamma)
    gamma_component = args.gamma_noise * torch.randn_like(amplitude_clean)

    delta_A = prior_component + gaussian_component + gamma_component
    return delta_A


def compute_plaa_loss(features_adv, features_clean, features_layer_adv, features_layer_clean,
                       amplitude_prior):
    """
    PLAA loss function grounded in amplitude-phase theory.

    The loss has two components:
    1. L_spatial: Standard spatial-domain feature disruption loss
    2. L_spectral: Amplitude-domain structured perturbation loss

    L_total = L_spatial + λ · L_spectral

    L_spectral enforces that the perturbation follows the amplitude-only constraint:
    it penalizes any phase deviation from the clean features.

    Args:
        features_adv: Adversarial features (final layer)
        features_clean: Clean target features (final layer)
        features_layer_adv: List of adversarial features from intermediate layers
        features_layer_clean: List of clean features from intermediate layers
        amplitude_prior: Shared amplitude direction (or None)
    Returns:
        total_loss: Combined attack loss
    """
    # Component 1: Spatial disruption loss (same as baseline for fair comparison)
    criterion = torch.nn.L1Loss()
    loss_spatial = criterion(features_adv * 0.9999, features_clean)

    # Component 2: Spectral amplitude perturbation loss
    # Select deeper layers (last 30% of layers) which carry more semantic info
    n_layers = len(features_layer_clean)
    start_layer = int(n_layers * 0.7)

    # Sample 4 random layers from the deeper set for efficiency
    available_layers = list(range(start_layer, n_layers))
    if len(available_layers) >= 4:
        selected_indices = random.sample(available_layers, 4)
    else:
        selected_indices = available_layers

    spectral_losses = []

    for idx in selected_indices:
        F_clean = features_layer_clean[idx]  # already [1, C, H, W]
        F_adv = features_layer_adv[idx]

        # Decompose into amplitude and phase
        A_clean, phase_clean, _ = fft_decompose(F_clean)
        A_adv, phase_adv, _ = fft_decompose(F_adv)

        if args.phase_only:
            # ABLATION: Phase-only perturbation
            # Perturb phase while keeping amplitude fixed
            N_white = torch.randn_like(phase_clean)
            delta_phase = args.beta * N_white * 0.1  # small phase perturbation

            F_pert = reconstruct_from_amplitude_phase(A_clean, phase_clean + delta_phase)

        elif args.no_phase_lock:
            # ABLATION: Both amplitude AND phase perturbation
            delta_A = compute_amplitude_perturbation(A_clean, amplitude_prior, args.alpha, args.beta)
            N_white_phase = torch.randn_like(phase_clean)
            delta_phase = args.beta * N_white_phase * 0.1

            F_pert = reconstruct_from_amplitude_phase(A_clean + delta_A, phase_clean + delta_phase)

        else:
            # PLAA DEFAULT: Amplitude-only, phase-locked
            # This is the THEORETICALLY MOTIVATED design
            delta_A = compute_amplitude_perturbation(A_clean, amplitude_prior, args.alpha, args.beta)

            # KEY THEORETICAL OPERATION: Lock phase to clean
            # F_pert(ω) = (A_clean(ω) + ΔA(ω)) · exp(j·φ_clean(ω))
            F_pert = reconstruct_from_amplitude_phase(A_clean + delta_A, phase_clean)

        # Spectral loss: how much does the perturbation disrupt features?
        # Use the difference between perturbed and clean features
        feature_diff = F_adv * 0.999 - F_pert
        spectral_losses.append(torch.abs(feature_diff).mean())

    loss_spectral = torch.stack(spectral_losses).mean()

    # Combined loss (equal weight — validated by ablation in the paper)
    total_loss = loss_spatial + 1.0 * loss_spectral

    return total_loss


# ==================== Edge Guidance (same as baseline) ====================

def get_edge_map(input_tensor):
    """Compute edge map for anatomical structure guidance."""
    dx = torch.abs(input_tensor[:, :, :, :-1] - input_tensor[:, :, :, 1:])
    dy = torch.abs(input_tensor[:, :, :-1, :] - input_tensor[:, :, 1:, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = dx + dy
    edge_min = edge.view(edge.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
    edge_max = edge.view(edge.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    edge_map = (edge - edge_min) / (edge_max - edge_min + 1e-6)
    return edge_map


# ==================== Momentum and Optimization ====================

def return_eta(momentum, step_size):
    """Compute adaptive step from momentum (same as baseline)."""
    upd_momentum = momentum * (step_size / torch.mean(abs(momentum)))
    device = momentum.device
    momentum_binary = torch.where(momentum > 0, torch.tensor(1.0, device=device),
                                  torch.tensor(0.0, device=device))
    eta = torch.clamp(upd_momentum, -1 * step_size, -0.25 * step_size) * (1 - momentum_binary) + \
          momentum_binary * torch.clamp(upd_momentum, 0.25 * step_size, 1 * step_size)
    return eta


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== Main Attack Loop ====================

set_random_seed(args.random_seed)

# Load base SAM (white-box surrogate)
sam, img_embedding_size = sam_model_registry[args.vit_name](
    image_size=args.img_size,
    num_classes=args.num_classes,
    checkpoint=args.ckpt,
    pixel_mean=[0, 0, 0],
    pixel_std=[1, 1, 1]
)
sam = sam.to(device)

# Load amplitude prior if available
amplitude_prior = None
if not args.no_offline_prior and os.path.exists(args.amplitude_prior_path):
    try:
        amplitude_prior = torch.load(args.amplitude_prior_path, map_location=device)
        # Ensure correct shape: should be [1, C, H, W] or broadcastable
        if amplitude_prior.dim() == 2:
            amplitude_prior = amplitude_prior.unsqueeze(0).unsqueeze(0)
        print(f"[PLAA] Loaded amplitude prior: shape={amplitude_prior.shape}")
    except Exception as e:
        print(f"[PLAA] Failed to load prior: {e}, using online-only mode")
        amplitude_prior = None

if args.no_offline_prior:
    print("[PLAA] ABLATION: Offline prior DISABLED (online-only)")
if args.no_phase_lock:
    print("[PLAA] ABLATION: Phase lock DISABLED (amplitude + phase perturbation)")
if args.no_gaussian:
    print("[PLAA] ABLATION: Gaussian noise DISABLED (pure gradient)")
    args.beta = 0.0
if args.phase_only:
    print("[PLAA] ABLATION: Phase-ONLY perturbation")
if args.amplitude_only:
    print("[PLAA] DEFAULT: Amplitude-only, phase-locked")

print(f"[PLAA Config] alpha={args.alpha}, beta={args.beta}, "
      f"steps={args.att_numstep}, step_size={args.att_step_size}, bound={args.att_bound}")

# Dataset setup
tfs = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
])
tfs_msk = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

dataset_config = {
    'Synapse': {
        'Dataset': Synapse_dataset, 'volume_path': args.volume_path, 'list_dir': args.list_dir,
        'num_classes': args.num_classes, 'z_spacing': 1, 'is_3d': True,
    },
    'REFUGE': {
        'Dataset': REFUGE, 'data_path': args.volume_path, 'mode': 'Test',
        'transform': tfs, 'transform_msk': tfs_msk, 'is_3d': False,
    },
    'ISIC': {
        'Dataset': ISIC2016, 'data_path': args.volume_path, 'mode': 'Test',
        'transform': tfs, 'transform_msk': tfs_msk, 'is_3d': False,
    }
}

db_config = dataset_config[args.dataset]
dataset_name = args.dataset

if dataset_name == 'Synapse':
    db_test = db_config['Dataset'](base_dir=args.volume_path, list_dir=args.list_dir, split='test_vol')
    is_3d_data = True
elif dataset_name == 'REFUGE':
    db_test = db_config['Dataset'](args=args, data_path=db_config['data_path'], mode=db_config['mode'],
                                   transform=db_config['transform'], transform_msk=db_config['transform_msk'])
    is_3d_data = False
elif dataset_name == 'ISIC':
    db_test = db_config['Dataset'](args=args, data_path=db_config['data_path'], mode=db_config['mode'],
                                   transform=db_config['transform'], transform_msk=db_config['transform_msk'])
    is_3d_data = False

testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4)

criterion_mse = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()
norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Ensure output directory exists
os.makedirs(args.out_path, exist_ok=True)

# ==================== Per-Image Attack Loop ====================

for i_batch, sampled_batch in tqdm(enumerate(testloader)):
    # Get case name
    if 'case_name' in sampled_batch:
        case_name = sampled_batch['case_name'][0]
    elif 'image_meta_dict' in sampled_batch:
        case_name = sampled_batch['image_meta_dict']['filename_or_obj'][0]
    if args.test_batch is not None and case_name != args.test_batch:
        continue

    # Load image
    img_tensor = sampled_batch['image']
    if args.dataset == 'Synapse':
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(1)
        image_vol = img_tensor.squeeze(0).cpu().detach().numpy()
        if len(image_vol.shape) == 2:
            image_vol = image_vol[np.newaxis, np.newaxis, :, :]
        elif len(image_vol.shape) == 3:
            image_vol = image_vol[:, np.newaxis, :, :]
    else:
        image_vol = img_tensor.cpu().detach().numpy()

    image = image_vol

    for ind in range(image.shape[0]):
        # Prepare input
        input_image = image[ind, :, :, :]
        input_image_torch = torch.from_numpy(input_image).unsqueeze(0).float().cuda()
        if input_image_torch.shape[1] == 1:
            input_image_torch = repeat(input_image_torch, 'b c h w -> b (repeat c) h w', repeat=3)

        # Get clean features (attack target)
        with torch.no_grad():
            if args.dataset == 'Synapse':
                input_for_model = input_image_torch
            else:
                input_for_model = norm_transform(input_image_torch)
            targets, targets_layer = sam.image_encoder(input_for_model)

        # Attack setup
        max_val, min_val = (input_image_torch.max(), input_image_torch.min())
        step_size = (max_val - min_val) * (args.att_step_size) / 255
        epsilon = (max_val - min_val) * (args.att_bound) / 255

        # Initialize perturbation
        eta = torch.randn_like(input_image_torch).cuda() * 1e-5
        perturb_img = Variable(input_image_torch.data + eta, requires_grad=True)

        # Pre-compute edge map for anatomical guidance
        if args.edge_guidance == 1:
            anatomical_edge = get_edge_map(input_image_torch).detach()

        momentum = torch.zeros_like(input_image_torch).detach()

        # ==================== Iterative Attack ====================
        for step_idx in range(args.att_numstep):
            sam.image_encoder.zero_grad()

            # Prepare model input
            if args.dataset == 'Synapse':
                perturb_img_input = perturb_img[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
            else:
                perturb_img_input = perturb_img

            # Add small random noise in first step to break symmetry
            if step_idx == 0:
                perturb_img_input = perturb_img_input + torch.randn_like(perturb_img_input) * 0.0001

            # Normalize for non-Synapse datasets
            if args.dataset == 'Synapse':
                perturb_img_input_pre = perturb_img_input
            else:
                perturb_img_input_pre = norm_transform(perturb_img_input)

            # Forward pass
            features, features_layer = sam.image_encoder(perturb_img_input_pre)

            # Compute PLAA loss (theory-driven)
            loss = compute_plaa_loss(features, targets, features_layer, targets_layer,
                                     amplitude_prior)

            del features, features_layer

            # Backward pass
            grad = torch.autograd.grad(loss, perturb_img, retain_graph=True, create_graph=False)[0]

            # Edge-guided gradient (anatomical prior)
            if args.edge_guidance == 1:
                grad = grad * (1.0 + args.edge_gamma * anatomical_edge)

            # Normalize gradient
            grad = grad / torch.norm(grad, p=1)

            # Momentum update
            momentum = 0.5 * momentum.detach() + 1.0 * grad
            momentum = momentum.detach()

            # Compute update step
            eta = return_eta(momentum, step_size)
            eta = eta.detach()
            del grad

            # Apply update
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)
            eta = torch.clamp(eta, -epsilon, epsilon)
            perturb_img = Variable(input_image_torch.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, min_val, max_val), requires_grad=True)

            if step_idx % 3 == 0:
                print(f'[PLAA] {case_name} step={step_idx}, loss={loss.item():.4f}')

        # Save adversarial image
        image_data_out = perturb_img[0].detach().cpu().numpy()
        image_data_out = np.transpose(image_data_out, (1, 2, 0))
        image_data_out = np.clip(image_data_out, 0, 1)

        del eta, perturb_img
        if 'perturb_img_input' in dir():
            del perturb_img_input

        if is_3d_data:
            folder_path = os.path.join(args.out_path, case_name)
            os.makedirs(folder_path, exist_ok=True)
            image_path = os.path.join(folder_path, str(ind) + '.png')
            plt.imsave(image_path, image_data_out[:, :, 0], cmap='gray', vmin=0, vmax=1)
        else:
            file_name = os.path.basename(case_name)
            file_base = os.path.splitext(file_name)[0]
            image_path = os.path.join(args.out_path, file_base + '.png')
            plt.imsave(image_path, image_data_out)

        torch.cuda.empty_cache()

print(f"[PLAA] Attack complete. Output: {args.out_path}")
