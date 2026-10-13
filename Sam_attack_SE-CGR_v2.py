"""
SE-CGR v2: Enhanced Frequency-Domain Adversarial Attack with:
  1. Frequency-Band Adaptive Noise Model (FreqBand)
  2. Structured Phase Perturbation (PhasePert)
  3. Layer-wise Spectral Aggregation (LayerAgg)
  4. Spectral Momentum Filtering (SpecMom)

Key theoretical improvements over baseline SE-CGR:
  - λ is now frequency-band-dependent: λ(ω) adapts to spectral energy distribution
  - Phase is NOT locked: small smooth perturbations exploit shared phase vulnerability
  - Layers contribute proportional to their spectral energy
  - Momentum is filtered in frequency domain to suppress model-specific noise
"""
import torch
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
import argparse
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from tqdm import tqdm
from torch.utils.data import DataLoader
from importlib import import_module
from segment_anything import sam_model_registry
import matplotlib.pyplot as plt
from datasets.dataset_synapse import Synapse_dataset
from einops import repeat
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
import torch.fft
import random
import time
import cv2
from torchvision.utils import save_image
from datasets.refuge import REFUGE
from datasets.isic import ISIC2016
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str, default='testset/test_vol_h5/')
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--output_dir', type=str, default='/output')
parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/', help='list_dir')
parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
parser.add_argument('--lora_ckpt', type=str, default='checkpoints/epoch_159.pth', help='The checkpoint from LoRA')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--consider_lora', action='store_true', help='Whether to consider lora when attack')
parser.add_argument('--test_batch', type=str, default=None)
parser.add_argument('--out_path', type=str, default='testset/advimages_ours_smoothedmulti_attack/')
parser.add_argument('--noise_sd', type=float, default=0.15)
parser.add_argument('--smoothed_momentum', type=float, default=0.25)
# V2-specific flags
parser.add_argument('--freq_bands', type=int, default=4, help='Number of frequency bands for adaptive λ')
parser.add_argument('--phase_bound', type=float, default=0.15, help='Max phase perturbation in radians')
parser.add_argument('--phase_smooth', type=float, default=3.0, help='Smoothness of phase perturbation (Gaussian sigma)')
parser.add_argument('--spec_momentum', type=int, default=1, help='Enable spectral momentum filtering')
parser.add_argument('--layer_aggregation', type=int, default=1, help='Enable layer-wise spectral aggregation')
# Legacy flags (kept for compatibility)
parser.add_argument('--SE', type=int, default=0)
parser.add_argument('--CGR', type=int, default=1)
parser.add_argument('--sftm_hpyer', type=int, default=4)
parser.add_argument('--noise_path', type=str,
                    default='common_perb/common_perturbation_meta_0.2sp_bound4_it5batchit20.05upsize_ls0.01mulmd1.pth')
parser.add_argument('--att_step_size', type=float, default=2.0)
parser.add_argument('--att_bound', type=float, default=10.0)
parser.add_argument('--att_numstep', type=int, default=10)
parser.add_argument('--dir_search', type=int, default=0)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--edge_guidance', type=int, default=1, help='Enable Edge-weighted Gradient Guidance')
parser.add_argument('--edge_gamma', type=float, default=3.0, help='Intensity of edge weighting (default: 3.0)')
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def input_diversity(x):
    resize_rate = 0.9
    diversity_prob = 0.1
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    if resize_rate < 1:
        img_size = img_resize
        img_resize = x.shape[-1]
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode="bilinear", align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    return padded if torch.rand(1) < diversity_prob else x


def normalize_tensor(input_tensor):
    len_tensor = len(input_tensor)
    min_vals = input_tensor.view(len_tensor, -1).min(dim=1, keepdim=True)[0].view(len_tensor, 1, 1, 1, 1)
    max_vals = input_tensor.view(len_tensor, -1).max(dim=1, keepdim=True)[0].view(len_tensor, 1, 1, 1, 1)
    normalized_tensor = (input_tensor - min_vals) / (max_vals - min_vals)
    return normalized_tensor


def return_mean_value(input_tensor):
    input_tensor = torch.abs(input_tensor)
    list_mean = []
    for i in range(len(input_tensor)):
        list_mean.append(input_tensor[i].mean())
    input_mean = torch.stack(list_mean)
    input_mean = input_mean.view(-1, 1, 1, 1, 1)
    return input_mean


def get_freq_band_mask(H, W, n_bands, device):
    """
    Create concentric frequency band masks for 2D FFT spectrum.
    Returns list of boolean masks, one per band, from low to high frequency.
    Band 0: center (lowest freq), Band n_bands-1: corners (highest freq).
    """
    # Create frequency grid
    fy = torch.fft.fftfreq(H, device=device).view(-1, 1)  # [H, 1]
    fx = torch.fft.fftfreq(W, device=device).view(1, -1)   # [1, W]
    # Normalized radius: 0 at center, 1 at corners
    radius = torch.sqrt(fy**2 + fx**2)  # [H, W]
    max_radius = radius.max()
    radius = radius / (max_radius + 1e-8)

    masks = []
    for b in range(n_bands):
        r_low = b / n_bands
        r_high = (b + 1) / n_bands
        if b == 0:
            mask = radius <= r_high
        else:
            mask = (radius > r_low) & (radius <= r_high)
        masks.append(mask)
    return masks  # list of [H, W] bool tensors


def create_smooth_phase_field(H, W, sigma, device):
    """
    Create a smooth random phase perturbation field.
    Uses low-resolution random field + upsampling + Gaussian smoothing.
    """
    # Low-resolution random phase field
    low_res = max(4, min(H, W) // 16)
    phase_low = torch.randn(1, 1, low_res, low_res, device=device) * np.pi * 0.1
    # Upsample to target size
    phase_up = F.interpolate(phase_low, size=(H, W), mode='bilinear', align_corners=False)
    # Gaussian smooth
    if sigma > 0:
        kernel_size = int(2 * sigma * 3 + 1) | 1  # odd
        kernel_size = min(kernel_size, 31)  # cap at 31
        # Create 2D Gaussian kernel
        kx = torch.arange(kernel_size, device=device).float() - kernel_size // 2
        ky = kx.clone()
        gx = torch.exp(-0.5 * (kx / sigma) ** 2)
        gy = torch.exp(-0.5 * (ky / sigma) ** 2)
        g2d = gx.view(1, -1) * gy.view(-1, 1)
        g2d = g2d / g2d.sum()
        g2d = g2d.view(1, 1, kernel_size, kernel_size)
        phase_smooth = F.conv2d(phase_up, g2d, padding=kernel_size // 2)
    else:
        phase_smooth = phase_up
    # Normalize to [-phase_bound, phase_bound]
    phase_smooth = phase_smooth / (phase_smooth.abs().max() + 1e-8) * args.phase_bound
    return phase_smooth.squeeze(0).squeeze(0)  # [H, W]


def spectral_momentum_filter(momentum_grad):
    """
    Filter momentum in frequency domain: suppress high-frequency noise
    by applying a Gaussian low-pass filter.
    """
    B, C, H, W = momentum_grad.shape
    # FFT
    grad_fft = torch.fft.fftn(momentum_grad, dim=(-2, -1), norm="ortho")
    # Create Gaussian low-pass filter in frequency domain
    fy = torch.fft.fftfreq(H, device=momentum_grad.device).view(-1, 1)
    fx = torch.fft.fftfreq(W, device=momentum_grad.device).view(1, -1)
    radius = torch.sqrt(fy**2 + fx**2)
    # Gaussian: sigma = 0.25 (keep ~68% of spectrum below 0.25*max_freq)
    sigma_freq = 0.25 * radius.max()
    gaussian_lpf = torch.exp(-0.5 * (radius / sigma_freq) ** 2)
    gaussian_lpf = gaussian_lpf.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    # Apply filter
    grad_fft_filtered = grad_fft * gaussian_lpf
    # IFFT
    grad_filtered = torch.fft.ifftn(grad_fft_filtered, dim=(-2, -1), norm="ortho").real
    return grad_filtered


def calculate_loss_v2(features, targets, features_layer, targets_layer):
    """
    Enhanced loss with:
    - Frequency-band adaptive noise model
    - Structured phase perturbation
    - Layer-wise spectral aggregation
    """
    loss_org = criterion(features * 0.9999, targets)

    pre_layer = int(len(features_layer) * 0.9)
    num_layers = 6
    indices = random.sample(range(num_layers), num_layers - 2)
    selected_features_layer = torch.stack([features_layer[i + pre_layer - num_layers] for i in indices])
    selected_targets_layer = torch.stack([targets_layer[i + pre_layer - num_layers] for i in indices])

    # [V2 IMPROVEMENT] Pre-compute frequency band masks (shared across layers for efficiency)
    _, _, H_feat, W_feat = selected_targets_layer[0].shape
    freq_masks = get_freq_band_mask(H_feat, W_feat, args.freq_bands, device)

    # [V2 IMPROVEMENT] Pre-compute smooth phase perturbation field (shared across layers)
    phase_pert_field = create_smooth_phase_field(H_feat, W_feat, args.phase_smooth, device)

    feature_difference_list = []
    layer_weights = []  # for layer aggregation

    for i in range(selected_features_layer.shape[0]):
        F_clean = selected_targets_layer[i].unsqueeze(0)  # [1, C, H, W]
        F_adv = selected_features_layer[i].unsqueeze(0)   # [1, C, H, W]

        # 1. FFT
        F_freq = torch.fft.fftn(F_clean, dim=(-2, -1), norm="ortho")
        A = torch.abs(F_freq)
        Phase_clean = F_freq.angle()

        # [V2 IMPROVEMENT] Frequency-band adaptive λ
        # Compute per-band diff_energy and spectrum_energy
        current_diff_spatial = F_adv * 0.999 - F_clean
        diff_fft = torch.fft.fftn(current_diff_spatial, dim=(-2, -1), norm="ortho")
        diff_A = torch.abs(diff_fft)  # differential amplitude

        # Build frequency-dependent Sigma_freq
        Sigma_freq = torch.zeros_like(A)
        gamma_val = 1.0e-4
        base_lambda = torch.mean(torch.abs(current_diff_spatial)) / (torch.mean(A) + 1e-6)

        for b, mask in enumerate(freq_masks):
            mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

            # Band-specific lambda: low freq → smaller λ, mid freq → larger λ, high freq → moderate λ
            # Theory: mid frequencies carry shared structural info that transfers best
            if b == 0:  # lowest freq
                lambda_band = base_lambda * 0.5  # preserve global structure
            elif b == args.freq_bands - 1:  # highest freq
                lambda_band = base_lambda * 0.7  # suppress model-specific high-freq
            else:  # mid frequencies
                lambda_band = base_lambda * 1.5  # amplify shared vulnerability

            # Compute band-local energy for adaptive scaling
            band_A = A * mask_expanded.float()
            band_energy = band_A.sum() / (mask_expanded.sum() + 1e-6)
            global_energy = A.mean()
            band_ratio = band_energy / (global_energy + 1e-6)
            lambda_band = lambda_band * (0.5 + band_ratio)

            Sigma_freq = Sigma_freq + (lambda_band * A + gamma_val) * mask_expanded.float()

        # [V2 IMPROVEMENT] Structured phase perturbation
        # Phase_pert = Phase_clean + smooth_phase_field (instead of just Phase_clean)
        Phase_use = Phase_clean + phase_pert_field.unsqueeze(0).unsqueeze(0)

        # Generate modulated noise with frequency-dependent sigma
        N_white = torch.randn_like(A, device=A.device)
        N_modulated = N_white * Sigma_freq

        # Construct spectral perturbation
        fft_c = N_modulated * torch.exp(1j * Phase_use)

        # Inverse FFT
        xi_struct_spatial = torch.fft.ifftn(fft_c, dim=(-2, -1), norm="ortho").real

        # Feature difference computation
        feature_difference = (F_adv * 0.999 - F_clean) - xi_struct_spatial
        feature_difference_list.append(feature_difference.squeeze(0))

        # [V2 IMPROVEMENT] Layer weight based on spectral energy
        if args.layer_aggregation:
            layer_energy = torch.mean(A)
            layer_weights.append(layer_energy)

    # Stack features
    input_mean = return_mean_value(selected_targets_layer)
    feature_difference = torch.stack(feature_difference_list)
    if len(input_mean.shape) > len(feature_difference.shape):
        input_mean = input_mean.view(feature_difference.shape[0], 1, 1, 1)
    feature_difference = feature_difference / (input_mean + 1e-6)

    # [V2 IMPROVEMENT] Layer-weighted aggregation
    if args.layer_aggregation and layer_weights:
        layer_weights_t = torch.stack(layer_weights)
        layer_weights_t = layer_weights_t / (layer_weights_t.sum() + 1e-6)
        l1_losses = 0.0
        for j in range(feature_difference.shape[0]):
            l1_losses = l1_losses + layer_weights_t[j] * torch.abs(feature_difference[j]).mean()
    else:
        l1_losses = (torch.abs(torch.mean(feature_difference, dim=0))).mean()

    loss = 1.0 * l1_losses + 1.0 * loss_org

    # Info loss (same as baseline)
    targets_flat = targets.view(*targets.shape[:-2], -1)
    features_flat = features.view(*features.shape[:-2], -1)
    features_softmax = F.softmax(features_flat, dim=2)
    targets_softmax = F.softmax(targets_flat, dim=2)
    info_loss = F.kl_div(features_softmax.log(), targets_softmax, reduction='batchmean')

    return loss + 0.0 * info_loss


def get_edge_map(input_tensor):
    dx = torch.abs(input_tensor[:, :, :, :-1] - input_tensor[:, :, :, 1:])
    dy = torch.abs(input_tensor[:, :, :-1, :] - input_tensor[:, :, 1:, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = dx + dy
    edge_min = edge.view(edge.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
    edge_max = edge.view(edge.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    edge_map = (edge - edge_min) / (edge_max - edge_min + 1e-6)
    return edge_map


def return_eta(momentum, step_size):
    upd_momentum = momentum * (step_size / torch.mean(abs(momentum)))
    device = momentum.device
    momentum_binary = torch.where(momentum > 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
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


# ===================== Main =====================
seed = args.random_seed
set_random_seed(seed)

sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                            num_classes=args.num_classes,
                                                            checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                            pixel_std=[1, 1, 1])
sam = sam.to(device)
output_path = args.out_path
scaler = GradScaler()
criterion_mse = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()
dataset_name = args.dataset

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
        'Dataset': Synapse_dataset,
        'volume_path': args.volume_path,
        'list_dir': args.list_dir,
        'num_classes': args.num_classes,
        'z_spacing': 1,
        'is_3d': True,
    },
    'REFUGE': {
        'Dataset': REFUGE,
        'data_path': args.volume_path,
        'mode': 'Test',
        'transform': tfs,
        'transform_msk': tfs_msk,
        'is_3d': False,
    },
    'ISIC': {
        'Dataset': ISIC2016,
        'data_path': args.volume_path,
        'mode': 'Test',
        'transform': tfs,
        'transform_msk': tfs_msk,
        'is_3d': False,
    }
}

db_config = dataset_config[dataset_name]

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

# Load common perturbation if available
if args.SE == 1:
    try:
        if os.path.exists(args.noise_path):
            common_perturbation = torch.load(args.noise_path)
            common_perturbation = common_perturbation.to(device)
        else:
            print(f"Warning: Noise path '{args.noise_path}' does not exist.")
            args.SE = 0
    except Exception as e:
        print(f"Error loading noise file: {e}")
        args.SE = 0

norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

print(f"[V2 Config] freq_bands={args.freq_bands}, phase_bound={args.phase_bound}, "
      f"phase_smooth={args.phase_smooth}, spec_momentum={args.spec_momentum}, "
      f"layer_aggregation={args.layer_aggregation}")

for i_batch, sampled_batch in tqdm(enumerate(testloader)):
    if 'case_name' in sampled_batch:
        case_name = sampled_batch['case_name'][0]
    elif 'image_meta_dict' in sampled_batch:
        case_name = sampled_batch['image_meta_dict']['filename_or_obj'][0]
    if case_name != args.test_batch and args.test_batch is not None:
        continue

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
        input_image = image[ind, :, :, :]
        input_image_torch = torch.from_numpy(input_image).unsqueeze(0).float().cuda()
        if input_image_torch.shape[1] == 1:
            input_image_torch = repeat(input_image_torch, 'b c h w -> b (repeat c) h w', repeat=3)

        momentum = torch.zeros_like(input_image_torch).detach()

        with torch.no_grad():
            if args.dataset == 'Synapse':
                input_for_model = input_image_torch
            else:
                input_for_model = norm_transform(input_image_torch)
            targets, targets_layer = sam.image_encoder(input_for_model)

        max_val, min_val = (input_image_torch.max(), input_image_torch.min())
        step_size = (input_image_torch.max() - input_image_torch.min()) * (args.att_step_size) / 255
        epsilon = (input_image_torch.max() - input_image_torch.min()) * (args.att_bound) / 255

        eta = torch.zeros_like(input_image_torch).cuda()
        perturb_img = Variable(input_image_torch.data, requires_grad=True)

        CGR = True if args.CGR == 1 else False
        momentum = torch.zeros_like(input_image_torch).detach()

        # Initialize momentum from common perturbation direction if available
        if args.SE == 1 and 'common_perturbation' in locals():
            prior_norm = torch.norm(common_perturbation, p=1)
            if prior_norm > 1e-6:
                prior_direction = (common_perturbation / prior_norm).detach()
                momentum = prior_direction

        eta = torch.zeros_like(input_image_torch).cuda()
        eta = torch.randn_like(eta) * 1e-5
        perturb_img = Variable(input_image_torch.data + eta, requires_grad=True)

        # Pre-compute edge map for guidance
        if args.edge_guidance == 1:
            anatomical_edge = get_edge_map(input_image_torch).detach()

        for step_idx in range(args.att_numstep):
            start_time = time.time()
            sam.image_encoder.zero_grad()
            if args.dataset == 'Synapse':
                perturb_img_input = perturb_img[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
            else:
                perturb_img_input = perturb_img

            if step_idx == 0 and args.SE == 0:
                perturb_img_input = perturb_img_input + torch.randn_like(perturb_img_input) * 0.0001

            if args.dataset == 'Synapse':
                perturb_img_input_pre = perturb_img_input
            else:
                perturb_img_input_pre = norm_transform(perturb_img_input)

            features, features_layer = sam.image_encoder(perturb_img_input_pre)

            # [V2] Use enhanced loss
            loss = calculate_loss_v2(features, targets, features_layer, targets_layer)

            del features, features_layer
            grad = torch.autograd.grad(loss, perturb_img, retain_graph=True, create_graph=False)[0]

            if args.edge_guidance == 1:
                grad = grad * (1.0 + args.edge_gamma * anatomical_edge)

            grad = grad / torch.norm(grad, p=1)

            # [V2 IMPROVEMENT] Spectral momentum filtering
            momentum = 0.5 * momentum.detach() + 1.0 * grad
            if args.spec_momentum:
                momentum = spectral_momentum_filter(momentum)
            momentum = momentum.detach()

            eta = return_eta(momentum, step_size)
            eta = eta.detach()
            del grad

            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)

            end_time = time.time()
            backward_time = "{:.3f}".format(end_time - start_time)
            print(f'[V2] step={step_idx}, loss={loss.item():.6f}, time={backward_time}s')

            eta = torch.clamp(eta, -epsilon, epsilon)
            perturb_img = Variable(input_image_torch.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, min_val, max_val), requires_grad=True)

        # Save adversarial image
        image_data_out = perturb_img[0].detach().cpu().numpy()
        image_data_out = np.transpose(image_data_out, (1, 2, 0))
        image_data_out = np.clip(image_data_out, 0, 1)

        del eta, perturb_img, perturb_img_input

        if is_3d_data:
            folder_path = os.path.join(output_path, case_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            image_path = os.path.join(folder_path, str(ind) + '.png')
            plt.imsave(image_path, image_data_out[:, :, 0], cmap='gray', vmin=0, vmax=1)
        else:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            file_name = os.path.basename(case_name)
            file_base = os.path.splitext(file_name)[0]
            image_path = os.path.join(output_path, file_base + '.png')
            plt.imsave(image_path, image_data_out)

        torch.cuda.empty_cache()
