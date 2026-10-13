import argparse
import os
import random
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import repeat
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from segment_anything import sam_model_registry
from datasets.dataset_synapse import Synapse_dataset
from datasets.refuge import REFUGE
from datasets.isic import ISIC2016

from shared_subspace_codd import load_shared_subspace, blend_gradient_with_subspace, normalize_l1


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str, default='testset/test_vol_h5/')
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--output_dir', type=str, default='/output')
parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/', help='list_dir')
parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
parser.add_argument('--lora_ckpt', type=str, default='checkpoints/epoch_159.pth', help='The checkpoint from LoRA')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--consider_lora', action='store_true', help='Whether to consider lora when attack')
parser.add_argument('--test_batch', type=str, default=None)
parser.add_argument('--out_path', type=str, default='testset/advimages_diia_codd/')
parser.add_argument('--noise_sd', type=float, default=0.15)
parser.add_argument('--smoothed_momentum', type=float, default=0.25)
parser.add_argument('--SE', type=int, default=0)
parser.add_argument('--CGR', type=int, default=1)
parser.add_argument('--sftm_hpyer', type=int, default=4)
parser.add_argument('--noise_path', type=str, default='common_perb/common_perturbation_meta.pth')
parser.add_argument('--att_step_size', type=float, default=2.0)
parser.add_argument('--att_bound', type=float, default=10.0)
parser.add_argument('--att_numstep', type=int, default=10)
parser.add_argument('--dir_search', type=int, default=0)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--edge_guidance', type=int, default=1, help='Enable edge-weighted gradient guidance')
parser.add_argument('--edge_gamma', type=float, default=3.0, help='Intensity of edge weighting')
# CODD / ADS parameters
parser.add_argument('--ads_rho', type=float, default=1.0, help='Power for spectral envelope Sigma = lambda * A^rho + gamma')
parser.add_argument('--ads_gamma', type=float, default=1e-4, help='Small positive floor in the anisotropic drift simulation')
parser.add_argument('--ads_tangential_beta', type=float, default=0.0, help='Optional tangential drift weight; keep 0.0 to preserve original strong behavior')
# Shared radial subspace (offline distilled, optional)
parser.add_argument('--shared_subspace_path', type=str, default='', help='Optional path to offline shared radial subspace bundle')
parser.add_argument('--shared_proj_w', type=float, default=0.15, help='Weight of online gradient projection bonus')
parser.add_argument('--shared_mean_w', type=float, default=0.0, help='Weight of offline mean direction bonus')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion_mse = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def return_mean_value(input_tensor: torch.Tensor) -> torch.Tensor:
    input_tensor = torch.abs(input_tensor)
    list_mean = [input_tensor[i].mean() for i in range(len(input_tensor))]
    input_mean = torch.stack(list_mean)
    input_mean = input_mean.view(-1, 1, 1, 1, 1)
    return input_mean


def get_edge_map(input_tensor: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(input_tensor[:, :, :, :-1] - input_tensor[:, :, :, 1:])
    dy = torch.abs(input_tensor[:, :, :-1, :] - input_tensor[:, :, 1:, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = dx + dy
    edge_min = edge.view(edge.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
    edge_max = edge.view(edge.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    return (edge - edge_min) / (edge_max - edge_min + 1e-6)


def return_eta(momentum: torch.Tensor, step_size: float) -> torch.Tensor:
    upd_momentum = momentum * (step_size / (torch.mean(torch.abs(momentum)) + 1e-12))
    device_local = momentum.device
    momentum_binary = torch.where(momentum > 0, torch.tensor(1.0, device=device_local), torch.tensor(0.0, device=device_local))
    eta = torch.clamp(upd_momentum, -1 * step_size, -0.25 * step_size) * (1 - momentum_binary) + \
          momentum_binary * torch.clamp(upd_momentum, 0.25 * step_size, 1 * step_size)
    return eta


def calculate_loss_codd(
    features: torch.Tensor,
    targets: torch.Tensor,
    features_layer,
    targets_layer,
    enable_ads: bool,
    ads_rho: float = 1.0,
    ads_gamma: float = 1e-4,
    ads_tangential_beta: float = 0.0,
):
    """
    CODD-compatible reinterpretation of the original strong Gaussian module.

    Important:
    - We preserve the original strong objective as much as possible.
    - The only conceptual change is the meaning of phase:
      phase is treated as the source carrier / local reference axis,
      NOT as a quantity assumed invariant in the target model.
    """
    loss_org = criterion(features * 0.9999, targets)

    pre_layer = int(len(features_layer) * 0.9)
    num_layers = 6
    indices = random.sample(range(num_layers), num_layers - 2)
    selected_features_layer = torch.stack([features_layer[i + pre_layer - num_layers] for i in indices])
    selected_targets_layer = torch.stack([targets_layer[i + pre_layer - num_layers] for i in indices])

    if enable_ads:
        feature_difference_list = []
        for i in range(selected_features_layer.shape[0]):
            F_clean = selected_targets_layer[i].unsqueeze(0)  # [1,C,H,W]
            F_adv = selected_features_layer[i].unsqueeze(0)   # [1,C,H,W]

            # Complex carrier of clean feature
            F_freq = torch.fft.fftn(F_clean, dim=(-2, -1), norm="ortho")
            A = torch.abs(F_freq)
            Phase_clean = F_freq.angle()

            current_diff_spatial = F_adv * 0.999 - F_clean
            diff_energy = torch.mean(torch.abs(current_diff_spatial))
            spectrum_energy = torch.mean(A)
            lambda_val = diff_energy / (spectrum_energy + 1e-6)

            # Anisotropic covariance envelope on the source spectral carrier
            Sigma_freq = lambda_val * (A + 1e-12).pow(ads_rho) + ads_gamma
            N_white = torch.randn_like(A, device=A.device)
            radial_coeff = N_white * Sigma_freq

            # Stable carrier-aligned radial component
            fft_c = radial_coeff * torch.exp(1j * Phase_clean)

            # Optional tangential component (disabled by default to preserve original attack strength)
            if ads_tangential_beta > 0:
                N_tan = torch.randn_like(A, device=A.device)
                fft_c = fft_c + ads_tangential_beta * (N_tan * Sigma_freq) * (1j * torch.exp(1j * Phase_clean))

            xi_struct_spatial = torch.fft.ifftn(fft_c, dim=(-2, -1), norm="ortho").real

            feature_difference = (F_adv * 0.999 - F_clean) - xi_struct_spatial
            feature_difference_list.append(feature_difference.squeeze(0))

        input_mean = return_mean_value(selected_targets_layer)
        feature_difference = torch.stack(feature_difference_list)
        if len(input_mean.shape) > len(feature_difference.shape):
            input_mean = input_mean.view(feature_difference.shape[0], 1, 1, 1)
        feature_difference = feature_difference / (input_mean + 1e-6)
        l1_losses = (torch.abs(torch.mean(feature_difference, dim=0))).mean()
        loss = 1.0 * l1_losses + 1.0 * loss_org
    else:
        input_mean = return_mean_value(selected_targets_layer)
        feature_difference = (selected_features_layer * 0.999 - selected_targets_layer) / (input_mean + 1e-6)
        gaussian_est = torch.randn_like(feature_difference) * 0.5 + 1
        feature_difference = torch.mul(feature_difference, gaussian_est)
        _ = (torch.abs(torch.mean(feature_difference, dim=0))).mean()
        loss = 1.0 * loss_org

    # keep the original info_loss placeholder with zero weight
    targets_flat = targets.view(*targets.shape[:-2], -1)
    features_flat = features.view(*features.shape[:-2], -1)
    features_softmax = F.softmax(features_flat, dim=2)
    targets_softmax = F.softmax(targets_flat, dim=2)
    info_loss = 1.0 * F.kl_div(features_softmax.log(), targets_softmax, reduction='batchmean')
    return loss + 0.0 * info_loss


def load_dataset():
    tfs = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    tfs_msk = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    if args.dataset == 'Synapse':
        db_test = Synapse_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split='test_vol')
        is_3d = True
    elif args.dataset == 'REFUGE':
        db_test = REFUGE(args=args, data_path=args.volume_path, mode='Test', transform=tfs, transform_msk=tfs_msk)
        is_3d = False
    elif args.dataset == 'ISIC':
        db_test = ISIC2016(args=args, data_path=args.volume_path, mode='Test', transform=tfs, transform_msk=tfs_msk)
        is_3d = False
    else:
        raise ValueError(args.dataset)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4)
    return testloader, is_3d


def get_case_name(sampled_batch):
    if 'case_name' in sampled_batch:
        return sampled_batch['case_name'][0]
    if 'image_meta_dict' in sampled_batch:
        return sampled_batch['image_meta_dict']['filename_or_obj'][0]
    return 'unknown'


def save_adv_image(image_tensor: np.ndarray, case_name: str, ind: int, is_3d: bool):
    image_data_out = np.transpose(image_tensor, (1, 2, 0))
    image_data_out = np.clip(image_data_out, 0, 1)
    if is_3d:
        folder_path = os.path.join(args.out_path, case_name)
        os.makedirs(folder_path, exist_ok=True)
        image_path = os.path.join(folder_path, str(ind) + '.png')
        plt.imsave(image_path, image_data_out[:, :, 0], cmap='gray', vmin=0, vmax=1)
    else:
        os.makedirs(args.out_path, exist_ok=True)
        file_name = os.path.basename(case_name)
        file_base = os.path.splitext(file_name)[0]
        image_path = os.path.join(args.out_path, file_base + '.png')
        plt.imsave(image_path, image_data_out)
    print('Saved:', image_path)


def maybe_load_common_perturbation() -> Optional[torch.Tensor]:
    if args.SE != 1:
        return None
    if not os.path.exists(args.noise_path):
        print(f"[Warning] noise_path not found: {args.noise_path}")
        return None
    try:
        cp = torch.load(args.noise_path, map_location='cpu')
        return cp.to(device)
    except Exception as e:
        print(f"[Warning] failed to load common perturbation: {e}")
        return None


def main():
    set_random_seed(args.random_seed)

    sam, _ = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    sam = sam.to(device).eval()

    norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    testloader, is_3d_data = load_dataset()
    common_perturbation = maybe_load_common_perturbation()

    shared_bundle = None
    if args.shared_subspace_path and os.path.exists(args.shared_subspace_path):
        try:
            shared_bundle = load_shared_subspace(args.shared_subspace_path, torch.device(device))
            print(f"[Info] loaded shared subspace from {args.shared_subspace_path}")
        except Exception as e:
            print(f"[Warning] failed to load shared subspace: {e}")
            shared_bundle = None

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        case_name = get_case_name(sampled_batch)
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
            input_image_torch = torch.from_numpy(input_image).unsqueeze(0).float().to(device)
            if input_image_torch.shape[1] == 1:
                input_image_torch = repeat(input_image_torch, 'b c h w -> b (repeat c) h w', repeat=3)

            with torch.no_grad():
                if args.dataset == 'Synapse':
                    input_for_model = input_image_torch
                else:
                    input_for_model = norm_transform(input_image_torch)
                targets, targets_layer = sam.image_encoder(input_for_model)

            max_val, min_val = input_image_torch.max(), input_image_torch.min()
            step_size = (max_val - min_val) * args.att_step_size / 255.0
            epsilon = (max_val - min_val) * args.att_bound / 255.0

            eta = torch.randn_like(input_image_torch) * 1e-5
            perturb_img = Variable(input_image_torch.data + eta, requires_grad=True)
            anatomical_edge = get_edge_map(input_image_torch).detach() if args.edge_guidance == 1 else None

            # warm-start momentum
            momentum = torch.zeros_like(input_image_torch).detach()
            if common_perturbation is not None:
                prior_norm = torch.norm(common_perturbation, p=1)
                if prior_norm > 1e-6:
                    momentum = (common_perturbation / prior_norm).detach()
            if shared_bundle is not None and args.shared_mean_w > 0:
                mean_dir = normalize_l1(shared_bundle.mean_direction)
                momentum = momentum + args.shared_mean_w * mean_dir
                momentum = momentum.detach()

            for t in range(args.att_numstep):
                start_time = time.time()
                sam.image_encoder.zero_grad()

                perturb_img_input = perturb_img[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1) if args.dataset == 'Synapse' else perturb_img
                if t == 0 and args.SE == 0:
                    perturb_img_input = perturb_img_input + torch.randn_like(perturb_img_input) * 0.0001

                if args.dataset == 'Synapse':
                    perturb_img_input_pre = perturb_img_input
                else:
                    perturb_img_input_pre = norm_transform(perturb_img_input)

                features, features_layer = sam.image_encoder(perturb_img_input_pre)
                loss = calculate_loss_codd(
                    features=features,
                    targets=targets,
                    features_layer=features_layer,
                    targets_layer=targets_layer,
                    enable_ads=(args.CGR == 1),
                    ads_rho=args.ads_rho,
                    ads_gamma=args.ads_gamma,
                    ads_tangential_beta=args.ads_tangential_beta,
                )

                grad = torch.autograd.grad(loss, perturb_img, retain_graph=False, create_graph=False)[0]
                if args.edge_guidance == 1:
                    grad = grad * (1.0 + args.edge_gamma * anatomical_edge)
                grad = grad / (torch.norm(grad, p=1) + 1e-12)
                grad = blend_gradient_with_subspace(
                    grad,
                    shared_bundle,
                    proj_weight=args.shared_proj_w,
                    mean_weight=0.0,
                )

                momentum = 0.5 * momentum.detach() + 1.0 * grad
                momentum = momentum.detach()
                eta = return_eta(momentum, float(step_size))
                eta = eta.detach()

                perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
                eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)
                perturb_img = Variable(input_image_torch.data + eta, requires_grad=True)
                perturb_img = Variable(torch.clamp(perturb_img, min_val, max_val), requires_grad=True)

                backward_time = time.time() - start_time
                print('step:', t, 'loss:', float(loss.item()), 'b_time:', f'{backward_time:.3f}')

            image_data_out = perturb_img[0].detach().cpu().numpy()
            save_adv_image(image_data_out, case_name, ind, is_3d_data)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
