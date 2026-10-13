import argparse
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from segment_anything import sam_model_registry
from datasets.dataset_synapse import Synapse_dataset
from datasets.refuge import REFUGE
from datasets.isic import ISIC2016

from shared_subspace_codd import fit_shared_subspace, save_shared_subspace, normalize_l1


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str, default='trainset/')
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--dataset', type=str, default='ISIC')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/')
parser.add_argument('--vit_name', type=str, default='vit_b')
parser.add_argument('--out_path', type=str, default='offline_shared_subspace_codd.pth')
parser.add_argument('--num_samples', type=int, default=128)
parser.add_argument('--attack_steps', type=int, default=3)
parser.add_argument('--att_step_size', type=float, default=2.0)
parser.add_argument('--att_bound', type=float, default=10.0)
parser.add_argument('--edge_guidance', type=int, default=1)
parser.add_argument('--edge_gamma', type=float, default=3.0)
parser.add_argument('--ads_rho', type=float, default=1.0)
parser.add_argument('--ads_gamma', type=float, default=1e-4)
parser.add_argument('--ads_tangential_beta', type=float, default=0.0)
parser.add_argument('--rank', type=int, default=8)
parser.add_argument('--random_seed', type=int, default=42)
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = torch.nn.L1Loss()


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def calculate_loss_codd(features, targets, features_layer, targets_layer):
    loss_org = criterion(features * 0.9999, targets)
    pre_layer = int(len(features_layer) * 0.9)
    num_layers = 6
    indices = random.sample(range(num_layers), num_layers - 2)
    selected_features_layer = torch.stack([features_layer[i + pre_layer - num_layers] for i in indices])
    selected_targets_layer = torch.stack([targets_layer[i + pre_layer - num_layers] for i in indices])

    feature_difference_list = []
    for i in range(selected_features_layer.shape[0]):
        F_clean = selected_targets_layer[i].unsqueeze(0)
        F_adv = selected_features_layer[i].unsqueeze(0)
        F_freq = torch.fft.fftn(F_clean, dim=(-2, -1), norm='ortho')
        A = torch.abs(F_freq)
        Phase_clean = F_freq.angle()
        diff_energy = torch.mean(torch.abs(F_adv * 0.999 - F_clean))
        spectrum_energy = torch.mean(A)
        lambda_val = diff_energy / (spectrum_energy + 1e-6)
        Sigma_freq = lambda_val * (A + 1e-12).pow(args.ads_rho) + args.ads_gamma
        N_white = torch.randn_like(A)
        fft_c = (N_white * Sigma_freq) * torch.exp(1j * Phase_clean)
        if args.ads_tangential_beta > 0:
            N_tan = torch.randn_like(A)
            fft_c = fft_c + args.ads_tangential_beta * (N_tan * Sigma_freq) * (1j * torch.exp(1j * Phase_clean))
        xi_struct_spatial = torch.fft.ifftn(fft_c, dim=(-2, -1), norm='ortho').real
        feature_difference = (F_adv * 0.999 - F_clean) - xi_struct_spatial
        feature_difference_list.append(feature_difference.squeeze(0))

    input_mean = return_mean_value(selected_targets_layer)
    feature_difference = torch.stack(feature_difference_list)
    if len(input_mean.shape) > len(feature_difference.shape):
        input_mean = input_mean.view(feature_difference.shape[0], 1, 1, 1)
    feature_difference = feature_difference / (input_mean + 1e-6)
    l1_losses = (torch.abs(torch.mean(feature_difference, dim=0))).mean()
    return 1.0 * l1_losses + 1.0 * loss_org


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
        db = Synapse_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split='train')
        is_3d = True
    elif args.dataset == 'REFUGE':
        # REFUGE 使用 {mode}-400 子目录，训练集为 Training-400（见 datasets/refuge.py）
        db = REFUGE(args=args, data_path=args.volume_path, mode='Training', transform=tfs, transform_msk=tfs_msk)
        is_3d = False
    elif args.dataset == 'ISIC':
        # CSV 文件名为 ISBI2016_ISIC_Part1_Training_GroundTruth.csv（见 datasets/isic.py）
        db = ISIC2016(args=args, data_path=args.volume_path, mode='Training', transform=tfs, transform_msk=tfs_msk)
        is_3d = False
    else:
        raise ValueError(args.dataset)
    return DataLoader(db, batch_size=1, shuffle=True, num_workers=4), is_3d


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
    loader, is_3d = load_dataset()

    gradients: List[torch.Tensor] = []
    seen = 0
    for sampled_batch in tqdm(loader):
        if seen >= args.num_samples:
            break
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
            if seen >= args.num_samples:
                break
            input_image = image[ind, :, :, :]
            input_image_torch = torch.from_numpy(input_image).unsqueeze(0).float().to(device)
            if input_image_torch.shape[1] == 1:
                input_image_torch = repeat(input_image_torch, 'b c h w -> b (repeat c) h w', repeat=3)

            with torch.no_grad():
                input_for_model = input_image_torch if args.dataset == 'Synapse' else norm_transform(input_image_torch)
                targets, targets_layer = sam.image_encoder(input_for_model)

            max_val, min_val = input_image_torch.max(), input_image_torch.min()
            step_size = (max_val - min_val) * args.att_step_size / 255.0
            epsilon = (max_val - min_val) * args.att_bound / 255.0
            perturb_img = Variable(input_image_torch.data + torch.randn_like(input_image_torch) * 1e-5, requires_grad=True)
            anatomical_edge = get_edge_map(input_image_torch).detach() if args.edge_guidance == 1 else None

            local_grads = []
            for _ in range(args.attack_steps):
                sam.image_encoder.zero_grad()
                perturb_img_input = perturb_img[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1) if args.dataset == 'Synapse' else perturb_img
                perturb_img_input_pre = perturb_img_input if args.dataset == 'Synapse' else norm_transform(perturb_img_input)
                features, features_layer = sam.image_encoder(perturb_img_input_pre)
                loss = calculate_loss_codd(features, targets, features_layer, targets_layer)
                grad = torch.autograd.grad(loss, perturb_img, retain_graph=False, create_graph=False)[0]
                if args.edge_guidance == 1:
                    grad = grad * (1.0 + args.edge_gamma * anatomical_edge)
                grad = normalize_l1(grad.detach())
                local_grads.append(grad.cpu())
                perturb_img = Variable(torch.clamp(perturb_img.data + step_size * grad.sign(), min_val, max_val), requires_grad=True)
                eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)
                perturb_img = Variable(input_image_torch.data + eta, requires_grad=True)

            gradients.append(torch.stack(local_grads, dim=0).mean(dim=0))
            seen += 1

    bundle = fit_shared_subspace(gradients, rank=args.rank)
    bundle.meta['dataset'] = args.dataset
    bundle.meta['num_samples'] = seen
    bundle.meta['attack_steps'] = args.attack_steps
    bundle.meta['ads_rho'] = args.ads_rho
    bundle.meta['ads_gamma'] = args.ads_gamma
    save_shared_subspace(bundle, args.out_path)
    print(f"Saved shared subspace to {args.out_path}")
    print(f"Rank={bundle.meta['rank']} Samples={seen}")


if __name__ == '__main__':
    main()
