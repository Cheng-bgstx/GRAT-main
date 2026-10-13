import sys
new_library_path = '/root/data1/fyc/GRAT-main/MedSAM'
sys.path.insert(0, new_library_path)
import torch
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
import argparse
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
# from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
from torch.utils.data import DataLoader
from importlib import import_module
from segment_anything import sam_model_registry
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_isic import ISIC_dataset
from datasets.data_transforms import ResizeAndPad, ToTensorAndNormalize
from einops import repeat
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
import torch.fft
import random
import torchvision.transforms as transforms
import time
import cv2
from one_sample_train import one_sample_train
import torch.distributions as dist
from torchvision.utils import save_image
from datasets.dataset_npz import NPZ_Dataset
from scipy.ndimage import label as label_instances
from scipy import ndimage
import glob
from skimage import transform


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
parser.add_argument('--volume_path', type=str, default='MedSAM/data/npy/CT_Abd/')
parser.add_argument('--output_dir', type=str, default='/output')
parser.add_argument('--img_size', type=int, default=1024, help='Input image size of the network')
parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--num_classes', type=int, default=13)
parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--test_batch', type=str, default=None)
parser.add_argument('--out_path', type=str, default='testset/advimages_ours_smoothedmulti_attack/')
parser.add_argument('--noise_sd', type=float, default=0.15)
parser.add_argument('--smoothed_momentum', type=float, default=0.25)
parser.add_argument('--UMI_GR', type=int, default=0)
parser.add_argument('--sftm_hpyer', type=int, default=4)
parser.add_argument('--noise_path', type=str,
                    default='testset/outcome/sa1b/universal_pert_sa1b_9900.pth')
parser.add_argument('--att_step_size', type=float, default=2.0)
parser.add_argument('--att_bound', type=float, default=10.0)
parser.add_argument('--att_numstep', type=int, default=10)
parser.add_argument('--dir_search', type=int, default=0)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--online_update_method', type=str, default=None,
                    choices=['average', 'sum', 'loss_weighted_average', 'pca', 'average_gradient'],
                    help='strategy')
parser.add_argument('--online_update_freq', type=int, default=2,
                    help='freq')
parser.add_argument('--save_vis', action='store_true', help="store")
parser.add_argument('--online_weight', type=float, default=1.0, help="Weight for the online initialization.")
parser.add_argument(
    '--attack_stage',
    type=str,
    default='image_encoder',
    choices=['image_encoder', 'mask_decoder'],
    help='选择要攻击的 SAM 阶段')
parser.add_argument(
    '--attack_decoder_layer',
    type=int,
    default=0,
    help='当攻击 mask_decoder 时，选择要攻击的 交互处的Transformer 层的索引 (例如 0 或 1)')
parser.add_argument(
    '--attack_decoder_feature',
    type=str,
    default='cross_t2i',
    choices=['self_attn','cross_t2i', 'mlp', 'cross_i2t','final_t2i','upscaling_embedding','mask_mlp', 'iou_head'],
    help='当攻击 mask_decoder 时，选择要攻击的特定特征')
args = parser.parse_args()

if torch.cuda.is_available():
    device = f"cuda:{args.gpu_id}"
else:
    device = "cpu"

print(f"Using device: {device}")


def input_diversity(x):
    resize_rate = 0.9
    diversity_prob = 0.1
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)

    if resize_rate < 1:
        img_size = img_resize
        img_resize = x.shape[-1]

    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(
        x, size=[rnd, rnd], mode="bilinear", align_corners=False
    )
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    padded = F.pad(
        rescaled,
        [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
        value=0,
    )
    return padded if torch.rand(1) < diversity_prob else x


def normalize_tensor(input_tensor):
    len_tensor = len(input_tensor)
    min_vals = input_tensor.view(len_tensor, -1).min(dim=1, keepdim=True)[0].view(len_tensor, 1, 1, 1, 1)
    max_vals = input_tensor.view(len_tensor, -1).max(dim=1, keepdim=True)[0].view(len_tensor, 1, 1, 1, 1)

    # Normalizing
    normalized_tensor = (input_tensor - min_vals) / (max_vals - min_vals)
    return normalized_tensor


def return_mean_value(input_tensor):
    input_tensor = torch.abs(input_tensor)
    list_mean = []
    for i in range(len(input_tensor)):
        list_mean.append(input_tensor[i].mean())
    input_mean = torch.stack(list_mean)
    # print(input_mean.shape)
    input_mean = input_mean.view(-1, 1, 1, 1, 1)
    return input_mean

def get_features_hook(storage_dict, name):
    """
    一个高阶函数，返回一个 PyTorch 钩子。
    这个钩子会将模块的输出保存到传入的 storage_dict 中，键为 name。
    """
    def hook(model, input, output):
        # 很多模块（如 Transformer 块）的输出可能是元组
        if isinstance(output, tuple):
            storage_dict[name] = output[0]
        else:
            storage_dict[name] = output
    return hook

def generate_point_prompts(instance_mask, num_random_points=5):
    # 获取所有前景点的坐标
    y_indices, x_indices = np.where(instance_mask > 0)
    if len(x_indices) == 0:
        return None, None
    # 1. 生成质心点
    center_y, center_x = ndimage.center_of_mass(instance_mask)
    centroid_point = [center_x, center_y]
    # 2. 生成随机点
    all_coords = list(zip(x_indices, y_indices))
    # 确保随机采样的点数不超过物体总像素数
    num_to_sample = min(num_random_points, len(all_coords))
    random_points_tuples = random.sample(all_coords, k=num_to_sample)
    # 转换为 (x, y) 格式
    random_points = [[x, y] for x, y in random_points_tuples]
    # 组合所有点
    point_coords = [centroid_point] + random_points
    # 为所有点生成标签 (1代表前景)
    point_labels = [1] * len(point_coords)
    return point_coords, point_labels


def get_prompts_from_mask(mask):
    prompts = []
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background

    for class_label in unique_labels:
        class_mask = (mask == class_label)
        labeled_instances, num_instances = label_instances(class_mask)
        if num_instances == 0:
            continue
        for i in range(1, num_instances + 1):
            # 为当前这一个实例创建掩码
            instance_mask = (labeled_instances == i).astype(np.uint8)

            # --- 为这个小实例生成提示 ---
            y_indices, x_indices = np.where(instance_mask > 0)
            if len(x_indices) == 0:
                continue
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            bbox = [x_min, y_min, x_max, y_max]
            point_coords, point_labels = generate_point_prompts(instance_mask, num_random_points=5)
            if point_coords:
                prompts.append({
                    "box": bbox,
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                    "label": int(class_label)
                })

    return prompts


def calculate_single_feature_loss(adv_feat, clean_feat):
    """辅助函数：计算两个展平特征之间的负余弦相似度"""
    adv_feat_flat = adv_feat.view(1, -1)
    clean_feat_flat = clean_feat.view(1, -1)

    # 目标是最大化差异，即最小化相似度。
    # 对于损失函数，我们最小化“负相似度”。
    loss = -F.cosine_similarity(adv_feat_flat, clean_feat_flat, dim=-1).mean()
    return loss

def calculate_decoder_loss(adv_features_dict, clean_features_dict, layer_index, feature_name):
    """
    计算对抗样本和干净样本在解码器指定中间层的特征差异。
    """

    # 情况 1: 攻击 "x2" 块内部 (需要 layer_index)
    if feature_name in ['self_attn', 'cross_t2i', 'mlp', 'cross_i2t']:
        target_key = f'layer_{layer_index}_{feature_name}'

        if target_key not in adv_features_dict or target_key not in clean_features_dict:
            print(f"警告: 在特征字典中未找到键 {target_key}。")
            return torch.tensor(0.0, device=device, requires_grad=True)

        adv_feat = adv_features_dict[target_key]
        clean_feat = clean_features_dict[target_key].detach()  # 目标是固定的

        return calculate_single_feature_loss(adv_feat, clean_feat)

    # 情况 2: 攻击 "mask_mlp" (ModuleList)
    elif feature_name == 'mask_mlp':
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        count = 0
        # 迭代所有可能的 mask token MLPs (通常是 4 个)
        for i in range(sam.mask_decoder.num_mask_tokens):  #
            target_key = f'mask_mlp_{i}'
            if target_key in adv_features_dict and target_key in clean_features_dict:
                adv_feat = adv_features_dict[target_key]
                clean_feat = clean_features_dict[target_key].detach()

                total_loss = total_loss + calculate_single_feature_loss(adv_feat, clean_feat)
                count += 1

        if count == 0:
            print(f"警告: 未找到任何 'mask_mlp_' 特征。")
            return torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss / count  # 返回平均损失
    # 情况 3: 攻击 "x2" 块之后的独立模块
    elif feature_name in ['final_t2i', 'iou_head', 'upscaling_embedding']:
        target_key = feature_name

        if target_key not in adv_features_dict or target_key not in clean_features_dict:
            print(f"警告: 在特征字典中未找到键 {target_key}。")
            return torch.tensor(0.0, device=device, requires_grad=True)

        adv_feat = adv_features_dict[target_key]
        clean_feat = clean_features_dict[target_key].detach()

        return calculate_single_feature_loss(adv_feat, clean_feat)

    # 兜底
    else:
        print(f"错误: 未知的 feature_name '{feature_name}'。")
        return torch.tensor(0.0, device=device, requires_grad=True)

def calculate_encoder_loss(features, targets, features_layer, targets_layer, GE):
    loss_org = criterion(features * 0.9999, targets)  # /torch.abs(targets).mean()
    loss_cosine = -F.cosine_similarity(features.view(1, -1), targets.view(1, -1))

    pre_layer = int(len(features_layer) * 0.9)
    num_layers = 6
    indices = random.sample(range(num_layers), num_layers - 2)
    selected_features_layer = torch.stack([features_layer[i + pre_layer - num_layers] for i in indices])
    selected_targets_layer = torch.stack([targets_layer[i + pre_layer - num_layers] for i in indices])
    input_mean = return_mean_value(selected_targets_layer)
    feature_difference = (selected_features_layer * 0.999 - selected_targets_layer) / input_mean
    Gaussian_est = torch.randn_like(feature_difference) * 0.5 + 1
    feature_difference = torch.mul(feature_difference, Gaussian_est)
    l1_losses = (torch.abs(torch.mean(feature_difference, dim=0))).mean()
    if GE:
        loss = 1 * l1_losses + 1 * loss_org
    else:
        loss = 0 * l1_losses + 1 * loss_org
    targets = targets.view(*targets.shape[:-2], -1)
    features = features.view(*features.shape[:-2], -1)
    features = F.softmax(features, dim=2)
    targets = F.softmax(targets, dim=2)
    info_loss = 1 * F.kl_div(features.log(), targets, reduction='batchmean')

    return loss + 0.0 * info_loss


def Minimize_domain_mean(features, mean, std_dev):
    target = torch.normal(mean, std_dev)
    l2 = criterion_mse(features * 0.9999, target)  # /torch.abs(targets).mean()
    l1 = criterion(features * 0.9999, target)
    loss = -1 * l1  # -10*l2
    return loss


def load_and_transform_images(folder_path):
    transformed_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                img_resized = img.resize((512, 512))
                img_array = np.asarray(img_resized) / 255.0
                transformed_images.append(img_array)
    transformed_images = np.array(transformed_images)
    transformed_images = torch.tensor(transformed_images).permute(0, 3, 1, 2).float()
    return transformed_images


def return_org_mean(model, data):
    feature_list = []
    for i in range(len(data)):
        input = data[i]
        input = input.unsqueeze(0)
        # input=input[0,0,:,:]
        # input=input.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        with torch.no_grad():
            features, features_layer = model.image_encoder(input)
        feature_list.append(features.cpu())
        del features, features_layer
        # if i ==1:
        # break
    # assert 1==0,print(features.shape)
    feature = torch.stack(feature_list)
    mean = feature.mean(dim=0)
    variance = feature.var(dim=0)
    std_dev = variance.sqrt()
    # mean=sum(feature_list)/len(feature_list)

    # assert 1==0,print(mean.shape)
    return mean, std_dev


def return_eta(momentum, step_size):
    upd_momentum = momentum * (step_size / torch.mean(abs(momentum)))
    device = momentum.device
    momentum_binary = torch.where(momentum > 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    eta = torch.clamp(upd_momentum, -1 * step_size, -0.25 * step_size) * (1 - momentum_binary) + \
          momentum_binary * torch.clamp(upd_momentum, 0.25 * step_size, 1 * step_size)
    return eta


def eta_search(input_tensor, eta, step, loss, model, targets, targets_layer, epsilon):
    var_steps = [-0.5, -0.25, -0.125, 0, 0.1]
    # var_steps= [-0.5]
    # input_tensor=input_tensor.detach()
    # eta=eta.detach()
    with torch.no_grad():
        step_weight = step / (abs(eta).max())
        adv_input = input_tensor + eta
        upd_input_list = []
        for var_step in var_steps:
            upd_eta = eta + var_step * step_weight * eta
            upd_eta = torch.clamp(upd_eta, -epsilon, epsilon)
            upd_input = input_tensor + upd_eta
            upd_input_list.append(upd_input)
        upd_input_tensors = torch.stack(upd_input_list, dim=0).squeeze(1)
        upd_input_list.append(adv_input)
        # print(upd_input_tensors.shape)
        loss_list = []
        features, features_layer = model.image_encoder(upd_input_tensors)
        for i in range(len(features)):
            #    print(features_layer[i].shape)
            new_features_layer = []
            for tensor in features_layer:
                new_tensor = tensor[i, :].unsqueeze(0)
                new_features_layer.append(new_tensor)
            scale = torch.mean(torch.abs(upd_input_list[i] - input_tensor)) / torch.mean(
                torch.abs(upd_input_list[4] - input_tensor))
            loss_upd = calculate_encoder_loss(features[i].unsqueeze(0), targets, new_features_layer,
                                      targets_layer, GE)  # /(scale**0.5)
            loss_list.append(loss_upd)
        # loss_list.append(loss)
        index = loss_list.index(max(loss_list))
        upd_eta = upd_input_list[index] - input_tensor
        print(index, loss_list, step_weight)
        del upd_input_list, loss_list
    return upd_eta


def return_random_img(image):
    n = image.shape[0]
    random_int = random.randrange(0, n)
    img = image[random_int]
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().cuda()
    img = repeat(img, 'b c h w -> b (repeat c) h w', repeat=3)
    return img


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    random.seed(args.random_seed + worker_id)
    np.random.seed(args.random_seed + worker_id)
    torch.manual_seed(args.random_seed + worker_id)
# --- START ---
def update_online_initialization(method, p_buffer, l_buffer, m_buffer, shape, step_size):
    new_init = torch.zeros(shape)

    if not p_buffer:
        return new_init

    if method == 'average':
        print("  -> Method: Averaging perturbations.")
        perturbations = torch.stack(p_buffer)
        new_init = torch.mean(perturbations, dim=0)

    elif method == 'sum':
        print("  -> Method: Summing perturbations.")
        perturbations = torch.stack(p_buffer)
        new_init = torch.sum(perturbations, dim=0)

    elif method == 'loss_weighted_average':
        print("  -> Method: Loss-weighted averaging.")
        if not l_buffer or len(l_buffer) != len(p_buffer):
            print("  -> Warning: Loss buffer is empty or mismatched. Falling back to simple average.")
            return update_online_initialization('average', p_buffer, [], [], shape, step_size)

        losses = torch.tensor(l_buffer, dtype=torch.float32)

        losses = F.softmax(losses, dim=0)

        perturbations = torch.stack(p_buffer)

        weights = losses.view(-1, 1, 1, 1, 1) if len(perturbations.shape) == 5 else losses.view(-1, 1, 1, 1)
        new_init = torch.sum(weights * perturbations, dim=0)

    elif method == 'pca':
        print("  -> Method: PCA.")
        perturbations = torch.stack(p_buffer)
        num_samples = perturbations.shape[0]

        if num_samples < 1:
            return new_init

        flat_perturbations = perturbations.view(num_samples, -1).numpy()

        n_components = min(1, num_samples)
        pca = PCA(n_components=n_components)
        pca.fit(flat_perturbations)

        principal_component = torch.from_numpy(pca.components_).float()

        avg_flat_perturbation = np.mean(flat_perturbations, axis=0)
        magnitude = np.dot(avg_flat_perturbation, pca.components_[0])

        new_init_flat = magnitude * principal_component
        new_init = new_init_flat.view(shape)

    elif method == 'average_gradient':
        print("  -> Method: Averaging gradient (momentum).")
        if not m_buffer:
            print("  -> Warning: Momentum buffer is empty. Returning zero init.")
            return new_init

        momentums = torch.stack(m_buffer)
        avg_momentum = torch.mean(momentums, dim=0)
        new_init = return_eta(avg_momentum, step_size)

    return new_init


seed = args.random_seed
set_random_seed(seed)
os.makedirs(args.output_dir, exist_ok=True)
vis_path = os.path.join(args.output_dir, 'visualizations')
if args.save_vis:
    os.makedirs(vis_path, exist_ok=True)
sam= sam_model_registry[args.vit_name](checkpoint=args.ckpt)

sam = sam.to(device)
sam.pixel_mean = torch.tensor([0.0, 0.0, 0.0], device=device)
sam.pixel_std = torch.tensor([1.0, 1.0, 1.0], device=device)
sam.eval()

output_path = args.out_path
scaler = GradScaler()
dataset_name = args.dataset

db_test = NPZ_Dataset(base_dir=args.volume_path, split='test')
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
print(f"找到 {len(testloader)} 个病例进行攻击...")
print(f"*** 攻击阶段: {args.attack_stage} ***")
if args.attack_stage == 'mask_decoder':
    print(f"*** 解码器目标: 特征 '{args.attack_decoder_feature}' ***")
    # --- START: MODIFIED (V2) ---
    if args.attack_decoder_feature in ['self_attn', 'cross_t2i', 'mlp', 'cross_i2t']:
        print(f"*** 目标层索引: Layer {args.attack_decoder_layer} ***")
    else:
        print(f"*** (这是一个全局/最终模块, --attack_decoder_layer 将被忽略) ***")

feature_hooks = {} # 用于存储中间特征的字典
hook_handles = [] # 用于保存钩子句柄，以便后续移除

if args.attack_stage == 'mask_decoder':
    print("正在为 MaskDecoder 注册前向钩子...")
    decoder_transformer = sam.mask_decoder.transformer  #
    # --- START: MODIFIED (V2) ---
    # 注册 1: 遍历 "x2" 块 (TwoWayTransformer.layers)
    max_layer_idx = len(decoder_transformer.layers) - 1
    for i, layer in enumerate(decoder_transformer.layers):
        # 1. 'self_attn'
        handle = layer.self_attn.register_forward_hook(
            get_features_hook(feature_hooks, f'layer_{i}_self_attn')
        )
        hook_handles.append(handle)

        # 2. 'cross_t2i'
        handle = layer.cross_attn_token_to_image.register_forward_hook(
            get_features_hook(feature_hooks, f'layer_{i}_cross_t2i')
        )
        hook_handles.append(handle)

        # 3. 'mlp'
        handle = layer.mlp.register_forward_hook(
            get_features_hook(feature_hooks, f'layer_{i}_mlp')
        )
        hook_handles.append(handle)

        # 4. 'cross_i2t'
        handle = layer.cross_attn_image_to_token.register_forward_hook(
            get_features_hook(feature_hooks, f'layer_{i}_cross_i2t')
        )
        hook_handles.append(handle)

    # 注册 2: "x2" 块之后的 'final_t2i'
    handle = decoder_transformer.final_attn_token_to_image.register_forward_hook(
        get_features_hook(feature_hooks, 'final_t2i')
    )
    hook_handles.append(handle)

    # 注册 3: 最终的 'upscaling_embedding'
    handle = sam.mask_decoder.output_upscaling.register_forward_hook(
        get_features_hook(feature_hooks, 'upscaling_embedding')
    )
    hook_handles.append(handle)

    # 注册 4: 最终的 'iou_head'
    handle = sam.mask_decoder.iou_prediction_head.register_forward_hook(
        get_features_hook(feature_hooks, 'iou_head')
    )
    hook_handles.append(handle)

    # 注册 5: 最终的 'mask_mlp' (ModuleList)
    for i, mlp in enumerate(sam.mask_decoder.output_hypernetworks_mlps):
        handle = mlp.register_forward_hook(
            get_features_hook(feature_hooks, f'mask_mlp_{i}')
        )
        hook_handles.append(handle)

    print(f"已成功在 MaskDecoder 上注册 {len(hook_handles)} 个钩子。")

# --- START:  ---
if args.online_update_method:
    print(f"strategy: {args.online_update_method}, freq: {args.online_update_freq}")
    perturbation_buffer = []
    loss_buffer = []
    momentum_buffer = []
    online_initialization = torch.tensor(0).to(device)
    slice_counter = 0
# --- END ---


for i_batch, sampled_batch in tqdm(enumerate(testloader),total=len(testloader)):  # every-time one image from testloader
    h, w = sampled_batch['image'].shape[2:]

    # the shape of the image is 1*512*512
    image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0]
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(
        0).cpu().detach().numpy()  # transform into batch*h*w
    adv_imgs_volume = []
    if case_name != args.test_batch and args.test_batch is not None:
        continue
    trans_analyze_path = 'trans_analyze.txt'
    with open(trans_analyze_path, 'a') as file:
        file.write('\nthe result for MI-FGSM log\n')
        file.write(str(args.UMI_GR))
        file.write('\n')
    common_perturbation = None
    original_gen_epsilon = None
    if args.UMI_GR == 1:
        loaded_noise = torch.load(args.noise_path, map_location='cpu')
        if isinstance(loaded_noise, dict) and 'perturbation' in loaded_noise:
            # 如果加载的是字典并且包含 'perturbation' 键 (MedSAM 原始方式)
            print("  -> Loaded perturbation from a dictionary (checkpoint).")
            common_perturbation = loaded_noise['perturbation']
        elif isinstance(loaded_noise, torch.Tensor):
            # 如果加载的是一个直接的张量 (原始攻击脚本方式)
            print("  -> Loaded perturbation directly from a tensor file.")
            common_perturbation = loaded_noise
        else:
            # 如果格式不被识别，则抛出错误
            raise TypeError(f"Unsupported noise file format in {args.noise_path}. "
                            f"Expected a torch.Tensor or a dict with a 'perturbation' key.")
        if original_gen_epsilon is None and common_perturbation is not None:
            #
            original_gen_epsilon = 10.0 / 255.0
            print(f"  -> 警告: 未在 checkpoint 中找到 Epsilon。假定默认值: 10.0/255.0")
        if common_perturbation is not None:
            common_perturbation = common_perturbation.to(device)
            print(f"  -> UMI perturbation moved to device: {common_perturbation.device}")

    for ind in range(image.shape[0]):

        # --- START:---
        if args.online_update_method:
            slice_counter += 1
        # --- END: ---

        input_image_np = image[ind, :, :]
        mask = label[ind, :, :]
        input_box_torch = None
        if args.attack_stage == 'mask_decoder':
            prompts = get_prompts_from_mask(mask)
            if not prompts:
                print(f"  -> 切片 {ind} 未能生成提示，跳过解码器攻击。")
                adv_imgs_volume.append(input_image_np)  # 添加原始图像
                continue
            scale_x = args.img_size / w
            scale_y = args.img_size / h
            scaled_boxes = []
            for p in prompts:
                box = p['box']  # [xmin, ymin, xmax, ymax]
                scaled_box = [
                box[0] * scale_x,
                box[1] * scale_y,
                box[2] * scale_x,
                box[3] * scale_y]
                scaled_boxes.append(scaled_box)
            input_box_torch = torch.tensor(scaled_boxes, device=device)
            if input_box_torch.dim() == 1:
                input_box_torch = input_box_torch.unsqueeze(0)

        mask_torch = torch.from_numpy(mask > 0).unsqueeze(0).unsqueeze(0).float().to(device)
        mask_torch = repeat(mask_torch, 'b c h w -> b (repeat c) h w', repeat=3)
        
        input_image_torch = torch.from_numpy(input_image_np).unsqueeze(0).unsqueeze(0).float().to(device)
        input_image_torch = repeat(input_image_torch, 'b c h w -> b (repeat c) h w', repeat=3)
        max_val, min_val = (input_image_torch.max(), input_image_torch.min())
        step_size = (input_image_torch.max() - input_image_torch.min()) * (args.att_step_size) / 255
        epsilon = (input_image_torch.max() - input_image_torch.min()) * (args.att_bound) / 255
        if ind == 0 and common_perturbation is not None:
            target_shape = input_image_torch.shape
            temp_common_perturbation = common_perturbation
            if temp_common_perturbation.shape != target_shape:
                print(f"\n[INFO] UMI 扰动尺寸不匹配。正在从 {temp_common_perturbation.shape} 调整至 {target_shape}")
                temp_common_perturbation = F.interpolate(
                    common_perturbation,
                    size=(target_shape[2], target_shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            target_epsilon = epsilon  #
            scale_factor = target_epsilon / original_gen_epsilon
            print(f"  -> 缩放 UMI 扰动。 原始 Epsilon: {original_gen_epsilon:.4f}, 目标 Epsilon: {target_epsilon:.4f}, 缩放因子: {scale_factor:.2f}")
            common_perturbation_ready_for_slice = temp_common_perturbation * scale_factor
        criterion_mse = torch.nn.MSELoss()
        criterion = torch.nn.L1Loss()
        criterion_cosine = torch.nn.CosineEmbeddingLoss()

        input_image = input_image_torch

        targets = None
        targets_layer = None
        clean_decoder_features = {}
        with torch.no_grad():
            infer_time = time.time()
            targets, targets_layer = sam.image_encoder(input_image_torch)
            if args.attack_stage == 'mask_decoder':
                # 编码提示
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=None,
                    boxes=input_box_torch,
                    masks=None
                )
                # 清除钩子字典，准备捕获特征
                feature_hooks.clear()
                masks, iou_pred = sam.mask_decoder(
                    image_embeddings=targets.detach(),  # 使用干净的图像嵌入
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings.detach(),
                    dense_prompt_embeddings=dense_embeddings.detach(),
                    multimask_output=False,
                )
                # 复制捕获到的特征
                clean_decoder_features = feature_hooks.copy()
                feature_hooks.clear()  # 再次清除以备下次使用

                if not clean_decoder_features:
                    print("错误：未能从干净的解码器前向传播中捕获任何特征。")
                    continue
            end_infer_time = time.time()
            print('infer_time_is:', (end_infer_time - infer_time))

        # transform_matrix,mean_std=pca_transform_matrix(noised_targets,1,11)
        if args.UMI_GR == 1 and common_perturbation is not None:
            adv_UMI = common_perturbation.clone().detach().cpu()
            save_image(adv_UMI[:, 0:1, :, :] * 10, 'noise_visualized/UMI.png')
            adv_UMI = adv_UMI.numpy()
            adv_UMI = (adv_UMI * 255).astype('uint8')
            noise_image = Image.fromarray(adv_UMI[0, 0, :, :])
            noise_image.save('noise_visualized/UMI_grey.png')
        momentum = torch.zeros_like(input_image_torch.detach())

        loss, smoothed_loss = 0, 0
        smoothed_direction = torch.tensor(0).to(device)
        eta, smoothed_eta = torch.tensor(0).to(device), torch.tensor(0).to(device)
        bias = 0
        eta_old = 0
        # input_image_torch = input_image_torch+ common_perturbation
        perturb_img = Variable(input_image_torch, requires_grad=True)
        loss_tensor_dict = {}
        if args.UMI_GR == 1:
            meta_ini = True
            GE = True
        else:
            meta_ini = False
            GE = False

        final_loss_for_image = 0.0

        for _ in range(args.att_numstep):
            start_time = time.time()
            pattern_name = 'MI-FGSM'
            if args.UMI_GR == 1:
                pattern_name = 'ours'

            if mask_torch.sum() > -1:

                grad = None
                perturb_img_input = perturb_img[0, 0, :, :]
                opt = torch.optim.SGD([perturb_img], lr=1e-3)
                opt.zero_grad()
                # opt1 = torch.optim.SGD([perturb_img_rot], lr=1e-3)
                # opt1.zero_grad()1
                sam.image_encoder.zero_grad()

                perturb_img_input = perturb_img_input.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                if _ == 0 and args.UMI_GR == 0:
                    perturb_img_input = perturb_img_input + torch.randn_like(perturb_img_input) * 0.0001
                features, features_layer = sam.image_encoder(perturb_img_input)
                if args.attack_stage == 'image_encoder':
                    loss = calculate_encoder_loss(features, targets, features_layer, targets_layer, GE)
                elif args.attack_stage == 'mask_decoder':
                    # 提示编码 (与干净的相同，因为提示是固定的)
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=None,
                        boxes=input_box_torch,
                        masks=None)
                    # 清除钩子字典
                    feature_hooks.clear()
                    # 运行解码器 (使用对抗性的图像嵌入 'features')
                    # (forward call)
                    masks, iou_pred = sam.mask_decoder(
                        image_embeddings=features,  # 关键点：传入对抗性特征
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings.detach(),
                        dense_prompt_embeddings=dense_embeddings.detach(),
                        multimask_output=False,
                    )
                    # 获取对抗性特征
                    adv_decoder_features = feature_hooks.copy()

                    # --- START: MODIFIED (V2) ---
                    # 计算解码器损失 (现在使用更智能的 V2 函数)
                    loss = calculate_decoder_loss(
                        adv_decoder_features,
                        clean_decoder_features,
                        args.attack_decoder_layer,
                        args.attack_decoder_feature
                    )
                if _ == args.att_numstep - 1:
                    final_loss_for_image = loss.item()

                loss_l2 = criterion_mse(features, targets).detach().cpu()
                loss_tensor_dict[loss.detach().cpu()] = perturb_img_input.detach().cpu()
                del features, features_layer
                if args.attack_stage == 'mask_decoder':
                    del adv_decoder_features, sparse_embeddings, dense_embeddings
                grad = torch.autograd.grad(loss, perturb_img, retain_graph=False, create_graph=False)[0]
                grad = grad / torch.norm(grad, p=1)

                mean_abs = torch.mean(torch.abs(grad))
                mask_grad = torch.abs(grad) > mean_abs
                # grad=grad*mask_grad
                momentum = 0.5 * momentum.detach() + 1 * grad
                momentum = momentum.detach()

                # eta = 1*step_size * momentum.sign()
                eta = return_eta(momentum, step_size)
                #新增
                eta = eta.detach()
                del grad
                perturb_img = (perturb_img.detach() + eta).requires_grad_()
                eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)
                #结束，上面的完成了对于单步的截断

                # --- START: ---
                if _ == 0:
                    if meta_ini:
                        eta = 2.5 * eta + 1 * common_perturbation_ready_for_slice
                        torch.save(eta, 'noise_pattern/UMI.pt')

                    if args.online_update_method and online_initialization.abs().sum() > 0:
                        print(f"  -> Applying online initialization with weight: {args.online_weight}")
                        eta = eta + args.online_weight * online_initialization
                # --- END:  ---
                #新删除
                #eta = eta.detach()
                #del grad

                #eta = eta
                #perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
                #eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)
                #结束，现在的含义是，停止了加入common-perb以后的截断，而是保留了common-perb的信息
                # if _==0 and meta_ini:
                # eta = 2.5*eta+1*common_perturbation
                # torch.save(eta,'noise_pattern/UMI.pt')

                if (_ == 5 and args.dir_search == 1):
                    upd_eta = eta - eta_old
                    eta_old = eta_old + upd_eta * 0.1
                    eta = eta_old.clone()

                end_time = time.time()
                backward_time = end_time - start_time
                backward_time = "{:.3f}".format(backward_time)
                start_time = time.time()
                end_time = time.time()
                search_time = end_time - start_time
                search_time = "{:.3f}".format(search_time)
                print('the loss is:', loss.item(), 'the smoothed loss is:', smoothed_loss, 'the b_time is:',
                      backward_time)  # ,'the avg energy is:',avg_energy)

                noise_path = 'noise_pattern/' + pattern_name + str(_)
                torch.save(eta, noise_path)
                print(noise_path)
                eta = torch.clamp(eta, -epsilon, epsilon)

                adv_noise_img = eta.clone().detach().cpu()
                save_image(adv_noise_img[:, 0:1, :, :] * 10, 'noise_visualized/' + str(_) + '.png')
                adv_noise_img = adv_noise_img.numpy()
                adv_noise_img = (adv_noise_img * 255).astype('uint8')
                adv_noise_img = Image.fromarray(adv_noise_img[0, 0, :, :])
                adv_noise_img.save('noise_visualized/' + str(_) + 'grey.png')

                perturb_img = Variable(input_image_torch.data + eta, requires_grad=True)
                perturb_img = Variable(torch.clamp(perturb_img, min_val, max_val), requires_grad=True)

        # 关键修复：保存resize后的对抗图像
        # 攻击时使用了resize预处理，所以保存的也应该是resize后的结果
        # 这样推理时resize一次，就能得到与攻击时相同的数值
        adv_slice_np = perturb_img[0, 0].cpu().detach().numpy().astype(np.float32)
        adv_imgs_volume.append(adv_slice_np)
                # --- START: ---
        if args.online_update_method:
            print(f"  -> Storing results for image {slice_counter} to buffer.")
            perturbation_buffer.append(eta.detach().cpu())
            loss_buffer.append(final_loss_for_image)
            momentum_buffer.append(momentum.detach().cpu())
            if slice_counter % args.online_update_freq == 0 and len(perturbation_buffer) > 0:
                print(
                    f"--- Reached update frequency ({args.online_update_freq}). Updating online initialization... ---")

                new_init = update_online_initialization(
                    method=args.online_update_method,
                    p_buffer=perturbation_buffer,
                    l_buffer=loss_buffer,
                    m_buffer=momentum_buffer,
                    shape=input_image_torch.shape,
                    step_size=step_size)
                online_initialization = new_init.to(device)

                print("--- Clearing buffers. ---")
                perturbation_buffer.clear()
                loss_buffer.clear()
                momentum_buffer.clear()
        # --- END: ---

            # --- 11. (修改点) 将整个病例的对抗样本保存为单个 .npz 文件 ---
    adv_imgs_volume_np = np.stack(adv_imgs_volume)
    output_npz_path = os.path.join(args.output_dir, f"adv_{case_name}.npz")

    original_data = np.load(os.path.join(args.volume_path, f"{case_name}.npz"))
    gts = original_data['gts']
    np.savez_compressed(output_npz_path, imgs=adv_imgs_volume_np, gts=gts)
    print(f"已保存对抗样本 volume 至: {output_npz_path}")

        # --- 12. (修改点) 可选的可视化保存 ---
    if args.save_vis:
        vis_slice_idx = adv_imgs_volume_np.shape[0] // 2
        original_img = image[vis_slice_idx]
        adv_img = adv_imgs_volume_np[vis_slice_idx]
        perturbation = np.abs(adv_img - original_img)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(original_img, cmap='gray');
        axes[0].set_title("Original");
        axes[0].axis('off')
        axes[1].imshow(perturbation, cmap='hot');
        axes[1].set_title("Perturbation");
        axes[1].axis('off')
        axes[2].imshow(adv_img, cmap='gray');
        axes[2].set_title("Adversarial");
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(vis_path, f"{case_name}.png"))
        plt.close()
        print(f"已保存可视化图片至: {os.path.join(vis_path, f'{case_name}.png')}")

print("正在移除所有 PyTorch 钩子...")
for handle in hook_handles:
    handle.remove()
print("脚本执行完毕。")

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)