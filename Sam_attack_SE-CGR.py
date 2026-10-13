import torch
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
import argparse
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
# from segment_anything.utils.transforms import ResizeLongestSide
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
parser.add_argument('--edge_guidance', type=int, default=1, help='Enable Edge-weighted Gradient Guidance (PSAA)')
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

    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)  # 随机生成一个张量，返回的是张量，所以rnd是张量
    rescaled = F.interpolate(
        x, size=[rnd, rnd], mode="bilinear", align_corners=False
    )  # 双线性插值的方法，将张量x转化为共两维，第零维有rnd个，第一维有rnd个（其中）
    h_rem = img_resize - rnd  # h_rem是张量，标量和张量计算得到张量
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,),
                            dtype=torch.int32)  # item函数将单元素张量转化为标量，就是原始形状为[1]，而且只含有单个元素的张量转化为标量
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    padded = F.pad(
        rescaled,
        [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
        value=0,
    )
    # 填充，有顺序，有值
    return padded if torch.rand(1) < diversity_prob else x


# 随机生成一个标量
def normalize_tensor(input_tensor):
    len_tensor = len(input_tensor)  # 获取第0维大小，也就是最外层的
    min_vals = input_tensor.view(len_tensor, -1).min(dim=1, keepdim=True)[0].view(len_tensor, 1, 1, 1, 1)
    max_vals = input_tensor.view(len_tensor, -1).max(dim=1, keepdim=True)[0].view(len_tensor, 1, 1, 1, 1)
    # 展平（第0维就是大小为len_tensor的，保持不变，后面的展平）  keepdim保持维度不变，找dim=1的最小或最大值，用[0]让只保留最小值不保留索引    再变维度，只要让维度数一样就可以运算
    # Normalizing
    normalized_tensor = (input_tensor - min_vals) / (
                max_vals - min_vals)  # 在维度数一样的情况下，每一层维度大小可以自动扩充一致，最终运算结果也是维度不变，改变的只是最高维度上元素的值
    return normalized_tensor


# 逐样本归一化

def return_mean_value(input_tensor):
    input_tensor = torch.abs(input_tensor)
    list_mean = []
    for i in range(len(input_tensor)):
        list_mean.append(input_tensor[i].mean())
    input_mean = torch.stack(list_mean)
    # print(input_mean.shape)
    input_mean = input_mean.view(-1, 1, 1, 1, 1)
    return input_mean


def calculate_loss(features, targets, features_layer, targets_layer, CGR,
                   momentum_update=None, step_size=None, epsilon=None):
    loss_org = criterion(features * 0.9999,
                         targets)  # /torch.abs(targets).mean()  #*0.9999是一种数值稳定性技巧，通过略微缩小特征值来避免可能的梯度爆炸或数值溢出问题。
    loss_cosine = -F.cosine_similarity(features.view(1, -1), targets.view(1, -1))  # 展开成形状为（1，num_elements），二维张量

    pre_layer = int(len(features_layer) * 0.9)
    num_layers = 6
    indices = random.sample(range(num_layers), num_layers - 2)
    selected_features_layer = torch.stack([features_layer[i + pre_layer - num_layers] for i in indices])
    selected_targets_layer = torch.stack([targets_layer[i + pre_layer - num_layers] for i in indices])
    input_mean = return_mean_value(selected_targets_layer)
    if CGR:  # 仅在攻击循环中且开启 GE 时执行
        feature_difference_list = []
        # 逐层应用频谱能量耦合噪声
        for i in range(selected_features_layer.shape[0]):
            F_clean = selected_targets_layer[i].unsqueeze(0)  # [1, C, H, W]
            F_adv = selected_features_layer[i].unsqueeze(0)  # [1, C, H, W]

            # 1. 特征频域化并计算幅度谱 A (用于建模方差)
            F_freq = torch.fft.fftn(F_clean, dim=(-2, -1), norm="ortho")
            A = torch.abs(F_freq)
            Phase_clean = F_freq.angle()  # [相位锁定关键点]

            # 2. 动态计算缩放因子 (恢复 lambda_val 计算)
            # 计算空间域当前差异的能量
            current_diff_spatial = F_adv * 0.999 - F_clean
            diff_energy = torch.mean(torch.abs(current_diff_spatial))
            # 计算频域幅度谱的能量
            spectrum_energy = torch.mean(A)

            # 自适应缩放：差异越大，注入的噪声越强
            lambda_val = 1.0 * (diff_energy / (spectrum_energy + 1e-6))

            # 3. 构建频域噪声方差模型 (Sigma_freq)
            gamma_val = 1.0e-4
            Sigma_freq = lambda_val * A + gamma_val

            # 4. 生成微调增量 c 的幅度 (N_modulated)
            # N_white 是随机源，Sigma_freq 是基于原图幅度的约束形状
            N_white = torch.randn_like(A, device=A.device)
            # 这里 mix_beta 作为一个额外的调节旋钮 (默认为1.0即可)
            N_modulated =  N_white * Sigma_freq

            # 5. 构造加性项 c (xi_struct) 并锁定相位
            # [关键修改]: 使用 N_modulated 作为幅度变化量，乘以 Clean Phase
            # 注意：N_modulated 可能是负数，这代表在该频率上幅度减小或反相，
            # 这符合"参数漂移"的物理意义。
            fft_c = N_modulated * torch.exp(1j * Phase_clean)

            # 逆变换回空间域
            xi_struct_spatial = torch.fft.ifftn(fft_c, dim=(-2, -1), norm="ortho").real

            # F_adv_augmented = F_adv + ξ_struct_spatial # 扰动特征
            # F_clean_augmented = F_clean + ξ_struct_spatial # 扰动目标

            # 采用特征差异项，并乘以幅度谱的均值作为权重，模拟偏差

            # [FINAL ADOPTED LOGIC: Use F_adv - F_clean, and let the gradient be corrected by the subsequent momentum filter.]

            # 特征差分项
            feature_difference = (F_adv * 0.999 - F_clean) - xi_struct_spatial

            feature_difference_list.append(feature_difference.squeeze(0))

        input_mean = return_mean_value(selected_targets_layer)

        feature_difference = torch.stack(feature_difference_list)  # 再次堆叠
        if len(input_mean.shape) > len(feature_difference.shape):
            input_mean = input_mean.view(feature_difference.shape[0], 1, 1, 1)

        feature_difference = feature_difference / (input_mean + 1e-6)  # 防止除零

        l1_losses = (torch.abs(torch.mean(feature_difference, dim=0))).mean()

        # 修正：原代码的归一化方式保留
        # l1_losses = (torch.abs(torch.mean(feature_difference, dim=0))).mean() / input_mean.mean()

        # l1_losses = (torch.abs(torch.mean(feature_difference, dim=0))).mean()

        loss = 1 * l1_losses + 1 * loss_org


        # ... (保持原有的 softmax 和 info_loss 计算) ...
    else:  # 非攻击循环，保留原始 loss 结构

        input_mean = return_mean_value(selected_targets_layer)
        feature_difference = (selected_features_layer * 0.999 - selected_targets_layer) / input_mean
        Gaussian_est = torch.randn_like(feature_difference) * 0.5 + 1
        feature_difference = torch.mul(feature_difference, Gaussian_est)
        l1_losses = (torch.abs(torch.mean(feature_difference, dim=0))).mean()
        loss = 1 * loss_org

        # --- 保持 info_loss ---
    targets_flat = targets.view(*targets.shape[:-2], -1)
    features_flat = features.view(*features.shape[:-2], -1)
    features_softmax = F.softmax(features_flat, dim=2)
    targets_softmax = F.softmax(targets_flat, dim=2)
    info_loss = 1 * F.kl_div(features_softmax.log(), targets_softmax, reduction='batchmean')

    return loss + 0.0 * info_loss

def get_edge_map(input_tensor):
    # input_tensor: [B, C, H, W]
    dx = torch.abs(input_tensor[:, :, :, :-1] - input_tensor[:, :, :, 1:])
    dy = torch.abs(input_tensor[:, :, :-1, :] - input_tensor[:, :, 1:, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = dx + dy
    edge_min = edge.view(edge.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
    edge_max = edge.view(edge.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    edge_map = (edge - edge_min) / (edge_max - edge_min + 1e-6)
    return edge_map


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
            with Image.open(image_path) as img:  # with a as b。a是要执行的语句，b是返回形式；其中执行完语句后会自动退出b
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
            loss_upd = calculate_loss(features[i].unsqueeze(0), targets, new_features_layer,
                                      targets_layer)  # /(scale**0.5)
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


seed = args.random_seed
set_random_seed(seed)
''
sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                            num_classes=args.num_classes,
                                                            checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                            pixel_std=[1, 1, 1])

sam = sam.to(device)
output_path = args.out_path
scaler = GradScaler()
criterion_mse = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()
criterion_cosine = torch.nn.CosineEmbeddingLoss()
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
        'is_3d': True, # 标记为 3D 数据
    },
    'REFUGE': {
        'Dataset': REFUGE,
        'data_path': args.volume_path, # 对应 args.volume_path 传入数据集根目录
        'mode': 'Test', # 注意大小写适配你的 refuge.py
        'transform': tfs,
        'transform_msk': tfs_msk,
        'is_3d': False, # 标记为 2D 数据
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

# 根据数据集类型初始化 DataLoader
if dataset_name == 'Synapse':
    db_test = db_config['Dataset'](base_dir=args.volume_path, list_dir=args.list_dir, split='test_vol')
    is_3d_data = True
elif dataset_name == 'REFUGE':
    db_test = db_config['Dataset'](args=args, data_path=db_config['data_path'], mode=db_config['mode'], transform=db_config['transform'], transform_msk=db_config['transform_msk'])
    is_3d_data = False
elif dataset_name == 'ISIC':
    db_test = db_config['Dataset'](args=args, data_path=db_config['data_path'], mode=db_config['mode'], transform=db_config['transform'], transform_msk=db_config['transform_msk'])
    is_3d_data = False
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4)
if args.SE == 1:
    try:
        # 检查文件是否存在
        if os.path.exists(args.noise_path):
            common_perturbation = torch.load(args.noise_path)
            common_perturbation = common_perturbation.to(device)  # 确保加载到 GPU
        else:
            print(f"Warning: Noise path '{args.noise_path}' does not exist. UMI-GR disabled.")
            args.SE = 0  # 强制关闭以防后续报错
            meta_ini = False
    except Exception as e:
        print(f"Error loading noise file: {e}")
        args.SE = 0
        meta_ini = False


norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

for i_batch, sampled_batch in tqdm(enumerate(testloader)):  # every-time one image from testloader
    if 'case_name' in sampled_batch:
        case_name = sampled_batch['case_name'][0]
    elif 'image_meta_dict' in sampled_batch:
        # REFUGE 和 ISIC 使用 image_meta_dict
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
    # the shape of the image is 1*512*512
    for ind in range(image.shape[0]):
        input_image = image[ind, :, :, :]
        input_image_torch = torch.from_numpy(input_image).unsqueeze(0).float().cuda()
        if input_image_torch.shape[1] == 1:
            # 如果是单通道 (Synapse)，需要 repeat 到 3 通道
            input_image_torch = repeat(input_image_torch, 'b c h w -> b (repeat c) h w', repeat=3)
        else:
            # 如果已经是 3 通道 (REFUGE/ISIC)，保持原样
            pass
        input_image = input_image_torch

        momentum = torch.zeros_like(input_image_torch).detach()

        with torch.no_grad():
            infer_time = time.time()
            if args.dataset == 'Synapse':
                # Synapse 不需要归一化
                input_for_model = input_image_torch
            else:
                # ISIC/REFUGE: 临时归一化，传给模型，但不改变 input_image_torch 本身
                input_for_model = norm_transform(input_image_torch)

            targets, targets_layer = sam.image_encoder(input_for_model)
            end_infer_time = time.time()
            print('infer_time_is:', (end_infer_time - infer_time))

        # transform_matrix,mean_std=pca_transform_matrix(noised_targets,1,11)
        # assert 1==0, print(common_perturbation.max(),common_perturbation.min())
        max_val, min_val = (input_image_torch.max(), input_image_torch.min())
        step_size = (input_image_torch.max() - input_image_torch.min()) * (
            args.att_step_size) / 255  # 归一化处理，将[0,255]（作为基准）映射到[min,max]之间
        epsilon = (input_image_torch.max() - input_image_torch.min()) * (args.att_bound) / 255
        # assert 1==0, print(max_val,min_val)
        ##攻击参数设置
        loss = 0
        eta = torch.zeros_like(input_image_torch).cuda()
        # input_image_torch = input_image_torch+ common_perturbation
        perturb_img = Variable(input_image_torch.data, requires_grad=True)
        loss_tensor_dict = {}
        if args.SE == 1:
            meta_ini = True
        else:
            meta_ini = False
        if args.CGR == 1:
            CGR  = True
        else:
            CGR  = False
        momentum = torch.zeros_like(input_image_torch).detach()

        # 仅当开启 SE 且存在通用扰动时
        if args.SE == 1 and 'common_perturbation' in locals():
            # 计算通用扰动的 L1 范数
            prior_norm = torch.norm(common_perturbation, p=1)
            if prior_norm > 1e-6:
                # 归一化：变成单位方向向量
                prior_direction = (common_perturbation / prior_norm).detach()
                # 赋值给动量，作为起步的"惯性"
                momentum = prior_direction

        # [修改] 2. 图像初始化：从 0 或微小噪声开始 (不再强加 Mask)
        eta = torch.zeros_like(input_image_torch).cuda()
        # 可选：加极微小的随机噪声打破对称性，防止梯度为0
        eta = torch.randn_like(eta) * 1e-5

        perturb_img = Variable(input_image_torch.data + eta, requires_grad=True)

        # [修改] 3. 预计算解剖边缘 (PSAA 引导图)
        # 只需要计算一次，后续循环复用
        if args.edge_guidance == 1:
            anatomical_edge = get_edge_map(input_image_torch).detach()
        for _ in range(args.att_numstep):
            start_time = time.time()
            grad = None
            sam.image_encoder.zero_grad()
            if args.dataset == 'Synapse':
                perturb_img_input = perturb_img[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
            else:
                perturb_img_input = perturb_img
            if _ == 0 and args.SE == 0:
                perturb_img_input = perturb_img_input + torch.randn_like(perturb_img_input) * 0.0001
            if args.dataset == 'Synapse':
                # Synapse 不需要归一化
                perturb_img_input_pre = perturb_img_input
            else:
                # ISIC/REFUGE: 临时归一化，传给模型，但不改变 input_image_torch 本身
                perturb_img_input_pre = norm_transform(perturb_img_input)
            features, features_layer = sam.image_encoder(perturb_img_input_pre)

            loss = calculate_loss(features, targets, features_layer, targets_layer, CGR)

            loss_l2 = criterion_mse(features, targets).detach().cpu()
            loss_tensor_dict[loss.detach().cpu()] = perturb_img_input.detach().cpu()
            del features, features_layer
            grad = torch.autograd.grad(loss, perturb_img, retain_graph=True, create_graph=False)[0]
            if args.edge_guidance == 1:
                # 逻辑：医学图像中，边缘是结构信息的载体。
                # 我们放大边缘处的梯度，抑制平坦背景的梯度。
                # (1.0 + edge_gamma * edge) -> 边缘处梯度放大 gamma 倍，背景保持 1 倍
                # 这迫使攻击能量自动"吸附"（Align）到解剖边界上
                grad = grad * (1.0 + args.edge_gamma * anatomical_edge)
            grad = grad / torch.norm(grad, p=1)
            mean_abs = torch.mean(torch.abs(grad))
            mask_grad = torch.abs(grad) > mean_abs
            # grad=grad*mask_grad
            momentum = 0.5 * momentum.detach() + 1 * grad
            momentum = momentum.detach()

            # eta = 1*step_size * momentum.sign()
            eta = return_eta(momentum, step_size)
            # grad=grad*mask_grad

            eta = eta.detach()
            del grad
            eta = eta
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)

            end_time = time.time()
            backward_time = end_time - start_time
            backward_time = "{:.3f}".format(backward_time)
            start_time = time.time()
            end_time = time.time()
            search_time = end_time - start_time
            search_time = "{:.3f}".format(search_time)
            print('the loss is:', loss.item(), 'the b_time is:',
                  backward_time)  # ,'the avg energy is:',avg_energy)


            eta = torch.clamp(eta, -epsilon, epsilon)


            perturb_img = Variable(input_image_torch.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, min_val, max_val), requires_grad=True)



        image_data_out = perturb_img[0].detach().cpu().numpy()  # [3, H, W]
        # 转回 [H, W, C] 并处理范围
        image_data_out = np.transpose(image_data_out, (1, 2, 0))
        # Clip 到 0-1 (因为 Dataset 一般除过 255)
        image_data_out = np.clip(image_data_out, 0, 1)
        del eta
        del perturb_img, perturb_img_input
        if is_3d_data:
            # Synapse: output/case_name/ind.png (保留切片结构)
            folder_path = os.path.join(output_path, case_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            image_path = os.path.join(folder_path, str(ind) + '.png')
            # 灰度图保存
            plt.imsave(image_path, image_data_out[:, :, 0], cmap='gray', vmin=0, vmax=1)
            print('Image saved:', image_path)
        else:
            # REFUGE/ISIC: output/case_name.png (直接保存)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

                # 处理文件名：只保留文件名部分，去除原有扩展名（如.jpg），改为.png
            file_name = os.path.basename(case_name)  # 防止case_name包含路径
            file_base = os.path.splitext(file_name)[0]  # 去除后缀
            image_path = os.path.join(output_path, file_base + '.png')
            plt.imsave(image_path, image_data_out)

            print('Saved png:', image_path)

        torch.cuda.empty_cache()
    # break



