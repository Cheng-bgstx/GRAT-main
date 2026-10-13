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
parser.add_argument('--UMI', type=int, default=0)
parser.add_argument('--GE', type=int, default=0)
parser.add_argument('--sftm_hpyer', type=int, default=4)
parser.add_argument('--noise_path', type=str,
                    default='common_perb/common_perturbation_meta_0.2sp_bound4_it5batchit20.05upsize_ls0.01mulmd1.pth')
parser.add_argument('--att_step_size', type=float, default=2.0)
parser.add_argument('--att_bound', type=float, default=10.0)
parser.add_argument('--att_numstep', type=int, default=10)
parser.add_argument('--dir_search', type=int, default=0)
parser.add_argument('--random_seed', type=int, default=42)

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


def calculate_loss(features, targets, features_layer, targets_layer, GE):
    loss_org = criterion(features * 0.9999,
                         targets)  # /torch.abs(targets).mean()  #*0.9999是一种数值稳定性技巧，通过略微缩小特征值来避免可能的梯度爆炸或数值溢出问题。
    loss_cosine = -F.cosine_similarity(features.view(1, -1), targets.view(1, -1))  # 展开成形状为（1，num_elements），二维张量

    pre_layer = int(len(features_layer) * 0.9)
    num_layers = 6
    indices = random.sample(range(num_layers), num_layers - 2)
    selected_features_layer = torch.stack([features_layer[i + pre_layer - num_layers] for i in indices])
    selected_targets_layer = torch.stack([targets_layer[i + pre_layer - num_layers] for i in indices])
    # 中间层特征提取，随机选取中间层的输出结果，就是从pre_layer，往前选取几层。-num_layers没啥意义就是确保i不会
    input_mean = return_mean_value(selected_targets_layer)
    feature_difference = (selected_features_layer * 0.999 - selected_targets_layer) / input_mean
    Gaussian_est = torch.randn_like(feature_difference) * 0.5 + 1
    feature_difference = torch.mul(feature_difference, Gaussian_est)  # 生成N(1,0.5²)的高斯噪声
    l1_losses = (torch.abs(torch.mean(feature_difference,
                                      dim=0))).mean()  # torch.mean求平均值函数，只要看物理含义就行：让对应dim上的大小变成1，keepdim=True代表保留所求dim的梯度；直接mean就是变成一个标量
    if GE:  # 需要注意，先对dim=0取平均，每一层的差异取平均加入
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
sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt)

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

if args.UMI == 1:
    try:
        # 检查文件是否存在
        if os.path.exists(args.noise_path):
            common_perturbation = torch.load(args.noise_path)
            common_perturbation = common_perturbation.to(device)  # 确保加载到 GPU
        else:
            print(f"Warning: Noise path '{args.noise_path}' does not exist. UMI-GR disabled.")
            args.UMI = 0  # 强制关闭以防后续报错
            meta_ini = False
    except Exception as e:
        print(f"Error loading noise file: {e}")
        args.UMI = 0
        meta_ini = False
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
            if input_image_torch.max() <= 1.0:
                input_image_torch = input_image_torch * 255.0

            # 确保输入是 3 通道 (SAM 需要 3 通道)
            if input_image_torch.shape[1] == 1:
                input_image_torch = repeat(input_image_torch, 'b c h w -> b (repeat c) h w', repeat=3)

            input_processed = sam.preprocess(input_image_torch)
            targets, targets_layer = sam.image_encoder(input_processed)
            end_infer_time = time.time()
            print('infer_time_is:', (end_infer_time - infer_time))

        # transform_matrix,mean_std=pca_transform_matrix(noised_targets,1,11)
        # assert 1==0, print(common_perturbation.max(),common_perturbation.min())
        step_size = args.att_step_size
        epsilon = args.att_bound
        # assert 1==0, print(max_val,min_val)
        ##攻击参数设置
        loss = 0
        eta = torch.tensor(0).cuda()
        # input_image_torch = input_image_torch+ common_perturbation
        perturb_img = Variable(input_image_torch.data, requires_grad=True)
        loss_tensor_dict = {}
        if args.UMI == 1:
            meta_ini = True
        else:
            meta_ini = False
        if args.GE == 1:
            GE = True
        else:
            GE = False
        for _ in range(args.att_numstep):
            start_time = time.time()
            grad = None
            sam.image_encoder.zero_grad()
            if args.dataset == 'Synapse':
                perturb_img_input = perturb_img[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
            else:
                perturb_img_input = perturb_img
            if _ == 0 and args.UMI == 0:
                perturb_img_input = perturb_img_input + torch.randn_like(perturb_img_input) * 0.0001
            perturb_img_processed = sam.preprocess(perturb_img_input)
            features, features_layer = sam.image_encoder(perturb_img_processed)

            loss = calculate_loss(features, targets, features_layer, targets_layer, GE)

            loss_l2 = criterion_mse(features, targets).detach().cpu()
            loss_tensor_dict[loss.detach().cpu()] = perturb_img_input.detach().cpu()
            del features, features_layer
            grad = torch.autograd.grad(loss, perturb_img, retain_graph=True, create_graph=False)[0]
            grad = grad / torch.norm(grad, p=1)

            mean_abs = torch.mean(torch.abs(grad))
            mask_grad = torch.abs(grad) > mean_abs
            # grad=grad*mask_grad
            momentum = 0.5 * momentum.detach() + 1 * grad
            momentum = momentum.detach()

            # eta = 1*step_size * momentum.sign()
            eta = return_eta(momentum, step_size)

            eta = eta.detach()
            del grad

            eta = eta
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)
            if _ == 0 and meta_ini:
                eta = 2.5 * eta + 1 * common_perturbation

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
            perturb_img = Variable(torch.clamp(perturb_img, 0, 255), requires_grad=True)



        image_data_out = perturb_img[0].detach().cpu().numpy()  # [3, H, W]
        # 转回 [H, W, C] 并处理范围
        image_data_out = np.transpose(image_data_out, (1, 2, 0))
        # Clip 到 0-1 (因为 Dataset 一般除过 255)
        image_data_out = np.clip(image_data_out, 0, 255)
        image_data_out = image_data_out.astype(np.uint8)
        del eta
        del perturb_img, perturb_img_input
        if is_3d_data:
            # Synapse: output/case_name/ind.png (保留切片结构)
            folder_path = os.path.join(output_path, case_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            image_path = os.path.join(folder_path, str(ind) + '.png')
            # 灰度图保存
            cv2.imwrite(image_path, image_data_out[:, :, 0])
            print('Image saved:', image_path)
        else:
            # REFUGE/ISIC: output/case_name.png (直接保存)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

                # 处理文件名：只保留文件名部分，去除原有扩展名（如.jpg），改为.png
            file_name = os.path.basename(case_name)  # 防止case_name包含路径
            file_base = os.path.splitext(file_name)[0]  # 去除后缀
            image_path = os.path.join(output_path, file_base + '.png')
            Image.fromarray(image_data_out).save(image_path)

            print('Saved png:', image_path)

        torch.cuda.empty_cache()
    # break



