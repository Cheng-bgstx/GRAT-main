
import os
import sys
from tqdm import tqdm
import logging
from PIL import Image
import argparse
import random
import numpy as np

# NumPy 1.24+ 移除 np.bool；旧版 medpy 仍用 np.bool（如 metric.binary.dc）
if not hasattr(np, "bool"):
    np.bool = np.bool_  # noqa: NPY001

import torch
import cv2
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from importlib import import_module
from segment_anything import sam_model_registry
import matplotlib.image as mpimg
from icecream import ic
from einops import repeat
from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk
from datasets.dataset_synapse import Synapse_dataset
from datasets.isic import ISIC2016
from datasets.refuge import REFUGE
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.ndimage import label as label_cc
class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: 'liver', 6: 'stomach',
                 7: 'aorta', 8: 'pancreas'}


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    # 正常计算
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    
    # 双方都为空（真阴性），预测完美
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1, 0
        
    # 【修复】：一方为空，另一方不为空（完全错误）
    else:
        # Dice 为 0
        # HD95 应该设为图像的最大可能距离（例如对角线长度）
        # 这里假设输入是 2D 或 3D，计算大概的对角线
        # 如果不知道具体尺寸，可以设为一个比你数据集中最大物体还要大的值，例如 100 或 300 (像素)
        # 或者计算当前 slice 的对角线: np.sqrt(pred.shape[0]**2 + pred.shape[1]**2)
        
        max_dist = np.sqrt(np.sum(np.array(pred.shape) ** 2)) # 动态计算对角线
        return 0, max_dist

COLOR_PARLETTE = np.array([
    [0, 0, 0],       # Class 0: Background
    [255, 0, 0],     # Class 1: Red
    [0, 255, 0],     # Class 2: Green
    [0, 0, 255],     # Class 3: Blue
    [255, 255, 0],   # Class 4: Yellow
    [0, 255, 255],   # Class 5: Cyan
    [255, 0, 255],   # Class 6: Magenta
    [255, 128, 0],   # Class 7: Orange
    [128, 0, 255],   # Class 8: Purple
    [128, 128, 128], # Class 9 (备用)
], dtype=np.uint8)

def apply_palette(mask, palette):
    """将整数掩码 (H, W) 转换为 RGB 彩色图 (H, W, 3)"""
    # 限制 mask 范围，防止越界
    mask = np.clip(mask, 0, len(palette) - 1).astype(int)
    return palette[mask]


def test_single_volume(image, image_clean, label, net, classes, multimask_output, patch_size, test_save_path=None,
                       case=None, z_spacing=1, is_3d=True):
    # 1. 预处理
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if image.max() > 1.1:
        image = image / 255.0

    # 2. 推理 (Inference)
    if is_3d:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice_img = image[ind, :, :]
            x, y = slice_img.shape[0], slice_img.shape[1]
            if x == patch_size[0] and y == patch_size[1]:
                slice_input = slice_img
            else:
                slice_input = zoom(slice_img, (patch_size[0] / x, patch_size[1] / y), order=3)
            inputs = torch.from_numpy(slice_input).unsqueeze(0).unsqueeze(0).float().cuda()
            inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
            net.eval()
            with torch.no_grad():
                outputs = net(inputs, multimask_output, patch_size[0])
                output_masks = outputs['masks']
                out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                out_h, out_w = out.shape
                if x != out_h or y != out_w:
                    pred = zoom(out, (x / out_h, y / out_w), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        # 2D 推理逻辑 ... (保持不变)
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (1, patch_size[0] / x, patch_size[1] / y), order=3)
        inputs = torch.from_numpy(image).unsqueeze(0).float().cuda()
        if inputs.shape[1] == 3:  # RGB
            inputs = norm_transform(inputs)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)

    # 3. 计算指标 (Metrics)
    metric_list = []
    if not is_3d and label.ndim == 3 and label.shape[0] == 2:
        pred_cup = (prediction == 2)
        pred_disc_ring = (prediction == 1)
        pred_disc_full = np.logical_or(pred_disc_ring, pred_cup)
        metric_disc = calculate_metric_percase(pred_disc_full, label[1])
        metric_cup = calculate_metric_percase(pred_cup, label[0])
        metric_list.append(metric_disc)
        metric_list.append(metric_cup)
    elif not is_3d:
        if label.ndim == 3: label = label.squeeze(0)
        metric_list.append(calculate_metric_percase(prediction == 1, label > 0))
    else:
        for i in range(1, classes + 1):
            metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # 4. 保存结果 (可视化 PNG + 3D NIfTI)
    if test_save_path is not None:
        # --- A. 保存可视化 PNG (取效果最好的一层) ---
        vis_dir = os.path.join(test_save_path, "vis_png")
        os.makedirs(vis_dir, exist_ok=True)

        if is_3d:
            slice_sums = np.sum(label, axis=(1, 2))
            if slice_sums.max() == 0:
                vis_ind = label.shape[0] // 2
            else:
                vis_ind = np.argmax(slice_sums)

            gt_data_slice = label[vis_ind]
            pred_data_slice = prediction[vis_ind]
            img_clean_slice = image_clean.squeeze(0).cpu().detach().numpy()[vis_ind]
        else:
            gt_data_slice = label
            pred_data_slice = prediction
            img_clean_slice = image_clean.squeeze(0).cpu().detach().numpy()
            if len(gt_data_slice.shape) == 3: gt_data_slice = gt_data_slice[0]

        # 归一化原始图像 (修复了这里原本的变量名错误)
        img_clean_show = (img_clean_slice - img_clean_slice.min()) / (
                    img_clean_slice.max() - img_clean_slice.min() + 1e-8)
        img_clean_show = (img_clean_show * 255).astype(np.uint8)

        # 保存 PNG
        if classes == 1:
            gt_vis = (gt_data_slice * 255).astype(np.uint8)
            pred_vis = (pred_data_slice * 255).astype(np.uint8)
        else:
            gt_vis = apply_palette(gt_data_slice, COLOR_PARLETTE)  # 注意: 你原代码里写的是 COLOR_PARLETTE (拼写错误)，这里改回 PALETTE
            pred_vis = apply_palette(pred_data_slice, COLOR_PARLETTE)
            gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR)
            pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(vis_dir, f"{case}_gt.png"), gt_vis)
        cv2.imwrite(os.path.join(vis_dir, f"{case}_adv_prediction.png"), pred_vis)
        cv2.imwrite(os.path.join(vis_dir, f"{case}_original.png"), img_clean_show)

        # --- B. 保存 3D NIfTI 文件 (这是你之前缺失的部分) ---
        # 只有在 3D 模式下且需要保存 nii 时才执行
        if is_3d:
            # 使用原始的 3D image 变量，而不是切片后的 img_clean_slice
            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            sitk.WriteImage(img_itk, os.path.join(test_save_path, f"{case}_img.nii.gz"))

            pred_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
            sitk.WriteImage(pred_itk, os.path.join(test_save_path, f"{case}_pred.nii.gz"))

            # 如果需要也可以保存 GT
            gt_itk = sitk.GetImageFromArray(label.astype(np.float32))
            sitk.WriteImage(gt_itk, os.path.join(test_save_path, f"{case}_gt.nii.gz"))

    return metric_list


def load_adv_images(folder_path, case_name, is_3d=True):
    # Synapse: 这是一个包含多个切片的文件夹
    if is_3d:
        case_folder = os.path.join(folder_path, case_name)
        if not os.path.exists(case_folder):
            return None

        # 按文件名数字排序 (0.png, 1.png...) 确保和 3D Label 顺序一致
        imgs_names = sorted([f for f in os.listdir(case_folder) if f.endswith(('.png', '.jpg'))],
                            key=lambda x: int(os.path.splitext(x)[0]))

        images = []
        for f in imgs_names:
            img_path = os.path.join(case_folder, f)
            img = Image.open(img_path).convert('L')  # 读灰度
            images.append(np.array(img))

        if not images: return None
        return np.array(images)  # 返回 [Depth, H, W]

    # ISIC / REFUGE: 这是一个单独的 PNG 文件
    else:
        # 1. 优先尝试读取 .npy 文件 (由 Sam_attack.py 生成)
        npy_path = os.path.join(folder_path, case_name + '.npy')
        if os.path.exists(npy_path):
            img = np.load(npy_path)  # 加载的数据形状通常是 [H, W, 3]，范围 0-1
            # 转换为 [3, H, W] 以适配模型输入
            img = np.transpose(img, (2, 0, 1))
            return img

        # 2. 如果没有 .npy，尝试读取 .png (旧逻辑兼容)
        img_path = os.path.join(folder_path, case_name + '.png')
        if not os.path.exists(img_path):
            # 尝试不带后缀的路径
            img_path = os.path.join(folder_path, case_name)
            if not os.path.exists(img_path):
                return None

        img = Image.open(img_path).convert('RGB')
        img = np.array(img)  # [H, W, 3]
        img = np.transpose(img, (2, 0, 1))  # [3, H, W]
        return img


def clean_batch_to_adv_tensor(image_clean, is_3d: bool):
    """
    将 dataloader 的干净图转成与 load_adv_images + unsqueeze(0) 后一致的 tensor，
    供 test_single_volume 使用（干净评测 / Benign）。
    """
    x = image_clean
    if torch.is_tensor(x):
        x = x.squeeze(0).float().cpu().numpy()
    else:
        x = np.asarray(x).squeeze(0)
    if is_3d:
        if x.ndim == 4 and x.shape[0] == 3:
            x = x[0]
        elif x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        images_array = np.asarray(x, dtype=np.float32)
    else:
        if x.ndim == 3 and x.shape[0] == 3:
            images_array = np.asarray(x, dtype=np.float32)
        elif x.ndim == 3 and x.shape[-1] == 3:
            images_array = np.transpose(np.asarray(x, dtype=np.float32), (2, 0, 1))
        else:
            images_array = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(images_array).float().unsqueeze(0)


def inference(args, multimask_output, db_config, model, adv_root_path, test_save_path=None):
    if args.dataset == 'Synapse':
        db_test = db_config['Dataset'](base_dir=args.volume_path, list_dir=args.list_dir, split='test_vol')
        is_3d_data = True
    else:
        # ISIC / REFUGE
        db_test = db_config['Dataset'](args=args, data_path=db_config['data_path'], mode='Test',
                                       transform=db_config['transform'], transform_msk=db_config['transform_msk'])
        is_3d_data = False
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4)
    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    metric_list = []
    len_cases = 0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        label = sampled_batch['label']
        image_clean = sampled_batch['image']
        # 获取 case_name (不同数据集 key 可能不同)
        if 'case_name' in sampled_batch:
            case_name = sampled_batch['case_name'][0]
        elif 'image_meta_dict' in sampled_batch:
            case_name = sampled_batch['image_meta_dict']['filename_or_obj'][0]
        if args.test_batch is not None and case_name != args.test_batch:
            continue
        if getattr(args, "use_clean_image", False):
            image_adv = clean_batch_to_adv_tensor(image_clean, is_3d_data)
        else:
            images_array = load_adv_images(adv_root_path, case_name, is_3d=is_3d_data)
            if images_array is None:
                logging.warning(f"Adv image not found for {case_name}, skipping.")
                continue
            image_adv = torch.tensor(images_array).float().unsqueeze(0)
        if is_3d_data:
            if image_adv.shape[1] != label.shape[1]:
                logging.warning(f"Mismatch: Adv {image_adv.shape} vs Label {label.shape}")
                continue
        metric_i = test_single_volume(
            image=image_adv,  # 对抗样本
            image_clean=image_clean,
            label=label,  # 原始 GT
            net=model,
            classes=args.num_classes,
            multimask_output=multimask_output,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=db_config.get('z_spacing', 1),
            is_3d=is_3d_data
        )
        metric_list.append(metric_i)
        len_cases = len_cases + 1
        case_mean_dice = np.mean(metric_i, axis=0)[0]
        case_mean_hd95 = np.mean(metric_i, axis=0)[1]
        if args.dataset == 'REFUGE':
            # REFUGE 特有逻辑：分别打印 Disc (Index 0) 和 Cup (Index 1)
            disc_dice = metric_i[0][0]
            disc_hd95 = metric_i[0][1]
            cup_dice = metric_i[1][0]
            cup_hd95 = metric_i[1][1]

            logging.info(
                f'idx {i_batch} case {case_name} | '
                f'Mean Dice: {case_mean_dice:.4f} Mean HD95: {case_mean_hd95:.2f} | '
                f'Disc Dice: {disc_dice:.4f} HD95: {disc_hd95:.2f} | '
                f'Cup Dice: {cup_dice:.4f} HD95: {cup_hd95:.2f}'
            )

        elif args.dataset == 'ISIC':
            # ISIC 只有一类 (Lesion)
            logging.info(
                f'idx {i_batch} case {case_name} | '
                f'Lesion Dice: {case_mean_dice:.4f} HD95: {case_mean_hd95:.2f}'
            )

        else:
            # Synapse (多类，打印平均值即可，太长了不需要每个 Case 都打印所有器官)
            logging.info(
                'idx %d case %s mean_dice %f mean_hd95 %f' %
                (i_batch, case_name, case_mean_dice, case_mean_hd95)
            )
    if len(metric_list) == 0:
        logging.error(
            "No cases evaluated (empty metric_list). "
            "For benign/clean eval use --use_clean_image; "
            "for adversarial check --out_path and PNG layout."
        )
        return float("nan"), float("nan")

    metric_array = np.array(metric_list)

    # 2. 计算每一类的平均值 (沿着 Case 维度 axis=0 取平均) -> shape = (Class数, 2)
    per_class_mean = np.mean(metric_array, axis=0)


    global_mean = np.mean(per_class_mean, axis=0)
    performance, mean_hd95 = global_mean[0], global_mean[1]

    # --- 针对 Synapse 数据集打印每一类的详细结果 ---
    if args.dataset == 'Synapse':
        logging.info("------ Synapse Per Class Performance ------")
        for i in range(args.num_classes):
            class_name = class_to_name.get(i + 1, f'Class {i + 1}')  # class_to_name key 从 1 开始
            class_dice = per_class_mean[i, 0]
            class_hd95 = per_class_mean[i, 1]
            logging.info(f'{class_name:<15} | Dice: {class_dice:.4f} | HD95: {class_hd95:.2f}')
        logging.info("-------------------------------------------")

    # --- 针对 REFUGE 数据集打印 Disc/Cup 平均结果 ---
    elif args.dataset == 'REFUGE':
        logging.info("------ REFUGE Per Class Performance ------")
        logging.info(f'Disc (Class 1)    | Dice: {per_class_mean[0, 0]:.4f} | HD95: {per_class_mean[0, 1]:.2f}')
        logging.info(f'Cup  (Class 2)    | Dice: {per_class_mean[1, 0]:.4f} | HD95: {per_class_mean[1, 1]:.2f}')
        logging.info("------------------------------------------")

    # 打印最终的总平均值
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    logging.info("Testing Finished!")
    return performance, mean_hd95


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--volume_path', type=str, default='testset/test_vol_h5/')
    parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/', help='list_dir')
    parser.add_argument('--output_dir', type=str, default='./output/adv_sam_lora')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='checkpoints/epoch_5_bd1_ld0.1_out1.pth',
                        help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
    parser.add_argument('--consider_lora', action='store_true', help='Whether to consider lora when attack')
    parser.add_argument('--test_batch', type=str, default=None)
    parser.add_argument('--out_path', type=str, default='testset/advimages-test_batch1_lowpass/')
    parser.add_argument(
        '--use_clean_image',
        action='store_true',
        help='Benign/clean eval: run on dataloader clean images (ignore out_path PNGs). '
        'Use for tables without pre-exporting clean_export_* folders.',
    )
    parser.add_argument('--noise_sd', type=float, default=0.15)
    parser.add_argument('--smoothed_momentum', type=float, default=0.25)
    parser.add_argument('--smoothed_freq', type=int, default=3)
    parser.add_argument('--sftm_hpyer', type=int, default=4)

    parser.add_argument('--att_step_size', type=float, default=2.0)
    parser.add_argument('--att_bound', type=float, default=10.0)
    parser.add_argument('--att_numstep', type=int, default=10)

    parser.add_argument('--noise_path', type=str,
                        default='common_perb/common_perturbation_meta_0.05sp_bound8_it6_0.2upsize_ls0.25.pth')
    args = parser.parse_args()
    if args.dataset == 'ISIC':
        args.num_classes = 1
    elif args.dataset == 'REFUGE':
        args.num_classes = 2
    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1
        },
        'REFUGE': {
            'Dataset': REFUGE,
            'data_path': args.volume_path,  # 注意这里是用 args.volume_path 传入 data_path
            'mode': 'Test',
            'transform': None,  # 后面会定义 tfs
            'transform_msk': None
        },
        'ISIC': {
            'Dataset': ISIC2016,
            'data_path': args.volume_path,
            'mode': 'Test',
            'transform': None,
            'transform_msk': None
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    folder_path = args.out_path
    tfs = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    tfs_msk = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # 更新 config
    dataset_config['REFUGE']['transform'] = tfs
    dataset_config['REFUGE']['transform_msk'] = tfs_msk
    dataset_config['ISIC']['transform'] = tfs
    dataset_config['ISIC']['transform_msk'] = tfs_msk
    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])

    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)
    net = net.cuda()  # Fix: ensure all LoRA params are on GPU after loading

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    performance, mean_hd95 = inference(args, multimask_output, dataset_config[dataset_name], net, folder_path,test_save_path)
    file_path = 'results_all.txt'

    file_path = 'Exp_log' + '/' + file_path[:-4] + '_layer' + str(args.sftm_hpyer) + '_stepsize' + str(
        args.att_step_size) + '_bound' + str(args.att_bound) + '_steps' + str(args.att_numstep) + file_path[-4:]
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'a') as file:
        file.write(f"path:{args.noise_path}, model:{args.lora_ckpt}\n")
        file.write(
            f"noise_sd: {args.noise_sd},sm_momentum: {args.smoothed_momentum},sm_freq: {args.smoothed_freq},sft_hyp: {args.sftm_hpyer},mean_dice: {performance},mean_hd95: {mean_hd95}\n")
    # print(performance,mean_hd95)
