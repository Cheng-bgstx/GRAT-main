import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from importlib import import_module
from datasets.dataset_synapse import Synapse_dataset


# 颜色映射表，用于显示 mask
def get_color_map(num_classes):
    # 简单的彩色映射，背景为黑色
    colors = np.array([
        [0, 0, 0],  # Background
        [255, 0, 0],  # Class 1: Red
        [0, 255, 0],  # Class 2: Green
        [0, 0, 255],  # Class 3: Blue
        [255, 255, 0],  # Class 4: Yellow
        [0, 255, 255],  # Class 5: Cyan
        [255, 0, 255],  # Class 6: Magenta
        [255, 128, 0],  # Class 7: Orange
        [128, 0, 255],  # Class 8: Purple
    ])
    return colors[:num_classes + 1]


def colorize_mask(mask, num_classes):
    cmap = get_color_map(num_classes)
    h, w = mask.shape
    img_color = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(1, num_classes + 1):
        img_color[mask == i] = cmap[i]
    return img_color


def load_npy_volume_from_folder(folder):
    if not os.path.exists(folder):
        return None
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    volume_list = []
    for f in files:
        volume_list.append(np.load(os.path.join(folder, f)))
    return np.array(volume_list) if len(volume_list) > 0 else None


def visualize_and_save(args):
    # 1. 准备目录
    vis_save_dir = os.path.join(args.out_path, "visualization")
    os.makedirs(vis_save_dir, exist_ok=True)
    print(f"🖼️  可视化结果将保存至: {vis_save_dir}")

    # 2. 加载模型
    print("⏳ 正在加载模型...")
    sam, _ = sam_model_registry[args.vit_name](image_size=args.img_size, num_classes=args.num_classes,
                                               checkpoint=args.ckpt, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()
    net.load_lora_parameters(args.lora_ckpt)
    net.eval()

    # 3. 加载数据集 (只用来获取 Clean Image 和 GT)
    dataset = Synapse_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split='test_vol')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 4. 循环处理
    for i, sample in enumerate(dataloader):
        # 原始数据
        clean_vol = sample['image'].squeeze(0).numpy()  # [D, H, W]
        label_vol = sample['label'].squeeze(0).numpy()  # [D, H, W]
        case_name = sample['case_name'][0]

        # 读取对抗样本
        adv_folder = os.path.join(args.out_path, case_name)
        adv_vol = load_npy_volume_from_folder(adv_folder)  # [D, H, W]

        if adv_vol is None:
            print(f"⚠️ 跳过 {case_name}: 未找到对抗样本 npy 文件")
            continue

        print(f"👉 正在处理 {case_name} ({adv_vol.shape[0]} 张切片)...")
        case_save_dir = os.path.join(vis_save_dir, case_name)
        os.makedirs(case_save_dir, exist_ok=True)

        # 确保对抗样本和原始数据维度一致
        # 有时候对抗样本可能只有部分切片，这里取交集长度
        depth = min(clean_vol.shape[0], adv_vol.shape[0])

        for slice_idx in range(depth):
            # 准备数据
            img_clean = clean_vol[slice_idx]
            img_adv = adv_vol[slice_idx]
            gt = label_vol[slice_idx]

            # 模型推理 (使用 Adv Image)
            input_tensor = torch.from_numpy(img_adv).float().cuda()
            # 转换为 [1, 3, H, W]
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

            with torch.no_grad():
                if args.num_classes > 1:
                    multimask_output = True
                else:
                    multimask_output = False

                outputs = net(input_tensor, multimask_output, args.input_size)
                output_masks = outputs['masks']
                # 获取预测 mask
                pred = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0).cpu().numpy()

            # Resize GT 和 Pred 以匹配显示 (如果尺寸不一致)
            H, W = img_clean.shape
            if gt.shape != (H, W):
                gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
            if pred.shape != (H, W):
                pred = cv2.resize(pred.astype(float), (W, H), interpolation=cv2.INTER_NEAREST).astype(int)

            # --- 绘图 ---
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))

            # 1. 原始图像 (灰度)
            ax[0].imshow(img_clean, cmap='gray')
            ax[0].set_title("Original Image")
            ax[0].axis('off')

            # 2. 对抗样本 (灰度 + 噪点肉眼可能看不清，但这是输入)
            ax[1].imshow(img_adv, cmap='gray')
            ax[1].set_title("Adversarial Image")
            ax[1].axis('off')

            # 3. Ground Truth (彩色)
            gt_color = colorize_mask(gt, args.num_classes)
            ax[2].imshow(gt_color)
            ax[2].set_title("Ground Truth")
            ax[2].axis('off')

            # 4. 预测结果 (彩色)
            pred_color = colorize_mask(pred, args.num_classes)
            ax[3].imshow(pred_color)
            ax[3].set_title("Prediction (Adv)")
            ax[3].axis('off')

            # 保存
            save_path = os.path.join(case_save_dir, f"slice_{slice_idx:03d}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close()

            # 仅处理前 5 张切片用于快速验证 (如果你想跑全部，注释掉下面两行)
            # if slice_idx >= 5:
            #     break

    print("✅ 可视化完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument('--volume_path', type=str, required=True, help='Testset path (h5)')
    parser.add_argument('--list_dir', type=str, required=True, help='List dir')
    parser.add_argument('--out_path', type=str, required=True, help='Path containing CaseXX/*.npy')

    # 模型参数
    parser.add_argument('--ckpt', type=str, required=True, help='SAM checkpoint')
    parser.add_argument('--lora_ckpt', type=str, required=True, help='LoRA checkpoint')
    parser.add_argument('--vit_name', type=str, default='vit_b')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    args = parser.parse_args()
    visualize_and_save(args)