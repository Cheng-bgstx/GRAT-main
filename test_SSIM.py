import os
import argparse
import numpy as np
import torch
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.image as mpimg
from skimage.metrics import structural_similarity as ssim

# 从您的项目结构中导入Synapse数据集加载器
# 这里假设 'datasets' 是与脚本位于同一目录下的一个包
from datasets.dataset_synapse import Synapse_dataset

# 类别ID到名称的映射，来自您的test_adv_all.py文件
CLASS_TO_NAME = {
    1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder',
    5: 'liver', 6: 'stomach', 7: 'aorta', 8: 'pancreas'
}


def load_adv_images_from_folder(folder_path):
    """
    从指定文件夹加载对抗性PNG图像并按数字排序。
    """
    images = []
    if not os.path.exists(folder_path):
        print(f"警告: 未找到对抗性图像文件夹: {folder_path}")
        return None

    try:
        img_filenames = sorted(os.listdir(folder_path), key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        print(f"警告: 无法在 {folder_path} 中按数字对文件名进行排序。将使用默认字母顺序排序。")
        img_filenames = sorted(os.listdir(folder_path))

    for filename in img_filenames:
        if filename.lower().endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = mpimg.imread(img_path)
            if img.ndim == 3:
                img = img[:, :, 0]
            images.append(img)

    return np.array(images)


def calculate_ssim_for_dataset(args):
    """
    遍历数据集，为每个类别计算SSIM，并将平均结果保存到CSV文件中。
    """
    # 配置并加载原始数据集
    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'z_spacing': 1
        }
    }
    db_config = dataset_config[args.dataset]
    db_test = db_config['Dataset'](base_dir=args.volume_path, list_dir=args.list_dir, split='test_vol')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    # 初始化一个字典来存储每个类别的所有SSIM分数
    # 格式: {class_id: [score1, score2, ...], ...}
    class_ssim_scores = {i: [] for i in range(1, args.num_classes + 1)}

    print("开始基于类别的SSIM评估...")

    # 遍历每个病例
    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
        case_name = sampled_batch['case_name'][0]

        # 1. 获取原始图像、标签和对抗性图像
        original_images = sampled_batch['image'].squeeze(0).cpu().numpy()
        label_images = sampled_batch['label'].squeeze(0).cpu().numpy()

        adv_case_path = os.path.join(args.adv_path, case_name)
        adv_images = load_adv_images_from_folder(adv_case_path)

        if adv_images is None: continue
        if original_images.shape[0] != adv_images.shape[0]:
            print(f"警告: 病例 {case_name} 的切片数量不匹配。")
            continue

        print(f"\n正在处理病例: {case_name}")

        # 遍历病例中的每个切片
        for i in range(original_images.shape[0]):
            original_slice = original_images[i].astype(np.float32)
            adv_slice = adv_images[i].astype(np.float32)
            label_slice = label_images[i]

            # 获取当前切片中存在的所有类别ID (忽略背景0)
            present_classes = np.unique(label_slice)
            present_classes = present_classes[present_classes != 0]

            if not present_classes.any():
                continue

            print(f"  切片 #{i}:")

            # 为每个存在的类别计算SSIM
            for class_id in present_classes:
                # 创建该类别的二值掩码
                mask = (label_slice == class_id)

                # 确保掩码区域足够大以进行有意义的计算
                if np.sum(mask) < 100:
                    continue

                # 计算完整的SSIM图，然后只在掩码区域内取平均值
                # 这是在特定区域计算SSIM的准确方法
                ssim_map = ssim(original_slice, adv_slice, data_range=1.0, full=True)[1]
                score = ssim_map[mask].mean()

                # 终端输出每个类别的SSIM
                class_name = CLASS_TO_NAME.get(class_id, f"未知类别{class_id}")
                print(f"    - 类别: {class_name:<15} SSIM: {score:.4f}")

                # 存储分数
                if class_id in class_ssim_scores:
                    class_ssim_scores[class_id].append(score)

    # --- 计算并保存最终结果 ---
    print("\n--- 评估完成，正在生成报告 ---")

    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['ClassID', 'ClassName', 'Average_SSIM'])

        all_scores = []
        for class_id, scores in class_ssim_scores.items():
            class_name = CLASS_TO_NAME.get(class_id, f"未知类别{class_id}")
            if scores:
                avg_score = np.mean(scores)
                all_scores.extend(scores)
                print(f"类别: {class_name:<15} | 平均SSIM: {avg_score:.4f} | (基于 {len(scores)} 个样本)")
                csv_writer.writerow([class_id, class_name, f"{avg_score:.6f}"])
            else:
                print(f"类别: {class_name:<15} | 平均SSIM: N/A (未找到该类别的有效样本)")
                csv_writer.writerow([class_id, class_name, "N/A"])

    if all_scores:
        overall_avg_ssim = np.mean(all_scores)
        print("\n-------------------------------------------")
        print(f"详细报告已保存至 {args.output_csv}")
        print(f"所有类别的总体平均SSIM: {overall_avg_ssim:.4f}")
        print("-------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="为每个语义类别评估SSIM，并将平均结果存为表格。")

    parser.add_argument('--volume_path', type=str, default='testset/test_vol_h5/',
                        help='包含原始图像和标签的数据集目录路径。')

    parser.add_argument('--adv_path', type=str, required=True,
                        help='包含生成的对抗性图像的目录路径。')

    parser.add_argument('--output_csv', type=str, default='ssim_per_class_results.csv',
                        help='用于保存按类别平均的SSIM结果的CSV文件路径。')

    parser.add_argument('--num_classes', type=int, default=8, help='数据集中的类别总数。')
    parser.add_argument('--dataset', type=str, default='Synapse', help='数据集名称。')
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/',
                        help='包含数据集列表文件的目录。')

    args = parser.parse_args()
    calculate_ssim_for_dataset(args)
