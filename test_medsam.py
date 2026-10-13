import numpy as np
import os
import glob
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import seaborn as sns


def calculate_iou(pred, gt):
    """Calculate Intersection over Union for binary masks"""
    intersection = np.logical_and(pred > 0, gt > 0).sum()
    union = np.logical_or(pred > 0, gt > 0).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return iou


def calculate_dice(pred, gt):
    """Calculate Dice coefficient for binary masks"""
    pred = pred > 0
    gt = gt > 0

    intersection = np.logical_and(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if pred_sum + gt_sum == 0:
        # Both masks are empty, perfect match (true negative)
        return 1.0

    dice = (2.0 * intersection) / (pred_sum + gt_sum)
    return dice


def calculate_metrics_per_instance(pred, gt):
    """Calculate metrics for each instance/class in the masks"""
    unique_labels = np.unique(gt)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background

    metrics = {
        'iou_per_instance': [],
        'dice_per_instance': [],
        'instance_ids': []
    }

    for label in unique_labels:
        gt_instance = (gt == label).astype(np.uint8)
        pred_instance = (pred == label).astype(np.uint8)

        iou = calculate_iou(pred_instance, gt_instance)
        dice = calculate_dice(pred_instance, gt_instance)

        metrics['iou_per_instance'].append(iou)
        metrics['dice_per_instance'].append(dice)
        metrics['instance_ids'].append(label)

    # Calculate overall metrics (treating all foreground as one)
    overall_iou = calculate_iou(pred > 0, gt > 0)
    overall_dice = calculate_dice(pred > 0, gt > 0)

    metrics['overall_iou'] = overall_iou
    metrics['overall_dice'] = overall_dice

    if len(metrics['iou_per_instance']) > 0:
        metrics['mean_iou'] = np.mean(metrics['iou_per_instance'])
        metrics['mean_dice'] = np.mean(metrics['dice_per_instance'])
    else:
        metrics['mean_iou'] = 0.0
        metrics['mean_dice'] = 0.0

    return metrics


def evaluate_npy_results(pred_path, gt_path):
    """Evaluate NPY format results"""
    # Get list of prediction files
    pred_files = sorted(glob.glob(os.path.join(pred_path, "seg_*.npy")))

    if len(pred_files) == 0:
        print(f"No prediction files found in {pred_path}")
        return None

    all_slice_metrics = []
    for pred_file in tqdm(pred_files, desc="Evaluating NPY files"):
        match = re.search(r'seg_(.*?)_label_(\d+)\.npy', os.path.basename(pred_file))
        if not match:
            continue
        basename = match.group(1)
        target_label = int(match.group(2))
        pred_slice = np.load(pred_file)

    # Load corresponding GT
        gt_file_path = os.path.join(gt_path, "gts", f"{basename}.npy")
        if not os.path.exists(gt_file_path):
            print(f"GT not found for {basename}, skipping...")
            continue
        gt_slice = np.load(gt_file_path)
        gt_instance = (gt_slice == target_label)

    # Calculate metrics
        if gt_instance.sum() > 0:
            dice = calculate_dice(pred_slice, gt_instance)
            iou = calculate_iou(pred_slice, gt_instance)
            all_slice_metrics.append({
                'case': basename,
                'label': target_label,
                'slice': 0,  # NPY 文件视为单个切片
                'dice': dice,
                'iou': iou
            })


    return all_slice_metrics


def evaluate_npz_results(pred_path, gt_path=None):
    """Evaluate NPZ format results (3D volumes)"""
    # Get list of prediction files
    pred_files = sorted(glob.glob(os.path.join(pred_path, "seg_*_label_*.npz")))
    if not pred_files:
        print(f"No prediction files found in {pred_path}")
        return None, None

    all_slice_metrics = []
    for pred_file in tqdm(pred_files, desc="Evaluating prediction files"):
        # NEW LOGIC: Parse case name and label from filename
        match = re.search(r'seg_(.*?)_label_(\d+)\.npz', os.path.basename(pred_file))
        if not match:
            print(f"Could not parse case and label from {pred_file}, skipping.")
            continue
        basename = match.group(1)
        target_label = int(match.group(2))
        if basename.startswith('adv_'):
            basename = basename[4:]
        try:
            data = np.load(pred_file)
            preds = data['segs']
            if gt_path:
                # Load GT from external path
                gt_file_path = os.path.join(gt_path, f"{basename}.npz")
                if not os.path.exists(gt_file_path):
                    print(f"External GT file not found at {gt_file_path}, skipping case {basename}.")
                    continue
                gts = np.load(gt_file_path)['gts']
            else:
                # Load GT from the bundled NPZ file
                if 'gts' not in data:
                    print(f"GT not found inside {pred_file}, and --gt_path was not provided. Skipping.")
                    continue
                gts = data['gts']
        except KeyError as e:
            print(f"文件 {pred_file} 缺少 'segs' 或 'gts' 键, 已跳过。")
            continue

        if preds.shape != gts.shape:
            print(f"文件 {pred_file} 中预测和GT的尺寸不匹配, 已跳过。")
            continue

        for slice_idx in range(preds.shape[0]):
            pred_slice, gt_slice = preds[slice_idx], gts[slice_idx]
            gt_instance = (gt_slice == target_label)

            # Only evaluate slices where the ground truth object exists
            if gt_instance.sum() > 0:
                dice = calculate_dice(pred_slice, gt_instance)
                all_slice_metrics.append({
                    'case': basename,
                    'label': target_label,
                    'slice': slice_idx,
                    'dice': dice
                })

    return all_slice_metrics


def save_results(metrics, output_path, freq=1):
    """Save evaluation results to CSV and generate plots"""
    if not metrics:
        print("No metrics to save.")
        return

    df = pd.DataFrame(metrics)
    class_stats = df.groupby('label')['dice'].agg(['mean', 'std', 'count']).reset_index()
    class_stats.rename(columns={'mean': 'mean_dice', 'std': 'std_dice', 'count': 'num_slices'}, inplace=True)
    overall_avg_dice = df['dice'].mean()
    overall_row = pd.DataFrame([{
        'label': 'Overall Avg',
        'mean_dice': overall_avg_dice,
        'std_dice': df['dice'].std(),
        'num_slices': len(df)
    }])
    summary_df = pd.concat([class_stats, overall_row], ignore_index=True)
    # Box plot for overall comparison
    summary_csv_path = os.path.join(output_path, 'summary_dice_scores.csv')
    summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
    print(f"Summary Dice scores saved to {summary_csv_path}")

    # 5. 在控制台打印简化后的结果
    print("\n" + "=" * 60)
    print("PER-CLASS EVALUATION RESULTS (Mean Dice)")
    print("=" * 60)
    for _, row in class_stats.iterrows():
        print(f"Class {str(row['label']):<12}: Mean Dice = {row['mean_dice']:.4f}")
    print("-" * 60)
    print(f"{'Overall Avg':<12}: Mean Dice = {overall_avg_dice:.4f}")
    print("=" * 60)

    # 6. 为每个类别生成一张趋势图
    plot_dir = os.path.join(output_path, 'dice_trend_plots')
    os.makedirs(plot_dir, exist_ok=True)

    print("\nGenerating per-class Dice trend plots...")
    # 按类别分组
    for label, group in tqdm(df.groupby('label'), desc="Creating plots"):
        # 按病例和切片排序，确保切片顺序的连续性
        group = group.sort_values(by=['case', 'slice']).reset_index(drop=True)

        num_slices = len(group)
        # 如果总切片数小于freq，则跳过，无法绘图
        if num_slices < freq:
            continue

        trend_data = []
        # 按 freq 的步长进行分块
        for i in range(0, num_slices, freq):
            chunk = group.iloc[i:i + freq]
            if not chunk.empty:
                avg_dice = chunk['dice'].mean()
                # X轴使用累积的切片数量
                cumulative_slices = i + len(chunk)
                trend_data.append({'slice_count': cumulative_slices, 'avg_dice': avg_dice})

        if not trend_data:
            continue

        trend_df = pd.DataFrame(trend_data)

        # 绘图
        plt.figure(figsize=(12, 6))
        plt.plot(trend_df['slice_count'], trend_df['avg_dice'], linestyle='-', marker='o', markersize=5)
        plt.title(f'Dice Score Trend for Class: {label}')
        plt.xlabel(f'Cumulative Slice Count (in chunks of {freq})')
        plt.ylabel('Average Dice Score')
        plt.grid(True, alpha=0.5)
        plt.ylim(-0.05, 1.05)

        plot_filename = f"dice_trend_class_{label}.png"
        plt.savefig(os.path.join(plot_dir, plot_filename), dpi=100)
        plt.close()

    print(f"Dice trend plots saved to {plot_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MedSAM segmentation results"
    )
    parser.add_argument(
        "-p", "--pred_path", type=str, required=True,
        help="Path to prediction results"
    )
    parser.add_argument(
        "-g", "--gt_path", type=str, default=None,
        help="Path to ground truth data (for npy mode, should contain 'gts' folder)"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--data_type", type=str, default="npy", choices=["npy", "npz"],
        help="Data type (npy for 2D slices, npz for 3D volumes)"
    )
    parser.add_argument('--online_update_freq', type=int, default=1,
                        help='freq')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    metrics_for_saving = None

    # Evaluate based on data type
    if args.data_type == "npy":
        if args.gt_path is None:
            raise ValueError("Ground truth path is required for npy evaluation")
        metrics_for_saving = evaluate_npy_results(args.pred_path, args.gt_path)
    else:
        metrics_npz = evaluate_npz_results(args.pred_path, args.gt_path)
        metrics_for_saving = metrics_npz
    if metrics_for_saving is None or len(metrics_for_saving) == 0:
        print("No results to evaluate")
        return

    # Save results
    save_results(metrics_for_saving, args.output_path, args.online_update_freq)


if __name__ == "__main__":
    main()