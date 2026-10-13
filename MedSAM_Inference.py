import sys
new_library_path = '/root/data1/fyc/GRAT-main/MedSAM'
sys.path.insert(0, new_library_path)
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from segment_anything import sam_model_registry
from skimage import transform
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import glob
from scipy import ndimage
import random
from PIL import Image
from scipy.ndimage import label as label_instances


# Visualization functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_resized=None, points_resized=None, H=None, W=None,
                     gt_slice_shape=None):
    if box_resized is not None:
        box_torch = torch.as_tensor(box_resized, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)
    else:
        box_torch = None

    if gt_slice_shape is not None:
        target_size = (gt_slice_shape[0], gt_slice_shape[1])
    else:
        target_size = (H, W)
    if points_resized is not None:
        points_coords = torch.as_tensor(points_resized['coords'], dtype=torch.float, device=img_embed.device)
        points_labels = torch.as_tensor(points_resized['labels'], dtype=torch.long, device=img_embed.device)
        if len(points_coords.shape) == 2:
            points_coords = points_coords[:, None, :]
            points_labels = points_labels[:, None]
        points_tuple = (points_coords, points_labels)
    else:
        points_tuple = None

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=points_tuple,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    return low_res_pred.squeeze().cpu().numpy()


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


def process_single_slice(img_slice, gt_slice, medsam_model, device, args):
    """Process a single slice with all instances"""
    if len(img_slice.shape) == 3:
        H, W, _ = img_slice.shape
    else:
        H, W = img_slice.shape
    # Prepare image
    if len(img_slice.shape) == 2:
        img_3c = np.repeat(img_slice[:, :, None], 3, axis=-1)
    else:
        img_3c = img_slice
    if img_3c.shape[2] == 4:
        img_3c = img_3c[:, :, :3]
    if H == args.image_size and W == args.image_size:
        # 图像已经是目标尺寸 (1024x1024)，跳过resize以保留扰动
        img_resized = img_3c
    else:
        # 图像不是目标尺寸 (例如，原始的干净图像)，执行resize
        img_resized = transform.resize(
            img_3c, (args.image_size, args.image_size), order=3, preserve_range=True, anti_aliasing=True
        )

    #img_resized = (img_resized - img_resized.min()) / np.clip(img_resized.max() - img_resized.min(), a_min=1e-8, a_max=None)

    img_resized_tensor = (
        torch.tensor(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    # Get image embedding
    with torch.no_grad():
        image_embedding , _= medsam_model.image_encoder(img_resized_tensor)

    # Get bounding boxes from GT
    prompts = get_prompts_from_mask(gt_slice)
    if not prompts:
        return np.zeros((H, W), dtype=np.uint8)

    combined_preds_by_class = {}

    for prompt in prompts:
        box_prompt = None
        point_prompt = None

        # 根据用户选择的prompt_type准备提示
        if args.prompt_type == "box":
            box_np = np.array([prompt["box"]])
            box_prompt = box_np / np.array([W, H, W, H]) * args.image_size

        elif args.prompt_type == "point":
            points_np = np.array(prompt["point_coords"])
            points_scaled = points_np / np.array([W, H]) * args.image_size
            point_prompt = {"coords": points_scaled, "labels": np.array(prompt["point_labels"])}

        pred_prob_map = medsam_inference(
            medsam_model, image_embedding, box_resized=box_prompt, points_resized=point_prompt, H=H, W=W,
            gt_slice_shape=gt_slice.shape
        )
        binary_mask = (pred_prob_map > 0.5).astype(np.uint8)
        organ_label = prompt["label"]
        if organ_label not in combined_preds_by_class:
            combined_preds_by_class[organ_label] = binary_mask
        else:
            combined_preds_by_class[organ_label] = np.logical_or(
                combined_preds_by_class[organ_label],
                binary_mask
            ).astype(np.uint8)

    return combined_preds_by_class


def process_npy_batch(args):
    """Process batch of NPY files"""
    device = args.device
    # Load model
    medsam_model = sam_model_registry["vit_b"](
        checkpoint=args.checkpoint)
    medsam_model = medsam_model.to(device)
    medsam_model.pixel_mean = torch.tensor([0.0, 0.0, 0.0], device=device)
    medsam_model.pixel_std = torch.tensor([1.0, 1.0, 1.0], device=device)
    medsam_model.eval()
    # Get list of NPY files
    npy_files = sorted(glob.glob(os.path.join(args.data_path, "imgs", "*.npy")))
    if len(npy_files) == 0:
        print(f"No NPY files found in {args.data_path}/imgs/")
        return
    print(f"Found {len(npy_files)} NPY files to process")
    # Process each file
    for npy_path in tqdm(npy_files):
        basename = os.path.basename(npy_path)
        case_name = basename.replace(".npy", "")
        # Load image and GT
        img = np.load(npy_path)
        gt_path = os.path.join(args.data_path, "gts", basename)
        if not os.path.exists(gt_path):
            print(f"GT not found for {basename}, skipping...")
            continue
        gt = np.load(gt_path)
        binary_masks_dict = process_single_slice(img, gt, medsam_model, device, args)

        # NEW LOGIC: Save each binary mask to a separate npy file
        for label, pred_mask in binary_masks_dict.items():
            save_path = os.path.join(args.seg_path, f"seg_{case_name}_label_{label}.npy")
            np.save(save_path, pred_mask)
            if args.save_vis:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].imshow(img, cmap='gray')
                axes[0].set_title("Original Image")
                axes[0].axis('off')
                gt_mask_vis = (gt == label)
                axes[1].imshow(img, cmap='gray');
                show_mask(gt_mask_vis, axes[1])
                axes[1].set_title(f"Ground Truth (Label {label})")
                axes[1].axis('off')
                axes[2].imshow(img, cmap='gray');
                show_mask(pred_mask, axes[2])
                axes[2].set_title(f"MedSAM Prediction (Label {label})")
                axes[2].axis('off')
                plt.tight_layout()
                vis_path = os.path.join(args.seg_path, f"vis_{case_name}_label_{label}.png")
                plt.savefig(vis_path, dpi=150)
                plt.close()

    print(f"\nProcessing complete. Results saved to {args.seg_path}")


def process_npz_batch(args):
    """Process batch of NPZ files (3D volumes)"""
    device = args.device

    # Load model
    medsam_model = sam_model_registry["vit_b"](
        checkpoint=args.checkpoint)
    medsam_model = medsam_model.to(device)
    medsam_model.pixel_mean = torch.tensor([0.0, 0.0, 0.0], device=device)
    medsam_model.pixel_std = torch.tensor([1.0, 1.0, 1.0], device=device)
    medsam_model.eval()

    # Get list of NPZ files
    npz_files = sorted(glob.glob(os.path.join(args.data_path, "*.npz")))

    if len(npz_files) == 0:
        print(f"No NPZ files found in {args.data_path}")
        return

    print(f"Found {len(npz_files)} NPZ files to process")

    # Process each volume
    for npz_path in tqdm(npz_files):
        basename = os.path.basename(npz_path).replace(".npz", "")

        # Load volume
        data = np.load(npz_path)
        imgs = data['imgs']  # Shape: (num_slices, H, W)
        gts = data['gts']  # Shape: (num_slices, H, W)
        class_volumes = {}
        all_slice_masks = [process_single_slice(imgs[i], gts[i], medsam_model, device, args) for i in
                           range(imgs.shape[0])]

        # NEW LOGIC: Reorganize sliced masks into volumes per class
        for slice_masks_dict in all_slice_masks:
            for label, mask in slice_masks_dict.items():
                if label not in class_volumes:
                    class_volumes[label] = []
                class_volumes[label].append(mask)

        print(f"\nProcessed {basename}, found classes: {list(class_volumes.keys())}")

        # NEW LOGIC: Save each class volume into a separate NPZ file
        for label, masks_list in class_volumes.items():
            # Check if the list of masks is not empty and has the correct number of slices
            if masks_list and len(masks_list) > 0:
                # We need to construct a full volume, even if a class is not in every slice
                full_volume = np.zeros_like(gts, dtype=np.uint8)
                slice_indices = np.where(np.array([label in d for d in all_slice_masks]))[0]

                # Reconstruct the volume from sparse slices
                for i, slice_idx in enumerate(slice_indices):
                    full_volume[slice_idx] = masks_list[i]

                # Save the binary segmentation volume for this class
                save_path = os.path.join(args.seg_path, f"seg_{basename}_label_{label}.npz")
                np.savez_compressed(save_path, segs=full_volume, gts=gts)
                print(f"  -> Saved segmentation for class {label} to {save_path}")

        # NEW LOGIC: Visualization saves one image per class for the middle slice
        if args.save_vis:
            vis_slice_idx = imgs.shape[0] // 2
            slice_masks_for_vis = all_slice_masks[vis_slice_idx]

            for label, pred_mask in slice_masks_for_vis.items():
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                axes[0].imshow(imgs[vis_slice_idx], cmap='gray')
                axes[0].set_title(f"Original Image (Slice {vis_slice_idx})")
                axes[0].axis('off')

                gt_mask_vis = (gts[vis_slice_idx] == label)
                axes[1].imshow(imgs[vis_slice_idx], cmap='gray')
                show_mask(gt_mask_vis, axes[1])
                axes[1].set_title(f"Ground Truth (Label {label})")
                axes[1].axis('off')

                axes[2].imshow(imgs[vis_slice_idx], cmap='gray')
                show_mask(pred_mask, axes[2])
                axes[2].set_title(f"MedSAM Prediction (Label {label})")
                axes[2].axis('off')

                plt.tight_layout()
                vis_path = os.path.join(args.seg_path, f"vis_{basename}_slice_{vis_slice_idx}_label_{label}.png")
                plt.savefig(vis_path, dpi=150)
                plt.close()
            print(f"  -> Saved visualizations for slice {vis_slice_idx} in {args.seg_path}")

    print(f"\nProcessing complete. Results saved to {args.seg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch MedSAM inference on NPY/NPZ data with automatic bounding box generation"
    )
    parser.add_argument(
        "-i", "--data_path", type=str, required=True,
        help="Path to the NPY data folder (should contain 'imgs' and 'gts' subfolders) or NPZ folder"
    )
    parser.add_argument(
        "-o", "--seg_path", type=str, required=True,
        help="Path to save segmentation results"
    )
    parser.add_argument(
        "--data_type", type=str, default="npy", choices=["npy", "npz"],
        help="Data type to process (npy for 2D slices, npz for 3D volumes)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use"
    )
    parser.add_argument(
        "-chk", "--checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth",
        help="Path to the trained MedSAM model"
    )
    parser.add_argument(
        "--save_vis", action="store_true",
        help="Save visualization of results"
    )
    parser.add_argument(
        "--image_size", type=int, default=1024,
        help="The input image size for the MedSAM model encoder (e.g., 512, 1024)"
    )
    parser.add_argument(
        "--prompt_type", type=str, default="box", choices=["box", "point"],
        help="Type of prompt to use for segmentation (default: box)"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.seg_path, exist_ok=True)

    # Process based on data type
    if args.data_type == "npy":
        process_npy_batch(args)
    else:
        process_npz_batch(args)

##!!!image size none of use   constrast of old(GRAT),process one slice's process changed(sam registery)