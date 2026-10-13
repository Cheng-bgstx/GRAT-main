#!/usr/bin/env python3
"""
Experiment 13: Qualitative Visualization.
Generates 6 rows x 6 columns visualization:
  Clean image | Adv image | Pert (x5) | Clean pred | Adv pred | GT

2 cases each from Synapse, ISIC, REFUGE (1 success + 1 edge case each).

Uses PIL for image assembly to avoid matplotlib _resample C-level bug.
"""
import os, sys, random
import numpy as np
import torch
import h5py
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from attack_codd import AttackConfig, run_codd_attack
from evaluate import load_sam_encoder, load_samed_model

DEVICE = torch.device("cuda:0")
IMAGE_SIZE = 512

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results", "qualitative")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIT = "vit_b"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
SYNAPSE_CKPT = f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"
ISIC_CKPT = f"{ROOT}/output_runs_correct/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth"
REFUGE_CKPT = f"{ROOT}/output_runs_correct/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth"


def generate_adversarial(source_sam, img_tensor):
    """Run SE-CGR attack and return adversarial image."""
    config = AttackConfig(
        att_step_size=2.0, att_bound=10.0, att_numstep=10,
        direction_mode="radial", probe_amplification=2.0,
        edge_guidance=True, seed=42,
    )
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    adv_np, _ = run_codd_attack(source_sam.image_encoder, img_tensor, config, return_trace=True)
    return adv_np  # [3, H, W]


def get_sam_prediction(net, image_np, num_classes):
    """Get segmentation prediction from SAMed model."""
    if image_np.ndim == 2:
        image_np = image_np[np.newaxis, :, :]
    if image_np.shape[0] == 1:
        image_t = torch.from_numpy(np.repeat(image_np, 3, axis=0)).unsqueeze(0).float().to(DEVICE)
    else:
        image_t = torch.from_numpy(image_np.copy()).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        outputs = net(image_t, True, IMAGE_SIZE)
        masks = outputs['masks']
    if num_classes > 1:
        masks = torch.softmax(masks, dim=1)
        pred = torch.argmax(masks, dim=1).squeeze(0).cpu().numpy()
    else:
        probs = torch.sigmoid(masks[0, 0]).cpu().numpy()
        pred = (probs > 0.5).astype(np.float32)
    return pred


def fix_image_for_display(data):
    """Ensure data is displayable: return [H,W] grayscale or [H,W,3] RGB uint8."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    while data.ndim > 3:
        data = data[0]
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        if data.shape[0] == 3:
            data = data.transpose(1, 2, 0)
        elif data.shape[-1] == 3:
            pass
        elif data.shape[0] == 1:
            return data[0]
        else:
            return data[0]
    if data.ndim == 3 and data.shape[-1] in (3, 4):
        if data.max() <= 1.0:
            data = (data * 255)
        return np.ascontiguousarray(data.clip(0, 255).astype(np.uint8))
    return np.ascontiguousarray(data)


def render_cell_pil(data, title, cell_size, is_segmask=False):
    """Convert a numpy array to a PIL Image cell with title overlay."""
    data = fix_image_for_display(data)

    if data.ndim == 2:
        vmin, vmax = data.min(), data.max()
        if is_segmask and data.max() > 1:
            # Multi-class segmentation: use tab10 colormap (via matplotlib.cm, no rendering)
            n_classes = int(data.max()) + 1
            cmap = plt.cm.get_cmap('tab10', n_classes)
            rgb = (cmap(data.astype(np.float64) / max(data.max(), 1))[:, :, :3] * 255).astype(np.uint8)
        else:
            if vmax > vmin:
                gray = ((data.astype(np.float64) - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            else:
                gray = np.zeros(data.shape[:2], dtype=np.uint8)
            rgb = np.stack([gray, gray, gray], axis=-1)
    elif data.ndim == 3 and data.shape[-1] >= 3:
        arr = data[..., :3].astype(np.float64)
        if arr.max() > 255 or arr.min() < 0:
            vmin, vmax = arr.min(), arr.max()
            if vmax > vmin:
                arr = (arr - vmin) / (vmax - vmin) * 255
            else:
                arr = np.zeros_like(arr)
        rgb = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        rgb = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)

    img = Image.fromarray(rgb).resize((cell_size, cell_size), Image.LANCZOS)

    # Title overlay
    draw = ImageDraw.Draw(img)
    bar_h = 22
    draw.rectangle([(0, 0), (cell_size, bar_h)], fill=(0, 0, 0))
    draw.text((4, 3), title, fill=(255, 255, 255))

    return img


def build_grid_pil(rows_data, output_path, cell_size=280):
    """rows_data: list of rows, each row is [(data, title, is_segmask), ...]."""
    n_rows = len(rows_data)
    n_cols = len(rows_data[0]) if rows_data else 0
    if n_rows == 0 or n_cols == 0:
        print("No data to save")
        return

    grid_w = n_cols * cell_size
    grid_h = n_rows * cell_size
    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))

    for r, row in enumerate(rows_data):
        for c, (data, title, is_segmask) in enumerate(row):
            cell = render_cell_pil(data, title, cell_size, is_segmask=is_segmask)
            grid.paste(cell, (c * cell_size, r * cell_size))

    grid.save(output_path, quality=90)
    print(f"Saved to {output_path}")


def main():
    print("Loading models...")
    source_sam = load_sam_encoder(VIT, SAM_CKPT, IMAGE_SIZE, 8, DEVICE)
    synapse_net = load_samed_model(VIT, SAM_CKPT, SYNAPSE_CKPT, IMAGE_SIZE, 8, device=DEVICE)

    rows_data = []

    # --- Synapse case 1: success (middle organ slice) ---
    print("Processing Synapse case 1...")
    tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    with h5py.File(os.path.join(tdir, sorted(os.listdir(tdir))[0]), 'r') as f:
        img, lbl = f['image'][:], f['label'][:]

    valid = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100],
                   key=lambda s: (lbl[s] > 0).sum(), reverse=True)
    slc = valid[len(valid)//2]

    clean_slice = img[slc].astype(np.float32)
    label_slice = lbl[slc]

    slc_3ch = np.stack([clean_slice]*3, axis=0)
    img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(DEVICE)
    adv_np = generate_adversarial(source_sam, img_t)

    clean_pred = get_sam_prediction(synapse_net, clean_slice, 8)
    adv_pred = get_sam_prediction(synapse_net, adv_np[0], 8)

    pert = np.clip(np.abs(adv_np.astype(np.float64) - slc_3ch.astype(np.float64)) * 5, 0, 255).astype(np.uint8)
    rows_data.append([
        (slc_3ch, 'Clean', False),
        (adv_np, 'Adversarial', False),
        (pert, 'Pert (x5)', False),
        (clean_pred, 'Clean Pred', True),
        (adv_pred, 'Adv Pred', True),
        (label_slice, 'GT', True),
    ])

    # --- Synapse case 2: edge case (small organ) ---
    print("Processing Synapse case 2...")
    small_slc = valid[-1] if len(valid) > 1 else valid[0]
    clean_slice2 = img[small_slc].astype(np.float32)
    label_slice2 = lbl[small_slc]
    slc_3ch2 = np.stack([clean_slice2]*3, axis=0)
    img_t2 = torch.from_numpy(slc_3ch2).unsqueeze(0).float().to(DEVICE)
    adv_np2 = generate_adversarial(source_sam, img_t2)
    clean_pred2 = get_sam_prediction(synapse_net, clean_slice2, 8)
    adv_pred2 = get_sam_prediction(synapse_net, adv_np2[0], 8)

    pert2 = np.clip(np.abs(adv_np2.astype(np.float64) - slc_3ch2.astype(np.float64)) * 5, 0, 255).astype(np.uint8)
    rows_data.append([
        (slc_3ch2, 'Clean', False),
        (adv_np2, 'Adversarial', False),
        (pert2, 'Pert (x5)', False),
        (clean_pred2, 'Clean Pred', True),
        (adv_pred2, 'Adv Pred', True),
        (label_slice2, 'GT', True),
    ])

    del synapse_net; torch.cuda.empty_cache()

    # --- ISIC cases ---
    print("Processing ISIC cases...")
    isic_net = load_samed_model(VIT, SAM_CKPT, ISIC_CKPT, IMAGE_SIZE, 1, device=DEVICE)
    import pandas as pd
    from torchvision import transforms
    data_dir = f"{ROOT}/data/ISIC"
    df = pd.read_csv(os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_GroundTruth.csv"))
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

    for i_case in range(2):
        row = df.iloc[i_case]
        img_pil = Image.open(os.path.join(data_dir, row['img'])).convert('RGB')
        gt = np.array(Image.open(os.path.join(data_dir, row['seg'])).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32) / 255.0

        img_t_orig = transform(img_pil) * 255  # [3, H, W] in [0, 255]
        img_t_tensor = img_t_orig.unsqueeze(0).float().to(DEVICE)
        adv_np_isic = generate_adversarial(source_sam, img_t_tensor)

        clean_pred_i = get_sam_prediction(isic_net, img_t_orig.numpy(), 1)
        adv_pred_i = get_sam_prediction(isic_net, adv_np_isic, 1)

        pert_isic = np.clip(np.abs(adv_np_isic.astype(np.float64) - img_t_orig.numpy().astype(np.float64)) * 5, 0, 255).astype(np.uint8)
        rows_data.append([
            (img_t_orig.numpy(), 'Clean', False),
            (adv_np_isic, 'Adversarial', False),
            (pert_isic, 'Pert (x5)', False),
            (clean_pred_i, 'Clean Pred', True),
            (adv_pred_i, 'Adv Pred', True),
            (gt, 'GT', True),
        ])

    del isic_net; torch.cuda.empty_cache()

    # --- REFUGE cases ---
    print("Processing REFUGE cases...")
    refuge_net = load_samed_model(VIT, SAM_CKPT, REFUGE_CKPT, IMAGE_SIZE, 2, device=DEVICE)
    refuge_dir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    folders = sorted([f for f in os.listdir(refuge_dir) if os.path.isdir(os.path.join(refuge_dir, f))])

    for i_case in range(2):
        folder = folders[i_case]
        img_path = os.path.join(refuge_dir, folder, f"{folder}.jpg")
        disc_path = os.path.join(refuge_dir, folder, f"{folder}_disc.bmp")
        cup_path = os.path.join(refuge_dir, folder, f"{folder}_cup.bmp")

        img_ref = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
        img_arr = np.array(img_ref).transpose(2, 0, 1).astype(np.float32)

        disc_raw = np.array(Image.open(disc_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        cup_raw = np.array(Image.open(cup_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        gt_ref = np.stack([
            (cup_raw < 128).astype(np.float32),
            (disc_raw <= 128).astype(np.float32),
            np.zeros_like(disc_raw),
        ], axis=0)

        img_t_ref = torch.from_numpy(img_arr).unsqueeze(0).float().to(DEVICE)
        adv_np_ref = generate_adversarial(source_sam, img_t_ref)
        clean_pred_r = get_sam_prediction(refuge_net, img_arr, 2)
        adv_pred_r = get_sam_prediction(refuge_net, adv_np_ref, 2)

        # GT visualization: disc=red, cup=blue overlay
        gt_vis = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        gt_vis[:, :, 0] = gt_ref[1]
        gt_vis[:, :, 2] = gt_ref[0]

        pert_ref = np.clip(np.abs(adv_np_ref.astype(np.float64) - img_arr.astype(np.float64)) * 5, 0, 255).astype(np.uint8)
        rows_data.append([
            (img_arr, 'Clean', False),
            (adv_np_ref, 'Adversarial', False),
            (pert_ref, 'Pert (x5)', False),
            (clean_pred_r, 'Clean Pred', True),
            (adv_pred_r, 'Adv Pred', True),
            (gt_vis, 'GT', False),  # GT is already RGB
        ])

    del refuge_net; torch.cuda.empty_cache()

    # Assemble and save with PIL (avoid matplotlib _resample bug)
    out_path = os.path.join(OUTPUT_DIR, "exp13_qualitative.png")
    build_grid_pil(rows_data, out_path, cell_size=280)
    print("DONE")


if __name__ == "__main__":
    main()
