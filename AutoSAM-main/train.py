import torch.optim as optim
import torch.utils.data
import torch
import torch.nn as nn
from tqdm import tqdm
import traceback
import random
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re

from models.model_single import MaskRefinement2D, ModelEmb3D, ModelEmb
from dataset.glas import get_glas_dataset
from dataset.MoNuBrain import get_monu_dataset
from dataset.polyp import get_polyp_dataset, get_tests_polyp_dataset
from dataset.LungData import get_lung_dataset
from dataset.isic import get_isic_dataset
from dataset.refuge import get_refuge_dataset
from dataset.synapse import get_synapse_dataset
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════
# Per-task configuration
# ═══════════════════════════════════════════════

TASK_CONFIG = {
    'isic':    {'num_classes': 1, 'mode': 'binary'},    # skin lesion
    'glas':    {'num_classes': 1, 'mode': 'binary'},    # gland
    'monu':    {'num_classes': 1, 'mode': 'binary'},    # nuclei
    'polyp':   {'num_classes': 1, 'mode': 'binary'},    # polyp
    'refuge':  {'num_classes': 2, 'mode': 'multi_bce'}, # cup + disc (2ch BCE)
    'synapse': {'num_classes': 9, 'mode': 'multi_ce'},  # 8 organs + bg (softmax CE)
    'lung':    {'num_classes': 1, 'mode': 'binary'},    # lung (3D)
}

TASKS_2D = {'isic', 'refuge', 'synapse', 'monu', 'glas', 'polyp'}
TASKS_3D = {'lung'}


# ═══════════════════════════════════════════════
# SAM multi-class expansion
# ═══════════════════════════════════════════════

def expand_sam_multimask(sam, num_classes):
    """Expand SAM's mask_decoder to output num_classes channels."""
    decoder = sam.mask_decoder
    old_num = decoder.num_multimask_outputs

    if num_classes <= old_num:
        decoder.num_multimask_outputs = num_classes
        decoder.num_mask_tokens = num_classes + 1
        print(f"SAM mask_decoder: using {num_classes} of {old_num} existing mask tokens")
        return sam

    new_num = num_classes + 1
    old_embed = decoder.mask_tokens
    old_weight = old_embed.weight.data  # (old_num_mask_tokens, D) includes IoU token

    old_num_tokens = old_weight.shape[0]  # actual token count (mask + IoU)

    # Expand mask_tokens embedding
    new_embed = nn.Embedding(new_num, decoder.transformer_dim).to(old_weight.device)
    new_embed.weight.data[:old_num_tokens] = old_weight
    nn.init.normal_(new_embed.weight.data[old_num_tokens:], std=0.02)
    decoder.mask_tokens = new_embed

    # Expand output_hypernetworks_mlps (one MLP per mask token)
    from segment_anything.modeling.mask_decoder import MLP
    old_mlps = list(decoder.output_hypernetworks_mlps)
    for i in range(old_num_tokens, new_num):
        new_mlp = MLP(decoder.transformer_dim, decoder.transformer_dim,
                      decoder.transformer_dim // 8, 3).to(old_weight.device)
        decoder.output_hypernetworks_mlps.append(new_mlp)

    decoder.num_mask_tokens = new_num
    decoder.num_multimask_outputs = num_classes
    print(f"SAM mask_decoder: expanded from {old_num_tokens} to {new_num} tokens "
          f"({num_classes} classes + 1 IoU)")
    return sam


def resize_sam_image_size(sam, new_size):
    """Resize SAM's position embeddings and internal sizes for a new image size.

    SAM is pre-trained with image_size=1024 (64x64 feature grid).
    For sam_size=512 (32x32 grid), we need to interpolate pos_embeddings.
    """
    old_size = sam.image_encoder.img_size  # typically 1024
    if old_size == new_size:
        print(f"SAM image_size already {new_size}, no resize needed")
        return sam

    patch_size = 16
    old_grid = old_size // patch_size
    new_grid = new_size // patch_size
    print(f"SAM: resizing image_size {old_size}→{new_size} (grid {old_grid}→{new_grid})")

    encoder = sam.image_encoder
    encoder.img_size = new_size

    # Interpolate position embedding
    if encoder.pos_embed is not None:
        pe = encoder.pos_embed.data  # (1, H, W, D)
        pe = pe.permute(0, 3, 1, 2)  # (1, D, H, W)
        pe = F.interpolate(pe, size=(new_grid, new_grid), mode='bilinear', align_corners=True)
        pe = pe.permute(0, 2, 3, 1)  # (1, H, W, D)
        encoder.pos_embed = nn.Parameter(pe)
        print(f"  pos_embed: {old_grid}×{old_grid} → {new_grid}×{new_grid}")

    # Interpolate relative position biases in attention blocks
    for i, blk in enumerate(encoder.blocks):
        if hasattr(blk, 'attn') and hasattr(blk.attn, 'rel_pos_h'):
            h, w = blk.attn.rel_pos_h.shape
            rph = blk.attn.rel_pos_h.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            rpw = blk.attn.rel_pos_w.unsqueeze(0).unsqueeze(0)
            # rel_pos has shape (2*grid-1, head_dim)
            new_rel_h = 2 * new_grid - 1
            rph = F.interpolate(rph, size=(new_rel_h, w), mode='bilinear', align_corners=True)
            rpw = F.interpolate(rpw, size=(new_rel_h, w), mode='bilinear', align_corners=True)
            blk.attn.rel_pos_h = nn.Parameter(rph.squeeze(0).squeeze(0))
            blk.attn.rel_pos_w = nn.Parameter(rpw.squeeze(0).squeeze(0))
    print(f"  rel_pos: interpolated for {len(encoder.blocks)} blocks")

    # Update prompt_encoder sizes
    prompt_enc = sam.prompt_encoder
    prompt_enc.image_embedding_size = (new_grid, new_grid)
    prompt_enc.input_image_size = (new_size, new_size)
    prompt_enc.mask_input_size = (4 * new_grid, 4 * new_grid)
    print(f"  prompt_encoder sizes updated")

    return sam


# ═══════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════

class SingleItemDataset(torch.utils.data.Dataset):
    def __init__(self, sample):
        self.sample = sample
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return self.sample


def norm_batch(x):
    bs = x.shape[0]
    Isize = x.shape[-1]
    min_value = x.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    max_value = x.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    x = (x - min_value) / (max_value - min_value + 1e-6)
    return x


def _spatial_dims(tensor):
    """Return spatial dims tuple for tensor: (2, 3) for 2D, (2, 3, 4) for 3D."""
    return tuple(range(2, tensor.dim()))


def Dice_loss_binary(y_true, y_pred, smooth=1e-6):
    """Binary Dice loss (per-sample). Works for both 2D and 3D."""
    y_pred = y_pred.clamp(0, 1)
    y_true = y_true.clamp(0, 1)
    sd = _spatial_dims(y_true)
    tp = torch.sum(y_true * y_pred, dim=sd)
    fn = torch.sum(y_true * (1 - y_pred), dim=sd)
    fp = torch.sum((1 - y_true) * y_pred, dim=sd)
    tversky = (tp + smooth) / (tp + 0.5 * fn + 0.5 * fp + smooth)
    return 1 - torch.mean(tversky)


def Dice_loss_multiclass(y_true, y_pred, smooth=1e-6):
    """Multi-class Dice loss (softmax probabilities, exclude background)."""
    # y_true: (B, C, H, W) one-hot, y_pred: (B, C, H, W) softmax probabilities
    num_classes = y_pred.shape[1]
    sd = _spatial_dims(y_true)
    dice = 0
    for c in range(1, num_classes):  # skip background (channel 0)
        y_true_c = y_true[:, c:c+1]
        y_pred_c = y_pred[:, c:c+1]
        tp = torch.sum(y_true_c * y_pred_c, dim=sd)
        fn = torch.sum(y_true_c * (1 - y_pred_c), dim=sd)
        fp = torch.sum((1 - y_true_c) * y_pred_c, dim=sd)
        dice_c = (2 * tp + smooth) / (2 * tp + fn + fp + smooth)
        dice += dice_c
    return 1 - torch.mean(dice / (num_classes - 1))


def get_dice_ji(predict, target):
    """Binary Dice/IoU for evaluation."""
    predict = predict.flatten() + 1
    target = target.flatten() + 1
    tp = np.sum((predict == 2) & (target == 2))
    fp = np.sum((predict == 2) & (target == 1))
    fn = np.sum((predict == 1) & (target == 2))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji


def get_dice_multiclass(predict, target, num_classes):
    """Multi-class Dice per class. predict & target are class-index maps."""
    dice_list = []
    for c in range(1, num_classes):
        pred_c = (predict == c).astype(np.float32)
        targ_c = (target == c).astype(np.float32)
        tp = (pred_c * targ_c).sum()
        fp = pred_c.sum() - tp
        fn = targ_c.sum() - tp
        denom = 2 * tp + fp + fn
        if denom > 0:
            d = float(2 * tp / denom)
        else:
            d = 0.0
        dice_list.append(d)
    return np.mean(dice_list) if dice_list else 0.0


def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return str(len(a))


def gen_step_binary(optimizer, gts, masks, criterion, accumulation_steps, step, model):
    """Binary loss step (ISIC)."""
    size = masks.shape[2:]
    gts_sized = F.interpolate(gts.unsqueeze(dim=1), size, mode='nearest').float()
    gts_sized = gts_sized.clamp(0, 1)

    loss_1 = criterion(masks, gts_sized)
    loss_2 = Dice_loss_binary(masks.sigmoid(), gts_sized)
    loss = loss_1 + loss_2
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def gen_step_multi_bce(optimizer, gts, masks, criterion, accumulation_steps, step, model):
    """Multi-channel BCE loss step (REFUGE: 2 channels)."""
    size = masks.shape[2:]
    # gts: (B, C, H, W), masks: (B, C, H, W) logits
    gts_sized = F.interpolate(gts, size, mode='nearest').float().clamp(0, 1)

    loss_1 = criterion(masks, gts_sized)
    loss_2 = Dice_loss_binary(masks.sigmoid(), gts_sized)
    loss = loss_1 + loss_2
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def gen_step_multi_ce(optimizer, gts, masks, ce_loss, accumulation_steps, step, model):
    """Multi-class CE+Dice loss (Synapse: 9 classes)."""
    size = masks.shape[2:]
    # gts: (B, H, W) class indices
    # interpolate expects (B, C, H, W), so unsqueeze channel dim
    gts_sized = F.interpolate(gts.unsqueeze(1).float(), size, mode='nearest').squeeze(1).long()

    ce = ce_loss(masks, gts_sized)  # CrossEntropyLoss
    probs = F.softmax(masks, dim=1)
    gts_onehot = F.one_hot(gts_sized, num_classes=masks.shape[1]).permute(0, 3, 1, 2).float()
    dice = Dice_loss_multiclass(gts_onehot, probs)
    loss = ce + dice
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def get_input_dict(imgs, original_sz, img_sz):
    batched_input = []
    for i, img in enumerate(imgs):
        input_size = tuple([int(x) for x in img_sz[i].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[i].squeeze().tolist()])
        batched_input.append({
            'image': img,
            'original_size': original_size,
            'image_size': input_size,
            'point_coords': None,
            'point_labels': None,
        })
    return batched_input


# ═══════════════════════════════════════════════
# SAM forward
# ═══════════════════════════════════════════════

def sam_call_2d(batched_input, sam, dense_embeddings, mask_refinement,
                num_classes, mode):
    """SAM forward with AutoSAM dense prompt embeddings.

    NOTE: images in batched_input are already preprocessed (normalized + padded)
    by the dataset loader. We do NOT call sam.preprocess() again to avoid
    double normalization.
    """
    with torch.no_grad():
        input_images = torch.stack([x["image"] for x in batched_input], dim=0)
        image_embeddings = sam.image_encoder(input_images)
        sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )

    multimask_output = (num_classes > 1)

    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
    )

    # low_res_masks: (B, num_classes, 256, 256) if multimask_output
    #                (B, 1, 256, 256) if not

    # Apply mask refinement (without final activation — loss handles it)
    refined = mask_refinement(low_res_masks, apply_activation=False)
    return refined


# ═══════════════════════════════════════════════
# 2D Training / Inference
# ═══════════════════════════════════════════════

def train_single_epoch_2d(ds, model, sam, optimizer, transform, epoch,
                          mask_refinement, num_classes, mode, task):
    loss_list = []
    pbar = tqdm(ds)
    Idim = int(args['Idim'])
    optimizer.zero_grad()

    # Loss
    if mode == 'multi_ce':
        ce_loss = nn.CrossEntropyLoss().to(sam.device)
    else:
        criterion = nn.BCEWithLogitsLoss().to(sam.device)

    for ix, batch in enumerate(pbar):
        imgs, gts, original_sz, img_sz = batch
        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)

        orig_imgs_small = F.interpolate(orig_imgs, (Idim, Idim), mode='bilinear', align_corners=True)
        dense_embeddings = model(orig_imgs_small)
        batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
        masks = sam_call_2d(batched_input, sam, dense_embeddings,
                            mask_refinement, num_classes, mode)
        # NOTE: do NOT norm_batch with BCEWithLogitsLoss/CrossEntropyLoss —
        # these expect raw logits. norm_batch would compress to [0,1] and
        # prevent learning (sigmoid([0,1]) ≈ [0.5,0.73] → loss stuck at ~0.69/ch)

        if mode == 'binary':
            loss = gen_step_binary(optimizer, gts, masks, criterion, 4, ix, model)
        elif mode == 'multi_bce':
            loss = gen_step_multi_bce(optimizer, gts, masks, criterion, 4, ix, model)
        elif mode == 'multi_ce':
            loss = gen_step_multi_ce(optimizer, gts, masks, ce_loss, 4, ix, model)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        loss_list.append(loss)
        pbar.set_description(f'(train | {task}) epoch {epoch} :: loss {np.mean(loss_list):.4f}')

    return np.mean(loss_list)


def inference_ds_2d(ds, model, sam, transform, epoch, args,
                    mask_refinement, num_classes, mode):
    pbar = tqdm(ds)
    model.eval()
    iou_list = []
    dice_list = []
    Idim = int(args['Idim'])

    for img_idx, batch in enumerate(pbar):
        imgs, gts, original_sz, img_sz = batch
        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)

        orig_imgs_small = F.interpolate(orig_imgs, (Idim, Idim), mode='bilinear', align_corners=True)
        dense_embeddings = model(orig_imgs_small)
        batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
        masks_logits = sam_call_2d(batched_input, sam, dense_embeddings,
                                   mask_refinement, num_classes, mode)

        if mode == 'binary':
            masks_pred = (masks_logits.sigmoid() > 0.5).float()
            gts_resized = F.interpolate(gts.unsqueeze(0), size=masks_pred.shape[2:], mode='nearest').squeeze(0)
            dice, ji = get_dice_ji(
                masks_pred.squeeze().detach().cpu().numpy(),
                gts_resized.squeeze().detach().cpu().numpy())
        elif mode == 'multi_bce':
            masks_pred = (masks_logits.sigmoid() > 0.5).float()
            gts_resized = F.interpolate(gts, size=masks_pred.shape[2:], mode='nearest')
            # Average Dice across channels
            dice_vals = []
            for c in range(num_classes):
                d, j = get_dice_ji(
                    masks_pred[0, c].detach().cpu().numpy(),
                    gts_resized[0, c].detach().cpu().numpy())
                dice_vals.append(d)
            dice = np.mean(dice_vals)
            ji = dice  # approximate
        elif mode == 'multi_ce':
            masks_pred = masks_logits.argmax(dim=1)  # (B, H, W) class indices
            # gts: (H, W) class indices for batch_size=1
            if gts.dim() == 2:
                gts = gts.unsqueeze(0)  # (1, H, W)
            gts_resized = F.interpolate(gts.unsqueeze(1).float(),
                                         size=masks_pred.shape[1:],
                                         mode='nearest').squeeze(1).long()
            dice = get_dice_multiclass(
                masks_pred.squeeze().detach().cpu().numpy(),
                gts_resized.squeeze().detach().cpu().numpy(),
                num_classes)
            ji = dice  # approximate
        else:
            raise ValueError(f"Unknown mode: {mode}")

        iou_list.append(ji)
        dice_list.append(dice)
        pbar.set_description(
            f'(Inference | {args["task"]}) Epoch {epoch} :: Dice {np.mean(dice_list):.4f} :: IoU {np.mean(iou_list):.4f}')

    return np.mean(iou_list)


# ═══════════════════════════════════════════════
# 3D Training / Inference (unchanged binary logic)
# ═══════════════════════════════════════════════

def train_single_epoch_3d(ds, model, sam, optimizer, transform, epoch,
                          mask_refinement, num_classes, mode):
    loss_list = []
    pbar = tqdm(ds)
    criterion = nn.BCEWithLogitsLoss()
    Idim = int(args['Idim'])
    NumSliceDim = int(args['NumSliceDim'])
    optimizer.zero_grad()

    consecutive_failures = 0
    max_failures = 10

    save_dir = os.path.join(args.get('output_dir', './output'), 'debug_training')
    os.makedirs(save_dir, exist_ok=True)

    i = 0
    for ix, batch in enumerate(pbar):
        for sample in batch:
            if consecutive_failures >= max_failures:
                return np.mean(loss_list) if loss_list else 0
            try:
                img_tensor, gts, original_sz, _ = sample
                orig_img = img_tensor.to(sam.device)
                gts = gts.to(sam.device)

                volume_depth = orig_img.shape[-1]
                if volume_depth <= NumSliceDim - 1:
                    continue
                if gts.max() < 0.9:
                    continue
                start_frame = np.random.randint(0, volume_depth - NumSliceDim + 1)
                selected_slices = orig_img[:, :, start_frame:start_frame + NumSliceDim]
                selected_gts = gts[:, :, start_frame:start_frame + NumSliceDim]
                selected_slices = selected_slices.permute(2, 0, 1)
                selected_gts = selected_gts.permute(2, 0, 1).unsqueeze(0)
                if selected_gts.max() < 0.9:
                    continue

                orig_imgs_small = F.interpolate(
                    selected_slices.unsqueeze(0).unsqueeze(1),
                    size=(NumSliceDim, Idim, Idim),
                    mode='trilinear', align_corners=True)
                if orig_imgs_small.shape[1] == 1:
                    orig_imgs_small = orig_imgs_small.repeat(1, 3, 1, 1, 1)

                dense_embeddings = model(orig_imgs_small)

                for slice_idx in range(NumSliceDim):
                    with torch.cuda.amp.autocast():
                        current_slice = orig_imgs_small[:, :, slice_idx, :, :]
                        current_dense_embeddings = dense_embeddings[:, :, slice_idx, :, :]
                        batched_input = get_input_dict([current_slice], [original_sz], [original_sz])
                        temp_mask = sam_call_2d(batched_input, sam, current_dense_embeddings,
                                               mask_refinement, num_classes, mode).unsqueeze(2)
                    if slice_idx == 0:
                        mask = temp_mask
                    else:
                        mask = torch.cat((mask, temp_mask), 2)

                loss = gen_step_binary(optimizer, selected_gts, mask, criterion, 4, ix, model)
                loss_list.append(loss)
                consecutive_failures = 0
                i += 1
            except Exception as e:
                consecutive_failures += 1
                continue

        pbar.set_description(f'(train) epoch {epoch} :: loss {np.mean(loss_list):.4f}')
    return np.mean(loss_list)


def inference_ds_3d(ds, model, sam, transform, epoch, args,
                    mask_refinement, num_classes, mode):
    pbar = tqdm(ds)
    model.eval()
    iou_list = []
    dice_list = []
    Idim = int(args['Idim'])
    NumSliceDim = int(args['NumSliceDim'])

    for img_idx, (imgs, gts, original_sz, img_sz) in enumerate(pbar):
        orig_imgs = imgs.to(sam.device)
        if orig_imgs.ndim == 4:
            orig_imgs = orig_imgs.unsqueeze(0)
        elif orig_imgs.ndim == 3:
            orig_imgs = orig_imgs.unsqueeze(0).unsqueeze(0)
        gts = gts.to(sam.device)

        orig_imgs_small = F.interpolate(orig_imgs, (NumSliceDim, Idim, Idim),
                                        mode='trilinear', align_corners=True)
        if orig_imgs_small.shape[1] == 1:
            orig_imgs_small = orig_imgs_small.repeat(1, 3, 1, 1, 1)

        dense_embeddings = model(orig_imgs_small)
        volume_depth = orig_imgs.shape[-1]
        if volume_depth < NumSliceDim:
            continue

        for slice_idx in range(NumSliceDim):
            current_slice = orig_imgs_small[:, :, slice_idx, :, :]
            current_dense_embeddings = dense_embeddings[:, :, slice_idx, :, :]
            batched_input = get_input_dict([current_slice], [original_sz], [original_sz])
            temp_mask = sam_call_2d(batched_input, sam, current_dense_embeddings,
                                   mask_refinement, num_classes, mode).unsqueeze(2)
            if slice_idx == 0:
                mask = temp_mask
            else:
                mask = torch.cat((mask, temp_mask), dim=2)

        gts_resized = F.interpolate(gts.unsqueeze(0), size=mask.shape[2:], mode='nearest').squeeze(0)
        dice, ji = get_dice_ji(
            mask.squeeze().detach().cpu().numpy(),
            gts_resized.squeeze().detach().cpu().numpy())
        iou_list.append(ji)
        dice_list.append(dice)

        pbar.set_description(
            f'(Inference | {args["task"]}) Epoch {epoch} :: Dice {np.mean(dice_list):.4f} :: IoU {np.mean(iou_list):.4f}')
    return np.mean(iou_list)


# ═══════════════════════════════════════════════
# Dataset dispatcher
# ═══════════════════════════════════════════════

def load_dataset(args, sam_trans):
    task = args.get('task', 'isic')
    if task == 'isic':
        return get_isic_dataset(args, sam_trans=sam_trans)
    elif task == 'refuge':
        return get_refuge_dataset(args, sam_trans=sam_trans)
    elif task == 'synapse':
        return get_synapse_dataset(args, sam_trans=sam_trans)
    elif task == 'monu':
        return get_monu_dataset(args, sam_trans=sam_trans)
    elif task == 'glas':
        return get_glas_dataset(args, sam_trans=sam_trans)
    elif task == 'polyp':
        return get_polyp_dataset(args, sam_trans=sam_trans)
    elif task == 'lung':
        return get_lung_dataset(args, sam_trans=sam_trans)
    else:
        raise ValueError(f"Unknown task: {task}")


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main(args=None, sam_args=None):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    task = args.get('task', 'isic')
    is_3d = task in TASKS_3D
    cfg = TASK_CONFIG[task]
    num_classes = cfg['num_classes']
    mode = cfg['mode']

    # SAM input size
    sam_size = int(args.get('sam_size', 512))
    embed_dim = sam_size // 16
    args['embed_dim'] = embed_dim
    if is_3d:
        args['Idim'] = embed_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Task: {task}, Classes: {num_classes}, "
          f"Mode: {mode}, SAM size: {sam_size}, Embed dim: {embed_dim}")

    # ── SAM ──
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=device)

    checkpoint = torch.load(sam_args['sam_checkpoint'], weights_only=True, map_location="cpu")
    valid_keys = set(sam.state_dict().keys())
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in valid_keys}
    sam.load_state_dict(filtered_checkpoint, strict=False)
    print("SAM checkpoint loaded.")

    # Resize SAM position embeddings for the target image size
    sam = resize_sam_image_size(sam, sam_size)

    # Expand SAM for multi-class
    sam = expand_sam_multimask(sam, num_classes)

    transform = ResizeLongestSide(sam_size)

    # ── Model ──
    if is_3d:
        model = ModelEmb3D(args=args).float().to(device)
    else:
        model = ModelEmb(args=args).float().to(device)

    mask_refinement = MaskRefinement2D(in_channels=num_classes,
                                       out_channels=num_classes).float().to(device)

    # Collect trainable parameters:
    # - AutoSAM embedding model (HarDNet + decoder)
    # - MaskRefinement2D
    # - SAM mask_decoder mask_tokens (needed for expanded classes like Synapse 9ch)
    trainable_params = list(model.parameters()) + list(mask_refinement.parameters())
    trainable_params += list(sam.mask_decoder.mask_tokens.parameters())
    # Also train output_hypernetworks_mlps and iou_prediction_head for multi-class
    if num_classes > 3:
        trainable_params += list(sam.mask_decoder.output_hypernetworks_mlps.parameters())
        trainable_params += list(sam.mask_decoder.iou_prediction_head.parameters())

    optimizer = optim.Adam(
        trainable_params,
        lr=float(args['learning_rate']),
        weight_decay=float(args['WD']))

    # Resume
    if args.get('resume'):
        if os.path.exists(args['resume']):
            model.load_state_dict(torch.load(args['resume'], map_location=device), strict=False)

    # ── Dataset ──
    print('Loading dataset...')
    trainset, testset = load_dataset(args, sam_trans=transform)
    print('Dataset loaded.')

    if args.get('debug_mode', False) and len(trainset) > 0:
        trainset = SingleItemDataset(trainset[0])

    if is_3d:
        ds = DataLoader(trainset, batch_size=int(args['Batch_size']), shuffle=True,
                        num_workers=int(args['nW']), drop_last=True, collate_fn=lambda x: x)
    else:
        ds = DataLoader(trainset, batch_size=int(args['Batch_size']), shuffle=True,
                        num_workers=int(args['nW']), drop_last=True)

    ds_val = DataLoader(testset, batch_size=1, shuffle=False,
                        num_workers=int(args['nW_eval']), drop_last=False)

    # ── Output dirs ──
    output_dir = args.get('output_dir', './output')
    results_dir = os.path.join(output_dir, args['folder'])
    os.makedirs(results_dir, exist_ok=True)
    args['path_best'] = os.path.join(results_dir, 'net_best.pth')
    path_best = os.path.join(results_dir, 'best.csv')
    os.makedirs(args.get('vis_folder', results_dir + '/vis'), exist_ok=True)

    best = 0
    f_best = open(path_best, 'w')

    # ── Training loop ──
    for epoch in range(int(args['epoches'])):
        print(f"Epoch {epoch}/{args['epoches']}")

        if is_3d:
            train_single_epoch_3d(ds, model.train(), sam.eval(), optimizer, transform,
                                  epoch, mask_refinement, num_classes, mode)
        else:
            train_single_epoch_2d(ds, model.train(), sam.eval(), optimizer, transform,
                                  epoch, mask_refinement, num_classes, mode, task)

        with torch.no_grad():
            if is_3d:
                IoU_val = inference_ds_3d(ds_val, model.eval(), sam, transform, epoch,
                                          args, mask_refinement, num_classes, mode)
            else:
                IoU_val = inference_ds_2d(ds_val, model.eval(), sam, transform, epoch,
                                          args, mask_refinement, num_classes, mode)

            if IoU_val > best:
                torch.save(model.state_dict(), args['path_best'])
                best = IoU_val
                print(f"Best IoU: {best:.4f}")
                f_best.write(f"{epoch},{best}\n")
                f_best.flush()

    f_best.close()
    print(f"Done. Best IoU: {best:.4f}, model: {args['path_best']}")


# ═══════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='AutoSAM Training')

    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float)
    parser.add_argument('-bs', '--Batch_size', default=3, type=int)
    parser.add_argument('-epoches', '--epoches', default=200, type=int)
    parser.add_argument('-nW', '--nW', default=0, type=int)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, type=int)
    parser.add_argument('-WD', '--WD', default=1e-4, type=float)
    parser.add_argument('-task', '--task', default='isic', type=str,
                        help='isic | refuge | synapse | monu | glas | polyp | lung')
    parser.add_argument('-depth_wise', '--depth_wise', action='store_true', default=False)
    parser.add_argument('-order', '--order', default=85, type=int)
    parser.add_argument('-Idim', '--Idim', default=64, type=int)
    parser.add_argument('-NumSliceDim', '--NumSliceDim', default=32, type=int)
    parser.add_argument('-rotate', '--rotate', default=22, type=float)
    parser.add_argument('-scale1', '--scale1', default=0.75, type=float)
    parser.add_argument('-scale2', '--scale2', default=1.25, type=float)
    parser.add_argument('-output_dir', '--output_dir', default='./output', type=str)
    parser.add_argument('-resume', '--resume', default='', type=str)
    parser.add_argument('-debug_mode', '--debug_mode', action='store_true', default=False)
    parser.add_argument('-sam_ckpt', '--sam_ckpt', default='./cp/sam_vit_h.pth', type=str)
    parser.add_argument('-sam_type', '--sam_type', default='vit_h', type=str)
    parser.add_argument('-sam_size', '--sam_size', default=512, type=int,
                        help='SAM input size: 512 (matches SAMed/MSA) or 1024 (original)')

    args = vars(parser.parse_args())

    # Create output directory with task name + timestamp
    import time
    vit = args.get('sam_type', 'unknown')
    run_name = f'{args["task"]}_{vit}_{time.strftime("%m%d_%H%M")}'
    os.makedirs('results', exist_ok=True)
    run_dir = os.path.join('results', run_name)
    os.makedirs(run_dir, exist_ok=True)
    args['folder'] = run_name
    args['path'] = os.path.join(run_dir, 'net_last.pth')
    args['path_best'] = os.path.join(run_dir, 'net_best.pth')
    args['vis_folder'] = os.path.join(run_dir, 'vis')
    os.makedirs(args['vis_folder'], exist_ok=True)
    args['visualize'] = True
    args['debug_mode'] = args.get('debug_mode', False)

    sam_args = {
        'sam_checkpoint': args.get('sam_ckpt', './cp/sam_vit_h.pth'),
        'model_type': args.get('sam_type', 'vit_h'),
        'generator_args': {
            'points_per_side': 8,
            'pred_iou_thresh': 0.95,
            'stability_score_thresh': 0.7,
            'crop_n_layers': 0,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 0,
            'point_grids': None,
            'box_nms_thresh': 0.7,
        },
        'gpu_id': 0,
    }

    main(args=args, sam_args=sam_args)
