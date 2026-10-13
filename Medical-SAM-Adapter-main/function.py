
import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import cfg
import models.sam.utils.transforms as samtrans
import pytorch_ssim
#from models.discriminatorlayer import discriminator
from conf import settings
from utils import *
from medpy import metric
# from lucent.modelzoo.util import get_model_layers
# from lucent.optvis import render, param, transform, objectives
# from lucent.modelzoo import inceptionv1

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.dataset == 'Synapse':
        lossfunc = loss_function 
    elif args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            # torch.cuda.empty_cache()
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:
                imgs, pt, masks, generated_labels = generate_click_prompt(imgs, masks)
                if args.dataset == 'Synapse' and masks.shape[1] == args.multimask_output:
                    masks = torch.argmax(masks, dim=1).unsqueeze(1).float()
                if pt.dim() == 3 and pt.shape[1] == 2:
                    pt = pt.permute(0, 2, 1)
                
                # 2. 定义缺失的 point_labels：全为 1 (表示正向提示点)
                # ptw 现在的形状是 [B, 9, 2]，num_points = 9
                b_size, num_points, _ = pt.size()
                point_labels = generated_labels.to(torch.int)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            if 'image_meta_dict' in pack:
                name = pack['image_meta_dict']['filename_or_obj']
            elif 'case_name' in pack:
                name = pack['case_name']  
            if args.thd:
                imgs, pt, masks = generate_click_prompt(imgs, masks)

                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')

                imgs = imgs.repeat(1,3,1,1)
                point_labels = torch.ones(imgs.size(0))

                imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
            showp = pt[..., [1, 0]]

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if point_labels.clone().flatten()[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                if(len(point_labels.shape)==1): # only one point prompt
                    coords_torch, labels_torch, showp = coords_torch.unsqueeze(1), labels_torch.unsqueeze(1), showp.unsqueeze(1)
                pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            # imgs = imgs.to(dtype = mask_type,device = GPUdevice)

            '''Train'''
            if args.mod == 'sam_adpt':
                for n, value in net.image_encoder.named_parameters(): 
                    if "Adapter" not in n:
                        value.requires_grad = False
                    else:
                        value.requires_grad = True
            elif args.mod == 'sam_lora' or args.mod == 'sam_adalora':
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.image_encoder)
                if args.mod == 'sam_adalora':
                    # Initialize the RankAllocator 
                    rankallocator = lora.RankAllocator(
                        net.image_encoder, lora_r=4, target_rank=8,
                        init_warmup=500, final_warmup=1500, mask_interval=10, 
                        total_step=3000, beta1=0.85, beta2=0.85, 
                    )
            else:
                for n, value in net.image_encoder.named_parameters(): 
                    value.requires_grad = True
                    
            origin_imgs = imgs.clone()
            imgs = net.preprocess(imgs)
            with torch.cuda.amp.autocast():
                imge= net.image_encoder(imgs)
                with torch.no_grad():
                    if args.net == 'sam' or args.net == 'mobile_sam':
                        se, de = net.prompt_encoder(
                            points=pt,
                            boxes=None,
                            masks=None,
                        )
                    elif args.net == "efficient_sam":
                        coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                        se = net.prompt_encoder(
                            coords=coords_torch,
                            labels=labels_torch,
                        )

                if args.net == 'sam':
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=(args.multimask_output > 1),
                    )
                elif args.net == 'mobile_sam':
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                    )
                elif args.net == "efficient_sam":
                    se = se.view(
                        se.shape[0],
                        1,
                        se.shape[1],
                        se.shape[2],
                    )
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        multimask_output=False,
                    )

                # Resize to the ordered output size
                pred = F.interpolate(pred,size=(args.out_size,args.out_size), mode="bilinear", align_corners=False)

            loss = lossfunc(pred.float(), masks)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            if args.mod == 'sam_adalora':
                scaler.scale(loss+lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                scaler.step(optimizer)
                scaler.update()
                rankallocator.update_and_mask(net, ind)
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            optimizer.zero_grad()

            '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name[:2]:
                        namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                    vis_image(origin_imgs/255,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

            pbar.update()

    return epoch_loss / len(train_loader)

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    dataset_size = len(val_loader.dataset)
    ave_res, mix_res = (0,0,0,0), (0,)*args.multimask_output*3
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    total_loss = 0
    total_dice = 0
    total_hd95 = 0
    total_iou = 0
    num_evals = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    real_val_step = 0
    if args.dataset == 'Synapse':
        lossfunc = loss_function 
    elif args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            if args.dataset == 'Synapse' and masksw.shape[1] == args.multimask_output:
                masksw = torch.argmax(masksw, dim=1).unsqueeze(1).float()
            cur_bsz = imgsw.shape[0]
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if args.dataset == 'Synapse':
                ptw = None
                point_labels = None
            # ----------------------------------------------------
            elif 'pt' not in pack or args.thd:
                imgsw, ptw, masksw, generated_labels = generate_click_prompt(imgsw, masksw)
                if ptw.dim() == 3 and ptw.shape[1] == 2:
                    ptw = ptw.permute(0, 2, 1)
                
                b_size, num_points, _ = ptw.size()
                point_labels = generated_labels.to(torch.int)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            if 'image_meta_dict' in pack:
                name = pack['image_meta_dict']['filename_or_obj']
            elif 'case_name' in pack:
                name = pack['case_name']
            
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            elif args.dataset == 'Synapse':
                # 3D 验证默认逐层：否则 [B*D,3,H,W] 一次送入 ViT（尤其 vit_h@512）会 OOM
                evl_ch = 1
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:,:,buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch
                real_val_step += 1
                if imgs.dim() == 5:
                    b, c, h, w, d = imgs.size()
                    # 将 Depth 维度并入 Batch 维度: [B*D, C, H, W]
                    imgs = imgs.permute(0, 4, 1, 2, 3).contiguous().view(-1, c, h, w)
                    masks = masks.permute(0, 4, 1, 2, 3).contiguous().view(-1, 1, h, w)

                    # 如果是单通道，重复为 3 通道以适配 SAM
                    if imgs.shape[1] == 1:
                        imgs = imgs.repeat(1, 3, 1, 1)
                if args.dataset == 'Synapse':
                    mask_for_prompt = masks.squeeze(1) 
                   
                    _, pt, _, generated_labels = generate_click_prompt(imgs, mask_for_prompt)
                    
                
                    pt = pt.permute(0, 2, 1)
                    
                   
                    point_labels = generated_labels.to(torch.int)
                    
                 
                    showp = pt[..., [1, 0]]
                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
                showp = pt[..., [1, 0]]

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                if point_labels.clone().flatten()[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    if(len(point_labels.shape)==1): # only one point prompt
                        coords_torch, labels_torch, showp = coords_torch.unsqueeze(1), labels_torch.unsqueeze(1), showp.unsqueeze(1)
                    pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                
                '''test'''
                with torch.no_grad():
                    origin_imgs = imgs.clone()
                    imgs = net.preprocess(imgs)
                    imge= net.image_encoder(imgs)
                    if args.net == 'sam' or args.net == 'mobile_sam':
                        se, de = net.prompt_encoder(
                            points=pt,
                            boxes=None,
                            masks=None,
                        )
                    elif args.net == "efficient_sam":
                        coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                        se = net.prompt_encoder(
                            coords=coords_torch,
                            labels=labels_torch,
                        )

                    if args.net == 'sam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de, 
                            multimask_output=(args.multimask_output > 1),
                        )
                    elif args.net == 'mobile_sam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de, 
                            multimask_output=False,
                        )
                    elif args.net == "efficient_sam":
                        se = se.view(
                            se.shape[0],
                            1,
                            se.shape[1],
                            se.shape[2],
                        )
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            multimask_output=False,
                        )
                    pred = F.interpolate(pred,size=(args.out_size,args.out_size), mode="bilinear", align_corners=False)
                    if args.vis and ind % args.vis == 0:
                        namecat = 'Test'
                        for na in name[:2]:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name + '+'
                        vis_image(origin_imgs / 255, pred, masks, os.path.join(args.path_helper['sample_path'],
                                                                               namecat + 'epoch+' + str(
                                                                                   epoch) + '.jpg'), reverse=False,points=showp)

                    # Resize to the ordered output size
                    if args.dataset == 'Synapse':

                        loss = lossfunc(pred.float(), masks)
                        total_loss += loss.item()

                        
                        dice, hd95, iou = eval_seg(pred, masks, threshold=None)
                        total_dice += dice
                        total_hd95 += hd95
                        total_iou += iou
                        num_evals += 1
                    else:
                       
                        tot += lossfunc(pred, masks) * cur_bsz
                        temp = eval_seg(pred, masks, threshold)
                        temp = tuple([number * cur_bsz for number in temp])
                        mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    if args.dataset == 'Synapse':
       
        if real_val_step == 0: real_val_step = 1
        return total_loss / real_val_step, (total_dice / real_val_step, total_hd95 / real_val_step, total_iou / real_val_step)
    else:

        return tot / len(val_loader.dataset), tuple([a / len(val_loader.dataset) for a in mix_res])

def transform_prompt(coord,label,h,w):
    coord = coord.transpose(0,1)
    label = label.transpose(0,1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
            :, :, : decoder_max_num_input_points, :
        ]
        label = label[
            :, :, : decoder_max_num_input_points
        ]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(
            rescaled_batched_points,
            (0, 0, 0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
        label = F.pad(
            label,
            (0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
    
    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points
    )

    return rescaled_batched_points,label


def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * 1024 / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * 1024 / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )