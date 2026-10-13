# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Junde Wu
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import cfg
import function
from conf import settings
#from models.discriminatorlayer import discriminator
from dataset import *
from utils import *

def main():

    args = cfg.parse_args()

    seed = args.seed
    set_seed(seed)

    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    if args.pretrain:
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights,strict=False)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

    '''load pretrained model'''
    if args.weights != 0:
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_tol = checkpoint['best_tol']

        net.load_state_dict(checkpoint['state_dict'],strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    out_root = getattr(args, 'out_root', '') or ''
    if out_root:
        os.makedirs(out_root, exist_ok=True)

    args.msa_exp_root = None
    if args.weights == 0:
        if out_root:
            from msa_snapshot_layout import build_samed_style_experiment_dir, make_path_helper_samed_style
            exp_dir_name = build_samed_style_experiment_dir(args)
            args.msa_exp_root = os.path.join(out_root, exp_dir_name)
            args.path_helper = make_path_helper_samed_style(args.msa_exp_root)
            print(f'=> MSA experiment dir (SAMed-style name): {args.msa_exp_root}')
        else:
            args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        logger.info(args)
    else:
        logger = create_logger(args.path_helper['log_path'])
        logger.info(args)

    nice_train_loader, nice_test_loader = get_dataloader(args)

    '''checkpoint path and tensorboard'''
    # iter_per_epoch = len(Glaucoma_training_loader)
    if getattr(args, 'msa_exp_root', None):
        # Align with SAMed: tensorboard under <snapshot>/log
        tb_dir = os.path.join(args.msa_exp_root, 'log')
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        ckpt_base = os.path.join(args.msa_exp_root, settings.CHECKPOINT_PATH)
        checkpoint_path = os.path.join(ckpt_base, args.net, settings.TIME_NOW)
    else:
        ckpt_base = os.path.join(out_root, settings.CHECKPOINT_PATH) if out_root else settings.CHECKPOINT_PATH
        tb_base = os.path.join(out_root, settings.LOG_DIR) if out_root else settings.LOG_DIR
        checkpoint_path = os.path.join(ckpt_base, args.net, settings.TIME_NOW)
        if not os.path.exists(tb_base):
            os.makedirs(tb_base, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(
                tb_base, args.net, settings.TIME_NOW))
    # input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
    # writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    '''begain training'''
    best_acc = 0.0
    best_tol = 1e4
    best_dice = 0.0

    total_epochs = args.max_epochs if getattr(args, 'max_epochs', -1) > 0 else settings.EPOCH
    for epoch in range(total_epochs):

        if epoch < 5 and args.val_freq <= 5:
            tol, mt = function.validation_sam(args, nice_test_loader, epoch, net, writer)
            mt = tuple(mt)
            if args.dataset == 'REFUGE':
                # eval_seg(c==2) 返回 6 项：iou_disc, iou_cup, disc_dice, cup_dice, disc_hd95, cup_hd95
                if len(mt) == 6:
                    iou_d, iou_c, d_dice, c_dice, hd_d, hd_c = mt
                    logger.info(
                        f'Total score: {tol}, IOU_DISC: {iou_d}, IOU_CUP: {iou_c}, DICE_DISC: {d_dice}, DICE_CUP: {c_dice}, '
                        f'HD95_DISC: {hd_d}, HD95_CUP: {hd_c} || @ epoch {epoch}.')
                else:
                    eiou_cup, eiou_disc, edice_cup, edice_disc = mt[:4]
                    logger.info(
                        f'Total score: {tol}, IOU_CUP: {eiou_cup}, IOU_DISC: {eiou_disc}, DICE_CUP: {edice_cup}, DICE_DISC: {edice_disc} || @ epoch {epoch}.')
            elif args.dataset == 'Synapse':
                avg_dice, avg_hd95, avg_iou = mt[0], mt[1], mt[2]
                logger.info(
                    f'Total Score: {tol:.4f} || Synapse Mean Dice: {avg_dice:.4f}, Mean HD95: {avg_hd95:.4f}, Mean IOU: {avg_iou:.4f} || @ epoch {epoch}.')
            else:
                # ISIC 等：eval_seg 单通道返回 (iou, dice, hd95)
                if len(mt) == 3:
                    eiou, edice, ehd95 = mt
                    logger.info(
                        f'Total score: {tol}, IOU: {eiou}, DICE: {edice}, HD95: {ehd95} || @ epoch {epoch}.')
                else:
                    eiou, edice = mt[0], mt[1]
                    logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        save_freq = int(getattr(args, 'save_freq', 0) or 0)
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            sd = net.module.state_dict() if args.distributed != 'none' else net.state_dict()
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_dice,
                'path_helper': args.path_helper,
            }, False, args.path_helper['ckpt_path'], filename=f"epoch_{epoch + 1}.pth")

        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == total_epochs-1:
            tol, mt = function.validation_sam(args, nice_test_loader, epoch, net, writer)
            mt = tuple(mt)
            if args.dataset == 'Synapse':
                avg_dice, avg_hd95, avg_iou = mt[0], mt[1], mt[2]
                logger.info(
                    f'Total Score: {tol:.4f} || Synapse Mean Dice: {avg_dice:.4f}, Mean HD95: {avg_hd95:.4f}, Mean IOU: {avg_iou:.4f} || @ epoch {epoch}.')
                current_perf = avg_dice
            elif args.dataset != 'REFUGE':
                if len(mt) == 3:
                    eiou, edice, ehd95 = mt
                    logger.info(
                        f'Total score: {tol}, IOU: {eiou}, DICE: {edice}, HD95: {ehd95} || @ epoch {epoch}.')
                else:
                    eiou, edice = mt[0], mt[1]
                    logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
                current_perf = mt[1] if len(mt) >= 2 else 0.0
            else:
                if len(mt) == 6:
                    iou_d, iou_c, d_dice, c_dice, hd_d, hd_c = mt
                    logger.info(
                        f'Total score: {tol}, IOU_DISC: {iou_d}, IOU_CUP: {iou_c}, DICE_DISC: {d_dice}, DICE_CUP: {c_dice}, '
                        f'HD95_DISC: {hd_d}, HD95_CUP: {hd_c} || @ epoch {epoch}.')
                    current_perf = (d_dice + c_dice) / 2
                else:
                    eiou_cup, eiou_disc, edice_cup, edice_disc = mt[:4]
                    logger.info(
                        f'Total score: {tol}, IOU_CUP: {eiou_cup}, IOU_DISC: {eiou_disc}, DICE_CUP: {edice_cup}, DICE_DISC: {edice_disc} || @ epoch {epoch}.')
                    current_perf = (edice_cup + edice_disc) / 2
                edice = current_perf

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            if current_perf > best_dice:
                best_dice = current_perf  # 别忘了更新 best_dice！原代码这里逻辑有点混乱
                best_tol = tol
                is_best = True

                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_dice,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="best_dice_checkpoint.pth")
            else:
                is_best = False

    writer.close()


if __name__ == '__main__':
    main()
