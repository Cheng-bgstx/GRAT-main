# train.py
#!/usr/bin/env	python3

""" valuate network using pytorch
    Junde Wu
"""

import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataset import *
from conf import settings
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function


def main():
    args = cfg.parse_args()
    if args.dataset == 'refuge' or args.dataset == 'refuge2':
        args.data_path = '../dataset'

    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

    '''load pretrained model'''
    assert args.weights != 0
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']

    state_dict = checkpoint['state_dict']
    if args.distributed != 'none':
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:] # remove `module.`
            name = 'module.' + k
            new_state_dict[name] = v
        # load params
    else:
        new_state_dict = state_dict

    net.load_state_dict(new_state_dict)

    # args.path_helper = checkpoint['path_helper']
    # logger = create_logger(args.path_helper['log_path'])
    # print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    # args.path_helper = set_log_dir('logs', args.exp_name)
    # logger = create_logger(args.path_helper['log_path'])
    # logger.info(args)

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    '''segmentation data'''
    nice_train_loader, nice_test_loader = get_dataloader(args)

    '''begain valuation'''
    best_acc = 0.0
    best_tol = 1e4

    if args.mod == 'sam_adpt':
        net.eval()
        if args.dataset == 'Synapse':
            tol, (avg_dice, avg_hd95, avg_iou) = function.validation_sam(args, nice_test_loader, start_epoch, net)
            
            logger.info(f'Total Score: {tol:.4f} || @ epoch {start_epoch}.')
            logger.info(f'Synapse Mean Dice: {avg_dice:.4f}, Mean HD95: {avg_hd95:.4f}, Mean IoU: {avg_iou:.4f}')
        elif args.dataset == 'isic':
            tol, (eiou, edice, ehd95) = function.validation_sam(args, nice_test_loader, start_epoch, net)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice}, HD95: {ehd95} || @ epoch {start_epoch}.')
        else:
            tol, (iou_d, iou_c, dice_d, dice_c, hd95_d, hd95_c) = function.validation_sam(args, nice_test_loader, start_epoch, net)
            avg_dice = (dice_d + dice_c) / 2.0
            avg_hd95 = (hd95_d + hd95_c) / 2.0
            logger.info(f'Total score: {tol} || @ epoch {start_epoch}.\n'
                        f'   IOU  - Disc: {iou_d:.4f}, Cup: {iou_c:.4f}\n'
                        f'   DICE - Disc: {dice_d:.4f}, Cup: {dice_c:.4f} (Avg: {avg_dice:.4f})\n'
                        f'   HD95 - Disc: {hd95_d:.4f}, Cup: {hd95_c:.4f} (Avg: {avg_hd95:.4f})')

if __name__ == '__main__':
    main()
