import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from torchvision import transforms
from icecream import ic


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


def _fp32_outputs_for_loss(outputs):
    """Autocast fp16 logits often overflow in CE+Dice; compute loss in float32."""
    out = dict(outputs)
    for k in ('low_res_logits', 'masks'):
        if k in out and torch.is_tensor(out[k]):
            out[k] = out[k].float()
    return out


def _amp_backward_step(model, optimizer, scaler, loss, max_norm=1.0):
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    params = model.module.parameters() if hasattr(model, 'module') else model.parameters()
    torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()


def trainer_synapse(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    skip_hard_nums = 0
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    outputs = model(image_batch, multimask_output, args.img_size)
                outputs = _fp32_outputs_for_loss(outputs)
                loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
                _amp_backward_step(model, optimizer, scaler, loss)
            else:
                outputs = model(image_batch, multimask_output, args.img_size)
                loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
                optimizer.zero_grad()
                loss.backward()
                if args.skip_hard and iter_num > 3 * args.warmup_period and loss.item() > 0.4:
                    skip_hard_nums += 1
                    print(f'Skip hard nums: {skip_hard_nums}')
                    continue
                optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** args.lr_exp
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                output_masks = outputs['masks']
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output_masks[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 20 # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_isic(args, model, snapshot_path, multimask_output, low_res):
    import hashlib
    from torch.utils.data import Subset
    from datasets.isic import ISIC2016
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # 1. 定义 Transforms
    # ISIC 图片通常较大，需要 Resize 到 args.img_size
    tf_train = transforms.Compose([

        transforms.Resize((args.img_size, args.img_size)),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    tf_mask = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # 2. 加载数据集 (mode='Train')
    db_train = ISIC2016(args, data_path=args.root_path, transform=tf_train, transform_msk=tf_mask, mode='Training')
    holdout = getattr(args, 'proxy_holdout_fraction', 0.0)
    if holdout:
        if not 0 < holdout < 1:
            raise ValueError('proxy_holdout_fraction must be in (0, 1)')
        names = db_train.name_list
        ordered = sorted(range(len(names)), key=lambda index: hashlib.sha256(
            f'{args.proxy_split_seed}:{names[index]}'.encode('utf-8')
        ).digest())
        count = max(1, min(len(names) - 1, round(len(names) * holdout)))
        db_train = Subset(db_train, ordered[count:])
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        import random
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    # 3. 定义 Loss
    # 注意：如果 num_classes=1，DiceLoss这里通常设为 num_classes+1 (背景+前景)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)

    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr

    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999),
                                weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9,
                              weight_decay=0.0001)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            # 获取数据
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch = image_batch.cuda()
            label_batch = label_batch.cuda()

            # 处理 label_batch
            # 原始 label_batch 可能是 [B, 1, H, W]，CrossEntropyLoss 需要 [B, H, W]
            if len(label_batch.shape) == 4:
                label_batch = label_batch.squeeze(1)

            # 生成 low_res_label (用于 calc_loss)
            # label_batch 是 [B, H, W]，interpolate 需要 [B, C, H, W]，所以先 unsqueeze 再 squeeze
            low_res_label_batch = F.interpolate(label_batch.unsqueeze(1).float(), size=(low_res, low_res),
                                                mode='nearest').squeeze(1)
            # 二值 mask 必须为 {0,1} 整型，否则 Dice 的 one-hot 与 CE 易数值异常
            low_res_label_batch = low_res_label_batch.round().long().clamp(0, 1)

            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    outputs = model(image_batch, multimask_output, args.img_size)
                outputs = _fp32_outputs_for_loss(outputs)
                loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss,
                                                     args.dice_param)
                _amp_backward_step(model, optimizer, scaler, loss)
            else:
                outputs = model(image_batch, multimask_output, args.img_size)
                loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 学习率调整
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** args.lr_exp
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
            iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            # 可视化
            if iter_num % 20 == 0:
                image = image_batch[1, :, :, :]  # RGB [3, H, W]
                # 反归一化以便显示 (假设使用 ImageNet 均值方差)
                image = (image - image.min()) / (image.max() - image.min())

                writer.add_image('train/Image', image, iter_num)

                output_masks = outputs['masks']
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output_masks[1, ...] * 50, iter_num)  # *50 为了让mask可见

                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # 保存模型
        save_interval = 20
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_refuge(args, model, snapshot_path, multimask_output, low_res):
    from datasets.refuge import REFUGE
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # 1. 定义 Transform (注意：这里不要包含 Normalize！)
    # 因为 refuge.py 会把这个 transform 同时用于 label (mask)，Norm 会破坏 label
    tf_train = transforms.Compose([

        transforms.Resize((args.img_size, args.img_size)),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])
    tf_mask = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # 2. 加载数据集
    # refuge.py 默认读取 'Training-400' 文件夹，请确保 args.root_path 指向包含该文件夹的父目录
    db_train = REFUGE(args, data_path=args.root_path, transform=tf_train, transform_msk=tf_mask, mode='Training')
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    # REFUGE 通常分为 3 类：背景(0), 视盘(1), 视杯(2)
    # 此时 args.num_classes 应该设为 2 (不含背景)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)

    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr

    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999),
                                weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9,
                              weight_decay=0.0001)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iterator = tqdm(range(max_epoch), ncols=70)

    # 定义归一化参数 (ImageNet Stats)，在循环内手动进

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            # 获取数据: image [B, 3, H, W], label [B, 2, H, W] (通道0是Cup, 通道1是Disc)
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # --- 关键步骤 2: 处理 Label ---
            # 将多通道 Label 转换为单通道多类别 Map (0:背景, 1:Disc, 2:Cup)
            # refuge.py 中 mask 是 float (mean of multi-raters)，先二值化
            gts = (label_batch > 0.5).float()
            print(f"Channel 0 sum: {gts[:, 0, ...].sum()}, Channel 1 sum: {gts[:, 1, ...].sum()}")
            # 创建空白 Label [B, H, W]
            target_label = torch.zeros(gts.shape[0], gts.shape[2], gts.shape[3]).cuda()

            # 逻辑：视杯(Cup)在视盘(Disc)内部。
            # 先赋值 Disc (Class 1)，再赋值 Cup (Class 2) 覆盖上去。
            # gts[:, 1, ...] 是 Disc, gts[:, 0, ...] 是 Cup
            target_label[gts[:, 1, ...] == 1] = 1  # 视盘区域
            target_label[gts[:, 0, ...] == 1] = 2  # 视杯区域 (覆盖视盘中心)

            # --- 关键步骤 3: 生成 low_res_label ---
            # 用于 Loss 计算的下采样标签
            low_res_label_batch = F.interpolate(target_label.unsqueeze(1), size=(low_res, low_res),
                                                mode='nearest').squeeze(1).long().clamp(0, 2)

            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    outputs = model(image_batch, multimask_output, args.img_size)
                outputs = _fp32_outputs_for_loss(outputs)
                loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss,
                                                     args.dice_param)
                _amp_backward_step(model, optimizer, scaler, loss)
            else:
                outputs = model(image_batch, multimask_output, args.img_size)
                loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 学习率调整 (同 Synapse)
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** args.lr_exp
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
            iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                # 可视化 (需要反归一化以便显示)
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)

                output_masks = outputs['masks']
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output_masks[0, ...] * 50, iter_num)

                labs = target_label[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # 保存模型逻辑 (同 Synapse)
        save_interval = 20
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
