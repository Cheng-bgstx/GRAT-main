import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_synapse import Synapse_dataset
import matplotlib.image as mpimg
from icecream import ic
from einops import repeat

class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: 'liver', 6: 'stomach',
                 7: 'aorta', 8: 'pancreas'}


def load_images_from_folder(folder):
    images = []
    imgs_names = os.listdir(folder)
    imgs_names.sort(key=lambda x: int(os.path.splitext(x)[0]))
    for filename in imgs_names:
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_path = os.path.join(folder, filename)
            img = mpimg.imread(img_path)[:, :, 0]
            # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # read gray scale image
            if img is not None:
                images.append(img)
    return np.array(images)


# test_adv_all_new.py

def inference(args, multimask_output, db_config, model, folder_path, test_save_path=None, ):
    db_test = db_config['Dataset'](base_dir=args.volume_path, list_dir=args.list_dir, split='test_vol')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4)
    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()

    # 初始化列表，用于存储每个案例的结果
    case_metrics_list = []

    # 遍历所有测试案例
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0]

        if case_name != args.test_batch and args.test_batch is not None:
            continue

        # 加载并重建3D对抗样本
        batch_path = folder_path + case_name
        if not os.path.exists(batch_path):
            logging.info(f"Skipping case {case_name} as its folder does not exist.")
            continue
        images_array = load_images_from_folder(batch_path)
        if images_array.size == 0:
            logging.info(f"Skipping case {case_name} as no images were found in its folder.")
            continue
        image_adv = torch.tensor(images_array).unsqueeze(0)

        # 调用核心评估函数
        metric_i, loss = test_single_volume(image_adv, image, label, model, classes=args.num_classes,
                                            multimask_output=multimask_output,
                                            patch_size=[args.img_size, args.img_size],
                                            input_size=[args.input_size, args.input_size],
                                            test_save_path=test_save_path, case=case_name,
                                            z_spacing=db_config['z_spacing'])

        # 提取当前案例的平均指标
        mean_dice_current = np.mean(metric_i, axis=0)[0]
        mean_hd95_current = np.mean(metric_i, axis=0)[1]
        loss_current = loss.item() if torch.is_tensor(loss) else loss

        # 将当前案例的结果存入列表
        case_metrics_list.append({
            'case_name': case_name,
            'mean_dice': mean_dice_current,
            'mean_hd95': mean_hd95_current,
            'mean_loss': loss_current
        })

        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
            i_batch, case_name, mean_dice_current, mean_hd95_current))

    if not case_metrics_list:
        logging.error("No cases were processed. Exiting.")
        return 0, 0, 0

    # --- 后续处理部分 ---

    metrics_df = pd.DataFrame(case_metrics_list)

    # 保存CSV
    csv_save_path = os.path.join(args.output_dir, 'per_case_metrics.csv')
    metrics_df.to_csv(csv_save_path, index=False)
    logging.info(f"Per-case metrics saved to {csv_save_path}")

    # 绘制并保存图表
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Evaluation Metrics per Case', fontsize=16)
    axes[0].plot(metrics_df['case_name'], metrics_df['mean_dice'], marker='o', linestyle='-')
    axes[0].set_title('Mean Dice Score');
    axes[0].set_ylabel('Dice Score');
    axes[0].grid(True)
    axes[1].plot(metrics_df['case_name'], metrics_df['mean_hd95'], marker='o', linestyle='-')
    axes[1].set_title('Mean HD95');
    axes[1].set_ylabel('HD95 (mm)');
    axes[1].grid(True)
    axes[2].plot(metrics_df['case_name'], metrics_df['mean_loss'], marker='o', linestyle='-')
    axes[2].set_title('Mean Loss');
    axes[2].set_ylabel('Loss');
    axes[2].set_xlabel('Case Name');
    axes[2].grid(True)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plot_save_path = os.path.join(args.output_dir, 'per_case_metrics_plot.png')
    plt.savefig(plot_save_path)
    logging.info(f"Metrics plot saved to {plot_save_path}")
    plt.close()

    # 优化：直接从DataFrame计算最终平均值，取代旧的metric_list/len_cases逻辑
    performance = metrics_df['mean_dice'].mean()
    mean_hd95 = metrics_df['mean_hd95'].mean()
    final_loss = metrics_df['mean_loss'].mean()

    logging.info('--- Overall Performance (from per-case metrics) ---')
    logging.info(f'mean_dice: {performance:.4f}, mean_hd95: {mean_hd95:.4f}, mean_loss: {final_loss:.4f}')
    logging.info("Testing Finished!")

    return performance, mean_hd95, final_loss


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--volume_path', type=str, default='testset/test_vol_h5/')
    parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/', help='list_dir')
    parser.add_argument('--output_dir', type=str, default='./output/adv_sam_lora')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='checkpoints/epoch_5_bd1_ld0.1_out1.pth',
                        help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
    parser.add_argument('--consider_lora', action='store_true', help='Whether to consider lora when attack')
    parser.add_argument('--test_batch', type=str, default=None)
    parser.add_argument('--out_path', type=str, default='testset/advimages-test_batch1_lowpass/')
    parser.add_argument('--noise_sd', type=float, default=0.15)
    parser.add_argument('--smoothed_momentum', type=float, default=0.25)
    parser.add_argument('--smoothed_freq', type=int, default=3)
    parser.add_argument('--sftm_hpyer', type=int, default=4)

    parser.add_argument('--att_step_size', type=float, default=2.0)
    parser.add_argument('--att_bound', type=float, default=10.0)
    parser.add_argument('--att_numstep', type=int, default=10)

    parser.add_argument('--noise_path', type=str,
                        default='common_perb/common_perturbation_meta_0.05sp_bound8_it6_0.2upsize_ls0.25.pth')
    parser.add_argument('--online_update_method', type=str, default=None,
                        help='用于在线更新初始化扰动的策略 (仅为日志记录)')
    parser.add_argument('--online_update_freq', type=int, default=0,
                        help='在线更新的频率 (仅为日志记录)')
    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.consider_lora:
        folder_path = 'testset/advimages-test_batch1_loar/'  # the adv_img path
    else:
        folder_path = 'testset/advimages_ours_smoothed_attack/'
    folder_path = args.out_path
    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])

    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    performance, mean_hd95, loss = inference(args, multimask_output, dataset_config[dataset_name], net, folder_path,
                                             test_save_path)
    file_path = 'results_all.txt'

    file_path = 'Exp_log' + '/' + file_path[:-4] + '_layer' + str(args.sftm_hpyer) + '_stepsize' + str(
        args.att_step_size) + '_bound' + str(args.att_bound) + '_steps' + str(args.att_numstep) + file_path[-4:]
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'a') as file:
        file.write(f"path:{args.noise_path}, model:{args.lora_ckpt}\n")
        file.write(
            f"noise_sd: {args.noise_sd},sm_momentum: {args.smoothed_momentum},sm_freq: {args.smoothed_freq},sft_hyp: {args.sftm_hpyer}"
            f"online_method: {args.online_update_method},online_freq: {args.online_update_freq},"  
            f"mean_dice: {performance},mean_hd95: {mean_hd95}, mean_loss:{loss}\n")
    # print(performance,mean_hd95)
