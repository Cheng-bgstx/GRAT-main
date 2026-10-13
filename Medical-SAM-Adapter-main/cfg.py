import argparse


def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=24, help='random seed')
    parser.add_argument('-net', type=str, default='sam', help='net type')
    parser.add_argument('-baseline', type=str, default='unet', help='baseline net type')
    parser.add_argument('-encoder', type=str, default='default', help='encoder type')
    parser.add_argument('-seg_net', type=str, default='transunet', help='net type')
    parser.add_argument('-mod', type=str, default='sam_adpt', help='mod type:seg,cls,val_ad')
    parser.add_argument('-exp_name', default='msa_test_isic', type=str, help='net type')
    parser.add_argument('-type', type=str, default='map', help='condition type:ave,rand,rand_map')
    parser.add_argument(
        '-vis',
        type=int,
        default=None,
        help='每多少步保存一次可视化；不设或 0 表示关闭（验证阶段不再报错）',
    )
    parser.add_argument('-reverse', type=bool, default=False, help='adversary reverse')
    parser.add_argument('-pretrain', type=bool, default=False, help='adversary reverse')
    parser.add_argument('-val_freq',type=int,default=5,help='interval between each validation')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-sim_gpu', type=int, default=0, help='split sim to this gpu')
    parser.add_argument('-epoch_ini', type=int, default=1, help='start epoch')
    parser.add_argument('-image_size', type=int, default=256, help='image_size')
    parser.add_argument('-out_size', type=int, default=256, help='output_size')
    parser.add_argument('-patch_size', type=int, default=2, help='patch_size')
    parser.add_argument('-dim', type=int, default=512, help='dim_size')
    parser.add_argument('-depth', type=int, default=1, help='depth')
    parser.add_argument('-heads', type=int, default=16, help='heads number')
    parser.add_argument('-mlp_dim', type=int, default=1024, help='mlp_dim')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=2, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-uinch', type=int, default=1, help='input channel of unet')
    parser.add_argument('-imp_lr', type=float, default=3e-4, help='implicit learning rate')
    parser.add_argument('-weights', type=str, default = 0, help='the weights file you want to test')
    parser.add_argument('-base_weights', type=str, default = 0, help='the weights baseline')
    parser.add_argument('-sim_weights', type=str, default = 0, help='the weights sim')
    parser.add_argument('-distributed', default='none' ,type=str,help='multi GPU ids to use')
    parser.add_argument('-dataset', default='isic' ,type=str,help='dataset name')
    parser.add_argument('-sam_ckpt', default=None , help='sam checkpoint address')
    parser.add_argument('-thd', type=bool, default=False , help='3d or not')
    parser.add_argument('-chunk', type=int, default=None , help='crop volume depth')
    parser.add_argument('-num_sample', type=int, default=4 , help='sample pos and neg')
    parser.add_argument('-roi_size', type=int, default=96 , help='resolution of roi')
    parser.add_argument(
        '-evl_chunk',
        type=int,
        default=None,
        help='验证/测试时每步沿 depth 维送入的切片数。Synapse 3D 未指定时代码里默认 1（避免整卷展平 batch 爆显存）；'
             '显存充足可设更大以加速验证。',
    )
    parser.add_argument('-mid_dim', type=int, default=None , help='middle dim of adapter or the rank of lora matrix')
    parser.add_argument('-grad_checkpoint', type=bool, default=False, help='use gradient checkpointing to save memory')
    parser.add_argument(
        '-multimask_output',
        type=int,
        default=1,
        help='mask decoder 输出通道数 / Synapse 多类评估用类别数。ISIC 用 1，REFUGE 用 2；'
             'Synapse(BTCV) 常见为 9（见 parse_args 里对默认 1 的自动纠正）。',
    )
    parser.add_argument(
    '-data_path',
    type=str,
    default='../data',
    help='The path of segmentation data')
    # '../dataset/RIGA/DiscRegion'
    # '../dataset/ISIC'
    parser.add_argument('-adv_path', type=str, default=None,help='Path to load adversarial images from. If None, load original images.')
    parser.add_argument('-list_dir', type=str, default='../dataset/Synapse/lists', help='list_dir')
    parser.add_argument(
        '-out_root',
        type=str,
        default='',
        help='If set, each run uses one folder under out_root whose name matches SAMed_h/train.py snapshot_path '
             '(Dataset_imgSize_pretrain_encoder_...); contains Model/, Log/, Samples/, log/ (TB), checkpoint/. '
             'Empty = legacy logs/<exp_name>_timestamp/ layout.',
    )
    parser.add_argument(
        '-max_iterations',
        type=int,
        default=30000,
        help='Only used for -out_root folder naming (same rule as SAMed when != 30000 adds _NNk)',
    )
    parser.add_argument(
        '-max_epochs',
        type=int,
        default=-1,
        help='Only used for -out_root folder naming; -1 means use settings.EPOCH (like SAMed max_epochs)',
    )
    parser.add_argument(
        '-save_freq',
        type=int,
        default=0,
        help='Save a resumable epoch checkpoint every N epochs; 0 disables trajectory snapshots.',
    )
    parser.add_argument(
        '-base_lr',
        type=float,
        default=None,
        help='If set, used in -out_root folder _lr* suffix (else args.lr), to match SAMed naming when desired',
    )
    parser.add_argument('-proxy_holdout_fraction', type=float, default=0.0,
                        help='Reserve this fraction of ISIC Training images for proxy calibration.')
    parser.add_argument('-proxy_split_seed', type=int, default=2026,
                        help='Stable image-name split seed for proxy calibration.')
    opt = parser.parse_args()

    # Synapse 多类分割：默认 multimask_output=1 时 eval_seg 中 range(1,1) 为空 → Mean of empty slice / NaN
    if getattr(opt, 'dataset', None) == 'Synapse' and getattr(opt, 'multimask_output', 1) == 1:
        import warnings as _w

        _w.warn(
            'dataset=Synapse 且未指定 -multimask_output：将自动设为 9（与 SAMed Synapse / 常见 BTCV 预处理一致）。'
            '若你的标签类别数不是 9，请显式传入 -multimask_output。',
            UserWarning,
            stacklevel=1,
        )
        opt.multimask_output = 9

    return opt

# required=True, 
