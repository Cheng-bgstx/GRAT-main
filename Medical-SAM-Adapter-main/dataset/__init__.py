import numpy as np
import hashlib
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.sampler import SubsetRandomSampler

from utils import *

from .atlas import Atlas
from .brat import Brat
from .ddti import DDTI
from .isic import ISIC2016
from .kits import KITS
from .lidc import LIDC
from .lnq import LNQ
from .pendal import Pendal
from .refuge import REFUGE
from .segrap import SegRap
from .stare import STARE
from .toothfairy import ToothFairy
from .wbc import WBC
from .dataset_synapse import Synapse_dataset,RandomGenerator

def get_dataloader(args):
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])

    transform_train_seg = transforms.Compose([
        transforms.Resize((args.out_size,args.out_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])

    transform_test_seg = transforms.Compose([
        transforms.Resize((args.out_size,args.out_size)),
        transforms.ToTensor(),
    ])
    
    if args.dataset == 'isic':
        '''isic data'''
        isic_train_dataset = ISIC2016(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')

        holdout = getattr(args, 'proxy_holdout_fraction', 0.0)
        if holdout:
            if not 0 < holdout < 1:
                raise ValueError('proxy_holdout_fraction must be in (0, 1)')
            names = isic_train_dataset.name_list
            ordered = sorted(range(len(names)), key=lambda index: hashlib.sha256(
                f'{args.proxy_split_seed}:{names[index]}'.encode('utf-8')
            ).digest())
            count = max(1, min(len(names) - 1, round(len(names) * holdout)))
            train_data = Subset(isic_train_dataset, ordered[count:])
            test_data = Subset(isic_train_dataset, ordered[:count])
        else:
            train_data = isic_train_dataset
            test_data = ISIC2016(args, args.data_path, transform = transform_test,
                                 transform_msk=transform_test_seg, mode='Test')
        nice_train_loader = DataLoader(train_data, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(test_data, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''
    elif args.dataset == 'Synapse':
        # 训练集：使用 RandomGenerator 进行数据增强 (旋转/翻转)
        # 注意：RandomGenerator 需要 output_size 和 low_res 参数
        # output_size 对应 args.out_size (512), low_res 通常是 output_size 的一半或原本大小，这里如果是 512 输入，low_res 可以设为 512 或 256
        # 如果你的 dataset_synapse.py 里的 __getitem__ 返回 low_res_label，则需要匹配
        train_dataset = Synapse_dataset(
            base_dir=args.data_path,
            list_dir=args.list_dir,
            split="train",
            transform=RandomGenerator(output_size=[args.image_size, args.image_size],
                                      low_res=[args.image_size, args.image_size])
        )

        if 'train_npz' in args.data_path:
            test_base_dir = args.data_path.replace('train_npz', 'test_vol_h5')
        else:
            # 如果路径不包含 train_npz，可能需要手动指定或假设结构
            test_base_dir = args.data_path.replace('train_npz', 'test_vol_h5')
        test_dataset = Synapse_dataset(
            base_dir=test_base_dir,
            list_dir=args.list_dir,
            split="test_vol"
        )

        nice_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    elif args.dataset == 'decathlon':
        nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list = get_decath_loader(args)


    elif args.dataset == 'REFUGE':
        '''REFUGE data'''
        refuge_train_dataset = REFUGE(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
        refuge_test_dataset = REFUGE(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'LIDC':
        '''LIDC data'''
        # dataset = LIDC(data_path = args.data_path)
        dataset = MyLIDC(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'DDTI':
        '''DDTI data'''
        refuge_train_dataset = DDTI(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
        refuge_test_dataset = DDTI(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'Brat':
        '''Brat data'''
        dataset = Brat(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'STARE':
        '''STARE data'''
        # dataset = LIDC(data_path = args.data_path)
        dataset = STARE(args, data_path = args.data_path, transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'kits':
        '''kits data'''
        dataset = KITS(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'WBC':
        '''WBC data'''
        dataset = WBC(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'segrap':
        '''segrap data'''
        dataset = SegRap(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'toothfairy':
        '''toothfairy data'''
        dataset = ToothFairy(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'atlas':
        '''atlas data'''
        dataset = Atlas(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'pendal':
        '''pendal data'''
        dataset = Pendal(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'lnq':
        '''lnq data'''
        dataset = LNQ(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    else:
        print("the dataset is not supported now!!!")
        
    return nice_train_loader, nice_test_loader
