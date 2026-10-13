import torch
import numpy as np
import os
import argparse
from torch.cuda.amp import autocast, GradScaler
from segment_anything import sam_model_registry
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.dataset_synapse import Synapse_dataset
from einops import repeat
from torch.autograd import Variable
from PIL import Image
import torch.fft
import random
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from torchvision import transforms
from datasets.refuge import REFUGE
from datasets.isic import ISIC2016

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--output_dir', type=str, default='/output')
parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/', help='list_dir')
parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
parser.add_argument('--lora_ckpt', type=str, default='checkpoints/epoch_159.pth', help='The checkpoint from LoRA')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--consider_lora', action='store_true', help='Whether to consider lora when attack')
parser.add_argument('--test_batch', type=str, default=None)
parser.add_argument('--out_path', type=str, default='testset/advimages_ours_smoothedmulti_attack/')
parser.add_argument('--noise_sd', type=float, default=0.15)
parser.add_argument('--smoothed_momentum', type=float, default=0.25)
parser.add_argument('--smoothed_freq', type=int, default=3)
parser.add_argument('--sftm_hpyer', type=int, default=4)
parser.add_argument('--loss_thres', type=float, default=0.25)
parser.add_argument('--batch_iter_time', type=int, default=4)
parser.add_argument('--attack_iter', type=int, default=5)
parser.add_argument('--mul_modedl', type=int, default=0)
parser.add_argument('--upd_size', type=float, default=0.2)
parser.add_argument('--step_size', type=float, default=0.2)
parser.add_argument('--noise_path', type=str, default='common_perb/common_perturbation_meta_0.05sp_bound8_it6_0.2upsize_ls0.25.pth')
parser.add_argument('--refuge_path', type=str, default='../data/REFUGE')
parser.add_argument('--isic_path', type=str, default='../data/ISIC2016')
parser.add_argument('--volume_path', type=str, default='testset/test_vol_h5/')

args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

data_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
])


def return_eta(momentum,step_size):
    upd_momentum=momentum*(step_size/torch.mean(abs(momentum)))
    momentum_binary = torch.where(momentum > 0, torch.tensor(1.0), torch.tensor(0.0))
    eta = torch.clamp(upd_momentum, -1*step_size, -0.25*step_size)*(1-momentum_binary) + momentum_binary*torch.clamp(upd_momentum, 0.25*step_size, 1*step_size)
    return eta


def get_length(length):
    num_block = 2
    length = int(length)
    rand = torch.rand(num_block)
    rand_norm = torch.round(rand * length / rand.sum()).int()
    max_idx = torch.argmax(rand_norm)
    rand_norm[max_idx] += length - rand_norm.sum()
    return tuple(rand_norm.tolist())

def shuffle_rotate(x,x_c):
    num_block = 2
    n, c, w, h = x.shape
    width_length, height_length = get_length(w), get_length(h)
    while min(width_length)<1 or min(height_length) <1:
         width_length, height_length = get_length(w), get_length(h)

    width_perm = torch.randperm(num_block)
    height_perm = torch.randperm(num_block)

    # Angle calculations simplified for demonstration
    angles = torch.randn(x.size(0)) * 2  # smaller angle for visibility
    angles=angles.int()
    # Split and permute width
    x_split_w = x.split(width_length, dim=2)
    x_w_perm = torch.cat([x_split_w[i] for i in width_perm], dim=2)
    # Split and permute height
    x_split_h_l = [x_split_w[i].split(height_length, dim=3) for i in width_perm]
    x_h_perm_list = []

    x_h_perm = torch.concat([torch.concat([TF.rotate(strip[i], int(angles), interpolation=TF.InterpolationMode.BILINEAR) for i in height_perm],axis=3 ) for strip in x_split_h_l],axis=2)

    # x_c=x
    x_c_split_w = x_c.split(width_length, dim=2)
    x_c_w_perm = torch.cat([x_c_split_w[i] for i in width_perm], dim=2)
    # Split and permute height
    x_c_split_h_l = [x_c_split_w[i].split(height_length, dim=3) for i in width_perm]
    x_c_h_perm_list = []

    x_c_h_perm = torch.concat([torch.concat([TF.rotate(strip[i], int(angles), interpolation=TF.InterpolationMode.BILINEAR) for i in height_perm],axis=3 ) for strip in x_c_split_h_l],axis=2)
    return [x_h_perm,x_c_h_perm]
def BSR(x,x_c,num_copies):
    x_t=[]
    x_c_t=[]
   
    for i in range(num_copies):
       trans_x= shuffle_rotate(x,x_c)
       x_t.append(trans_x[0])
       x_c_t.append(trans_x[1])
    #    print(trans_x[0]==trans_x[1])
    return torch.concat(x_t,axis=0),torch.concat(x_c_t,axis=0)

sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])
sam=sam.to(device)

output_path=args.out_path
scaler = GradScaler()
dataset_config = {
    'Synapse': {
        'Dataset': Synapse_dataset,
        'volume_path': args.volume_path,
        'list_dir': args.list_dir,
        'split': 'train',  # Synapse 特有的参数
        'transform': None
    },
    'REFUGE': {
        'Dataset': REFUGE,
        'volume_path': args.refuge_path,
        'split': 'Training',  # REFUGE 特有的参数 (Training/Validation/Test)
        'transform': data_transform
    },
    'ISIC': {
        'Dataset': ISIC2016,
        'volume_path': args.isic_path,
        'split': 'Training',  # ISIC 特有的参数 (Training/Test)
        'transform': data_transform
    }
}

# --- 修改 Dataset 初始化逻辑 ---
db_config = dataset_config[args.dataset]

# 根据数据集类型传递不同的参数
if args.dataset == 'Synapse':
    db_test = db_config['Dataset'](base_dir=db_config['volume_path'], list_dir=db_config['list_dir'],
                                   split=db_config['split'])
elif args.dataset == 'REFUGE':
    # REFUGE 需要 args, data_path, transform, mode
    db_test = db_config['Dataset'](args=args, data_path=db_config['volume_path'], transform=db_config['transform'],
                                   mode=db_config['split'])
elif args.dataset == 'ISIC':
    # ISIC 需要 args, data_path, transform, mode
    db_test = db_config['Dataset'](args=args, data_path=db_config['volume_path'], transform=db_config['transform'],
                                   mode=db_config['split'])

testloader = DataLoader(db_test, batch_size=1, shuffle=True, num_workers=4)

if os.path.exists(args.noise_path):
    print(f"Loading existing perturbation from {args.noise_path}")
    common_perturbation = torch.load(args.noise_path).to(device)
else:
    print("Initializing new common perturbation...")
    common_perturbation = torch.zeros((1, 3, args.img_size, args.img_size)).to(device)

criterion_mse = torch.nn.MSELoss()
norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
for i_batch, sampled_batch in tqdm(enumerate(testloader)): #every-time one image from testloader
    if 'case_name' in sampled_batch:
        case_name = sampled_batch['case_name'][0]
    elif 'image_meta_dict' in sampled_batch:
        # REFUGE 和 ISIC 使用 image_meta_dict
        case_name = sampled_batch['image_meta_dict']['filename_or_obj'][0]
    if case_name !=args.test_batch and args.test_batch is not None:
            continue
    img_tensor = sampled_batch['image']
    if args.dataset == 'Synapse':
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(1)
        image_vol = img_tensor.squeeze(0).cpu().detach().numpy()
        if len(image_vol.shape) == 2:
            image_vol = image_vol[np.newaxis, np.newaxis, :, :]
        elif len(image_vol.shape) == 3:
            image_vol = image_vol[:, np.newaxis, :, :]

    else:
        image_vol = img_tensor.cpu().detach().numpy()

    image = image_vol
    batch_iter_times=args.batch_iter_time
    loss_thres=args.loss_thres
    attack_iters=args.attack_iter
    for iters in range(batch_iter_times):
        attack_iters=args.attack_iter
        step_size=args.step_size/255
            # step_size=step_size/((iters+1))
        epsilon=10/255
        count=0
        np.random.shuffle(image)
        for ind in range(image.shape[0]):  ### iter batch 里面的每张图片
            input_image = image[ind, :, :,:]
            input_image_torch = torch.from_numpy(input_image).unsqueeze(0).float().cuda()
            if input_image_torch.shape[1] == 1:
                # 如果是单通道 (Synapse)，需要 repeat 到 3 通道
                input_image_torch = repeat(input_image_torch, 'b c h w -> b (repeat c) h w', repeat=3)
            else:
                # 如果已经是 3 通道 (REFUGE/ISIC)，保持原样
                pass
            momentum = torch.zeros_like(input_image_torch).detach()
            input_image = input_image_torch
            with torch.no_grad():
                if args.dataset == 'Synapse':
                    # Synapse 不需要归一化
                    input_for_model = input_image_torch
                else:
                    # ISIC/REFUGE: 临时归一化，传给模型，但不改变 input_image_torch 本身
                    input_for_model = norm_transform(input_image_torch)

                targets, targets_layer = sam.image_encoder(input_for_model)

            max_val,min_val=1,0
            eta=torch.tensor(0).cuda()
            perturb_img = Variable(input_image_torch.data + common_perturbation, requires_grad=True)
            num_copies=1
            tansform_p=random.uniform(0,1)
            for _ in range(attack_iters):
                grad = None
                if args.dataset == 'Synapse':
                    perturb_img_input = perturb_img[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
                else:
                    perturb_img_input = perturb_img
                avg_grad = torch.zeros_like(perturb_img).detach().cuda()
                # print(perturb_img_input.shape,num_copies)
                if tansform_p>0.7:
                    perturb_img_input_batch,input_image_torch_batch = BSR(perturb_img_input,input_image_torch,num_copies)
                else:
                    perturb_img_input_batch,input_image_torch_batch =perturb_img_input,input_image_torch 
                # perturb_img = Variable(torch.clamp(perturb_img, min_val, max_val), requires_grad=True)
                sam.image_encoder.zero_grad()

                if args.dataset == 'Synapse':
                    perb_input = perturb_img_input_batch[0].unsqueeze(0)
                else:
                    # 对 Clean Image Batch 进行归一化
                    perb_input = norm_transform(perturb_img_input_batch[0].unsqueeze(0))

                features,features_layer = sam.image_encoder(perb_input)
                with torch.no_grad():
                    if args.dataset == 'Synapse':
                        clean_input = input_image_torch_batch[0].unsqueeze(0)
                    else:
                        # 对 Clean Image Batch 进行归一化
                        clean_input = norm_transform(input_image_torch_batch[0].unsqueeze(0))
                    targets_c, targets_layer_c = sam.image_encoder(clean_input)

                loss=criterion_mse(features*0.9999, targets_c)
                if loss > loss_thres:
                    eta = perturb_img.data - input_image_torch.data
                    print('this image fullfill requirement')
                    # if _ ==0:
                    count=count+1
                    break 
                del features,features_layer
                grad=torch.autograd.grad(loss, perturb_img,
                                        retain_graph=True, create_graph=False)[0]
                loss = loss.detach()
                grad=grad / torch.norm(grad, p=1)
                momentum = 0.5 * momentum.detach() +grad# / torch.norm(grad, p=1)
                momentum=momentum.detach()
                eta = step_size * momentum.sign()
                del grad
            
                perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
                eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)
                print('the loss is:', loss,count,'/',len(image),loss_thres,iters)#,'the avg energy is:',avg_energy)
                perturb_img = Variable(input_image_torch.data + eta, requires_grad=True)

            noise_slice = perturb_img.data - input_image.data
            upd_size=(epsilon)/(step_size*attack_iters*image.shape[0])
            common_perturbation = torch.clamp((common_perturbation+upd_size*(eta-common_perturbation)),-epsilon, epsilon)
            print('the max and min value is',common_perturbation.max(),common_perturbation.min())
            del eta
            # cv2.imwrite('adv_noise.jpg', adv_noise)
            del perturb_img,perturb_img_input
            torch.cuda.empty_cache()
    torch.save(common_perturbation.clone().detach(),args.noise_path)
    torch.save(momentum.clone().detach(),'momentum_common.pth')
    epsilon_val = 10.0 / 255.0
    save_img = (common_perturbation[0].detach().cpu() - (epsilon_val)) / (epsilon_val - (-epsilon_val)) # 归一化到0-1以便显示
    save_image(save_img, 'universal_noise_vis.png')



