import torch
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
import argparse
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.utils.data import DataLoader
from importlib import import_module
import matplotlib.pyplot as plt
from datasets.dataset_synapse import Synapse_dataset
from einops import repeat
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
import torch.fft
import random
import time
import cv2
from torchvision.utils import save_image
from datasets.refuge import REFUGE
from datasets.isic import ISIC2016
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str, default='testset/test_vol_h5/')
parser.add_argument('--num_classes', type=int, default=8)
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
parser.add_argument('--UMI_GR', type=int, default=0)
parser.add_argument('--sftm_hpyer', type=int, default=4)
parser.add_argument('--noise_path', type=str, default='common_perb/common_perturbation_meta_0.2sp_bound4_it5batchit20.05upsize_ls0.01mulmd1.pth')
parser.add_argument('--att_step_size', type=float, default=2.0)
parser.add_argument('--att_bound', type=float, default=10.0)
parser.add_argument('--att_numstep', type=int, default=10)
parser.add_argument('--dir_search', type=int, default=0)
parser.add_argument('--random_seed', type=int, default=42)


args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def input_diversity(x):
    resize_rate=0.9
    diversity_prob=0.1
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)

    if resize_rate < 1:
        img_size = img_resize
        img_resize = x.shape[-1]

    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(
        x, size=[rnd, rnd], mode="bilinear", align_corners=False
    )
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    padded = F.pad(
        rescaled,
        [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
        value=0,
    )

    return padded if torch.rand(1) < diversity_prob else x

def normalize_tensor(input_tensor):
    len_tensor=len(input_tensor)
    min_vals = input_tensor.view(len_tensor, -1).min(dim=1, keepdim=True)[0].view(len_tensor, 1, 1, 1, 1)
    max_vals = input_tensor.view(len_tensor, -1).max(dim=1, keepdim=True)[0].view(len_tensor, 1, 1, 1, 1)

    # Normalizing
    normalized_tensor = (input_tensor - min_vals) / (max_vals - min_vals)
    return normalized_tensor

def return_mean_value(input_tensor):
    input_tensor=torch.abs(input_tensor)
    list_mean=[]
    for i in range(len(input_tensor)):
        list_mean.append(input_tensor[i].mean()) 
    input_mean=torch.stack(list_mean)
    # print(input_mean.shape)
    input_mean=input_mean.view(-1, 1,1,1,1)
    return input_mean



def calculate_loss(features,targets,features_layer,targets_layer,GE):

    loss_org=criterion(features*0.9999, targets)#/torch.abs(targets).mean()
    loss_cosine=-F.cosine_similarity(features.view(1, -1), targets.view(1, -1))
    # loss_mse=criterion_mse(features*0.9999, targets)
    pre_layer= int(len(features_layer)*0.9)
    num_layers=6
    indices = random.sample(range(num_layers), num_layers-2)
    selected_features_layer = torch.stack([features_layer[i+pre_layer-num_layers] for i in indices])
    selected_targets_layer = torch.stack([targets_layer[i+pre_layer-num_layers] for i in indices])

    input_mean=return_mean_value(selected_targets_layer)
    feature_difference= (selected_features_layer*0.999-selected_targets_layer)/input_mean
    Gaussian_est =  torch.randn_like(feature_difference) * 1 + 1
    feature_difference=torch.mul(feature_difference,Gaussian_est)
    l1_losses= (torch.abs(torch.mean(feature_difference,dim=0))).mean()
    if GE == True:
        loss =1*l1_losses+1*loss_org
    else:
        loss =0*l1_losses+1*loss_org
    targets=targets.view(*targets.shape[:-2], -1)
    features = features.view(*features.shape[:-2], -1)
    features=F.softmax(features,dim=2)
    targets=F.softmax(targets,dim=2)
    info_loss =1*F.kl_div(features.log(), targets, reduction='batchmean')

    return loss+0.0*info_loss



def Minimize_domain_mean(features,mean,std_dev):
    target= torch.normal(mean, std_dev)
    l2=criterion_mse(features*0.9999, target)#/torch.abs(targets).mean()
    l1=criterion(features*0.9999, target)
    loss =-1*l1#-10*l2
    return loss 





def load_and_transform_images(folder_path):
    transformed_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                # 调整图片大小
                img_resized = img.resize((512, 512))
                # 将图片转换为numpy数组并归一化
                img_array = np.asarray(img_resized) / 255.0
                transformed_images.append(img_array)
    transformed_images=np.array(transformed_images)
    transformed_images=torch.tensor(transformed_images).permute(0,3,1,2).float()
    return transformed_images

def return_org_mean(model,data):
    feature_list=[]
    for i in range(len(data)):
        input=data[i]
        input=input.unsqueeze(0)
        input=input[0,0,:,:]
        input=input.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        with torch.no_grad():
            features,features_layer=model.image_encoder(input)
        feature_list.append(features)
        # if i ==1:
            # break
    
    # assert 1==0,print(features.shape)
    feature=torch.stack(feature_list)
    mean = feature.mean(dim=0)
    variance = feature.var(dim=0)
    std_dev = variance.sqrt()
    # mean=sum(feature_list)/len(feature_list)
    
    # assert 1==0,print(mean.shape)
    return mean,std_dev

def return_eta(momentum,step_size):
    upd_momentum=momentum*(step_size/torch.mean(abs(momentum)))
    momentum_binary = torch.where(momentum > 0, torch.tensor(1.0), torch.tensor(0.0))
    eta = torch.clamp(upd_momentum, -1*step_size, -0.25*step_size)*(1-momentum_binary) + momentum_binary*torch.clamp(upd_momentum, 0.25*step_size, 1*step_size)
    return eta

def eta_search(input_tensor,eta,step,loss,model,targets,targets_layer,epsilon):
    var_steps = [-0.5,-0.25,-0.125,0,0.1]

    with torch.no_grad():
        step_weight=step/(abs(eta).max())
        adv_input=input_tensor+eta
        upd_input_list=[]
        for var_step in var_steps:
            upd_eta = eta+var_step*step_weight*eta
            upd_eta = torch.clamp(upd_eta, -epsilon, epsilon)
            upd_input=input_tensor+upd_eta
            upd_input_list.append(upd_input)
        upd_input_tensors=torch.stack(upd_input_list,dim=0).squeeze(1)
        upd_input_list.append(adv_input)
        # print(upd_input_tensors.shape)
        loss_list=[]
        features,features_layer=model.image_encoder(upd_input_tensors)
        for i in range(len(features)):
        #    print(features_layer[i].shape)
            new_features_layer = []
            for tensor in features_layer:
                new_tensor = tensor[i, :].unsqueeze(0)  
                new_features_layer.append(new_tensor)
            scale=torch.mean(torch.abs(upd_input_list[i]-input_tensor))/torch.mean(torch.abs(upd_input_list[4]-input_tensor))
            loss_upd=calculate_loss(features[i].unsqueeze(0),targets,new_features_layer,targets_layer)#/(scale**0.5)
            loss_list.append(loss_upd)
        # loss_list.append(loss)
        index= loss_list.index(max(loss_list))
        upd_eta=upd_input_list[index]-input_tensor
        print(index,loss_list,step_weight)
        del upd_input_list,loss_list
    return upd_eta

def return_random_img(image):
    n = image.shape[0]
    random_int = random.randrange(0, n)
    img = image[random_int]
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().cuda()
    img = repeat(img, 'b c h w -> b (repeat c) h w', repeat=3)
    return img

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
dataset_name = args.dataset

tfs = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
])
tfs_msk = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

dataset_config = {
    'Synapse': {
        'Dataset': Synapse_dataset,
        'volume_path': args.volume_path,
        'list_dir': args.list_dir,
        'num_classes': args.num_classes,
        'z_spacing': 1,
        'is_3d': True, # 标记为 3D 数据
    },
    'REFUGE': {
        'Dataset': REFUGE,
        'data_path': args.volume_path, # 对应 args.volume_path 传入数据集根目录
        'mode': 'Test', # 注意大小写适配你的 refuge.py
        'transform': tfs,
        'transform_msk': tfs_msk,
        'is_3d': False, # 标记为 2D 数据
    },
    'ISIC': {
        'Dataset': ISIC2016,
        'data_path': args.volume_path,
        'mode': 'Test',
        'transform': tfs,
        'transform_msk': tfs_msk,
        'is_3d': False,
    }
}

db_config = dataset_config[dataset_name]

# 根据数据集类型初始化 DataLoader
if dataset_name == 'Synapse':
    db_test = db_config['Dataset'](base_dir=args.volume_path, list_dir=args.list_dir, split='test_vol')
    is_3d_data = True
elif dataset_name == 'REFUGE':
    db_test = db_config['Dataset'](args=args, data_path=db_config['data_path'], mode=db_config['mode'], transform=db_config['transform'], transform_msk=db_config['transform_msk'])
    is_3d_data = False
elif dataset_name == 'ISIC':
    db_test = db_config['Dataset'](args=args, data_path=db_config['data_path'], mode=db_config['mode'], transform=db_config['transform'], transform_msk=db_config['transform_msk'])
    is_3d_data = False
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4)


criterion_mse=torch.nn.MSELoss()
criterion=torch.nn.L1Loss()
criterion_cosine = torch.nn.CosineEmbeddingLoss()

common_perturbation = None
# 2. 仅当 UMI_GR 启用时才加载文件
if args.UMI_GR == 1:
    try:
        # 检查文件是否存在
        if os.path.exists(args.noise_path):
            common_perturbation = torch.load(args.noise_path)
            common_perturbation = common_perturbation.to(device)  # 确保加载到 GPU
        else:
            print(f"Warning: Noise path '{args.noise_path}' does not exist. UMI-GR disabled.")
            args.UMI_GR = 0  # 强制关闭以防后续报错
            meta_ini = False
    except Exception as e:
        print(f"Error loading noise file: {e}")
        args.UMI_GR = 0
        meta_ini = False

norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

for i_batch, sampled_batch in tqdm(enumerate(testloader)): #every-time one image from testloader

    if 'case_name' in sampled_batch:
        case_name = sampled_batch['case_name'][0]
    elif 'image_meta_dict' in sampled_batch:
        case_name = sampled_batch['image_meta_dict']['filename_or_obj'][0]
        # 去掉路径和后缀
        case_name = os.path.splitext(os.path.basename(case_name))[0]

    if args.test_batch is not None and case_name != args.test_batch:
        continue
    img_tensor = sampled_batch['image']

    if args.dataset == 'Synapse':
        # Synapse数据处理：[B, D, H, W] -> [B, 1, D, H, W] -> Squeeze -> Numpy
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(1)
        image_vol = img_tensor.squeeze(0).cpu().detach().numpy()
        # 调整维度为 [Depth, C, H, W] 格式方便遍历
        if len(image_vol.shape) == 2:
            image_vol = image_vol[np.newaxis, np.newaxis, :, :]
        elif len(image_vol.shape) == 3:
            image_vol = image_vol[:, np.newaxis, :, :]
    else:
        # 2D 数据 (REFUGE/ISIC): [B, C, H, W] -> [1, C, H, W] (Batch=1)
        image_vol = img_tensor.cpu().detach().numpy()  # [1, 3, H, W]

        # image_vol 现在是 numpy 数组，形状为 [Depth, C, H, W] (3D) 或 [1, C, H, W] (2D)
    image = image_vol

    for ind in range(image.shape[0]):
        input_image = image[ind, :, :, :]  # [C, H, W]
        input_image_torch = torch.from_numpy(input_image).unsqueeze(0).float().cuda()  # [1, C, H, W]

        # 通道对齐：如果单通道，转为 3 通道
        if input_image_torch.shape[1] == 1:
            input_image_torch = repeat(input_image_torch, 'b c h w -> b (repeat c) h w', repeat=3)

        # 此时 input_image_torch 必定是 [1, 3, H, W]
        momentum = torch.zeros_like(input_image_torch).detach()

        input_image=input_image_torch


        with torch.no_grad():
            infer_time= time.time()
            if args.dataset == 'Synapse':
                # Synapse 不需要归一化
                input_for_model = input_image_torch
            else:
                # ISIC/REFUGE: 临时归一化，传给模型，但不改变 input_image_torch 本身
                input_for_model = norm_transform(input_image_torch)

            targets, targets_layer = sam.image_encoder(input_for_model)
            end_infer_time= time.time()
            print('infer_time_is:',(end_infer_time-infer_time))
        
        max_val,min_val=(input_image_torch.max(),input_image_torch.min())
        step_size=(input_image_torch.max()-input_image_torch.min())*(args.att_step_size)/255
        epsilon=(input_image_torch.max()-input_image_torch.min())*(args.att_bound)/255
        # momentum =momentum
        momentum_old = momentum
        loss=0,0
        eta=torch.tensor(0).cuda()
        perturb_img = Variable(input_image_torch, requires_grad=True)
        loss_tensor_dict = {}
        if args.UMI_GR==1:
            meta_ini = True
            GE = True
        else:
            meta_ini = False
            GE = False   
        num_copies=8
        # if ind==5:
            # break
        for _ in range(args.att_numstep):
            start_time = time.time()
            grad = None
            if args.dataset == 'Synapse':
                perturb_img_input = perturb_img[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
            else:
                perturb_img_input = perturb_img
            if _ ==0 and args.UMI_GR==0:
                perturb_img_input=perturb_img_input+torch.randn_like(perturb_img_input)*0.0001
            avg_grad = torch.zeros_like(perturb_img).detach().cuda()
            perturb_img_input_batch, input_image_torch_batch = BSR(perturb_img_input, input_image_torch, num_copies)
            for index_bat in range(num_copies):
                sam.image_encoder.zero_grad()
                if args.dataset == 'Synapse':
                    # Synapse 不需要归一化
                    input_for_model_pre = perturb_img_input_batch[index_bat]
                else:
                    # ISIC/REFUGE: 临时归一化，传给模型，但不改变 input_image_torch 本身
                    input_for_model_pre = norm_transform(perturb_img_input_batch[index_bat])
                features,features_layer = sam.image_encoder(input_for_model_pre.unsqueeze(0))
                with torch.no_grad():
                    if args.dataset == 'Synapse':
                        # Synapse 不需要归一化
                        input_for_model_prepre = input_image_torch_batch[index_bat]
                    else:
                        # ISIC/REFUGE: 临时归一化，传给模型，但不改变 input_image_torch 本身
                        input_for_model_prepre = norm_transform(input_image_torch_batch[index_bat])
                    targets_c,targets_layer_c=sam.image_encoder(input_for_model_prepre.unsqueeze(0))
                loss=1*calculate_loss(features,targets_c,features_layer,targets_layer_c,GE)#+0*calculate_loss(features,targets,features_layer,targets_layer,GE)
                loss_l2=criterion_mse(features, targets).detach().cpu()
                # loss_r=calculate_loss(features_r,targets_rand,features_layer_r,targets_layer_rand)
                loss = loss#+0.3*loss_r
                loss_tensor_dict[loss.detach().cpu()] = perturb_img_input.detach().cpu()
                g1= torch.autograd.grad(loss, perturb_img,
                                        retain_graph=True, create_graph=False)[0]
                avg_grad += g1
                loss = loss.detach()
                del features,features_layer
                
            
            eta= (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True).detach()
            eta = 0.5 * momentum + eta
            momentum = eta
                # smoothed_eta=smoothed_eta*0.7
            del g1,avg_grad
            eta=step_size * eta.sign()
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)
            if _==0 and meta_ini:
                eta = 5*eta+0.5*common_perturbation
                # eta=eta_l
            if (_ ==10 and meta_ini):
                perturb_img = Variable(input_image_torch.data + 0.25*eta, requires_grad=True)
                eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)

            # eta = return_proj(eta.detach(),(common_perturbation/ torch.norm(common_perturbation, p=1)))
            end_time = time.time()
            backward_time = end_time - start_time
            backward_time = "{:.3f}".format(backward_time)
            start_time = time.time()
            # eta = eta_search(input_image_torch,eta,step_size,loss,sam,targets,targets_layer,epsilon)
            end_time = time.time()
            search_time = end_time - start_time
            search_time = "{:.3f}".format(search_time)
            print('the loss is:', loss.item(),'the b_time is:', backward_time)#,'the avg energy is:',avg_energy)
            

            eta = torch.clamp(eta, -epsilon, epsilon)
            perturb_img = Variable(input_image_torch.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, min_val, max_val), requires_grad=True)

        image_data_out = perturb_img[0].detach().cpu().numpy()  # [3, H, W]

        # 2. 删除变量释放显存
        del eta, perturb_img, perturb_img_input

        # 3. 后处理保存
        image_data_out = np.transpose(image_data_out, (1, 2, 0))  # [H, W, 3]
        image_data_out = np.clip(image_data_out, 0, 1)

        if is_3d_data:
            # Synapse: output/case_name/ind.png (保留切片结构)
            folder_path = os.path.join(output_path, case_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            image_path = os.path.join(folder_path, str(ind) + '.png')
            # 灰度图保存
            plt.imsave(image_path, image_data_out[:, :, 0], cmap='gray', vmin=0, vmax=1)
            print('Image saved:', image_path)
        else:
            if not os.path.exists(output_path):
                os.makedirs(output_path)

                # 处理文件名：只保留文件名部分，去除原有扩展名（如.jpg），改为.png
            file_name = os.path.basename(case_name)  # 防止case_name包含路径
            file_base = os.path.splitext(file_name)[0]  # 去除后缀
            image_path = os.path.join(output_path, file_base + '.png')

            # 保存为 PNG
            # image_data_out 已经是 [H, W, C] 格式且数值在 [0, 1] 之间，可以直接用 plt.imsave
            # 如果是单通道灰度图，plt.imsave 会自动应用 colormap，如果需要纯灰度，可以加 cmap='gray'
            # 这里的 image_data_out 通常是 3 通道的（因为前面 repeat 过了），所以直接保存即可
            plt.imsave(image_path, image_data_out)

            print('Saved png:', image_path)

        torch.cuda.empty_cache()
