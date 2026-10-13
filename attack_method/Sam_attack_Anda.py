import torch
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
import argparse
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
# from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry
import matplotlib.pyplot as plt
from datasets.dataset_synapse import Synapse_dataset
from einops import repeat
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
import torch.fft
import random
import torchvision.transforms as transforms
import time
import cv2
from one_sample_train import one_sample_train
import torch.distributions as dist
from torchvision.utils import save_image
from anda_ct import attack
from datasets.refuge import REFUGE
from datasets.isic import ISIC2016
import torchvision.transforms as transforms
from torch.utils.checkpoint import checkpoint

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

parser.add_argument('--batch_size', type=int, default=10, help='mini-batch size (default: 1)')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 0)')
parser.add_argument('--max_epsilon', default=16.0, type=float, help='max magnitude of adversarial perturbations')
parser.add_argument('--num_iter', default=10, type=int, help='max iteration')
parser.add_argument('--n_ens', default=2, type=int, help='augmentation number')
parser.add_argument('--aug_max', default=0.3, type=float, help='augmentation degree of the attack')
parser.add_argument('--input_csv', default='./datasets/dev.csv', type=str, help='csv info of clean examples')
parser.add_argument('--input_dir', default='./datasets/images/', type=str, help='directory of clean examples')
# parser.add_argument('--output_dir', default='', type=str, help='directory of crafted adversarial examples')
parser.add_argument('--victim_model', default='vgg19', type=str, help='directory for test')
parser.add_argument('--device', default='2', type=str, help='gpu device')

parser.add_argument('--kernel_size', default=7, type=int, help='kernel_size')
parser.add_argument('--sigma', default=3, type=int, help='sigma of kernel')
parser.add_argument('--kernel_name', default='gaussian', type=str)
parser.add_argument('--m', default=5, type=int, help='the number of scale copies')
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
    Gaussian_est =  torch.randn_like(feature_difference) * 0.5 + 1
    feature_difference=torch.mul(feature_difference,Gaussian_est)
    l1_losses= (torch.abs(torch.mean(feature_difference,dim=0))).mean()
    if GE:
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
        # input=input[0,0,:,:]
        # input=input.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        with torch.no_grad():
            features,features_layer=model.image_encoder(input)
        feature_list.append(features.cpu())
        del features,features_layer
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
    # var_steps= [-0.5]
    # input_tensor=input_tensor.detach()
    # eta=eta.detach()
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




def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def make_checkpoint_wrapper(block):
    original_forward = block.forward
    def checkpointed_forward(x):
        return checkpoint(original_forward, x, use_reentrant=False)
    return checkpointed_forward


seed = args.random_seed
set_random_seed(seed)

sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])

sam=sam.to(device)
for block in sam.image_encoder.blocks:
    block.forward = make_checkpoint_wrapper(block)
output_path=args.out_path
scaler = GradScaler()
criterion_mse = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()
dataset_name = args.dataset

# 定义 Transform
tfs = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
])
tfs_msk = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

# --- [修改点 1]：Dataset Config 参照 Universal 结构，但使用测试集 Split ---
dataset_config = {
    'Synapse': {
        'Dataset': Synapse_dataset,
        'volume_path': args.volume_path,
        'list_dir': args.list_dir,
        'split': 'test_vol',  # 对应 Universal 的 'train'，这里改为 'test_vol'
        'transform': None
    },
    'REFUGE': {
        'Dataset': REFUGE,
        'volume_path': args.volume_path,
        'split': 'Test',      # 对应 Universal 的 'Training'，这里改为 'Test'
        'transform': tfs,
        'transform_msk': tfs_msk
    },
    'ISIC': {
        'Dataset': ISIC2016,
        'volume_path': args.volume_path,
        'split': 'Test',      # 对应 Universal 的 'Training'，这里改为 'Test'
        'transform': tfs,
        'transform_msk': tfs_msk
    }
}

db_config = dataset_config[dataset_name]

# 初始化 DataLoader (逻辑参照 Universal，但参数适配 Test)
if dataset_name == 'Synapse':
    db_test = db_config['Dataset'](base_dir=db_config['volume_path'], list_dir=db_config['list_dir'],
                                   split=db_config['split'])
    is_3d_data = True
elif dataset_name == 'REFUGE':
    db_test = db_config['Dataset'](args=args, data_path=db_config['volume_path'],
                                   mode=db_config['split'], # REFUGE类接收 mode 参数
                                   transform=db_config['transform'], transform_msk=db_config['transform_msk'])
    is_3d_data = False
elif dataset_name == 'ISIC':
    db_test = db_config['Dataset'](args=args, data_path=db_config['volume_path'],
                                   mode=db_config['split'], # ISIC类接收 mode 参数
                                   transform=db_config['transform'], transform_msk=db_config['transform_msk'])
    is_3d_data = False

testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4)

# --- UMI_GR 噪声加载逻辑 (放在循环外) ---
if args.UMI_GR == 1:
    try:
        if os.path.exists(args.noise_path):
            common_perturbation = torch.load(args.noise_path)
            common_perturbation = common_perturbation.to(device)
            print(f"Loaded common perturbation from {args.noise_path}")
        else:
            print(f"Warning: Noise path '{args.noise_path}' does not exist. UMI-GR disabled.")
            args.UMI_GR = 0
            meta_ini = False
    except Exception as e:
        print(f"Error loading noise file: {e}")
        args.UMI_GR = 0
        meta_ini = False


if args.dir_search==1:
    args.att_numstep=args.att_numstep+5

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

        # --- [修改点 2]：数据维度处理逻辑 (完全参照 Universal 脚本) ---
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

        #transform_matrix,mean_std=pca_transform_matrix(noised_targets,1,11)


        # assert 1==0, print(common_perturbation.max(),common_perturbation.min())
        max_val,min_val=(input_image_torch.max(),input_image_torch.min())
        step_size=(input_image_torch.max()-input_image_torch.min())*(args.att_step_size)/255
        epsilon=(input_image_torch.max()-input_image_torch.min())*(args.att_bound)/255
        # if ind==0:
            # attacker = L2T(model_name = sam,epsilon=epsilon,alpha=step_size, targeted = False, epoch = 10, num_scale = 3)
        # assert 1==0, print(max_val,min_val) 

        momentum_old = momentum

        loss,smoothed_loss=0,0
        smoothed_direction=torch.tensor(0).cuda()
        eta,smoothed_eta=torch.tensor(0).cuda(),torch.tensor(0).cuda()
        bias=0
        eta_old=0
        # input_image_torch = input_image_torch+ common_perturbation
        perturb_img = Variable(input_image_torch, requires_grad=True)
        loss_tensor_dict = {}
        if args.UMI_GR==1:
            meta_ini = True
            GE = True

        else:
            meta_ini = False
            GE = False    

        with autocast():
            perturb_img = attack(args,perturb_img, targets, sam, 10, epsilon, step_size, minibatch=False, sample=False).detach()
            perturb_img = Variable(perturb_img.data, requires_grad=True)
        
        for _ in range(args.att_numstep):
            start_time = time.time()
            grad = None
            if args.dataset == 'Synapse':
                perturb_img_input = perturb_img[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
            else:
                perturb_img_input = perturb_img
            # opt1 = torch.optim.SGD([perturb_img_rot], lr=1e-3)
            # opt1.zero_grad()1
            sam.image_encoder.zero_grad()

            if _ ==0 and args.UMI_GR==0:
                perturb_img_input=perturb_img_input+torch.randn_like(perturb_img_input)*0.0001

            if args.dataset == 'Synapse':
                # Synapse 不需要归一化
                perturb_img_input_pre = perturb_img_input
            else:
                # ISIC/REFUGE: 临时归一化，传给模型，但不改变 input_image_torch 本身
                perturb_img_input_pre = norm_transform(perturb_img_input)

            features,features_layer = sam.image_encoder(perturb_img_input_pre)
                # features_r,features_layer_r = sam.image_encoder(perturb_img_input_t)

            if _<0 and meta_ini:

                grad=perturb_img.grad.data

                # momentum = 0.5 * momentum.detach() +grad / torch.norm(grad, p=1)
                # momentum=momentum.detach()
                eta = 1*step_size * grad.sign()
                # with torch.no_grad():
                    # targets,targets_layer=sam.image_encoder(perturb_img_input)
            else:
                loss=calculate_loss(features,targets,features_layer,targets_layer,GE)
                # loss_r=calculate_loss(features_r,targets_rand,features_layer_r,targets_layer_rand)
                loss = loss#+0.3*loss_r
                loss_l2=criterion_mse(features, targets).detach().cpu()
                loss_tensor_dict[loss.detach().cpu()] = perturb_img_input.detach().cpu()
                del features,features_layer
                grad= torch.autograd.grad(loss, perturb_img,
                retain_graph=True, create_graph=False)[0]


                grad=grad / torch.norm(grad, p=1)

                # grad=return_proj(grad.detach(),(common_perturbation).detach())
                mean_abs = torch.mean(torch.abs(grad))
                mask_grad = torch.abs(grad)>mean_abs
                # grad=grad*mask_grad
                momentum = 0.5 * momentum.detach()+1 *grad# / torch.norm(grad, p=1)
                momentum=momentum.detach()

                # eta = 1*step_size * momentum.sign()
                eta = return_eta(momentum,step_size)
                # eta = 1*step_size * (momentum+smoothed_direction).sign()
                # smoothed_direction=smoothed_direction*0.7
            eta=eta.detach()
                # smoothed_eta=smoothed_eta*0.7
            del grad

            eta=eta
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)
            if _==0 and meta_ini:
                eta = 2.5*eta+1*common_perturbation
                torch.save(eta,'noise_pattern/UMI.pt')
                # eta = 0.25*eta
                # eta=eta_l
            if (_ ==5 and args.dir_search==1):
                upd_eta=eta-eta_old
                eta_old=eta_old+upd_eta*0.1
                eta = eta_old.clone()

            end_time = time.time()
            backward_time = end_time - start_time
            backward_time = "{:.3f}".format(backward_time)
            start_time = time.time()
            # eta = eta_search(input_image_torch,eta,step_size,loss,sam,targets,targets_layer,epsilon)
            end_time = time.time()
            search_time = end_time - start_time
            search_time = "{:.3f}".format(search_time)
            print('the loss is:', loss.item(),'the smoothed loss is:', smoothed_loss,'the b_time is:', backward_time)#,'the avg energy is:',avg_energy)

            # if (_ ==10):
            #     upd_eta=eta-eta_old
            #     eta_old=eta_old+upd_eta*0.05
            #     torch.save(eta,'noise_pattern/UMI_10.pt')
            #     eta = eta_old
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