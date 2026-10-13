# 模型 Clean Dice 验证命令

> 权重位置：`/root/data2/fyc/fyc/GRAT-main/checkpoints/best_weights/`

## 1. SAMed Synapse (3 models)

```bash
cd /root/data2/fyc/fyc/GRAT-main

CUDA_VISIBLE_DEVICES=1 python test_samed.py \
  --dataset Synapse --vit_name vit_b --num_classes 8 \
  --volume_path data/Synapse_preprocess/test_vol_h5 \
  --list_dir SAMed_h/lists/lists_Synapse \
  --lora_ckpt checkpoints/best_weights/SAMed_Synapse_vit_b.pth \
  --ckpt checkpoints/sam_vit_b_01ec64.pth \
  --img_size 512 --module sam_lora_image_encoder --consider_lora --use_clean_image

CUDA_VISIBLE_DEVICES=1 python test_samed.py \
  --dataset Synapse --vit_name vit_l --num_classes 8 \
  --volume_path data/Synapse_preprocess/test_vol_h5 \
  --list_dir SAMed_h/lists/lists_Synapse \
  --lora_ckpt checkpoints/best_weights/SAMed_Synapse_vit_l.pth \
  --ckpt checkpoints/sam_vit_l_0b3195.pth \
  --img_size 512 --module sam_lora_image_encoder --consider_lora --use_clean_image

python test_samed.py \
  --dataset Synapse --vit_name vit_h --num_classes 8 \
  --volume_path data/Synapse_preprocess/test_vol_h5 \
  --list_dir SAMed_h/lists/lists_Synapse \
  --lora_ckpt checkpoints/best_weights/SAMed_Synapse_vit_h.pth \
  --ckpt checkpoints/sam_vit_h_4b8939.pth \
  --img_size 512 --module sam_lora_image_encoder --consider_lora --use_clean_image
```

## 2. SAMed ISIC (3 models)

```bash
CUDA_VISIBLE_DEVICES=1 python test_samed.py \
  --dataset ISIC --vit_name vit_b --num_classes 1 \
  --volume_path data/ISIC \
  --list_dir SAMed_h/lists/lists_Synapse \
  --lora_ckpt checkpoints/best_weights/SAMed_ISIC_vit_b.pth \
  --ckpt checkpoints/sam_vit_b_01ec64.pth \
  --img_size 512 --module sam_lora_image_encoder --consider_lora --use_clean_image

CUDA_VISIBLE_DEVICES=1 python test_samed.py \
  --dataset ISIC --vit_name vit_l --num_classes 1 \
  --volume_path data/ISIC \
  --list_dir SAMed_h/lists/lists_Synapse \
  --lora_ckpt checkpoints/best_weights/SAMed_ISIC_vit_l.pth \
  --ckpt checkpoints/sam_vit_l_0b3195.pth \
  --img_size 512 --module sam_lora_image_encoder --consider_lora --use_clean_image

CUDA_VISIBLE_DEVICES=1 python test_samed.py \
  --dataset ISIC --vit_name vit_h --num_classes 1 \
  --volume_path data/ISIC \
  --list_dir SAMed_h/lists/lists_Synapse \
  --lora_ckpt checkpoints/best_weights/SAMed_ISIC_vit_h.pth \
  --ckpt checkpoints/sam_vit_h_4b8939.pth \
  --img_size 512 --module sam_lora_image_encoder --consider_lora --use_clean_image
```

## 3. SAMed REFUGE (3 models)

```bash
CUDA_VISIBLE_DEVICES=2 python test_samed.py \
  --dataset REFUGE --vit_name vit_b --num_classes 2 \
  --volume_path data/REFUGE-Multirater \
  --list_dir SAMed_h/lists/lists_Synapse \
  --lora_ckpt checkpoints/best_weights/SAMed_REFUGE_vit_b.pth \
  --ckpt checkpoints/sam_vit_b_01ec64.pth \
  --img_size 512 --module sam_lora_image_encoder --consider_lora --use_clean_image

CUDA_VISIBLE_DEVICES=2 python test_samed.py \
  --dataset REFUGE --vit_name vit_l --num_classes 2 \
  --volume_path data/REFUGE-Multirater \
  --list_dir SAMed_h/lists/lists_Synapse \
  --lora_ckpt checkpoints/best_weights/SAMed_REFUGE_vit_l.pth \
  --ckpt checkpoints/sam_vit_l_0b3195.pth \
  --img_size 512 --module sam_lora_image_encoder --consider_lora --use_clean_image

CUDA_VISIBLE_DEVICES=2 python test_samed.py \
  --dataset REFUGE --vit_name vit_h --num_classes 2 \
  --volume_path data/REFUGE-Multirater \
  --list_dir SAMed_h/lists/lists_Synapse \
  --lora_ckpt checkpoints/best_weights/SAMed_REFUGE_vit_h.pth \
  --ckpt checkpoints/sam_vit_h_4b8939.pth \
  --img_size 512 --module sam_lora_image_encoder --consider_lora --use_clean_image
```

## 4. MSA (9 models)

```bash
cd /root/data2/fyc/fyc/GRAT-main/Medical-SAM-Adapter-main

# MSA Synapse
CUDA_VISIBLE_DEVICES=2 python val.py -dataset Synapse -net sam -mod sam_adpt -encoder vit_b \
  -image_size 512 -out_size 512 -data_path ../data/Synapse_preprocess/test_vol_h5 \
  -list_dir ../data/Synapse_preprocess/lists \
  -weights ../checkpoints/best_weights/MSA_Synapse_vit_b.pth \
  -sam_ckpt ../checkpoints/sam_vit_b_01ec64.pth -gpu_device 0

CUDA_VISIBLE_DEVICES=2 python val.py -dataset Synapse -net sam -mod sam_adpt -encoder vit_l \
  -image_size 512 -out_size 512 -data_path ../data/Synapse_preprocess/test_vol_h5 \
  -list_dir ../data/Synapse_preprocess/lists \
  -weights ../checkpoints/best_weights/MSA_Synapse_vit_l.pth \
  -sam_ckpt ../checkpoints/sam_vit_l_0b3195.pth -gpu_device 0

CUDA_VISIBLE_DEVICES=2 python val.py -dataset Synapse -net sam -mod sam_adpt -encoder vit_h \
  -image_size 512 -out_size 512 -data_path ../data/Synapse_preprocess/test_vol_h5 \
  -list_dir ../data/Synapse_preprocess/lists \
  -weights ../checkpoints/best_weights/MSA_Synapse_vit_h.pth \
  -sam_ckpt ../checkpoints/sam_vit_h_4b8939.pth -gpu_device 0

# MSA ISIC
CUDA_VISIBLE_DEVICES=2 python val.py -dataset isic -net sam -mod sam_adpt -encoder vit_b \
  -image_size 512 -out_size 512 -data_path ../data/ISIC \
  -weights ../checkpoints/best_weights/MSA_ISIC_vit_b.pth \
  -sam_ckpt ../checkpoints/sam_vit_b_01ec64.pth -gpu_device 0

CUDA_VISIBLE_DEVICES=2 python val.py -dataset isic -net sam -mod sam_adpt -encoder vit_l \
  -image_size 512 -out_size 512 -data_path ../data/ISIC \
  -weights ../checkpoints/best_weights/MSA_ISIC_vit_l.pth \
  -sam_ckpt ../checkpoints/sam_vit_l_0b3195.pth -gpu_device 0

CUDA_VISIBLE_DEVICES=2 python val.py -dataset isic -net sam -mod sam_adpt -encoder vit_h \
  -image_size 512 -out_size 512 -data_path ../data/ISIC \
  -weights ../checkpoints/best_weights/MSA_ISIC_vit_h.pth \
  -sam_ckpt ../checkpoints/sam_vit_h_4b8939.pth -gpu_device 0

# MSA REFUGE
CUDA_VISIBLE_DEVICES=2 python val.py -dataset REFUGE -net sam -mod sam_adpt -encoder vit_b \
  -image_size 512 -out_size 512 -multimask_output 2 \
  -data_path ../data/REFUGE-Multirater \
  -weights ../checkpoints/best_weights/MSA_REFUGE_vit_b.pth \
  -sam_ckpt ../checkpoints/sam_vit_b_01ec64.pth -gpu_device 0

CUDA_VISIBLE_DEVICES=2 python val.py -dataset REFUGE -net sam -mod sam_adpt -encoder vit_l \
  -image_size 512 -out_size 512 -multimask_output 2 \
  -data_path ../data/REFUGE-Multirater \
  -weights ../checkpoints/best_weights/MSA_REFUGE_vit_l.pth \
  -sam_ckpt ../checkpoints/sam_vit_l_0b3195.pth -gpu_device 0

CUDA_VISIBLE_DEVICES=2 python val.py -dataset REFUGE -net sam -mod sam_adpt -encoder vit_h \
  -image_size 512 -out_size 512 -multimask_output 2 \
  -data_path ../data/REFUGE-Multirater \
  -weights ../checkpoints/best_weights/MSA_REFUGE_vit_h.pth \
  -sam_ckpt ../checkpoints/sam_vit_h_4b8939.pth -gpu_device 0
```

## 5. MSA 训练命令

> 工作目录：`/root/data2/fyc/fyc/GRAT-main`

```bash
# MSA Synapse vit_b (bs=12)
python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset Synapse \
  -encoder vit_b -sam_ckpt checkpoints/sam_vit_b_01ec64.pth \
  -data_path data/Synapse_preprocess/train_npz -list_dir data/Synapse_preprocess/lists \
  -image_size 512 -out_size 512 -b 12 -gpu_device 0 \
  -out_root OUT_ROOT/MedicalSAMAdapter -exp_name unused_when_out_root

# MSA Synapse vit_l (bs=12)
python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset Synapse \
  -encoder vit_l -sam_ckpt checkpoints/sam_vit_l_0b3195.pth \
  -data_path data/Synapse_preprocess/train_npz -list_dir data/Synapse_preprocess/lists \
  -image_size 512 -out_size 512 -b 12 -gpu_device 0 \
  -out_root OUT_ROOT/MedicalSAMAdapter -exp_name unused_when_out_root

# MSA Synapse vit_h (bs=6)
python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset Synapse \
  -encoder vit_h -sam_ckpt checkpoints/sam_vit_h_4b8939.pth \
  -data_path data/Synapse_preprocess/train_npz -list_dir data/Synapse_preprocess/lists \
  -image_size 512 -out_size 512 -b 6 -gpu_device 0 \
  -out_root OUT_ROOT/MedicalSAMAdapter -exp_name unused_when_out_root

# MSA ISIC vit_b (bs=24)
python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset isic \
  -encoder vit_b -sam_ckpt checkpoints/sam_vit_b_01ec64.pth \
  -data_path data/ISIC -image_size 512 -out_size 512 -b 24 -gpu_device 0 \
  -out_root OUT_ROOT/MedicalSAMAdapter -exp_name unused_when_out_root

# MSA ISIC vit_l (bs=16)
python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset isic \
  -encoder vit_l -sam_ckpt checkpoints/sam_vit_l_0b3195.pth \
  -data_path data/ISIC -image_size 512 -out_size 512 -b 16 -gpu_device 0 \
  -out_root OUT_ROOT/MedicalSAMAdapter -exp_name unused_when_out_root

# MSA ISIC vit_h (bs=8)
python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset isic \
  -encoder vit_h -sam_ckpt checkpoints/sam_vit_h_4b8939.pth \
  -data_path data/ISIC -image_size 512 -out_size 512 -b 8 -gpu_device 0 \
  -out_root OUT_ROOT/MedicalSAMAdapter -exp_name unused_when_out_root

# MSA REFUGE vit_b (bs=24, multimask_output=2)
python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset REFUGE \
  -encoder vit_b -sam_ckpt checkpoints/sam_vit_b_01ec64.pth \
  -data_path data/REFUGE-Multirater -image_size 512 -out_size 512 -b 24 -gpu_device 0 \
  -out_root OUT_ROOT/MedicalSAMAdapter -exp_name unused_when_out_root -multimask_output 2

# MSA REFUGE vit_l (bs=16, multimask_output=2)
python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset REFUGE \
  -encoder vit_l -sam_ckpt checkpoints/sam_vit_l_0b3195.pth \
  -data_path data/REFUGE-Multirater -image_size 512 -out_size 512 -b 16 -gpu_device 0 \
  -out_root OUT_ROOT/MedicalSAMAdapter -exp_name unused_when_out_root -multimask_output 2

# MSA REFUGE vit_h (bs=8, multimask_output=2)
python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset REFUGE \
  -encoder vit_h -sam_ckpt checkpoints/sam_vit_h_4b8939.pth \
  -data_path data/REFUGE-Multirater -image_size 512 -out_size 512 -b 8 -gpu_device 0 \
  -out_root OUT_ROOT/MedicalSAMAdapter -exp_name unused_when_out_root -multimask_output 2
```

## 6. 预期结果

| | SAMed Synapse | SAMed ISIC | SAMed REFUGE | MSA Synapse | MSA ISIC | MSA REFUGE |
|---|------|------|------|------|------|------|
| vit_b | 0.80 | 0.93 | 0.73 | 0.89 | 0.91 | 0.85 |
| vit_l | 0.81 | 0.93 | 0.86 | 0.91 | 0.93 | 0.87 |
| vit_h | 0.83 | 0.94 | 0.83 | 0.92 | 0.92 | 0.88 |
