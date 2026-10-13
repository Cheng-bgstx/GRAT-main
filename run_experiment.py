import os
import subprocess

# --- 1. 参数配置 ---
# 请根据您的实际路径和需求修改这些变量
VOLUME_PATH = "testset/test_vol_h5/"
LIST_DIR = "./lists/lists_Synapse/"
ADV_IMAGES_OUTPUT_PATH = "testset/adv_images_online_black_UMI_loss_freq50/" # 为本次实验新建一个对抗样本输出目录
EVALUATION_OUTPUT_PATH = "./evaluation_results_online_black_UMI_loss_freq50/" # 为本次评估结果新建一个输出目录
LORA_CKPT = "checkpoints/epoch_159.pth"
SAM_CKPT = "checkpoints/sam_vit_b_01ec64.pth"

# 在线更新策略参数
ONLINE_METHOD = "loss_weighted_average"
ONLINE_FREQ = 50


# --- 2. 确保输出目录存在 ---
os.makedirs(ADV_IMAGES_OUTPUT_PATH, exist_ok=True)
os.makedirs(EVALUATION_OUTPUT_PATH, exist_ok=True)


# --- 3. 构建并执行命令 ---

# === 命令1：运行 online_Sam_attack.py 生成对抗样本 ===
attack_command = [
    "python", "online_Sam_attack_newUMI.py",
    "--volume_path", VOLUME_PATH,
    "--list_dir", LIST_DIR,
    "--out_path", ADV_IMAGES_OUTPUT_PATH,
    #"--lora_ckpt", LORA_CKPT,
    "--ckpt", SAM_CKPT,
    "--UMI_GR", "1",
    "--noise_path", "testset/outcome/sa1b/universal_pert_sa1b_9900.pth",
    "--online_update_method", ONLINE_METHOD,
    "--online_update_freq", str(ONLINE_FREQ)
]

print("--- 阶段1：开始生成对抗样本 ---")
print(f"执行命令: {' '.join(attack_command)}")
subprocess.run(attack_command, check=True) # 您可以取消这行注释来实际运行
print("--- 阶段1：生成对抗样本完成 ---\n")


# === 命令2：运行 test_adv_all_new.py 进行评估 ===
# 修正了参数以匹配 test_adv_all_new.py 的要求
test_command = [
    "python", "test_adv_all_new.py",
    "--volume_path", VOLUME_PATH,                 # 新增：评估需要知道原始h5数据的位置
    "--list_dir", LIST_DIR,                       # 新增：评估需要列表文件来加载数据集
    "--out_path", ADV_IMAGES_OUTPUT_PATH,         # 修正：评估脚本用 --out_path 来读取对抗图片
    "--output_dir", EVALUATION_OUTPUT_PATH,       # 评估结果的输出目录
    "--lora_ckpt", LORA_CKPT,
    "--ckpt", SAM_CKPT,
    "--online_update_method", ONLINE_METHOD,      # 新增：传递给评估脚本用于日志记录
    "--online_update_freq", str(ONLINE_FREQ)      # 新增：传递给评估脚本用于日志记录
]

print("--- 阶段2：开始评估对抗样本 ---")
print(f"执行命令: {' '.join(test_command)}")
subprocess.run(test_command, check=True) # 您可以取消这行注释来实际运行
print("--- 阶段2：评估完成 ---")
print(f"\n评估结果（CSV和图表）已保存至: {EVALUATION_OUTPUT_PATH}")