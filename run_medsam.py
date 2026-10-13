import os
import subprocess

# --- 1. 参数配置 ---
# 请根据您的实际路径和需求修改这些变量

# --- 路径配置 ---
# 原始数据路径 (包含 .npz 文件的文件夹)
ORIGINAL_DATA_PATH = "MedSAM/data/npz/CT_Abd_Full/"
# MedSAM预训练模型路径
MEDSAM_CHECKPOINT = "checkpoints/medsam_vit_b.pth"
# 所有输出的根目录
PIPELINE_OUTPUT_DIR = "pipeline_output_py"


# --- 脚本参数配置 ---
# Attack 脚本的特定参数
ATTACK_BOUND = 10.0
ATTACK_STEPS = 10
ATTACK_STEP_SIZE = 2.0

# Inference 脚本的特定参数
PROMPT_TYPE = "box"  # 可选项: "box" 或 "point"

# Test 脚本的特定参数 (控制'Dice per Slice'图上点的密度)
TEST_PLOT_FREQ = 5

# 是否在 Attack 和 Inference 阶段保存可视化结果
SAVE_VISUALIZATIONS = True


# --- 2. 自动生成并确保输出目录存在 ---
# 为每个阶段创建独立的子目录
ADV_SAMPLE_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "1_adversarial_samples")
INFERENCE_SEG_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "2_inference_segmentations")
TEST_RESULTS_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "3_test_results")

print("--- 初始化：创建输出目录 ---")
os.makedirs(ADV_SAMPLE_DIR, exist_ok=True)
os.makedirs(INFERENCE_SEG_DIR, exist_ok=True)
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
print(f"对抗样本将保存至: {ADV_SAMPLE_DIR}")
print(f"分割结果将保存至: {INFERENCE_SEG_DIR}")
print(f"评估报告将保存至: {TEST_RESULTS_DIR}\n")


# --- 3. 构建并执行命令 ---

# === 阶段 1：运行 online_Sam_attack_medsam.py 生成对抗样本 ===
print("--- 阶段 1：开始生成对抗样本 ---")
attack_command = [
    "python", "online_Sam_attack_medsam.py",
    "--volume_path", ORIGINAL_DATA_PATH,
    "--output_dir", ADV_SAMPLE_DIR,
    "--ckpt", MEDSAM_CHECKPOINT,
    "--att_bound", str(ATTACK_BOUND),
    "--att_numstep", str(ATTACK_STEPS),
    "--att_step_size", str(ATTACK_STEP_SIZE),
]
if SAVE_VISUALIZATIONS:
    attack_command.append("--save_vis")

print(f"执行命令: {' '.join(attack_command)}")
# 使用 check=True, 如果脚本执行失败 (返回非零退出码), 则会抛出异常
subprocess.run(attack_command, check=True)
print("--- 阶段 1：生成对抗样本完成 ---\n")


# === 阶段 2：运行 MedSAM_Inference.py 对抗样本进行分割 ===
print("--- 阶段 2：开始在对抗样本上运行分割推理 ---")
inference_command = [
    "python", "MedSAM_Inference.py",
    "--data_path", ADV_SAMPLE_DIR,
    "--seg_path", INFERENCE_SEG_DIR,
    "--data_type", "npz",
    "--checkpoint", MEDSAM_CHECKPOINT,
    "--prompt_type", PROMPT_TYPE,
]
if SAVE_VISUALIZATIONS:
    inference_command.append("--save_vis")

print(f"执行命令: {' '.join(inference_command)}")
subprocess.run(inference_command, check=True)
print("--- 阶段 2：分割推理完成 ---\n")


# === 阶段 3：运行 test_medsam.py 评估分割结果 ===
print("--- 阶段 3：开始评估分割结果 ---")
test_command = [
    "python", "test_medsam.py",
    "--pred_path", INFERENCE_SEG_DIR,
    "--output_path", TEST_RESULTS_DIR,
    "--data_type", "npz",
    "--online_update_freq", str(TEST_PLOT_FREQ), # 此参数在test脚本中用于控制绘图频率
]

print(f"执行命令: {' '.join(test_command)}")
subprocess.run(test_command, check=True)
print("--- 阶段 3：评估完成 ---\n")

print("======================================================")
print("========== PIPELINE COMPLETED SUCCESSFULLY ==========")
print("======================================================")
print(f"所有结果已保存在主目录: {PIPELINE_OUTPUT_DIR}")