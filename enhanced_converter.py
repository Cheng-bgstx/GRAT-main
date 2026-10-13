import os
import h5py
import numpy as np
from PIL import Image


def convert_3d_h5_to_2d_png_slices(h5_folder, png_folder, h5_dataset_name='image'):
    """
    将指定文件夹内的所有三维 .h5 文件批量转换为独立的二维 .png 切片文件。

    参数:
    h5_folder (str): 存放 .h5 文件的源文件夹路径。
    png_folder (str): 存放转换后的 .png 文件的目标文件夹路径。
    h5_dataset_name (str): .h5 文件中包含三维影像数据的数据集（dataset）名称。
    """
    # 检查并创建目标文件夹
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)
        print(f"创建文件夹: {png_folder}")

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(h5_folder):
        if filename.endswith(".h5"):
            h5_path = os.path.join(h5_folder, filename)

            try:
                with h5py.File(h5_path, 'r') as f:
                    # 检查数据集是否存在
                    if h5_dataset_name not in f:
                        print(f"警告: 在 '{filename}' 中未找到数据集 '{h5_dataset_name}'。跳过此文件。")
                        print(f"'{filename}' 中可用的数据集有: {list(f.keys())}")
                        continue

                    # 读取整个三维体积数据
                    volume_data = f[h5_dataset_name][()]

                # 检查数据是否至少是三维的
                if volume_data.ndim < 3:
                    print(f"警告: '{filename}' 中的数据不是三维的（维度为 {volume_data.shape}）。无法按切片处理，已跳过。")
                    continue

                # 获取文件名（不含扩展名），用于创建PNG文件名
                base_filename = os.path.splitext(filename)[0]
                num_slices = volume_data.shape[0]
                print(f"处理文件: {filename}，包含 {num_slices} 个切片...")

                # --- 核心改动：遍历三维体积的每一个切片 ---
                for i in range(num_slices):
                    # 提取单个二维切片
                    # 假设数据格式是 (切片数, 高度, 宽度)
                    slice_data = volume_data[i, :, :]

                    # --- 数据处理与标准化 (与之前相同) ---
                    # 确保数据类型为浮点数以便计算
                    slice_data = slice_data.astype(np.float32)
                    min_val = np.min(slice_data)
                    max_val = np.max(slice_data)

                    # 标准化到 0-255
                    if max_val - min_val > 0:
                        slice_data = 255 * (slice_data - min_val) / (max_val - min_val)
                    else:
                        slice_data = np.zeros_like(slice_data)

                    # 转换为8位无符号整数
                    slice_data = slice_data.astype(np.uint8)

                    # 使用Pillow创建图像对象
                    img = Image.fromarray(slice_data, 'L')  # 'L' 表示灰度图像

                    # --- 文件名管理 ---
                    # 构建包含切片索引的输出文件名，例如 "case_0001_slice_001.png"
                    # zfill(3) 会将数字补零，例如 1 -> 001, 10 -> 010
                    slice_filename = f"{base_filename}_slice_{str(i).zfill(3)}.png"
                    png_path = os.path.join(png_folder, slice_filename)

                    # 保存图像
                    img.save(png_path)

                print(f"成功将 {filename} 转换为 {num_slices} 个PNG切片。")

            except Exception as e:
                print(f"处理文件 {h5_path} 时发生错误: {e}")


# --- 主程序入口 ---
if __name__ == "__main__":
    # --- !!! 请在这里修改您的文件夹路径 !!! ---
    # 源文件夹：存放 .h5 文件的位置
    # 在Windows上，使用原始字符串 r'...' 是个好习惯
    source_folder = r"D:\code\GRAT-main\data\predata"

    # 目标文件夹：存放转换后的 .png 文件的位置
    destination_folder = r"D:\code\GRAT-main\data\predata"

    # --- !!! 请在这里修改您的数据集名称 !!! ---
    # HDF5 文件中三维影像数据对应的数据集（key）名称
    dataset_key = 'image'  # <-- 这是一个示例，请务必根据您的文件进行修改！

    # 检查路径是否为示例路径
    if source_folder == r"path/to/your/h5_files" or destination_folder == r"path/to/your/png_files":
        print("=" * 60)
        print("!!! 请注意 !!!")
        print("请先在代码中修改 'source_folder' 和 'destination_folder' 变量，")
        print("使其指向您实际的文件夹路径。")
        print("同时，也请确认 'dataset_key' 变量是否与您H5文件中的结构匹配。")
        print("=" * 60)
    else:
        # 执行转换
        convert_3d_h5_to_2d_png_slices(source_folder, destination_folder, dataset_key)
        print("\n所有文件转换完成！")