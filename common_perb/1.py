import torch
from torchvision.utils import save_image
import os

# ================= Configuration =================
# 1. Path to your input .pth file
# (Replace this with your actual file path)
pth_path = '/root/data1/fyc/GRAT-main/common_perb/Synapse.pth'

# 2. Path for the output .png file
output_png_path = '/root/data1/fyc/GRAT-main/common_perb/noise_visualization.png'
# =================================================

def convert_pth_to_png():
    # Check if file exists
    if not os.path.exists(pth_path):
        print(f"Error: File not found at {pth_path}")
        return

    # 1. Load the .pth file
    # map_location='cpu' ensures it runs even without a GPU
    try:
        noise_tensor = torch.load(pth_path, map_location='cpu')
        print(f"File loaded successfully. Tensor shape: {noise_tensor.shape}")
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # 2. Adjust Dimensions
    # If shape is [1, 3, H, W] (Batch size 1), squeeze it to [3, H, W]
    if noise_tensor.dim() == 4:
        noise_tensor = noise_tensor.squeeze(0)
    
    # 3. Normalization (Crucial Step)
    # The raw noise values are likely very small (e.g., -10/255 to +10/255).
    # We apply Min-Max Normalization to stretch them to [0, 1] for visualization.
    min_val = noise_tensor.min()
    max_val = noise_tensor.max()
    
    print(f"Raw value range: Min={min_val:.5f}, Max={max_val:.5f}")

    if max_val - min_val > 1e-6: # Avoid division by zero
        # Formula: (x - min) / (max - min)
        # This maps the smallest value to Black (0) and the largest to White (1)
        vis_tensor = (noise_tensor - min_val) / (max_val - min_val)
    else:
        vis_tensor = noise_tensor
        print("Warning: value range is extremely small or constant. Output might be blank.")

    # 4. Save as PNG
    save_image(vis_tensor, output_png_path)
    print(f"Conversion complete! Image saved at: {output_png_path}")

if __name__ == '__main__':
    convert_pth_to_png()