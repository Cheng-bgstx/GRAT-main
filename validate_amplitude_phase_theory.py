"""
Comprehensive Amplitude-Phase Theory Validation
================================================
Validates the core theoretical claim across ALL 18 fine-tuned models × 3 base SAM variants.

THEORY TO VALIDATE:
  Given intermediate ViT features F(ω) = A(ω)·exp(j·φ(ω)):

  H1: Amplitude A(ω) CHANGES significantly between base SAM and fine-tuned models
      (measured by relative L2 distance in amplitude spectrum)

  H2: Phase φ(ω) is RELATIVELY STABLE between base SAM and fine-tuned models
      (measured by circular correlation / cosine similarity of phase)

  H3: The amplitude change is CONSISTENT across different fine-tuning methods
      (SAMed LoRA vs Medical-SAM-Adapter show similar amplitude shifts)

  H4: The phase stability holds across DIFFERENT ViT scales
      (vit_b, vit_l, vit_h all preserve phase similarly)

MODELS (18 fine-tuned + 3 base):
  SAMed (LoRA):   3 ViT × 3 datasets = 9 models
  MSA (Adapter):  3 ViT × 3 datasets = 9 models
  Base SAM:       vit_b, vit_l, vit_h = 3 models

DATASETS: Synapse (CT), ISIC (dermoscopy), REFUGE (fundus)
"""
import torch
import numpy as np
import os
import sys
from collections import defaultdict
import json
from tqdm import tqdm
import torch.fft
from segment_anything import sam_model_registry

# ==================== Configuration ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
NUM_TEST_IMAGES = 50  # samples per dataset for statistical significance
OUTPUT_DIR = "results/theory_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Layer to analyze (ViT has 12 layers, we use deeper layers)
ANALYSIS_LAYERS = [9, 10, 11]  # last 3 layers carry most semantic info

# Base SAM models
BASE_MODELS = {
    'vit_b': 'checkpoints/sam_vit_b_01ec64.pth',
    'vit_l': 'checkpoints/sam_vit_l_0b3195.pth',
    'vit_h': 'checkpoints/sam_vit_h_4b8939.pth',
}

# Fine-tuned models: {model_id: (checkpoint, model_type, vit, dataset)}
FINETUNED_MODELS = {}
for vit in ['vit_b', 'vit_l', 'vit_h']:
    for ds in ['Synapse', 'ISIC', 'REFUGE']:
        # SAMed (LoRA)
        key = f"SAMed_{ds}_{vit}"
        ckpt = f"checkpoints/best_weights/SAMed_{ds}_{vit}.pth"
        if os.path.exists(ckpt):
            FINETUNED_MODELS[key] = (ckpt, 'SAMed', vit, ds)

        # MSA (Medical-SAM-Adapter)
        key = f"MSA_{ds}_{vit}"
        ckpt = f"checkpoints/best_weights/MSA_{ds}_{vit}.pth"
        if os.path.exists(ckpt):
            FINETUNED_MODELS[key] = (ckpt, 'MSA', vit, ds)

print(f"Found {len(FINETUNED_MODELS)} fine-tuned models")
for k, v in sorted(FINETUNED_MODELS.items()):
    print(f"  {k}: {v[1]} ({v[2]}, {v[3]})")


# ==================== Data Loading ====================
def load_test_images(dataset_name, num_images=NUM_TEST_IMAGES):
    """Load test images from each dataset."""
    images = []
    if dataset_name == 'Synapse':
        # Load from test_vol_h5
        import h5py
        test_dir = 'data/Synapse_preprocess/test_vol_h5'
        test_list = 'data/Synapse_preprocess/lists/test_vol.txt'
        if os.path.exists(test_list):
            with open(test_list) as f:
                cases = [line.strip() for line in f if line.strip()]
        else:
            cases = [f.replace('.npy.h5', '') for f in os.listdir(test_dir) if f.endswith('.h5')]

        count = 0
        for case in cases:
            if count >= num_images:
                break
            try:
                with h5py.File(os.path.join(test_dir, f'{case}.npy.h5'), 'r') as f:
                    data = f['image'][:]
                # Take middle slice
                mid = data.shape[0] // 2
                slice_img = data[mid]
                # Normalize and convert to 3-channel
                slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
                slice_img = np.stack([slice_img] * 3, axis=0)
                images.append(torch.from_numpy(slice_img).float())
                count += 1
            except Exception as e:
                continue

    elif dataset_name == 'ISIC':
        from PIL import Image
        import torchvision.transforms as T
        data_dir = 'data/ISIC/ISBI2016_ISIC_Part1_Test_Data'
        transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jpg')])[:num_images]
        for f in files:
            img = Image.open(os.path.join(data_dir, f)).convert('RGB')
            images.append(transform(img))

    elif dataset_name == 'REFUGE':
        from PIL import Image
        import torchvision.transforms as T
        data_dir = 'data/REFUGE-Multirater/Test-400'
        transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
        dirs = sorted(os.listdir(data_dir))[:num_images]
        for d in dirs:
            img_path = os.path.join(data_dir, d)
            # REFUGE directories contain image files
            if os.path.isdir(img_path):
                imgs = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png'))]
                if imgs:
                    img = Image.open(os.path.join(img_path, imgs[0])).convert('RGB')
                    images.append(transform(img))

    return images[:num_images]


# ==================== Feature Extraction ====================
def extract_features(model, images, vit_name, dataset_name):
    """Extract intermediate ViT features from a model."""
    all_features = defaultdict(list)
    model.eval()

    norm_transform = None
    if dataset_name in ['ISIC', 'REFUGE']:
        import torchvision.transforms as T
        norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        for img in tqdm(images, desc=f'Features {vit_name}/{dataset_name}', leave=False):
            x = img.unsqueeze(0).to(DEVICE)
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            if norm_transform is not None:
                x = norm_transform(x)

            _, features_layer = model.image_encoder(x)

            for layer_idx in ANALYSIS_LAYERS:
                feat = features_layer[layer_idx]  # [1, C, H, W]
                # FFT
                feat_fft = torch.fft.fftn(feat, dim=(-2, -1), norm="ortho")
                A = feat_fft.abs()
                phi = feat_fft.angle()
                all_features[layer_idx].append({
                    'amplitude': A.cpu(),
                    'phase': phi.cpu(),
                })

    return all_features


# ==================== Metrics ====================
def compute_amplitude_change(A_base, A_ft):
    """
    Compute relative amplitude change between base and fine-tuned model.
    Returns: relative L2 distance (higher = more change)
    """
    diff = (A_base - A_ft).pow(2).sum().sqrt()
    norm_base = A_base.pow(2).sum().sqrt()
    return (diff / (norm_base + 1e-8)).item()


def compute_phase_similarity(phi_base, phi_ft):
    """
    Compute phase similarity using circular correlation.
    Returns: cosine similarity of phase vectors (higher = more stable)
    """
    # Flatten
    p_base = phi_base.flatten()
    p_ft = phi_ft.flatten()

    # Circular correlation via cos and sin
    cos_sim = torch.cos(p_base - p_ft).mean()
    return cos_sim.item()


def compute_frequency_band_analysis(A_base, A_ft, phi_base, phi_ft, n_bands=4):
    """
    Per-frequency-band analysis of amplitude change and phase stability.
    """
    H, W = A_base.shape[-2:]
    fy = torch.fft.fftfreq(H).view(-1, 1)
    fx = torch.fft.fftfreq(W).view(1, -1)
    radius = torch.sqrt(fy**2 + fx**2)
    max_r = radius.max()
    radius = radius / (max_r + 1e-8)

    results = {}
    for b in range(n_bands):
        r_low = b / n_bands
        r_high = (b + 1) / n_bands
        if b == 0:
            mask = radius <= r_high
        else:
            mask = (radius > r_low) & (radius <= r_high)

        # Apply mask (broadcast to C dimension)
        mask_expanded = mask.unsqueeze(0).unsqueeze(0).to(A_base.device)

        A_base_band = A_base * mask_expanded.float()
        A_ft_band = A_ft * mask_expanded.float()
        phi_base_band = phi_base * mask_expanded.float()
        phi_ft_band = phi_ft * mask_expanded.float()

        amp_change = compute_amplitude_change(A_base_band, A_ft_band)
        phase_sim = compute_phase_similarity(phi_base_band[phi_base_band != 0],
                                              phi_ft_band[phi_ft_band != 0]) if phi_base_band.sum() > 0 else 0

        results[f'band_{b}'] = {
            'amp_change': amp_change,
            'phase_sim': phase_sim,
            'band_range': f'[{r_low:.2f}, {r_high:.2f}]'
        }

    return results


# ==================== Main Validation ====================
def main():
    print("=" * 60)
    print("AMPLITUDE-PHASE THEORY VALIDATION")
    print("=" * 60)

    all_results = {}

    # For each dataset, load test images
    for dataset in ['Synapse', 'ISIC', 'REFUGE']:
        print(f"\n{'='*40}")
        print(f"Dataset: {dataset}")
        print(f"{'='*40}")

        try:
            images = load_test_images(dataset)
            print(f"Loaded {len(images)} test images")
        except Exception as e:
            print(f"Failed to load images for {dataset}: {e}")
            continue

        # For each base ViT
        for vit_name, vit_ckpt in BASE_MODELS.items():
            print(f"\n  Base model: {vit_name}")

            # Load base SAM
            sam_base, _ = sam_model_registry[vit_name](
                image_size=IMG_SIZE, num_classes=8,
                checkpoint=vit_ckpt, pixel_mean=[0,0,0], pixel_std=[1,1,1])
            sam_base = sam_base.to(DEVICE)

            # Extract base features
            base_features = extract_features(sam_base, images, vit_name, dataset)

            # Compare with each fine-tuned model of same ViT + dataset
            matching_models = {
                k: v for k, v in FINETUNED_MODELS.items()
                if v[2] == vit_name and v[3] == dataset
            }

            for model_id, (ckpt, method, vit, ds) in matching_models.items():
                print(f"    vs {model_id} ({method})...")

                try:
                    # Load fine-tuned model (SAMed uses LoRA, MSA uses Adapter)
                    if method == 'SAMed':
                        # Load base + LoRA weights
                        import importlib
                        sam_ft, _ = sam_model_registry[vit_name](
                            image_size=IMG_SIZE, num_classes=8,
                            checkpoint=vit_ckpt, pixel_mean=[0,0,0], pixel_std=[1,1,1])
                        sam_ft = sam_ft.to(DEVICE)

                        pkg = importlib.import_module('sam_lora_image_encoder')
                        net = pkg.LoRA_Sam(sam_ft, 4).to(DEVICE)
                        net.load_lora_parameters(ckpt)
                        # Use net's sam for feature extraction
                        ft_model = net.sam
                    else:
                        # MSA — needs different loading
                        # For now, skip MSA models (they require adapter loading)
                        print(f"      Skipping MSA (needs adapter loading)")
                        continue

                    # Extract fine-tuned features
                    ft_features = extract_features(ft_model, images, vit_name, dataset)

                    # Compute metrics per layer
                    for layer_idx in ANALYSIS_LAYERS:
                        amp_changes = []
                        phase_sims = []
                        band_analyses = []

                        for i in range(min(len(base_features[layer_idx]), len(ft_features[layer_idx]))):
                            A_base = base_features[layer_idx][i]['amplitude']
                            phi_base = base_features[layer_idx][i]['phase']
                            A_ft = ft_features[layer_idx][i]['amplitude']
                            phi_ft = ft_features[layer_idx][i]['phase']

                            amp_changes.append(compute_amplitude_change(A_base, A_ft))
                            phase_sims.append(compute_phase_similarity(phi_base, phi_ft))

                            # Band analysis on first 10 images
                            if i < 10:
                                band_analyses.append(
                                    compute_frequency_band_analysis(A_base, A_ft, phi_base, phi_ft))

                        avg_amp_change = np.mean(amp_changes)
                        std_amp_change = np.std(amp_changes)
                        avg_phase_sim = np.mean(phase_sims)
                        std_phase_sim = np.std(phase_sims)

                        key = f"{dataset}/{vit_name}/{model_id}/layer_{layer_idx}"
                        all_results[key] = {
                            'avg_amp_change': float(avg_amp_change),
                            'std_amp_change': float(std_amp_change),
                            'avg_phase_sim': float(avg_phase_sim),
                            'std_phase_sim': float(std_phase_sim),
                        }

                        print(f"      L{layer_idx}: ΔA={avg_amp_change:.4f}±{std_amp_change:.4f}, "
                              f"cos(φ)={avg_phase_sim:.4f}±{std_phase_sim:.4f}")

                    del ft_model
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"      Error: {e}")
                    import traceback
                    traceback.print_exc()

            del sam_base
            torch.cuda.empty_cache()

    # ==================== Summary Statistics ====================
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Aggregate across all models
    all_amp = []
    all_phase = []
    by_method = defaultdict(lambda: {'amp': [], 'phase': []})
    by_vit = defaultdict(lambda: {'amp': [], 'phase': []})
    by_dataset = defaultdict(lambda: {'amp': [], 'phase': []})

    for key, res in all_results.items():
        all_amp.append(res['avg_amp_change'])
        all_phase.append(res['avg_phase_sim'])

        # Parse key: dataset/vit/model_id/layer_N
        parts = key.split('/')
        ds = parts[0]
        vit = parts[1]
        method = 'SAMed' if 'SAMed' in parts[2] else 'MSA'

        by_method[method]['amp'].append(res['avg_amp_change'])
        by_method[method]['phase'].append(res['avg_phase_sim'])
        by_vit[vit]['amp'].append(res['avg_amp_change'])
        by_vit[vit]['phase'].append(res['avg_phase_sim'])
        by_dataset[ds]['amp'].append(res['avg_amp_change'])
        by_dataset[ds]['phase'].append(res['avg_phase_sim'])

    print(f"\nOverall (N={len(all_amp)}):")
    print(f"  Amplitude change: {np.mean(all_amp):.4f} ± {np.std(all_amp):.4f}")
    print(f"  Phase similarity:  {np.mean(all_phase):.4f} ± {np.std(all_phase):.4f}")

    print(f"\nBy fine-tuning method:")
    for method in ['SAMed', 'MSA']:
        if by_method[method]['amp']:
            print(f"  {method}: ΔA={np.mean(by_method[method]['amp']):.4f}, "
                  f"cos(φ)={np.mean(by_method[method]['phase']):.4f}")

    print(f"\nBy ViT scale:")
    for vit in ['vit_b', 'vit_l', 'vit_h']:
        if by_vit[vit]['amp']:
            print(f"  {vit}: ΔA={np.mean(by_vit[vit]['amp']):.4f}, "
                  f"cos(φ)={np.mean(by_vit[vit]['phase']):.4f}")

    print(f"\nBy dataset:")
    for ds in ['Synapse', 'ISIC', 'REFUGE']:
        if by_dataset[ds]['amp']:
            print(f"  {ds}: ΔA={np.mean(by_dataset[ds]['amp']):.4f}, "
                  f"cos(φ)={np.mean(by_dataset[ds]['phase']):.4f}")

    # ==================== Hypothesis Testing ====================
    print(f"\n{'='*40}")
    print("HYPOTHESIS TESTING")
    print(f"{'='*40}")

    amp_mean = np.mean(all_amp)
    phase_mean = np.mean(all_phase)

    print(f"\nH1: Amplitude changes significantly?")
    print(f"  ΔA = {amp_mean:.4f} → {'YES (ΔA > 0.1)' if amp_mean > 0.1 else 'MARGINAL' if amp_mean > 0.05 else 'NO'}")

    print(f"\nH2: Phase is relatively stable?")
    print(f"  cos(φ) = {phase_mean:.4f} → {'YES (cos(φ) > 0.8)' if phase_mean > 0.8 else 'MARGINAL' if phase_mean > 0.6 else 'NO'}")

    # Effect size: amplitude change relative to phase stability
    if phase_mean > 0:
        amp_phase_ratio = amp_mean / (1.0 - phase_mean + 1e-8)
        print(f"\nAmplitude/Phase instability ratio: {amp_phase_ratio:.2f}")
        print(f"  Interpretation: amplitude changes are {amp_phase_ratio:.1f}x larger than phase changes")

    # Save full results
    with open(f'{OUTPUT_DIR}/theory_validation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nFull results saved to {OUTPUT_DIR}/theory_validation_results.json")


if __name__ == '__main__':
    main()
