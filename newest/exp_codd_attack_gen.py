#!/usr/bin/env python3
"""
CODD Attack Generator: Generate adversarial images using CODD theory-driven attack.
Outputs images compatible with test_samed.py and Medical-SAM-Adapter-main/val.py.

Key insight: CODD default (use_random_noise=True) = SE-CGR baseline.
CODD adds: directional ablation (radial/tangential/isotropic/misaligned) and SRS warm-start.

Usage:
  python exp_codd_attack_gen.py --vit vit_b --dataset ISIC --gpu 0 --mode radial
  python exp_codd_attack_gen.py --vit vit_b --dataset ISIC --gpu 0 --mode tangential
  python exp_codd_attack_gen.py --vit vit_b --dataset ISIC --gpu 0 --mode radial --srs

  # Full sweep:
  for vit in vit_b vit_l vit_h; do
    for ds in Synapse ISIC REFUGE; do
      for mode in radial tangential isotropic misaligned; do
        python exp_codd_attack_gen.py --vit $vit --dataset $ds --gpu $G --mode $mode &
      done
    done
  done
"""
import os, sys, argparse, random, gc, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from attack_codd import AttackConfig, run_codd_attack, distill_srs
from theory import SharedSubspace
from evaluate import load_sam_encoder

# ============================================================
SAM_CKPT = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}

IMAGE_SIZE = 512
OUTPUT_BASE = os.path.join(ROOT, "testset", "codd_attack")

# Attack defaults matching SE-CGR paper
ATTACK_DEFAULTS = {
    "att_step_size": 2.0,
    "att_bound": 10.0,
    "att_numstep": 10,
    "momentum_decay": 0.5,
    "ads_rho": 1.0,
    "ads_gamma": 1e-4,
    "edge_guidance": True,
    "edge_gamma": 3.0,
}

SRS_DIR = os.path.join(NEWEST, "codd_master_results", "srs")
# ============================================================


def load_synapse_test_slices(case_limit=3, slice_limit=12):
    """Load Synapse test H5 volumes -> list of 2D slices."""
    test_dir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    all_slices = []
    for fname in sorted(os.listdir(test_dir))[:case_limit]:
        path = os.path.join(test_dir, fname)
        if fname.endswith('.h5'):
            import h5py
            with h5py.File(path, 'r') as f:
                img = f['image'][:].astype(np.float32)
                lbl = f['label'][:].astype(np.float32)
        else:
            data = np.load(path, allow_pickle=True)
            img = data['image'].astype(np.float32) if isinstance(data, np.ndarray) else data['image'].astype(np.float32)
            lbl = data['label'].astype(np.float32) if isinstance(data, np.ndarray) else np.zeros_like(img)

        # Find organ-containing slices
        valid = [s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100]
        if not valid:
            valid = list(range(min(img.shape[0], slice_limit)))
        elif len(valid) > slice_limit:
            step = len(valid) / slice_limit
            valid = [valid[int(i * step)] for i in range(slice_limit)]

        case_name = os.path.splitext(fname)[0].replace('.npy', '')
        for s in valid:
            slc = img[s].copy()
            if slc.max() > 1.0:
                slc = np.clip(slc, -200, 300)
                slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-6) * 255.0
            from scipy.ndimage import zoom
            h_s = IMAGE_SIZE / slc.shape[0]
            w_s = IMAGE_SIZE / slc.shape[1]
            slc = zoom(slc, (h_s, w_s), order=3)
            all_slices.append({"case_name": f"{case_name}_slice{s}", "image": slc})
    return all_slices


def load_isic_test_images(case_limit=32):
    """Load ISIC test images."""
    from PIL import Image
    test_dir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_Data"
    cases = []
    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith('.jpg'):
            continue
        if len(cases) >= case_limit:
            break
        img = Image.open(os.path.join(test_dir, fname)).convert('RGB')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32)
        cases.append({"case_name": os.path.splitext(fname)[0], "image": img_np})
    return cases


def load_refuge_test_images(case_limit=32):
    """Load REFUGE test images."""
    from PIL import Image
    test_dir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    subdirs = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    cases = []
    for subdir in subdirs:
        if len(cases) >= case_limit:
            break
        subdir_path = os.path.join(test_dir, subdir)
        imgs = sorted([f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png', '.bmp'))])
        if not imgs:
            continue
        img = Image.open(os.path.join(subdir_path, imgs[0])).convert('RGB')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32)
        cases.append({"case_name": subdir, "image": img_np})
    return cases


def run_attack_on_cases(source_sam, cases, config, output_dir, srs_bundle=None, device=None):
    """Generate adversarial images for all test cases."""
    if device is None:
        device = next(source_sam.parameters()).device

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for idx, case in enumerate(cases):
        case_name = case["case_name"]
        img_np = case["image"]

        # Prepare image tensor (SAM expects [0, 255])
        if img_np.ndim == 2:
            img_3ch = np.stack([img_np] * 3, axis=0)
        elif img_np.ndim == 3 and img_np.shape[-1] == 3:
            img_3ch = img_np.transpose(2, 0, 1)
        else:
            img_3ch = img_np

        img_t = torch.from_numpy(img_3ch).float().unsqueeze(0).to(device)
        if img_t.max() > 2.0:
            pass  # already 0-255
        else:
            img_t = img_t * 255.0

        try:
            adv_np, trace = run_codd_attack(
                source_sam.image_encoder, img_t, config,
                srs_bundle=srs_bundle, return_trace=True,
            )

            # Save adversarial image
            if case["case_name"].startswith("case0") or "slice" in case["case_name"]:
                # Synapse 3D volume — create subdirectory per case
                vol_name = case["case_name"].split("_slice")[0]
                vol_dir = os.path.join(output_dir, vol_name)
                os.makedirs(vol_dir, exist_ok=True)
                slice_idx = case["case_name"].split("_slice")[1] if "_slice" in case["case_name"] else "0"
                out_path = os.path.join(vol_dir, f"{slice_idx}.png")
            elif case_name.isdigit():
                # REFUGE directory format
                case_dir = os.path.join(output_dir, case_name)
                os.makedirs(case_dir, exist_ok=True)
                out_path = os.path.join(case_dir, "adv.png")
            else:
                # ISIC — flat file
                out_path = os.path.join(output_dir, f"{case_name}.png")

            # Save as PNG
            import matplotlib.image as mpimg
            adv_save = adv_np.copy()
            if adv_save.ndim == 3 and adv_save.shape[0] == 3:
                adv_save = adv_save.transpose(1, 2, 0)
            adv_save = np.clip(adv_save, 0, 255).astype(np.uint8)
            if adv_save.ndim == 2:
                mpimg.imsave(out_path, adv_save, cmap='gray')
            else:
                mpimg.imsave(out_path, adv_save)

            results.append({
                "case_name": case_name,
                "final_loss": trace.get("final_loss", float("nan")),
                "max_perturbation": trace.get("max_perturbation", float("nan")),
            })
        except Exception as e:
            print(f"  ERROR {case_name}: {e}")
            continue

        if (idx + 1) % 8 == 0:
            print(f"  [{idx + 1}/{len(cases)}] cases done")
        del img_t, adv_np
        gc.collect()
        torch.cuda.empty_cache()

    print(f"  Generated {len(results)}/{len(cases)} adversarial images → {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="CODD Attack Generator")
    parser.add_argument("--vit", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--dataset", type=str, default="ISIC", choices=["Synapse", "ISIC", "REFUGE"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", type=str, default="radial",
                       choices=["radial", "tangential", "isotropic", "misaligned", "random_noise"])
    parser.add_argument("--srs", action="store_true", help="Use SRS warm-start")
    parser.add_argument("--srs_dataset", type=str, default="", help="SRS proxy dataset (default: same as --dataset)")
    parser.add_argument("--numstep", type=int, default=10)
    parser.add_argument("--bound", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_cases", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    print(f"CODD Attack Generator — {args.vit} {args.dataset} mode={args.mode} SRS={args.srs}")
    print(f"GPU: cuda:{args.gpu}, steps={args.numstep}, bound={args.bound}/255")

    # Output path
    srs_tag = "_srs" if args.srs else ""
    output_dir = os.path.join(OUTPUT_BASE, f"{args.vit}_{args.dataset}_{args.mode}{srs_tag}")
    print(f"Output: {output_dir}")

    # Load source SAM
    print("Loading source SAM...")
    source_sam = load_sam_encoder(args.vit, SAM_CKPT[args.vit], IMAGE_SIZE, device=device)

    # Load test cases
    if args.dataset == "Synapse":
        cases = load_synapse_test_slices(case_limit=3 if args.max_cases == 0 else args.max_cases)
        print(f"Loaded {len(cases)} Synapse slices")
    elif args.dataset == "ISIC":
        n = 32 if args.max_cases == 0 else args.max_cases
        cases = load_isic_test_images(case_limit=n)
        print(f"Loaded {len(cases)} ISIC images")
    elif args.dataset == "REFUGE":
        n = 32 if args.max_cases == 0 else args.max_cases
        cases = load_refuge_test_images(case_limit=n)
        print(f"Loaded {len(cases)} REFUGE images")

    if not cases:
        print("No test cases found!")
        return

    # Create attack config
    config_kwargs = dict(ATTACK_DEFAULTS)
    config_kwargs.update({
        "att_numstep": args.numstep,
        "att_bound": args.bound,
        "direction_mode": "radial" if args.mode == "random_noise" else args.mode,
        "seed": args.seed,
    })
    config = AttackConfig(**config_kwargs)

    # Load SRS if requested
    srs_bundle = None
    if args.srs:
        srs_dataset = args.srs_dataset or args.dataset
        srs_path = os.path.join(SRS_DIR, f"srs_{args.vit}_{srs_dataset}.pt")
        if os.path.exists(srs_path):
            print(f"Loading SRS: {srs_path}")
            srs_bundle = SharedSubspace.load(srs_path, device)
            config.use_srs = True
            config.srs_path = srs_path
            config.srs_proj_weight = 0.15
            config.srs_mean_weight = 0.1
        else:
            print(f"SRS not found: {srs_path}. Distilling...")
            os.makedirs(SRS_DIR, exist_ok=True)

            # Quick proxy data loading
            proxy_cases = []
            if srs_dataset == "Synapse":
                proxy_cases_slc = load_synapse_test_slices(case_limit=3, slice_limit=32)
                # Use test slices as proxy (we don't have separate train loader here)
            elif srs_dataset == "ISIC":
                from PIL import Image
                train_dir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Training_Data"
                for fname in sorted(os.listdir(train_dir))[:64]:
                    if fname.endswith('.jpg'):
                        img = Image.open(os.path.join(train_dir, fname)).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
                        proxy_cases.append({"case_name": fname, "image": np.array(img).astype(np.float32)})
            elif srs_dataset == "REFUGE":
                from PIL import Image
                train_dir = f"{ROOT}/data/REFUGE-Multirater/Training-400"
                train_subdirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
                for subdir in train_subdirs[:64]:
                    sp = os.path.join(train_dir, subdir)
                    if os.path.isdir(sp):
                        imgs = sorted([f for f in os.listdir(sp) if f.endswith(('.jpg', '.png'))])
                        if imgs:
                            img = Image.open(os.path.join(sp, imgs[0])).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
                            proxy_cases.append({"case_name": subdir, "image": np.array(img).astype(np.float32)})

            if proxy_cases:
                class ProxyDL:
                    def __init__(self, data):
                        self.data = data
                    def __iter__(self):
                        self._i = 0
                        return self
                    def __next__(self):
                        if self._i >= len(self.data):
                            raise StopIteration
                        item = self.data[self._i]
                        self._i += 1
                        img = item["image"]
                        if img.ndim == 3 and img.shape[-1] == 3:
                            img = img.transpose(2, 0, 1)
                        elif img.ndim == 2:
                            img = np.stack([img]*3, axis=0)
                        img_t = torch.from_numpy(img).float().unsqueeze(0)
                        if img_t.max() <= 2.0:
                            img_t = img_t * 255.0
                        return {"image": img_t}

                srs_bundle = distill_srs(
                    source_sam.image_encoder,
                    ProxyDL(proxy_cases[:64]),
                    AttackConfig(**ATTACK_DEFAULTS, direction_mode="radial"),
                    num_samples=64, rank=8, device=device, is_3d=(srs_dataset == "Synapse"),
                )
                srs_bundle.save(srs_path)
                config.use_srs = True
                config.srs_path = srs_path
                config.srs_proj_weight = 0.15
                config.srs_mean_weight = 0.1
                print(f"SRS distilled and saved to {srs_path}")
            else:
                print("WARNING: No proxy data, disabling SRS")
                args.srs = False

    # Run attack
    t0 = time.time()
    run_attack_on_cases(source_sam, cases, config, output_dir, srs_bundle, device)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/len(cases):.1f}s/case)")

    # Cleanup
    del source_sam
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
