#!/usr/bin/env python3
"""
Experiment 2: Carrier Null Test for all 6 vit_b targets.
Decomposes the SAME drift (G_target - G_source) using 3 different carrier bases:
  - Source carrier:  e^{j*Phi_s}        (original source phase)
  - Shuffled carrier: e^{j*shuffle(Phi_s)} (spatially scrambled phase)
  - Random carrier:  e^{j*rand(0,2pi)}   (uniform random phase)

If the source carrier uniquely captures the drift structure:
  source_radial_ratio >> shuffled_radial_ratio ≈ random_radial_ratio ≈ 0.5
"""
import os, sys, json, gc
import numpy as np
import torch

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder, load_samed_model, load_medsa_model

DEVICE = torch.device("cuda:0")
IMAGE_SIZE = 512

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = [
    ("Synapse", "SAMed", f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", None, 8, 9),
    ("Synapse", "MSA", None, f"{ROOT}/output_runs_correct/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s24/Model/checkpoint_best.pth", 8, 9),
    ("ISIC", "SAMed", f"{ROOT}/output_runs_correct/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", None, 1, 1),
    ("ISIC", "MSA", None, f"{ROOT}/output_runs_correct/MedicalSAMAdapter/isic_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/checkpoint_best.pth", 1, 1),
    ("REFUGE", "SAMed", f"{ROOT}/output_runs_correct/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", None, 2, 2),
    ("REFUGE", "MSA", None, f"{ROOT}/output_runs_correct/MedicalSAMAdapter/REFUGE_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/checkpoint_best.pth", 2, 2),
]

VIT_NAME = "vit_b"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
NUM_SAMPLES = 20


def decompose_with_carrier(feat_source, feat_target, carrier_phase, eps=1e-8):
    """
    Decompose drift Delta_G = G_target - G_source using an arbitrary carrier basis.

    Args:
        feat_source, feat_target: [B, C, H, W] features
        carrier_phase: [B, 1, H, W] phase to use as carrier

    Returns:
        radial_ratio: E_radial / (E_radial + E_tangential)
    """
    fs = feat_source.mean(dim=1, keepdim=True)
    ft = feat_target.mean(dim=1, keepdim=True)

    Gs = torch.fft.fft2(fs, norm="ortho")
    Gt = torch.fft.fft2(ft, norm="ortho")

    Delta = Gt - Gs
    carrier = torch.exp(1j * carrier_phase)  # e^{j*Phi}

    # Project onto this carrier basis
    local = Delta * torch.conj(carrier)
    alpha = torch.real(local)   # radial
    beta = torch.imag(local)    # tangential

    radial_energy = float(alpha.square().sum().item())
    tangential_energy = float(beta.square().sum().item())
    total = radial_energy + tangential_energy

    return radial_energy / (total + eps)


def get_samples(dataset):
    if dataset == "Synapse":
        import h5py
        tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
        samples = []
        for h5f in sorted(os.listdir(tdir)):
            if len(samples) >= NUM_SAMPLES: break
            with h5py.File(os.path.join(tdir, h5f), 'r') as f:
                img, lbl = f['image'][:], f['label'][:]
            valid = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100],
                          key=lambda s: (lbl[s] > 0).sum(), reverse=True)
            for s in valid:
                slc = img[s].astype(np.float32)
                samples.append(torch.from_numpy(np.stack([slc]*3, axis=0)))
                if len(samples) >= NUM_SAMPLES: break
        return samples
    elif dataset == "ISIC":
        from PIL import Image
        from torchvision import transforms
        import pandas as pd
        data_dir = f"{ROOT}/data/ISIC"
        df = pd.read_csv(os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_GroundTruth.csv"))
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        samples = []
        for _, row in df.iterrows():
            img = Image.open(os.path.join(data_dir, row['img'])).convert('RGB')
            samples.append(transform(img))
            if len(samples) >= NUM_SAMPLES: break
        return samples
    else:
        from PIL import Image
        from torchvision import transforms
        data_dir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
        folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        samples = []
        for folder in folders:
            img_path = os.path.join(data_dir, folder, f"{folder}.jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                samples.append(transform(img))
            if len(samples) >= NUM_SAMPLES: break
        return samples


def main():
    print("Loading source SAM (vit_b)...")
    source_sam = load_sam_encoder(VIT_NAME, SAM_CKPT, IMAGE_SIZE, 8, DEVICE)
    source_sam.eval()

    all_results = {}

    for dataset, model_type, samed_ckpt, msa_ckpt, num_cls, mmo in TARGETS:
        tag = f"{dataset}/{model_type}"
        ckpt_path = samed_ckpt or msa_ckpt
        if not os.path.exists(ckpt_path):
            print(f"\nSKIP {tag}: checkpoint not found")
            continue

        print(f"\n{'='*60}\n{tag}\n{'='*60}")

        if model_type == "SAMed":
            target = load_samed_model(VIT_NAME, SAM_CKPT, samed_ckpt, IMAGE_SIZE, num_cls, device=DEVICE)
        else:
            target = load_medsa_model(VIT_NAME, SAM_CKPT, msa_ckpt, IMAGE_SIZE, mmo, DEVICE)

        samples = get_samples(dataset)
        print(f"  Processing {len(samples)} samples...")

        source_rrs, shuffled_rrs, random_rrs = [], [], []

        for i, sample in enumerate(samples):
            img_t = sample.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                src_feat, _ = source_sam.image_encoder(img_t)

                if model_type == "SAMed":
                    tgt_feat, _ = target.sam.image_encoder(img_t)
                else:
                    tgt_feat = target.image_encoder(target.preprocess(img_t))

            # Compute source carrier phase from source features
            fs = src_feat.mean(dim=1, keepdim=True)
            Gs = torch.fft.fft2(fs, norm="ortho")
            source_phase = torch.angle(Gs)  # [1, 1, H, W]

            # 1. Source carrier: use original phase
            source_rr = decompose_with_carrier(src_feat, tgt_feat, source_phase)
            source_rrs.append(source_rr)

            # 2. Shuffled carrier: random permutation of phase positions
            B, C, H, W = source_phase.shape
            phase_flat = source_phase.view(B*C, -1)
            perm = torch.randperm(H*W, device=source_phase.device)
            shuffled_phase = phase_flat[:, perm].view_as(source_phase)
            shuffled_rr = decompose_with_carrier(src_feat, tgt_feat, shuffled_phase)
            shuffled_rrs.append(shuffled_rr)

            # 3. Random carrier: random phase
            random_phase = torch.rand_like(source_phase) * 2 * np.pi
            random_rr = decompose_with_carrier(src_feat, tgt_feat, random_phase)
            random_rrs.append(random_rr)

        result = {
            "source_carrier": {"mean": float(np.mean(source_rrs)), "std": float(np.std(source_rrs))},
            "shuffled_carrier": {"mean": float(np.mean(shuffled_rrs)), "std": float(np.std(shuffled_rrs))},
            "random_carrier": {"mean": float(np.mean(random_rrs)), "std": float(np.std(random_rrs))},
        }
        all_results[tag] = result

        print(f"  Source:   {result['source_carrier']['mean']:.4f} ± {result['source_carrier']['std']:.4f}")
        print(f"  Shuffled: {result['shuffled_carrier']['mean']:.4f} ± {result['shuffled_carrier']['std']:.4f}")
        print(f"  Random:   {result['random_carrier']['mean']:.4f} ± {result['random_carrier']['std']:.4f}")
        diff = result['source_carrier']['mean'] - result['shuffled_carrier']['mean']
        print(f"  Source >> shuffled: {diff > 0.05} (delta={diff:.4f})")

        del target; gc.collect(); torch.cuda.empty_cache()

    output = {"config": {"vit": VIT_NAME, "num_samples": NUM_SAMPLES}, "results": all_results}
    out_path = os.path.join(OUTPUT_DIR, "exp2_carrier_null.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
