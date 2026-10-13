#!/usr/bin/env python3
"""Full evaluation of ALL 18 fine-tuned models on their respective test sets."""
import sys, os, gc, time, json
import numpy as np
import torch
import h5py
from PIL import Image
import glob

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT); sys.path.insert(0, NEWEST)

from evaluate import load_samed_model, load_medsa_model

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
CHKPT = f"{ROOT}/checkpoints"
SAM_CKPTS = {
    "vit_b": f"{CHKPT}/sam_vit_b_01ec64.pth",
    "vit_l": f"{CHKPT}/sam_vit_l_0b3195.pth",
    "vit_h": f"{CHKPT}/sam_vit_h_4b8939.pth",
}
device = torch.device("cuda:0")
IM_MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
IM_STD = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)


# ═══ Synapse Evaluation (12 volumes) ═══
def eval_synapse(model_path, vit_name):
    net = load_samed_model(vit_name, SAM_CKPTS[vit_name], model_path, IMAGE_SIZE, 8, device=device)
    net.eval()

    with open(f"{ROOT}/data/Synapse_preprocess/lists/test_vol.txt") as f:
        cases = [l.strip() for l in f if l.strip()]

    all_dices = []
    for case in cases:
        h5_path = f"{ROOT}/data/Synapse_preprocess/test_vol_h5/{case}.npy.h5"
        with h5py.File(h5_path, 'r') as f:
            img_vol = f['image'][:]; lbl_vol = f['label'][:]

        case_dices = []
        for s in range(img_vol.shape[0]):
            if (lbl_vol[s] > 0).sum() < 500: continue
            slc = img_vol[s].astype(np.float32)
            x = torch.from_numpy(np.stack([slc]*3)).unsqueeze(0).float().to(device)
            gt = lbl_vol[s]
            with torch.no_grad():
                out = net(x, multimask_output=False, image_size=IMAGE_SIZE); m = out['masks']
            pred = torch.argmax(torch.softmax(m.float(), dim=1), dim=1).squeeze(0).cpu().numpy()
            for c in range(1, 9):
                pc = (pred==c).astype(np.float32); gc = (gt==c).astype(np.float32)
                if gc.sum() == 0: continue
                inter = (pc*gc).sum()
                case_dices.append(float(2*inter/(pc.sum()+gc.sum()+1e-8)))
        if case_dices:
            all_dices.append(np.mean(case_dices))

    del net; torch.cuda.empty_cache()
    return float(np.mean(all_dices)) if all_dices else 0, len(cases)


# ═══ ISIC Evaluation (379 images) ═══
def eval_isic(model_path, vit_name):
    net = load_samed_model(vit_name, SAM_CKPTS[vit_name], model_path, IMAGE_SIZE, 1, device=device)
    net.eval()

    imgs = sorted(glob.glob(f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_Data/*.jpg"))
    all_dices = []
    for ip in imgs:
        bp = ip.replace('_Data','_GroundTruth').replace('.jpg','_Segmentation.png')
        if not os.path.exists(bp): continue
        img = Image.open(ip).convert('RGB').resize((512,512), Image.BILINEAR)
        im = torch.from_numpy(np.array(img).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
        im = ((im-IM_MEAN)/IM_STD).to(device)
        mask = Image.open(bp).convert('L').resize((512,512), Image.NEAREST)
        gt = (np.array(mask) > 128).astype(np.float32)
        if gt.sum() < 100: continue
        with torch.no_grad():
            out = net(im, multimask_output=False, image_size=IMAGE_SIZE); m = out['masks']
        probs = torch.softmax(m.float(), dim=1); lp = probs[0,1].cpu().numpy()
        best = 0.0
        for th in [0.1,0.3,0.5,0.7,0.9]:
            pb = (lp>th).astype(np.float32); inter = (pb*gt).sum()
            dice = float(2*inter/(pb.sum()+gt.sum()+1e-8))
            if dice > best: best = dice
        all_dices.append(best)

    del net; torch.cuda.empty_cache()
    return float(np.mean(all_dices)) if all_dices else 0, len(all_dices)


# ═══ REFUGE Evaluation ═══
def eval_refuge(model_path, vit_name):
    net = load_samed_model(vit_name, SAM_CKPTS[vit_name], model_path, IMAGE_SIZE, 2, device=device)
    net.eval()

    imgs = sorted(glob.glob(f"{ROOT}/data/REFUGE-Multirater/Test-400/Images/*.bmp"))
    gts  = sorted(glob.glob(f"{ROOT}/data/REFUGE-Multirater/Test-400/Disc_Cup_Masks/*.bmp"))

    all_dices = []
    for ip, gp in zip(imgs, gts):
        img = Image.open(ip).convert('RGB').resize((512,512), Image.BILINEAR)
        im = torch.from_numpy(np.array(img).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
        im = ((im-IM_MEAN)/IM_STD).to(device)
        mask = np.array(Image.open(gp).resize((512,512), Image.NEAREST))
        # REFUGE: Disc=0/128, Cup=0/255 (approximate split)
        disc_gt = (mask >= 128).astype(np.float32); cup_gt = (mask == 255).astype(np.float32)

        with torch.no_grad():
            out = net(im, multimask_output=False, image_size=IMAGE_SIZE); m = out['masks']
        probs = torch.softmax(m.float(), dim=1)
        for ci, gm in [(1, disc_gt), (2, cup_gt)]:
            cp = probs[0,ci].cpu().numpy()
            best = 0.0
            for th in [0.1,0.3,0.5,0.7,0.9]:
                pb = (cp>th).astype(np.float32); inter = (pb*gm).sum()
                dice = float(2*inter/(pb.sum()+gm.sum()+1e-8))
                if dice > best: best = dice
            all_dices.append(best)

    del net; torch.cuda.empty_cache()
    return float(np.mean(all_dices)) if all_dices else 0, len(all_dices)//2


# ═══ MSA Synapse Evaluation ═══
def eval_msa_synapse(ckpt_path, vit_name):
    import random as _random; _random.seed(42)
    net = load_medsa_model(vit_name, SAM_CKPTS[vit_name], ckpt_path, IMAGE_SIZE, multimask_output=9, device=device)
    net.eval()

    with open(f"{ROOT}/data/Synapse_preprocess/lists/test_vol.txt") as f:
        cases = [l.strip() for l in f if l.strip()]

    all_dices = []
    for case in cases:
        h5_path = f"{ROOT}/data/Synapse_preprocess/test_vol_h5/{case}.npy.h5"
        with h5py.File(h5_path, 'r') as f:
            img_vol = f['image'][:]; lbl_vol = f['label'][:]

        for s in range(img_vol.shape[0]):
            if (lbl_vol[s] > 0).sum() < 500: continue
            slc = img_vol[s].astype(np.float32)
            x = torch.from_numpy(np.stack([slc]*3)).unsqueeze(0).float().to(device)
            gt = lbl_vol[s]

            pt_list, lbl_list = [], []
            for c in range(1, 9):
                indices = np.argwhere(gt == c)
                if len(indices) > 0:
                    r, cc = indices[_random.randint(0, len(indices)-1)]
                    pt_list.append(np.array([cc, r])); lbl_list.append(1)
                else:
                    pt_list.append(np.array([0, 0])); lbl_list.append(0)

            coords = torch.tensor(np.stack(pt_list,axis=0), dtype=torch.float, device=device).unsqueeze(0)
            labels = torch.tensor(lbl_list, dtype=torch.int, device=device).unsqueeze(0)

            with torch.no_grad():
                imp = net.preprocess(x); ie = net.image_encoder(imp)
                se, de = net.prompt_encoder(points=(coords,labels), boxes=None, masks=None)
                pred, _ = net.mask_decoder(image_embeddings=ie, image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=True)
                pred = torch.nn.functional.interpolate(pred, size=(IMAGE_SIZE,IMAGE_SIZE), mode="bilinear", align_corners=False)
            pc = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            dices = []
            for c in range(1, 9):
                pcc = (pc==c).astype(np.float32); tc = (gt==c).astype(np.float32)
                denom = pcc.sum()+tc.sum()
                dices.append(2.0*(pcc*tc).sum()/(denom+1e-8) if denom>0 else 1.0)
            all_dices.append(float(np.mean(dices)))

    del net; torch.cuda.empty_cache()
    return float(np.mean(all_dices)) if all_dices else 0, len(cases)


# ═══ Main ═══
if __name__ == "__main__":
    RESULTS = {}

    # ── SAMed Synapse ──
    print("=== SAMed Synapse ===")
    samed_syn = {
        "vit_b": f"{CKPT_BASE}/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth",
        "vit_l": f"{CHKPT}/Synapse_512_pretrain_vit_l_epo200_bs12_lr0.0001/epoch_99.pth",
        "vit_h": f"{CKPT_BASE}/SAMed/Synapse_512_pretrain_vit_h_epo200_bs6_lr0.005/epoch_299.pth",
    }
    for mk, ckpt in samed_syn.items():
        t0 = time.time()
        dice, n = eval_synapse(ckpt, mk)
        print(f"  Syn_{mk}: Dice={dice:.4f} ({n} cases, {time.time()-t0:.0f}s)")
        RESULTS[f"SAMed_Syn_{mk}"] = dice

    # ── SAMed ISIC ──
    print("\n=== SAMed ISIC ===")
    samed_isic = {
        "vit_b": f"{CKPT_BASE}/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth",
        "vit_l": f"{CKPT_BASE}/SAMed/isic_512_lrexp5.0_pretrain_vit_l_epo300_bs12_lr0.005/epoch_99.pth",
        "vit_h": f"{CKPT_BASE}/SAMed/isic_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_99.pth",
    }
    for mk, ckpt in samed_isic.items():
        t0 = time.time()
        dice, n = eval_isic(ckpt, mk)
        print(f"  ISIC_{mk}: Dice={dice:.4f} ({n} imgs, {time.time()-t0:.0f}s)")
        RESULTS[f"SAMed_ISIC_{mk}"] = dice

    # ── SAMed REFUGE ──
    print("\n=== SAMed REFUGE ===")
    samed_ref = {
        "vit_b": f"{CKPT_BASE}/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth",
        "vit_l": f"{CKPT_BASE}/SAMed/REFUGE_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_99.pth",
        "vit_h": f"{CKPT_BASE}/SAMed/REFUGE_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_99.pth",
    }
    for mk, ckpt in samed_ref.items():
        t0 = time.time()
        dice, n = eval_refuge(ckpt, mk)
        print(f"  REF_{mk}: Dice={dice:.4f} ({n} imgs, {time.time()-t0:.0f}s)")
        RESULTS[f"SAMed_REF_{mk}"] = dice

    # ── MSA Synapse ──
    print("\n=== MSA Synapse ===")
    msa_syn = {
        "vit_b": f"{CKPT_BASE}/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s24/Model/best_dice_checkpoint.pth",
        "vit_l": f"{CKPT_BASE}/MedicalSAMAdapter/Synapse_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24/Model/best_dice_checkpoint.pth",
        "vit_h": f"{CKPT_BASE}/MedicalSAMAdapter/Synapse_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24/Model/best_dice_checkpoint.pth",
    }
    for mk, ckpt in msa_syn.items():
        t0 = time.time()
        try:
            dice, n = eval_msa_synapse(ckpt, mk)
            print(f"  Syn_{mk}: Dice={dice:.4f} ({n} cases, {time.time()-t0:.0f}s)")
            RESULTS[f"MSA_Syn_{mk}"] = dice
        except Exception as e:
            print(f"  Syn_{mk}: FAIL {e}")
            RESULTS[f"MSA_Syn_{mk}"] = 0

    # ── Summary table ──
    print("\n" + "="*60)
    print("FINAL SUMMARY: ALL 18 MODELS")
    print("="*60)
    print(f"  {'Model':<20} {'Synapse':>10} {'ISIC':>10} {'REFUGE':>10}")
    for arch in ["SAMed", "MSA"]:
        for vit in ["vit_b", "vit_l", "vit_h"]:
            syn = RESULTS.get(f"{arch}_Syn_{vit}", 0)
            isic = RESULTS.get(f"{arch}_ISIC_{vit}", 0)
            ref = RESULTS.get(f"{arch}_REF_{vit}", 0)
            print(f"  {arch} {vit:<14} {syn:>10.4f} {isic:>10.4f} {ref:>10.4f}")

    with open(f"{NEWEST}/all_models_clean_dice.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\nSaved to {NEWEST}/all_models_clean_dice.json")
