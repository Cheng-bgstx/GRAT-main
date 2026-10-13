#!/usr/bin/env python3
"""
PASD Comprehensive Transfer Attack Experiment.

CRITICAL: Attack generated on BASE SAM encoder (no LoRA/Adapter).
          Evaluated on VICTIM model (SAMed with LoRA, MSA with Adapter).

Main table: PASD vs baselines across all 18 models × 3 budgets.
Ablation:  PASD component ablation on Synapse SAMed models.
"""

import os, sys, time, json, gc
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder, load_samed_model, load_medsa_model
from attack_pasa import run_pasd_attack, PASD_MODES

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPTS = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}

device = torch.device("cuda:0")
IM_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD_CPU = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ── Model registries ─────────────────────────────────────────────────────────

SAMED_MODELS = {
    "Syn_vit_b": {"vit":"vit_b","ckpt":"SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth","n_class":8,"ds":"Synapse"},
    "Syn_vit_l": {"vit":"vit_l","ckpt":"SAMed/Synapse_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_249.pth","n_class":8,"ds":"Synapse"},
    "Syn_vit_h": {"vit":"vit_h","ckpt":"SAMed/Synapse_512_pretrain_vit_h_epo200_bs6_lr0.005/epoch_299.pth","n_class":8,"ds":"Synapse"},
    "ISIC_vit_b":{"vit":"vit_b","ckpt":"SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth","n_class":1,"ds":"ISIC"},
    "ISIC_vit_l":{"vit":"vit_l","ckpt":"SAMed/isic_512_lrexp5.0_pretrain_vit_l_epo300_bs12_lr0.005/epoch_59.pth","n_class":1,"ds":"ISIC"},
    "ISIC_vit_h":{"vit":"vit_h","ckpt":"SAMed/isic_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_249.pth","n_class":1,"ds":"ISIC"},
    "REF_vit_b": {"vit":"vit_b","ckpt":"SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_199.pth","n_class":2,"ds":"REFUGE"},
    "REF_vit_l": {"vit":"vit_l","ckpt":"SAMed/REFUGE_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_299.pth","n_class":2,"ds":"REFUGE"},
    "REF_vit_h": {"vit":"vit_h","ckpt":"SAMed/REFUGE_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_299.pth","n_class":2,"ds":"REFUGE"},
}

MSA_MODELS = {
    "MSA_ISIC_vit_b":{"vit":"vit_b","dir":"isic_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24","ds":"ISIC","mm":9},
    "MSA_ISIC_vit_l":{"vit":"vit_l","dir":"isic_512_pretrain_vit_l_epo100_bs16_lr0.0001_s24","ds":"ISIC","mm":9},
    "MSA_ISIC_vit_h":{"vit":"vit_h","dir":"isic_512_pretrain_vit_h_epo100_bs8_lr0.0001_s24","ds":"ISIC","mm":9},
    "MSA_Syn_vit_b": {"vit":"vit_b","dir":"Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s24","ds":"Synapse","mm":9},
    "MSA_Syn_vit_l": {"vit":"vit_l","dir":"Synapse_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24","ds":"Synapse","mm":9},
    "MSA_Syn_vit_h": {"vit":"vit_h","dir":"Synapse_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24","ds":"Synapse","mm":9},
    "MSA_REF_vit_b": {"vit":"vit_b","dir":"REFUGE_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24","ds":"REFUGE","mm":9},
    "MSA_REF_vit_l": {"vit":"vit_l","dir":"REFUGE_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24","ds":"REFUGE","mm":9},
    "MSA_REF_vit_h": {"vit":"vit_h","dir":"REFUGE_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24","ds":"REFUGE","mm":9},
}

# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(dataset_name, n=12):
    if dataset_name == "Synapse":
        from exp_paid_optimal_attack import load_synapse_slices
        return load_synapse_slices(n)
    elif dataset_name == "ISIC":
        tdir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_Data"
        mdir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_GroundTruth"
        jpgs = sorted([f for f in os.listdir(tdir) if f.endswith('.jpg')])[:n]
        imgs = []
        for jpg in jpgs:
            img = Image.open(os.path.join(tdir, jpg)).convert('RGB').resize((512,512), Image.BILINEAR)
            img_np = np.array(img).astype(np.float32)
            base = jpg.replace('.jpg','')
            mask = Image.open(os.path.join(mdir, base+'_Segmentation.png')).convert('L')
            mask = mask.resize((512,512), Image.NEAREST)
            gt = (np.array(mask) > 128).astype(np.float32)
            if gt.sum() < 100: continue
            img_t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float()
            img_t = (img_t/255.0 - IM_MEAN_CPU) / IM_STD_CPU
            imgs.append((img_t, gt))
        return imgs
    elif dataset_name == "REFUGE":
        from exp_paid_optimal_attack import load_refuge_images
        return load_refuge_images(n)
    raise ValueError(f"Unknown dataset: {dataset_name}")

# ── Evaluation functions ─────────────────────────────────────────────────────

def evaluate_samed(net, img_t, gt, ds):
    with torch.no_grad():
        out = net(img_t, multimask_output=False, image_size=IMAGE_SIZE)
        m = out['masks']
    if ds == "Synapse":
        pred = torch.argmax(torch.softmax(m.float(), dim=1), dim=1).squeeze(0).cpu().numpy()
        dices = []
        for c in range(1, 8):
            pc = (pred == c).astype(np.float32)
            gc = (gt == c).astype(np.float32)
            if gc.sum() == 0: continue
            inter = (pc * gc).sum()
            dices.append(float(2.0*inter/(pc.sum()+gc.sum()+1e-8)))
        return float(np.mean(dices)) if dices else 0.0
    elif ds == "ISIC":
        probs = torch.softmax(m.float(), dim=1)
        lp = probs[0,1,:,:].cpu().numpy()
        best = 0.0
        for th in [0.1,0.3,0.5,0.7,0.9]:
            pb = (lp > th).astype(np.float32)
            inter = (pb*gt).sum()
            dice = float(2.0*inter/(pb.sum()+gt.sum()+1e-8))
            if dice > best: best = dice
        return best
    elif ds == "REFUGE":
        probs = torch.softmax(m.float(), dim=1)
        cup_gt, disc_gt = gt[0], gt[1]
        dices = []
        for ci, gm in [(2, cup_gt), (1, disc_gt)]:
            cp = probs[0,ci,:,:].cpu().numpy()
            best = 0.0
            for th in [0.1,0.3,0.5,0.7,0.9]:
                pb = (cp>th).astype(np.float32)
                inter = (pb*gm).sum()
                dice = float(2.0*inter/(pb.sum()+gm.sum()+1e-8))
                if dice > best: best = dice
            dices.append(best)
        return float(np.mean(dices))

def evaluate_msa(model, img_t, gt, ds):
    import random as _random
    _random.seed(42)
    if ds == "ISIC":
        msk = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0).to(device)
        indices = torch.nonzero(msk[0,0])
        if indices.size(0)==0:
            pt = torch.tensor([[0.0,0.0]], device=device)
            lp = torch.tensor([[0]], dtype=torch.int, device=device)
        else:
            com = indices.float().mean(dim=0)
            pt = com.flip(0).unsqueeze(0).unsqueeze(0)
            lp = torch.tensor([[1]], dtype=torch.int, device=device)
        im = img_t.to(dtype=torch.float32, device=device)
        with torch.no_grad():
            imp = model.preprocess(im)
            ie = model.image_encoder(imp)
            se, de = model.prompt_encoder(points=(pt.float(),lp), boxes=None, masks=None)
            pred,_ = model.mask_decoder(image_embeddings=ie, image_pe=model.prompt_encoder.get_dense_pe(),
                                        sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=False)
            pred = F.interpolate(pred, size=(IMAGE_SIZE,IMAGE_SIZE), mode="bilinear", align_corners=False)
        probs = torch.sigmoid(pred).detach().cpu().numpy()
        best = 0.0
        for th in [0.1,0.3,0.5,0.7,0.9]:
            pb = (probs[0,0]>th).astype(np.float32)
            inter = (pb*gt).sum()
            dice = float(2.0*inter/(pb.sum()+gt.sum()+1e-8))
            if dice > best: best = dice
        return best
    elif ds == "Synapse":
        pt_list, lbl_list = [], []
        for c in range(1,9):
            indices = np.argwhere(gt==c)
            if len(indices)>0:
                r,cc = indices[_random.randint(0,len(indices)-1)]
                pt_list.append(np.array([cc,r]))
                lbl_list.append(1)
            else:
                pt_list.append(np.array([0,0]))
                lbl_list.append(0)
        coords = torch.tensor(np.stack(pt_list,axis=0), dtype=torch.float, device=device).unsqueeze(0)
        labels = torch.tensor(lbl_list, dtype=torch.int, device=device).unsqueeze(0)
        im = img_t.to(dtype=torch.float32, device=device)
        with torch.no_grad():
            imp = model.preprocess(im)
            ie = model.image_encoder(imp)
            se, de = model.prompt_encoder(points=(coords,labels), boxes=None, masks=None)
            pred,_ = model.mask_decoder(image_embeddings=ie, image_pe=model.prompt_encoder.get_dense_pe(),
                                        sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=True)
            pred = F.interpolate(pred, size=(IMAGE_SIZE,IMAGE_SIZE), mode="bilinear", align_corners=False)
        pc = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
        dices = []
        for c in range(1,9):
            pcc = (pc==c).astype(np.float32)
            tc = (gt==c).astype(np.float32)
            denom = pcc.sum()+tc.sum()
            dices.append(2.0*(pcc*tc).sum()/(denom+1e-8) if denom>0 else 1.0)
        return float(np.mean(dices))
    elif ds == "REFUGE":
        pts = torch.tensor([[[0.0,0.0]]], device=device)
        pl = torch.tensor([[1]], dtype=torch.int, device=device)
        im = img_t.to(dtype=torch.float32, device=device)
        with torch.no_grad():
            imp = model.preprocess(im)
            ie = model.image_encoder(imp)
            se, de = model.prompt_encoder(points=(pts,pl), boxes=None, masks=None)
            pred,_ = model.mask_decoder(image_embeddings=ie, image_pe=model.prompt_encoder.get_dense_pe(),
                                        sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=False)
            pred = F.interpolate(pred, size=(IMAGE_SIZE,IMAGE_SIZE), mode="bilinear", align_corners=False)
        probs = torch.sigmoid(pred[0])
        cup_p = (probs[0]>0.5).float().cpu().numpy()
        disc_p = (probs[1]>0.5).float().cpu().numpy()
        cup_g = (gt[0]>0.5).astype(np.float32)
        disc_g = (gt[1]>0.5).astype(np.float32)
        dc = 2.0*(cup_p*cup_g).sum()/(cup_p.sum()+cup_g.sum()+1e-8)
        dd = 2.0*(disc_p*disc_g).sum()/(disc_p.sum()+disc_g.sum()+1e-8)
        return float((dc+dd)/2.0)

# ── Baseline attacks ─────────────────────────────────────────────────────────

def attack_mifgsm_pixel(net, x, eps_pix, step_pix, steps, ds, n_class, model_type):
    """MI-FGSM: pixel-space attack maximizing Dice degradation."""
    from attack_codd import _get_edge_map
    max_val, min_val = x.max(), x.min()
    x_adv = x.clone().detach() + torch.randn_like(x)*1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)

    # Get clean prediction
    with torch.no_grad():
        if model_type == "samed":
            clean_out = net(x, multimask_output=False, image_size=IMAGE_SIZE)
            clean_probs = torch.softmax(clean_out['masks'].float(), dim=1).detach()
        else:
            imp = net.preprocess(x)
            clean_feats = net.image_encoder(imp).detach()

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        if model_type == "samed":
            out = net(x_adv, multimask_output=False, image_size=IMAGE_SIZE)
            probs = torch.softmax(out['masks'].float(), dim=1)
            if ds == "Synapse":
                loss = 0.0
                for c in range(1, n_class+1):
                    inter = (probs[:,c:c+1]*clean_probs[:,c:c+1]).sum()
                    denom = probs[:,c:c+1].sum()+clean_probs[:,c:c+1].sum()+1e-8
                    loss = loss + 2.0*inter/denom
                loss = -loss/n_class
            elif ds == "ISIC":
                inter = (probs[:,1:2]*clean_probs[:,1:2]).sum()
                denom = probs[:,1:2].sum()+clean_probs[:,1:2].sum()+1e-8
                loss = -2.0*inter/denom
            elif ds == "REFUGE":
                loss = 0.0
                for c in [1,2]:
                    inter = (probs[:,c:c+1]*clean_probs[:,c:c+1]).sum()
                    denom = probs[:,c:c+1].sum()+clean_probs[:,c:c+1].sum()+1e-8
                    loss = loss + 2.0*inter/denom
                loss = -loss/2
        else:
            imp = net.preprocess(x_adv)
            feats = net.image_encoder(imp)
            loss = F.l1_loss(feats, clean_feats)

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0+3.0*edge_map)
        grad = grad/(torch.norm(grad, p=1)+1e-12)
        momentum = 0.5*momentum+grad
        x_adv = x_adv.detach()+step_pix*momentum.sign()
        eta = torch.clamp(x_adv-x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x+eta, min_val, max_val).detach()
    return x_adv.detach()


def attack_random(x, eps_pix):
    noise = torch.randn_like(x)
    noise = noise/(noise.abs().max()+1e-8)*eps_pix
    return torch.clamp(x+noise, x.min(), x.max())


# ── Attack Encoder Cache ─────────────────────────────────────────────────────

_attack_encoder_cache = {}

def get_attack_encoder(vit_name):
    """Get BASE SAM encoder (no LoRA) for attack generation. Cached per vit."""
    cache_key = vit_name
    if cache_key not in _attack_encoder_cache:
        sam = load_sam_encoder(vit_name, SAM_CKPTS[vit_name], IMAGE_SIZE, num_classes=8, device=device)
        _attack_encoder_cache[cache_key] = sam.image_encoder
        _attack_encoder_cache[cache_key].eval()
    return _attack_encoder_cache[cache_key]


# ── Main experiment ──────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Methods for main comparison
MAIN_METHODS = ["pasd", "clean_phase", "white_amp", "secgr", "none"]
# Methods for ablation (full set)
ABLATION_METHODS = ["pasd", "clean_phase", "white_amp", "secgr", "random_phase", "no_gaussian", "none"]

EPS_LIST = [4.0/255, 8.0/255, 16.0/255]
STEPS = 20
N_IMAGES = 12

# PASD hyperparameters
PASD_SIGMA = 1.0
PASD_ALPHA = 1.0
PASD_AMP = 2.0
PASD_K = 2
PASD_LAYERS = 6

MAIN_OUTPUT = os.path.join(OUTPUT_DIR, "pasd_main_results.json")
ABLATION_OUTPUT = os.path.join(OUTPUT_DIR, "pasd_ablation_results.json")


def run_transfer_experiment(model_dict, model_type, methods, output_path, label,
                             limit_models=None):
    """
    Run transfer attack experiment.

    Attack generated on BASE SAM encoder (no LoRA/Adapter).
    Evaluated on VICTIM model (SAMed with LoRA / MSA with Adapter).
    """
    all_results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} prior results from {output_path}")

    completed = set()
    for r in all_results:
        completed.add((r["model"], r["eps"], r.get("steps", STEPS), r["method"]))

    models_to_run = model_dict
    if limit_models:
        models_to_run = {k: v for k, v in model_dict.items() if k in limit_models}

    for model_key in sorted(models_to_run.keys()):
        cfg = models_to_run[model_key]
        ds = cfg["ds"]
        vit = cfg["vit"]

        # Skip if all combos done
        pending = False
        for eps_frac in EPS_LIST:
            for mk in methods:
                if (model_key, f"{eps_frac*255:.0f}/255", STEPS, mk) not in completed:
                    pending = True
                    break
            if pending: break
        if not pending:
            print(f"  [{label}] SKIP {model_key}: all combos complete")
            continue

        print(f"\n{'='*80}")
        print(f"  [{label}] {model_key} | {vit} | {ds} | victim={model_type}")
        print(f"{'='*80}")

        # ── Load VICTIM for evaluation only ──
        try:
            if model_type == "samed":
                victim = load_samed_model(vit, SAM_CKPTS[vit],
                    os.path.join(CKPT_BASE, cfg["ckpt"]),
                    IMAGE_SIZE, cfg["n_class"], device=device)
                n_class = cfg["n_class"]
            elif model_type == "msa":
                msa_dir = os.path.join(CKPT_BASE, "MedicalSAMAdapter", cfg["dir"])
                msa_ckpt = os.path.join(msa_dir, "Model", "best_dice_checkpoint.pth")
                victim = load_medsa_model(vit, SAM_CKPTS[vit],
                    msa_ckpt, IMAGE_SIZE, multimask_output=cfg["mm"], device=device)
                n_class = None
            victim.eval()
        except Exception as e:
            print(f"  SKIP: Victim load failed: {e}")
            continue

        # ── Load BASE SAM encoder for attack generation ──
        try:
            attack_encoder = get_attack_encoder(vit)
        except Exception as e:
            print(f"  SKIP: Attack encoder load failed: {e}")
            del victim; torch.cuda.empty_cache(); continue

        # ── Load data ──
        images = load_data(ds, n=N_IMAGES)
        print(f"  Loaded {len(images)} images")

        for eps_frac in EPS_LIST:
            ep_str = f"{eps_frac*255:.0f}/255"
            for method_key in methods:
                combo_key = (model_key, ep_str, STEPS, method_key)
                if combo_key in completed:
                    continue

                dices, clean_dices, times = [], [], []

                for img_data in images:
                    # Parse image
                    if ds == "Synapse":
                        x = torch.from_numpy(img_data[0]).unsqueeze(0).float().to(device)
                        gt = img_data[1]
                    elif ds == "ISIC":
                        x = img_data[0].to(device)
                        gt = img_data[1]
                    elif ds == "REFUGE":
                        x = img_data[0].unsqueeze(0).float().to(device)
                        gt = img_data[1]

                    rg = x.max().item() - x.min().item()
                    eps_pix = eps_frac * rg
                    step_pix = 2.0 / 255.0 * rg  # Fixed step size matching original SE-CGR (NOT eps/steps)

                    # Clean evaluation on VICTIM
                    if model_type == "samed":
                        clean_dice = evaluate_samed(victim, x, gt, ds)
                    else:
                        clean_dice = evaluate_msa(victim, x, gt, ds)
                    if clean_dice < 0.01:
                        continue

                    try:
                        t0 = time.time()

                        if method_key in PASD_MODES:
                            pm = PASD_MODES[method_key]["probe_mode"]
                            x_adv = run_pasd_attack(
                                image_encoder=attack_encoder,
                                input_image=x,
                                eps_pix=eps_pix, step_pix=step_pix, steps=STEPS,
                                sigma=PASD_SIGMA, alpha=PASD_ALPHA,
                                probe_amplification=PASD_AMP, K=PASD_K,
                                probe_mode=pm, num_layers=PASD_LAYERS,
                            )
                        elif method_key == "mifgsm":
                            x_adv = attack_mifgsm_pixel(victim, x, eps_pix, step_pix, STEPS, ds, n_class, model_type)
                        elif method_key == "random":
                            x_adv = attack_random(x, eps_pix)
                        else:
                            raise ValueError(f"Unknown method: {method_key}")

                        # Adversarial evaluation on VICTIM
                        if model_type == "samed":
                            adv_dice = evaluate_samed(victim, x_adv, gt, ds)
                        else:
                            adv_dice = evaluate_msa(victim, x_adv, gt, ds)

                        times.append(time.time() - t0)
                        dices.append(adv_dice)
                        clean_dices.append(clean_dice)
                    except Exception as e:
                        print(f"    Error [{method_key}]: {e}")
                        continue

                    del x_adv
                    torch.cuda.empty_cache()

                if len(dices) == 0:
                    print(f"  eps={ep_str} {method_key:<22} NO VALID RESULTS")
                    continue

                mc = float(np.mean(clean_dices))
                ma = float(np.mean(dices))
                md = mc - ma
                mt = float(np.mean(times))

                label_m = PASD_MODES[method_key]["label"] if method_key in PASD_MODES else method_key

                all_results.append({
                    "model": model_key, "dataset": ds, "vit": vit,
                    "model_type": model_type,
                    "eps": ep_str, "steps": STEPS,
                    "method": method_key, "method_label": label_m,
                    "mean_clean": mc, "mean_adv": ma,
                    "mean_delta": md, "mean_time": mt, "n_valid": len(dices),
                })

                print(f"  eps={ep_str} {label_m:<28}"
                      f"Clean={mc:.4f} Adv={ma:.4f} Δ={md:+.4f} ({mt:.1f}s) [{len(dices)}]")

                # Save incrementally
                with open(output_path, "w") as f:
                    json.dump(all_results, f, indent=2)

        del victim
        torch.cuda.empty_cache()
        gc.collect()

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[{label}] Saved {len(all_results)} results to {output_path}")
    return all_results


def print_summary(results, title):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    for eps_str in sorted(set(r["eps"] for r in results)):
        n_steps_set = sorted(set(r.get("steps", STEPS) for r in results))
        for n_steps in n_steps_set:
            print(f"\n── eps={eps_str}, steps={n_steps} ──")
            methods_seen = sorted(set(r["method"] for r in results
                                      if r["eps"]==eps_str and r.get("steps",STEPS)==n_steps))
            models_seen = sorted(set(r["model"] for r in results
                                    if r["eps"]==eps_str and r.get("steps",STEPS)==n_steps))
            # Header
            header = f"  {'Model':<22}"
            for m in methods_seen:
                label = PASD_MODES[m]["label"] if m in PASD_MODES else m
                header += f" {label:>14}"
            print(header)
            print("  " + "-"*(len(header)-2))
            for mk in models_seen:
                row = f"  {mk:<22}"
                best_val, best_m = -999, ""
                for m in methods_seen:
                    entries = [r for r in results
                              if r["model"]==mk and r["eps"]==eps_str
                              and r.get("steps",STEPS)==n_steps and r["method"]==m]
                    md = entries[0]["mean_delta"] if entries else float("nan")
                    row += f" {md:>+14.4f}"
                    if md > best_val:
                        best_val, best_m = md, m
                # Mark winner
                row += f"  ← {PASD_MODES[best_m]['label'] if best_m in PASD_MODES else best_m}"
                print(row)
            # Average
            row = f"  {'AVERAGE':<22}"
            for m in methods_seen:
                vals = [r["mean_delta"] for r in results
                       if r["eps"]==eps_str and r.get("steps",STEPS)==n_steps
                       and r["method"]==m]
                avg = np.mean(vals) if vals else float("nan")
                row += f" {avg:>+14.4f}"
            print("  " + "-"*(len(row)-2))
            print(row)


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=str, default="all",
                    choices=["all","main_samed","main_msa","ablation","summary"])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--n_images", type=int, default=N_IMAGES)
    args = ap.parse_args()

    device = torch.device(args.device)
    N_IMAGES = args.n_images

    # ── Phase 1: Main comparison ──
    if args.phase in ("all", "main_samed"):
        print("\n" + "="*90)
        print("PHASE 1a: MAIN — SAMed Models (base SAM → SAMed transfer)")
        print("="*90)
        run_transfer_experiment(SAMED_MODELS, "samed", MAIN_METHODS,
                                MAIN_OUTPUT, "SAMed")

    if args.phase in ("all", "main_msa"):
        print("\n" + "="*90)
        print("PHASE 1b: MAIN — MSA Models (base SAM → MSA transfer)")
        print("="*90)
        run_transfer_experiment(MSA_MODELS, "msa", MAIN_METHODS,
                                MAIN_OUTPUT, "MSA")

    # ── Phase 2: Ablation ──
    ABLATION_MODEL_KEYS = ["Syn_vit_b", "Syn_vit_l", "Syn_vit_h"]
    if args.phase in ("all", "ablation"):
        print("\n" + "="*90)
        print("PHASE 2: ABLATION — SAMed Synapse Models")
        print("="*90)
        run_transfer_experiment(SAMED_MODELS, "samed", ABLATION_METHODS,
                                ABLATION_OUTPUT, "Ablation",
                                limit_models=ABLATION_MODEL_KEYS)

    # ── Summary ──
    if args.phase in ("all", "main_samed", "main_msa", "summary"):
        main_results = []
        if os.path.exists(MAIN_OUTPUT):
            with open(MAIN_OUTPUT) as f:
                main_results = json.load(f)
        if main_results:
            print_summary(main_results, "MAIN RESULTS")

    if args.phase in ("all", "ablation", "summary"):
        abl_results = []
        if os.path.exists(ABLATION_OUTPUT):
            with open(ABLATION_OUTPUT) as f:
                abl_results = json.load(f)
        if abl_results:
            print_summary(abl_results, "ABLATION RESULTS")

    print("\nDone — PASD experiment complete.")
