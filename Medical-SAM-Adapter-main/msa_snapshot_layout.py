"""
Experiment folder naming aligned with SAMed_h/train.py snapshot_path rules
so that -out_root/<name>/ mirrors SAMed's output layout under --output/<name>/.
"""
import os

from conf import settings


def build_samed_style_experiment_dir(args) -> str:
    """
    Mirror SAMed_h/train.py:
      {dataset}_{img_size}[_pretrain]_{vit_name}[_XXk][_epoN][_bsN][_lrX][_sY]
    """
    dataset_name = args.dataset
    snapshot = f"{dataset_name}_{args.image_size}"
    snapshot += "_pretrain"
    snapshot += f"_{args.encoder}"

    max_iter = getattr(args, "max_iterations", 30000)
    if max_iter != 30000:
        snapshot += "_" + str(max_iter)[:2] + "k"

    max_epochs = getattr(args, "max_epochs", -1)
    if max_epochs is None or max_epochs < 0:
        max_epochs = settings.EPOCH
    if max_epochs != 30:
        snapshot += "_epo" + str(max_epochs)

    snapshot += "_bs" + str(args.b)

    base_lr = getattr(args, "base_lr", None)
    lr_for_name = base_lr if base_lr is not None else args.lr
    if abs(float(lr_for_name) - 0.01) > 1e-12:
        snapshot += "_lr" + str(lr_for_name)

    if args.seed != 1234:
        snapshot += "_s" + str(args.seed)

    return snapshot


def make_path_helper_samed_style(exp_root: str):
    """Same subdirs as set_log_dir (Model/Log/Samples) but under exp_root without timestamp."""
    os.makedirs(exp_root, exist_ok=True)
    ckpt_path = os.path.join(exp_root, "Model")
    log_path = os.path.join(exp_root, "Log")
    sample_path = os.path.join(exp_root, "Samples")
    for p in (ckpt_path, log_path, sample_path):
        os.makedirs(p, exist_ok=True)
    return {
        "prefix": exp_root,
        "ckpt_path": ckpt_path,
        "log_path": log_path,
        "sample_path": sample_path,
    }
