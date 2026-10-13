#!/usr/bin/env bash
set -euo pipefail

err() { echo "[ERR] $*" >&2; exit 1; }
warn() { echo "[WARN] $*" >&2; }
ok() { echo "[OK] $*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM_CKPT_VIT_B="${SAM_CKPT_VIT_B:-$REPO_ROOT/checkpoints/sam_vit_b_01ec64.pth}"
SAM_CKPT_VIT_L="${SAM_CKPT_VIT_L:-$REPO_ROOT/checkpoints/sam_vit_l_0b3195.pth}"
SAM_CKPT_VIT_H="${SAM_CKPT_VIT_H:-$REPO_ROOT/checkpoints/sam_vit_h_4b8939.pth}"
SYNAPSE_TRAIN_NPZ="${SYNAPSE_TRAIN_NPZ:-$REPO_ROOT/data/Synapse_preprocess/train_npz}"
SYNAPSE_LIST_DIR="${SYNAPSE_LIST_DIR:-$REPO_ROOT/data/Synapse_preprocess/lists}"
ISIC_ROOT="${ISIC_ROOT:-$REPO_ROOT/data/ISIC}"
REFUGE_ROOT="${REFUGE_ROOT:-$REPO_ROOT/data/REFUGE-Multirater}"

for f in "$SAM_CKPT_VIT_B" "$SAM_CKPT_VIT_L" "$SAM_CKPT_VIT_H"; do
  [[ -f "$f" ]] || err "SAM checkpoint not found: $f"
done
ok "SAM checkpoints exist"

[[ -d "$SYNAPSE_TRAIN_NPZ" ]] || err "Synapse train_npz dir missing: $SYNAPSE_TRAIN_NPZ"
[[ -d "$SYNAPSE_LIST_DIR" ]] || err "Synapse list_dir missing: $SYNAPSE_LIST_DIR"
[[ -f "$SYNAPSE_LIST_DIR/train.txt" ]] || err "Missing train.txt: $SYNAPSE_LIST_DIR/train.txt"
[[ -f "$SYNAPSE_LIST_DIR/test_vol.txt" ]] || warn "Missing test_vol.txt: $SYNAPSE_LIST_DIR/test_vol.txt (needed for val/test)"
ok "Synapse lists exist"

first_id="$(grep -m1 -v '^\s*$' "$SYNAPSE_LIST_DIR/train.txt" || true)"
[[ -n "${first_id:-}" ]] || err "Synapse train.txt is empty"
npz_path="$SYNAPSE_TRAIN_NPZ/${first_id}.npz"
[[ -f "$npz_path" ]] || err "Synapse sample npz missing: $npz_path"
ok "Synapse sample npz exists: $npz_path"

[[ -d "$ISIC_ROOT" ]] || warn "ISIC_ROOT is not a directory: $ISIC_ROOT"
[[ -d "$REFUGE_ROOT" ]] || warn "REFUGE_ROOT is not a directory: $REFUGE_ROOT"

echo
ok "Basic checks passed. Next:"
echo "  - Run training via TRAINING.md or scripts/train_*.sh"

