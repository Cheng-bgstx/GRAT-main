#!/usr/bin/env bash
set -euo pipefail

repo_id="${1:?usage: scripts/upload_large_assets_to_hf.sh <namespace/repo> [dataset|model]}"
repo_type="${2:-dataset}"
token_args=()

if [ -n "${HF_TOKEN:-}" ]; then
  token_args=(--token "$HF_TOKEN")
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI is required. Install or activate huggingface_hub first." >&2
  exit 1
fi

if ! hf auth whoami "${token_args[@]}" >/dev/null 2>&1; then
  echo "Hugging Face CLI is not logged in. Run: hf auth login, or set HF_TOKEN." >&2
  exit 1
fi

hf repo create "$repo_id" --type "$repo_type" --exist-ok "${token_args[@]}"

upload_path() {
  local path="$1"
  if [ -e "$path" ]; then
    hf upload "$repo_id" "$path" "$path" --repo-type "$repo_type" "${token_args[@]}"
  fi
}

upload_path migration_assets.tsv
upload_path checkpoints
upload_path common_perb
upload_path ddta
upload_path newest/offline_priors

find new -maxdepth 1 -type f -name '*.pth' -print0 2>/dev/null | while IFS= read -r -d '' path; do
  upload_path "$path"
done

if [ "${INCLUDE_BULK:-0}" = "1" ]; then
  upload_path data
  upload_path testset
  upload_path results
  upload_path output
  upload_path output_runs_correct
  upload_path adv_outputs
  upload_path new/test_runs
  upload_path newest/output
  upload_path robot/pace_gen_transfer.tar.gz
fi

find . -maxdepth 3 -type f \( \
  -name '*.pth' -o -name '*.pt' -o -name '*.ckpt' -o -name '*.safetensors' -o \
  -name '*.bin' -o -name '*.h5' -o -name '*.hdf5' -o -name '*.npy' -o \
  -name '*.npz' -o -name '*.nii' -o -name '*.nii.gz' -o -name '*.tar' -o \
  -name '*.tar.gz' -o -name '*.tgz' -o -name '*.zip' \
\) -print0 | while IFS= read -r -d '' path; do
  rel="${path#./}"
  case "$rel" in
    checkpoints/*|data/*|testset/*|results/*|output/*|output_runs_correct/*|adv_outputs/*|common_perb/*|ddta/*|new/test_runs/*|newest/output/*|newest/offline_priors/*|robot/pace_gen_transfer.tar.gz|new/*.pth)
      ;;
    *)
      hf upload "$repo_id" "$rel" "$rel" --repo-type "$repo_type" "${token_args[@]}"
      ;;
  esac
done
