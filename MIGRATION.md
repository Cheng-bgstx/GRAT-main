# Migration Notes

This repository is prepared for a split migration:

- GitHub: source code, docs, configs, scripts, and small reference assets.
- Hugging Face Hub: reusable model weights, checkpoints, offline priors, and an asset manifest.

## Why weights should not go to normal GitHub

The current working tree contains files far above GitHub's normal Git object limit, including multi-GB checkpoints and archives. GitHub can store large files through Git LFS, but this project already exceeds the size where Hugging Face Hub is the more practical target for checkpoints and research artifacts.

The `.gitignore` keeps these large assets out of the GitHub source repository. The `.gitattributes` file still marks the common large binary formats for Git LFS, so intentional future tracking uses LFS instead of normal Git objects.

## Largest local assets found

Examples from the local scan:

- `robot/pace_gen_transfer.tar.gz` around 7.9 GB
- `checkpoints/sam_vit_h_4b8939.pth` around 2.6 GB
- `checkpoints/sam_vit_l_0b3195.pth` around 1.2 GB
- `data/SA-1B/sa_000000.tar` through `sa_000002.tar` around 800 MB each
- many `testset/**/*.h5`, `data/**/*.h5`, `results/**/*.nii.gz`, and `output/**/*.pth` artifacts

By default, do not upload `data/`, `testset/`, `output_runs_correct/`, `results/`, or `output/`. They are large raw data, generated adversarial samples, predictions, logs, and historical training outputs. Keep them local unless a specific experiment requires publishing them and the dataset licenses allow redistribution.

The default Hugging Face upload script includes only the 18 reusable SAMed and Medical-SAM-Adapter weights:

- `migration_assets.tsv`
- 9 SAMed checkpoints from `output_runs_correct/SAMed/`
- 9 Medical-SAM-Adapter `best_dice_checkpoint.pth` files from `output_runs_correct/MedicalSAMAdapter/`

Generate a full local manifest with:

```bash
find . -type f \( -name '*.pth' -o -name '*.pt' -o -name '*.ckpt' -o -name '*.safetensors' -o -name '*.bin' -o -name '*.h5' -o -name '*.hdf5' -o -name '*.npy' -o -name '*.npz' -o -name '*.nii' -o -name '*.nii.gz' -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.zip' \) -printf '%s\t%p\n' | sort -nr > migration_assets.tsv
```

## GitHub push

After creating an empty GitHub repository:

```bash
git remote add origin git@github.com:<owner>/<repo>.git
git push -u origin main
```

If HTTPS is preferred:

```bash
git remote add origin https://github.com/<owner>/<repo>.git
git push -u origin main
```

## Hugging Face upload

Log in first:

```bash
hf auth login
```

Then upload large assets with:

```bash
scripts/upload_large_assets_to_hf.sh <namespace>/<dataset-or-model-repo> dataset
```

To upload bulk generated data as well, opt in explicitly:

```bash
INCLUDE_BULK=1 scripts/upload_large_assets_to_hf.sh <namespace>/<dataset-or-model-repo> dataset
```

Use `model` instead of `dataset` if the target repository should be a model repository.
