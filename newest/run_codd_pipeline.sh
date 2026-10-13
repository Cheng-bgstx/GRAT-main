#!/bin/bash
# CODD Master Experiment Pipeline
# ==================================
# Phase 1: Generate CODD adversarial images (parallel, 6 GPUs)
# Phase 2: Evaluate SAMed victims (test_samed.py)
# Phase 3: Evaluate MSA victims (Medical-SAM-Adapter-main/val.py)
# Phase 4: Compile results
#
# Usage: bash newest/run_codd_pipeline.sh [phase] [--dry-run]

set -e
ROOT=/root/data2/fyc/fyc/GRAT-main
NEWEST=$ROOT/newest
OUTDIR=$NEWEST/codd_master_results
mkdir -p $OUTDIR
cd $ROOT

# ============================================================
# Configuration
# ============================================================
VITS="vit_b vit_l vit_h"
DATASETS="ISIC REFUGE Synapse"
MODES="radial tangential isotropic misaligned"
# MODES_FULL="radial tangential isotropic misaligned"  # full ablation

# SAM checkpoints
declare -A SAM_CKPT
SAM_CKPT[vit_b]=$ROOT/checkpoints/sam_vit_b_01ec64.pth
SAM_CKPT[vit_l]=$ROOT/checkpoints/sam_vit_l_0b3195.pth
SAM_CKPT[vit_h]=$ROOT/checkpoints/sam_vit_h_4b8939.pth

# Victim checkpoints (symlinks from checkpoints/best_weights/)
BEST=$ROOT/checkpoints/best_weights

# ============================================================
# Phase 1: Generate adversarial images
# ============================================================
phase1() {
    echo "=== Phase 1: CODD Attack Generation ==="

    GPU=0
    for VIT in $VITS; do
        for DS in $DATASETS; do
            for MODE in $MODES; do
                echo "Launch: $VIT $DS $MODE on GPU $GPU"
                python $NEWEST/exp_codd_attack_gen.py \
                    --vit $VIT --dataset $DS --gpu $GPU --mode $MODE \
                    --max_cases 32 &
                GPU=$(( (GPU + 1) % 6 ))
                sleep 2  # stagger launches
            done
        done
    done
    wait
    echo "Phase 1 done."
}

phase1_quick() {
    echo "=== Phase 1 Quick: CODD Attack (radial only, all models) ==="

    GPU=0
    for VIT in $VITS; do
        for DS in $DATASETS; do
            echo "Launch: $VIT $DS radial on GPU $GPU"
            CUDA_VISIBLE_DEVICES=$GPU python $NEWEST/exp_codd_attack_gen.py \
                --vit $VIT --dataset $DS --gpu 0 --mode radial \
                --max_cases 32 &
            GPU=$(( (GPU + 1) % 6 ))
            sleep 3
            # Wait if all 6 GPUs are busy
            if [ $GPU -eq 0 ]; then wait; fi
        done
    done
    wait
    echo "Phase 1 Quick done."
}

# ============================================================
# Phase 2: Evaluate SAMed
# ============================================================
phase2_samed() {
    echo "=== Phase 2: SAMed Evaluation ==="

    for VIT in vit_b vit_l vit_h; do
        for DS in ISIC REFUGE Synapse; do
            # Find the checkpoint
            CKPT=$(readlink -f $BEST/SAMed_${DS}_${VIT}.pth 2>/dev/null || echo "")
            if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
                echo "SKIP SAMed $DS $VIT: no checkpoint"
                continue
            fi

            # Determine num_classes
            case $DS in
                Synapse) NCLS=8 ;;
                ISIC) NCLS=1 ;;
                REFUGE) NCLS=2 ;;
            esac

            # Clean evaluation
            echo "SAMed $DS $VIT: clean eval"
            python $ROOT/test_samed.py \
                --volume_path $ROOT/data/$DS \
                --dataset $DS \
                --num_classes $NCLS \
                --list_dir $ROOT/data/$DS \
                --output_dir $OUTDIR/samed_clean \
                --ckpt ${SAM_CKPT[$VIT]} \
                --lora_ckpt $CKPT \
                --vit_name $VIT \
                --out_path __clean__ \
                --max_cases 32 \
                2>&1 | tee $OUTDIR/eval_${DS}_${VIT}_SAMed_clean.log

            # Radial attack evaluation
            ADV_DIR=$ROOT/testset/codd_attack/${VIT}_${DS}_radial
            if [ -d "$ADV_DIR" ]; then
                echo "SAMed $DS $VIT: radial adv eval → $ADV_DIR"
                python $ROOT/test_samed.py \
                    --volume_path $ROOT/data/$DS \
                    --dataset $DS \
                    --num_classes $NCLS \
                    --list_dir $ROOT/data/$DS \
                    --output_dir $OUTDIR/samed_radial \
                    --ckpt ${SAM_CKPT[$VIT]} \
                    --lora_ckpt $CKPT \
                    --vit_name $VIT \
                    --out_path $ADV_DIR \
                    --max_cases 32 \
                    2>&1 | tee $OUTDIR/eval_${DS}_${VIT}_SAMed_radial.log
            fi
        done
    done
    echo "Phase 2 done."
}

# ============================================================
# Phase 3: Evaluate MSA
# ============================================================
phase3_msa() {
    echo "=== Phase 3: MSA Evaluation ==="
    MSA_ROOT=$ROOT/Medical-SAM-Adapter-main

    for VIT in vit_b vit_l vit_h; do
        for DS in ISIC REFUGE Synapse; do
            CKPT=$(readlink -f $BEST/MSA_${DS}_${VIT}.pth 2>/dev/null || echo "")
            if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
                echo "SKIP MSA $DS $VIT: no checkpoint"
                continue
            fi

            case $DS in
                Synapse) MM_OUT=9 ;;
                ISIC|REFUGE) MM_OUT=1 ;;
            esac

            # Clean
            echo "MSA $DS $VIT: clean eval"
            cd $MSA_ROOT
            python val.py \
                --net sam \
                --encoder $VIT \
                --dataset $DS \
                --weights $CKPT \
                --sam_ckpt ${SAM_CKPT[$VIT]} \
                --gpu_device 0 \
                --batch_size 1 \
                --multimask_output $MM_OUT \
                --clean_eval \
                2>&1 | tee $OUTDIR/eval_${DS}_${VIT}_MSA_clean.log
            cd $ROOT

            # Radial attack
            ADV_DIR=$ROOT/testset/codd_attack/${VIT}_${DS}_radial
            if [ -d "$ADV_DIR" ]; then
                echo "MSA $DS $VIT: radial adv eval"
                cd $MSA_ROOT
                python val.py \
                    --net sam \
                    --encoder $VIT \
                    --dataset $DS \
                    --weights $CKPT \
                    --sam_ckpt ${SAM_CKPT[$VIT]} \
                    --gpu_device 0 \
                    --batch_size 1 \
                    --multimask_output $MM_OUT \
                    --adv_dir $ADV_DIR \
                    2>&1 | tee $OUTDIR/eval_${DS}_${VIT}_MSA_radial.log
                cd $ROOT
            fi
        done
    done
    echo "Phase 3 done."
}

# ============================================================
# Phase 4: Compile results
# ============================================================
phase4_compile() {
    echo "=== Phase 4: Compiling Results ==="
    python $NEWEST/compile_codd_results.py --input_dir $OUTDIR --output $OUTDIR/codd_final_summary.json
    echo "Phase 4 done."
}

# ============================================================
# Main
# ============================================================
PHASE=${1:-all}

case $PHASE in
    all)
        phase1_quick
        ;;
    phase1)
        phase1_quick
        ;;
    phase1_full)
        phase1
        ;;
    phase2)
        phase2_samed
        ;;
    phase3)
        phase3_msa
        ;;
    phase4)
        phase4_compile
        ;;
    dry-run)
        echo "Dry run — would launch:"
        echo "  9 experiments × 4 modes = 36 attack jobs"
        echo "  18 SAMed evaluations + 18 MSA evaluations"
        ;;
    *)
        echo "Usage: $0 {all|phase1|phase1_full|phase2|phase3|phase4|dry-run}"
        ;;
esac
