#!/bin/bash
# CODD Full Experiment Launch — All 9 combinations × radial attack
# Usage: bash newest/launch_codd_full.sh
set -e

ROOT=/root/data2/fyc/fyc/GRAT-main
NEWEST=$ROOT/newest
mkdir -p $NEWEST/codd_master_results

echo "=========================================="
echo "CODD Full Experiment Pipeline"
echo "Phase 1: Attack Generation (parallel 6 GPUs)"
echo "Phase 2: Evaluation (all 18 victims)"
echo "=========================================="
echo ""

# ============================================================
# Phase 1: Generate adversarial images
# ============================================================
echo "=== Phase 1: Attack Generation ==="

declare -A GPU_MAP
GPU_MAP[0]=0
GPU_MAP[1]=1
GPU_MAP[2]=2
GPU_MAP[3]=3
GPU_MAP[4]=4
GPU_MAP[5]=5

task_id=0
for VIT in vit_b vit_l vit_h; do
    for DS in Synapse ISIC REFUGE; do
        GPU=$(( task_id % 6 ))
        LOG=$NEWEST/codd_master_results/attack_${VIT}_${DS}.log
        echo "  [$task_id] $VIT $DS radial → GPU $GPU (log: $LOG)"
        CUDA_VISIBLE_DEVICES=$GPU python $NEWEST/exp_codd_attack_gen.py \
            --vit $VIT --dataset $DS --gpu 0 --mode radial \
            --max_cases 32 > $LOG 2>&1 &
        task_id=$(( task_id + 1 ))
        sleep 1
    done
done

echo "  Waiting for $task_id attack jobs to complete..."
wait
echo "Phase 1 done."

# ============================================================
# Phase 2: Evaluate all victims
# ============================================================
echo ""
echo "=== Phase 2: Evaluation ==="

task_id=0
for VIT in vit_b vit_l vit_h; do
    for MODE in radial; do
        GPU=$(( task_id % 6 ))
        LOG=$NEWEST/codd_master_results/eval_${VIT}_${MODE}.log
        echo "  [$task_id] $VIT $MODE eval → GPU $GPU"
        CUDA_VISIBLE_DEVICES=$GPU python $NEWEST/eval_codd_unified.py \
            --vit $VIT --gpu 0 --mode $MODE --max_cases 16 \
            > $LOG 2>&1 &
        task_id=$(( task_id + 1 ))
        sleep 2
    done
done

echo "  Waiting for $task_id eval jobs to complete..."
wait
echo "Phase 2 done."

# ============================================================
# Phase 3: Compile all results
# ============================================================
echo ""
echo "=== Phase 3: Compile Results ==="
python $NEWEST/compile_codd_results.py \
    --input_dir $NEWEST/codd_master_results \
    --output $NEWEST/codd_master_results/codd_final_summary.json 2>&1
echo "Phase 3 done."
echo ""
echo "=== Pipeline Complete ==="
echo "Results: $NEWEST/codd_master_results/"
