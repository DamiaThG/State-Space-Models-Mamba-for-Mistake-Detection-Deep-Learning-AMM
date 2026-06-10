#!/bin/bash
# ============================================================
# SLURM Job Script — TempAgg Baseline Training
# Cluster: GCluster (gcluster.dmi.unict.it)
# Sottometti con: sbatch scripts/train_baseline.sh
#
# NOTA: Questo script è il wrapper SLURM+Apptainer.
# La logica di training è in src/training/train_baseline.py.
# ============================================================

#SBATCH --job-name=tempagg_baseline
#SBATCH --output=experiments/logs/tempagg_%j.out
#SBATCH --error=experiments/logs/tempagg_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --gres=shard:22000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ---------- Ambiente ----------
echo "================================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "Start     : $(date)"
echo "Working   : $(pwd)"
echo "================================================"

# ---------- Logging dir ----------
mkdir -p experiments/logs
mkdir -p experiments/checkpoints

# ---------- Lancio nel container ----------
# La baseline usa l'immagine di default (latest.sif)
APPTAINER_IMAGE="/shared/sifs/latest.sif"

apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    $APPTAINER_IMAGE \
    bash -c "
        export WANDB_MODE=offline
        export PYTHONUNBUFFERED=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        PYTHONPATH=. python src/training/train_baseline.py \
            --processed_dir   data/processed \
            --annots_dir      data/annotations/assembly101-mistake-detection/annots \
            --batch_size      4 \
            --num_workers     4 \
            --hidden_dim      256 \
            --dropout         0.3 \
            --spanning_scales 8 16 24 \
            --recent_scales   30 90 150 \
            --max_seq_len     500 \
            --epochs          50 \
            --lr              5e-5 \
            --weight_decay    1e-3 \
            --seed            42 \
            --wandb_project   mistake-detection \
            --wandb_run_name  \"tempagg-baseline-$SLURM_JOB_ID\" \
            --ckpt_dir        experiments/checkpoints
    "

echo "================================================"
echo "End: $(date)"
echo "================================================"
