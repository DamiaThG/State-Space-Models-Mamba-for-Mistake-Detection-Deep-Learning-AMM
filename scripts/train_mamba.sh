#!/bin/bash
# ============================================================
# Script unico — Mamba SSM Training
# Cluster: GCluster (gcluster.dmi.unict.it)
#
# Usi:
#   ./scripts/train_mamba.sh                    # da dentro il container (interattivo)
#   bash  scripts/train_mamba.sh                # da fuori il container
#   sbatch scripts/train_mamba.sh               # SLURM (batch)
#   ./scripts/train_mamba.sh --lr 1e-4          # con override (in qualsiasi modalità)
#
# Lo script rileva automaticamente se è già dentro Apptainer
# e si comporta di conseguenza.
# ============================================================

#SBATCH --job-name=mamba_ssm
#SBATCH --output=experiments/logs/mamba_%j.out
#SBATCH --error=experiments/logs/mamba_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --gres=shard:22000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

# ---------- Info ambiente ----------
echo "================================================"
echo "Job ID    : ${SLURM_JOB_ID:-N/A}"
echo "Node      : $(hostname)"
echo "Start     : $(date)"
echo "Working   : $(pwd)"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "================================================"

# ---------- Directory ----------
mkdir -p experiments/logs
mkdir -p experiments/checkpoints

# ---------- Variabili d'ambiente comuni ----------
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- Parametri training ----------
TRAIN_CMD=(
    python src/training/train_mamba.py
        --processed_dir   data/processed
        --batch_size      4
        --num_workers     4
        --accumulate_grad_batches 2
        --d_model         512
        --n_layers        6
        --dropout         0.2
        --max_seq_len     25000
        --use_checkpointing
        --epochs          100
        --lr              5e-5
        --weight_decay    1e-3
        --focal_gamma     2.0
        --class_weight_exp 1.0 #se max_seq_len = none, questo è da aumentare a 1.5, altrimenti il modello si perde totalmente mistake e correction
        --seed            42
        --wandb_project   mistake-detection
        --wandb_run_name  "mamba-ssm-wholevid-${SLURM_JOB_ID:-interactive}"
        --ckpt_dir        experiments/checkpoints
        "$@"
)

# ---------- Esecuzione ----------
if [ -n "${APPTAINER_NAME:-}${SINGULARITY_NAME:-}" ]; then
    # Già dentro il container: lancia Python direttamente
    echo "Modalità: dentro Apptainer — lancio Python direttamente"
    PYTHONPATH=. "${TRAIN_CMD[@]}"
else
    # Fuori dal container: usa apptainer exec
    echo "Modalità: fuori Apptainer — lancio via apptainer exec"
    APPTAINER_IMAGE="mamba_env.sif"
    apptainer exec --nv \
        --env PYTHONPATH=/workspace \
        --bind "$(pwd)":/workspace \
        --pwd /workspace \
        "$APPTAINER_IMAGE" \
        "${TRAIN_CMD[@]}"
fi

echo "================================================"
echo "End: $(date)"
echo "================================================"
