#!/bin/bash
# ============================================================
# Script unico — xLSTM Training
# Cluster: GCluster (gcluster.dmi.unict.it)
#
# Usi:
#   ./scripts/train_xlstm.sh                    # da dentro il container (interattivo)
#   bash  scripts/train_xlstm.sh                # da fuori il container
#   sbatch scripts/train_xlstm.sh               # SLURM (batch)
#   ./scripts/train_xlstm.sh --lr 1e-4          # con override (in qualsiasi modalità)
#
# Lo script rileva automaticamente se è già dentro Apptainer
# e si comporta di conseguenza.
# ============================================================

#SBATCH --job-name=xlstm
#SBATCH --output=experiments/logs/xlstm_%j.out
#SBATCH --error=experiments/logs/xlstm_%j.err
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
# Fix per il JIT compiler di xLSTM che cerca i percorsi CUDA
export CUDA_HOME=/opt/conda

# ---------- Parametri training ----------
TRAIN_CMD=(
    python src/training/train_xlstm.py
        --processed_dir   data/processed
        --batch_size      2
        --num_workers     4
        --accumulate_grad_batches 4
        --d_model         512
        --n_layers        6
        --dropout         0.4
        --max_seq_len     8000
        --use_checkpointing
        --epochs          50
        --lr              2e-5
        --weight_decay    1e-2
        --focal_gamma     2.0
        --class_weight_exp 1.0
        --seed            42
        --wandb_project   mistake-detection
        --wandb_run_name  "xlstm-wholevid-${SLURM_JOB_ID:-interactive}"
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
    APPTAINER_IMAGE="/shared/sifs/latest.sif"
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
