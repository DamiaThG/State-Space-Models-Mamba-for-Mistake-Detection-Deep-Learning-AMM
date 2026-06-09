#!/bin/bash
# ============================================================
# Runner Script — Mamba SSM Training
# Esegui direttamente dentro il container Apptainer (mamba-docker)
# Uso: ./scripts/run_train_mamba.sh [opzioni extra per il training]
# ============================================================

set -euo pipefail

# ---------- Verifica ambiente ----------
echo "================================================"
echo "Runner    : run_train_mamba.sh"
echo "Node      : $(hostname)"
echo "Start     : $(date)"
echo "Working   : $(pwd)"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python    : $(python --version 2>&1)"
echo "================================================"

# ---------- Directory ----------
mkdir -p experiments/logs
mkdir -p experiments/checkpoints

# ---------- Environment ----------
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
# Riduce la frammentazione della VRAM (consigliato da PyTorch per OOM)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- Training ----------
# I parametri di default possono essere sovrascritti passando argomenti allo script.
# Esempio: ./scripts/run_train_mamba.sh --batch_size 8 --lr 1e-4
PYTHONPATH=. python src/training/train_mamba_whole_video.py \
    --processed_dir   data/processed \
    --batch_size      4 \
    --num_workers     4 \
    --accumulate_grad_batches 2 \
    --d_model         512 \
    --n_layers        6 \
    --dropout         0.2 \
    --max_seq_len     25000 \
    --use_checkpointing \
    --epochs          50 \
    --lr              5e-5 \
    --weight_decay    1e-3 \
    --seed            42 \
    --wandb_project   mistake-detection \
    --wandb_run_name  "mamba-ssm-wholevid-interactive" \
    --ckpt_dir        experiments/checkpoints \
    "$@"

echo "================================================"
echo "End: $(date)"
echo "================================================"
