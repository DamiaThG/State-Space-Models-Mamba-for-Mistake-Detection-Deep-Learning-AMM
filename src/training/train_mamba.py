"""
Training Loop — Mistake Detection (Mamba Whole Video)
=====================================================
Struttura dedicata a PyTorch Lightning per l'addestramento di Mamba
sull'intero video, utilizzando il `WholeVideoDataset` e il 
`LengthGroupedSampler`.

Uso rapido:
    python src/training/train_mamba.py \\
        --processed_dir   data/processed \\
        --epochs          50 \\
        --batch_size      4 \\
        --max_seq_len     20000 \\
        --use_checkpointing \\
        --wandb_project   mistake-detection
"""

import argparse
import os
import random
import sys
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# --- WORKAROUND PER TORCHVISION ROTTO NEL CONTAINER ---
# Nel container la versione di torch (2.12) e torchvision (0.21) non combaciano,
# quindi le estensioni C++ di torchvision non vengono caricate. Questo fa
# crashare l'importazione quando PyTorch cerca di registrare gli operatori mancanti.
# Creiamo un mock temporaneo per ignorare gli errori "does not exist".
original_register_fake = torch.library.register_fake

def mock_register_fake(name, *args, **kwargs):
    def decorator(fn):
        try:
            return original_register_fake(name, *args, **kwargs)(fn)
        except RuntimeError as e:
            if "does not exist" in str(e):
                return fn  # Ignora l'errore per gli operatori C++ mancanti
            raise
    return decorator

torch.library.register_fake = mock_register_fake

import torchvision

# Ripristino la funzione originale
torch.library.register_fake = original_register_fake
# --------------------------------------------------------
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

# Importazioni dalla root del progetto (eseguire sempre dalla root)
from src.models.mamba_model import MambaMistakeDetector
from src.datasets.dataloader import build_whole_video_dataloaders
from src.training.lightning_module import MistakeDetectionLightningModule
from src.training.metrics_callback import MetricsCallback


# ---------------------------------------------------------------------------
# Riproducibilità
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mamba Whole Video Training")

    # Dati
    p.add_argument("--processed_dir",   default="data/processed")
    p.add_argument("--val_split",       type=float, default=0.15)
    p.add_argument("--test_split",      type=float, default=0.15)
    p.add_argument("--batch_size",      type=int,   default=4,
                   help="Batch size (più piccolo per whole video: es. 4 o 8)")
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--max_seq_len",     type=int,   default=20000,
                   help="Limite massimo frame (tail-truncation). Consigliato: 20000")

    # Architettura — Mamba
    p.add_argument("--d_model",  type=int, default=512)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--dropout",  type=float, default=0.2)
    p.add_argument("--use_checkpointing", action="store_true",
                   help="Attiva il Gradient Checkpointing per limitare la VRAM")

    # Training
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-5)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--focal_gamma",  type=float, default=2.0,
                   help="Esponente Focal Loss (gamma=0 → CrossEntropyLoss pesata standard)")
    p.add_argument("--class_weight_exp", type=float, default=1.5,
                   help="Esponente per attenuare o accentuare i pesi delle classi")

    # Logging
    p.add_argument("--wandb_project",   default="mistake-detection")
    p.add_argument("--wandb_run_name",  default=None)
    p.add_argument("--no_wandb",        action="store_true")

    # Output
    p.add_argument("--ckpt_dir",        default="experiments/checkpoints")
    p.add_argument("--resume",          action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.max_seq_len is not None and args.max_seq_len <= 0:
        args.max_seq_len = None

    # ── Configurazione Logging (solo stdout — FileHandler aggiunto da MetricsCallback) ────
    log_dir = Path("experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # args già parsati: creiamo subito il callback così cattura TUTTI i log nel file
    metrics_cb = MetricsCallback(model_name="mamba", args=args)

    logging.info(f"Avvio addestramento Mamba Whole Video. Run: {metrics_cb.run_id}")

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Dispositivo: {device_str}")

    # ── DataLoader a Video Intero ──────────────────────────────────────────
    train_loader, val_loader, test_loader = build_whole_video_dataloaders(
        processed_dir   = args.processed_dir,
        batch_size      = args.batch_size,
        val_split       = args.val_split,
        test_split      = args.test_split,
        num_workers     = args.num_workers,
        pin_memory      = (device_str == "cuda"),
        max_seq_len     = args.max_seq_len,
        seed            = args.seed,
    )

    # ── Modello Mamba ──────────────────────────────────────────────────────
    model = MambaMistakeDetector(
        input_dim=2048,
        d_model=args.d_model,
        n_layers=args.n_layers,
        num_classes=3,
        dropout=args.dropout,
        use_checkpointing=args.use_checkpointing,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Modello: Mamba (Whole Video) — Parametri trainabili: {total_params:,}")

    # ── LightningModule ────────────────────────────────────────────────────
    # Riutilizziamo lo stesso modulo di training della baseline per la logica
    # di loss, metriche, class weighting e ottimizzazione.
    lit_model = MistakeDetectionLightningModule(
        model        = model,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        focal_gamma  = args.focal_gamma,
        class_weight_exp = args.class_weight_exp,
    )

    # ── Logger & Callbacks ─────────────────────────────────────────────────
    run_name = args.wandb_run_name or f"mamba-wholevid-{metrics_cb.run_id}"

    loggers = []
    if not args.no_wandb:
        wandb_logger = WandbLogger(
            project = args.wandb_project,
            name    = run_name,
            config  = vars(args),
        )
        loggers.append(wandb_logger)

    ckpt_callback = ModelCheckpoint(
        dirpath      = args.ckpt_dir,
        filename     = "mamba-wholevid-{epoch:02d}-{val/f1_macro:.4f}",
        monitor      = "val/f1_macro",
        mode         = "max",
        save_top_k   = 3,
        save_last    = True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # N.B. Monitoriamo la macro F1 per premiare la reale detection.
    early_stop = EarlyStopping(
        monitor   = "val/f1_macro",
        mode      = "max",
        patience  = 15,
        min_delta = 0.001,
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs               = args.epochs,
        accelerator              = "auto",
        devices                  = "auto",
        precision                = "16-mixed",
        logger                   = loggers if loggers else False,
        callbacks                = [ckpt_callback, lr_monitor, early_stop, metrics_cb],
        log_every_n_steps        = 10,
        gradient_clip_val        = 1.0,
        accumulate_grad_batches  = args.accumulate_grad_batches,
    )

    ckpt_path = None
    if args.resume:
        last_ckpt = os.path.join(args.ckpt_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            logging.info(f"Ripresa dell'addestramento dal checkpoint: {last_ckpt}")
            ckpt_path = last_ckpt
        else:
            logging.warning(f"Flag --resume usato, ma {last_ckpt} non trovato. Partenza da zero.")

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=ckpt_path)

    # ── Valutazione finale sul test set ──────────────────────────────────────────────
    logging.info(f"Best checkpoint: {ckpt_callback.best_model_path}")

    if test_loader is not None:
        trainer.test(lit_model, dataloaders=test_loader, ckpt_path="best")

    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
