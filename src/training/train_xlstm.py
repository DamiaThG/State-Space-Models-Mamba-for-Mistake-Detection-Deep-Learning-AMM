import os
import argparse
import datetime
import random
import sys
import logging
from pathlib import Path
from datetime import datetime as dt

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import wandb

from src.datasets.dataloader import build_whole_video_dataloaders
from src.training.lightning_module import MistakeDetectionLightningModule
from src.models.xlstm_model import xLSTMMistakeDetector
from src.training.metrics_callback import MetricsCallback

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for xLSTM on Assembly101 whole videos")
    parser.add_argument("--processed_dir", type=str, required=True, help="Directory containing processed data")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--accumulate_grad_batches", type=int, default=2, help="Gradient accumulation steps")
    
    parser.add_argument("--d_model", type=int, default=512, help="xLSTM embedding dimension")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of xLSTM blocks")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--max_seq_len", type=int, default=25000, help="Max sequence length for videos")
    parser.add_argument("--use_checkpointing", action="store_true", help="Enable gradient checkpointing per-block")
    
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for AdamW")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Esponente Focal Loss (gamma=0 → CrossEntropyLoss pesata standard)")
    parser.add_argument("--class_weight_exp", type=float, default=1.5,
                        help="Esponente per attenuare o accentuare i pesi delle classi")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="mistake-detection", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--ckpt_dir", type=str, default="experiments/checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # ── Configurazione Logging (solo stdout — FileHandler aggiunto da MetricsCallback) ────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # args già parsati: creiamo subito il callback così cattura TUTTI i log nel file
    metrics_cb = MetricsCallback(model_name="xlstm", args=args)

    logging.info(f"Avvio addestramento xLSTM Whole Video. Run: {metrics_cb.run_id}")

    # 1. Impostazioni base
    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Dispositivo: {device_str}")
    
    if args.wandb_run_name is None:
        args.wandb_run_name = f"xlstm-wholevid-{metrics_cb.run_id}"

    # 2. Dataloaders (whole video)
    logging.info("Inizializzazione dataloaders...")
    train_loader, val_loader, test_loader = build_whole_video_dataloaders(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        val_split=0.15,
        test_split=0.15,
        num_workers=args.num_workers,
        pin_memory=(device_str == "cuda"),
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    
    # 3. Model
    logging.info("Inizializzazione modello xLSTM...")
    model = xLSTMMistakeDetector(
        input_dim=2048,
        d_model=args.d_model,
        n_layers=args.n_layers,
        num_classes=3,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        use_checkpointing=args.use_checkpointing
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Modello: xLSTM (Whole Video) — Parametri trainabili: {total_params:,}")

    # 4. Lightning Module
    total_steps = len(train_loader) * args.epochs // args.accumulate_grad_batches
    logging.info(f"Estimated total steps: {total_steps}")
    
    lightning_module = MistakeDetectionLightningModule(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        class_weight_exp=args.class_weight_exp,
    )
    
    # 5. Callbacks & Logger
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="xlstm-wholevid-{epoch:02d}-val_f1={val/f1_macro:.4f}",
        monitor="val/f1_macro",
        mode="max",
        save_top_k=3,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stop = EarlyStopping(
        monitor   = "val/f1_macro",
        mode      = "max",
        patience  = 15,
        min_delta = 0.001,
    )
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name,
        log_model=False,
        save_dir="experiments"
    )
    
    wandb_logger.experiment.config.update(vars(args))
    wandb_logger.watch(model, log="all", log_freq=100)
    
    # 6. Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if device_str == "cuda" else "cpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stop, metrics_cb],
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision="16-mixed" if device_str == "cuda" else "32-true",
        gradient_clip_val=1.0,
        log_every_n_steps=10
    )
    
    # 7. Training
    logging.info("Avvio training...")
    ckpt_path = None
    if args.resume:
        last_ckpt = os.path.join(args.ckpt_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            logging.info(f"Ripresa dell'addestramento dal checkpoint: {last_ckpt}")
            ckpt_path = last_ckpt
        else:
            logging.warning(f"Flag --resume usato, ma {last_ckpt} non trovato. Partenza da zero.")

    trainer.fit(lightning_module, train_loader, val_loader, ckpt_path=ckpt_path)

    # 8. Testing (sul best checkpoint)
    logging.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    trainer.test(lightning_module, test_loader, ckpt_path="best")

    wandb.finish()
    
if __name__ == "__main__":
    main()
