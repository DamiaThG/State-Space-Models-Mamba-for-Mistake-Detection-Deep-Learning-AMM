"""
Training Script — TempAgg Baseline
====================================
Script dedicato per il training della TempAgg Baseline su Assembly101.
Usa i dataloader a sequenze di azioni (build_split_dataloaders).

Uso rapido:
    python src/training/train_baseline.py \\
        --processed_dir   data/processed \\
        --annots_dir      data/annotations/assembly101-mistake-detection/annots \\
        --epochs          50 \\
        --batch_size      16 \\
        --wandb_project   mistake-detection
"""

import argparse
import os
import random
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from src.models.baseline import TempAggMistakeDetector
from src.datasets.dataloader import build_split_dataloaders
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
# Funzioni standalone (alternativa a Lightning per debug rapido)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:      nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion:  nn.CrossEntropyLoss,
    device:     torch.device,
) -> Dict[str, float]:
    """Valutazione standalone (senza Lightning)."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_preds  = []
    all_labels = []

    for batch in dataloader:
        features = batch["features"].to(device)
        labels   = batch["labels"].to(device)
        mask     = batch["attention_mask"].to(device)

        logits = model(features, mask)
        B, T, C = logits.shape
        logits_flat = logits.reshape(B * T, C)
        labels_flat = labels.reshape(B * T)

        loss = criterion(logits_flat, labels_flat)
        total_loss += loss.item()
        n_batches  += 1

        preds = logits_flat.argmax(dim=-1)
        valid = labels_flat != -1
        all_preds.append(preds[valid].cpu())
        all_labels.append(labels_flat[valid].cpu())

    torch.cuda.empty_cache()

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    metrics = {"loss": total_loss / max(n_batches, 1)}
    class_names = ["correct", "mistake", "correction"]
    for cls_idx, cls_name in enumerate(class_names):
        tp = ((all_preds == cls_idx) & (all_labels == cls_idx)).sum().float()
        fp = ((all_preds == cls_idx) & (all_labels != cls_idx)).sum().float()
        fn = ((all_preds != cls_idx) & (all_labels == cls_idx)).sum().float()
        prec = (tp / (tp + fp + 1e-8)).item()
        rec  = (tp / (tp + fn + 1e-8)).item()
        metrics[f"precision_{cls_name}"] = prec
        metrics[f"recall_{cls_name}"]    = rec

    return metrics


def train_epoch(
    model:      nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.CrossEntropyLoss,
    device:     torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """Singola epoca di training standalone (senza Lightning)."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    from tqdm import tqdm
    pbar = tqdm(dataloader, desc="Training", mininterval=2.0, file=sys.stdout)

    for batch in pbar:
        features = batch["features"].to(device)
        labels   = batch["labels"].to(device)
        mask     = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        logits = model(features, mask)

        B, T, C = logits.shape
        loss = criterion(logits.reshape(B * T, C), labels.reshape(B * T))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TempAgg Baseline — Training")

    # Dati
    p.add_argument("--processed_dir",   default="data/processed")
    p.add_argument("--annots_dir",
                   default="data/annotations/assembly101-mistake-detection/annots")
    p.add_argument("--val_split",       type=float, default=0.15)
    p.add_argument("--test_split",      type=float, default=0.15)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--num_workers",     type=int,   default=4)

    # Architettura — TempAgg
    p.add_argument("--hidden_dim",      type=int,   default=512)
    p.add_argument("--spanning_scales", type=int,   nargs="+", default=[8, 16, 24])
    p.add_argument("--recent_scales",   type=int,   nargs="+", default=[30, 90, 150])
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--max_seq_len",     type=int,   default=None)

    # Training
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-5)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--focal_gamma",  type=float, default=2.0,
                   help="Esponente Focal Loss (gamma=0 → CrossEntropyLoss pesata standard)")

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

    # ── Configurazione Logging (solo stdout — FileHandler aggiunto da MetricsCallback) ────
    log_dir = Path("experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # args già parsati: creiamo subito il callback così cattura TUTTI i log nel file
    metrics_cb = MetricsCallback(model_name="baseline", args=args)

    logging.info(f"Avvio addestramento TempAgg Baseline. Run: {metrics_cb.run_id}")

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Dispositivo: {device_str}")

    # ── DataLoader ─────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_split_dataloaders(
        processed_dir   = args.processed_dir,
        annotations_dir = args.annots_dir,
        batch_size      = args.batch_size,
        val_split       = args.val_split,
        test_split      = args.test_split,
        num_workers     = args.num_workers,
        pin_memory      = (device_str == "cuda"),
        max_seq_len     = args.max_seq_len,
        seed            = args.seed,
    )

    # ── Modello ────────────────────────────────────────────────────────────
    model = TempAggMistakeDetector(
        input_dim=2048,
        hidden_dim=args.hidden_dim,
        num_classes=3,
        spanning_scales=args.spanning_scales,
        recent_scales=args.recent_scales,
        dropout=args.dropout,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Modello: TempAgg Baseline — Parametri trainabili: {total_params:,}")

    # ── LightningModule ────────────────────────────────────────────────────
    lit_model = MistakeDetectionLightningModule(
        model        = model,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        focal_gamma  = args.focal_gamma,
    )

    # ── Logger & Callbacks ─────────────────────────────────────────────────
    run_name = args.wandb_run_name or f"tempagg-baseline-{metrics_cb.run_id}"

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
        filename     = "tempagg-{epoch:02d}-{val/loss:.4f}",
        monitor      = "val/loss",
        mode         = "min",
        save_top_k   = 3,
        save_last    = True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stop = EarlyStopping(
        monitor   = "val/recall_mistake",
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
        deterministic            = "warn",
    )

    ckpt_path = None
    if args.resume:
        last_ckpt = os.path.join(args.ckpt_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            logging.info(f"Ripresa dal checkpoint: {last_ckpt}")
            ckpt_path = last_ckpt
        else:
            logging.warning(f"Flag --resume usato, ma {last_ckpt} non trovato. Partenza da zero.")

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=ckpt_path)

    logging.info(f"Best checkpoint: {ckpt_callback.best_model_path}")

    if test_loader is not None:
        trainer.test(lit_model, dataloaders=test_loader, ckpt_path="best")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
