"""
Training Loop — TempAgg Baseline per Mistake Detection
=======================================================
Struttura basata su PyTorch Lightning + Weights & Biases.

Uso rapido:
    python src/training/training_loop.py \
        --processed_dir   data/processed \
        --annots_dir      data/annotations/assembly101-mistake-detection/annots \
        --epochs          50 \
        --batch_size      16 \
        --wandb_project   mistake-detection

Per il cluster SLURM: usa `scripts/train_baseline.sh` (generato alla fine).
"""

import argparse
import os
import random
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from torchmetrics import Precision, Recall

# Importazioni dalla root del progetto (eseguire sempre dalla root)
from src.models.baseline import TempAggMistakeDetector
from src.datasets.dataloader import build_split_dataloaders


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
# Class Weights (sbilanciamento Assembly101)
# ---------------------------------------------------------------------------
# Distribuzione approssimativa:
#   correct    (0): ~77.4%  → peso 1.0 (baseline)
#   mistake    (1): ~15.9%  → peso inversamente proporzionale
#   correction (2):  ~6.7%  → peso inversamente proporzionale

def compute_class_weights(
    correct_frac:    float = 0.774,
    mistake_frac:    float = 0.159,
    correction_frac: float = 0.067,
) -> torch.Tensor:
    """
    Restituisce un tensore di pesi [3] per CrossEntropyLoss.

    Usa i pesi quadratici (inv^2) invece dei lineari per enfatizzare
    maggiormente le classi rare, in particolare 'correction' che con
    pesi lineari veniva completamente ignorata dal modello.
    Risultati approssimativi:
        correct    (0): 1.0
        mistake    (1): ~23.7
        correction (2): ~133.5
    """
    inv = torch.tensor([
        1.0 / correct_frac,
        1.0 / mistake_frac,
        1.0 / correction_frac,
    ])
    # Pesi quadratici: amplifica molto di più le classi rare
    weights = inv ** 2
    # Normalizza rispetto alla classe più frequente
    weights = weights / weights.min()
    return weights   # [1.0, ~23.7, ~133.5]


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class TempAggLightningModule(L.LightningModule):
    """
    LightningModule che incapsula TempAggMistakeDetector.

    Gestisce:
      - Forward pass con attention_mask
      - CrossEntropyLoss con class weighting e ignore_index=-1
      - Masking del padding prima del calcolo della loss
      - Precision e Recall per classe (torchmetrics)
      - Logging su WandB
    """

    def __init__(
        self,
        # Architettura
        input_dim:       int   = 2048,
        hidden_dim:      int   = 512,
        num_classes:     int   = 3,
        spanning_scales        = [8, 16, 32, 64],
        recent_scales          = [1, 2, 4],
        dropout:         float = 0.1,
        # Ottimizzazione
        lr:              float = 1e-4,
        weight_decay:    float = 1e-5,
        warmup_steps:    int   = 500,
        # Loss
        correct_frac:    float = 0.774,
        mistake_frac:    float = 0.159,
        correction_frac: float = 0.067,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # ── Modello ────────────────────────────────────────────────────────
        self.model = TempAggMistakeDetector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            spanning_scales=spanning_scales,
            recent_scales=recent_scales,
            dropout=dropout,
        )

        # ── Loss ───────────────────────────────────────────────────────────
        class_weights = compute_class_weights(
            correct_frac, mistake_frac, correction_frac
        )
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=-1,   # ignora il padding
            reduction="mean",
        )

        # ── Metriche (per classe 1=mistake, 2=correction) ──────────────────
        metric_kwargs = dict(
            task="multiclass",
            num_classes=num_classes,
            average=None,        # una metrica per classe
            ignore_index=-1,
        )
        self.val_precision = Precision(**metric_kwargs)
        self.val_recall    = Recall(**metric_kwargs)
        self.test_precision = Precision(**metric_kwargs)
        self.test_recall    = Recall(**metric_kwargs)

    # ------------------------------------------------------------------
    def forward(
        self,
        features:       torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(features, attention_mask)

    # ------------------------------------------------------------------
    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcola logits, loss e predizioni.

        Masking del padding:
            - Flatten logits → [B*T, 3] e labels → [B*T]
            - CrossEntropyLoss con ignore_index=-1 gestisce automaticamente
              il padding (labels = -1 dove attention_mask = False)
        """
        features = batch["features"]          # [B, T, 2048]
        labels   = batch["labels"]            # [B, T]  (-1 = padding)
        mask     = batch["attention_mask"]    # [B, T]  bool

        logits = self(features, mask)         # [B, T, 3]

        # Flatten per la loss
        B, T, C = logits.shape
        logits_flat = logits.reshape(B * T, C)   # [B*T, 3]
        labels_flat = labels.reshape(B * T)      # [B*T]

        loss = self.criterion(logits_flat, labels_flat)

        preds = logits_flat.argmax(dim=-1)    # [B*T]
        return loss, preds, labels_flat

    # ------------------------------------------------------------------
    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, _, _ = self._shared_step(batch, "train")
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        loss, preds, labels = self._shared_step(batch, "val")

        # Maschera i padding prima di aggiornare le metriche
        valid = labels != -1
        self.val_precision.update(preds[valid], labels[valid])
        self.val_recall.update(preds[valid], labels[valid])

        self.log("val/loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        self._log_per_class_metrics("val", self.val_precision, self.val_recall)
        self.val_precision.reset()
        self.val_recall.reset()
        torch.cuda.empty_cache()   # svuota cache VRAM dopo la validazione

    # ------------------------------------------------------------------
    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        loss, preds, labels = self._shared_step(batch, "test")
        valid = labels != -1
        self.test_precision.update(preds[valid], labels[valid])
        self.test_recall.update(preds[valid], labels[valid])
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        self._log_per_class_metrics("test", self.test_precision, self.test_recall)
        self.test_precision.reset()
        self.test_recall.reset()

    # ------------------------------------------------------------------
    def _log_per_class_metrics(
        self,
        prefix:    str,
        precision: Precision,
        recall:    Recall,
    ) -> None:
        """
        Logga Precision e Recall per singola classe.
        Le classi minoritarie (1=mistake, 2=correction) sono le più rilevanti.
        """
        class_names = ["correct", "mistake", "correction"]
        prec_vals = precision.compute()   # [3]
        rec_vals  = recall.compute()      # [3]

        metrics: Dict[str, float] = {}
        for i, name in enumerate(class_names):
            metrics[f"{prefix}/precision_{name}"] = prec_vals[i].item()
            metrics[f"{prefix}/recall_{name}"]    = rec_vals[i].item()

        # F1 per le classi minoritarie
        for i in [1, 2]:
            name = class_names[i]
            p, r = prec_vals[i].item(), rec_vals[i].item()
            f1 = (2 * p * r / (p + r + 1e-8))
            metrics[f"{prefix}/f1_{name}"] = f1

        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # ── LR Schedule: LinearWarmup (3 epoche) → CosineAnnealing ────────────
        # Il warmup evita la memorizzazione immediata osservata all'epoca 0-1.
        warmup_epochs = 3
        total_epochs  = self.trainer.max_epochs if self.trainer else 50
        warmup_sched  = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = 0.1,        # parte da lr * 0.1
            end_factor   = 1.0,        # arriva a lr
            total_iters  = warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max    = total_epochs - warmup_epochs,
            eta_min  = self.hparams.lr * 1e-2,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers  = [warmup_sched, cosine_sched],
            milestones  = [warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }


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
    """
    Valutazione standalone (senza Lightning).

    Returns:
        dict con loss, precision e recall per classe.
    """
    model.eval()
    total_loss    = 0.0
    n_batches     = 0
    all_preds  = []
    all_labels = []

    for batch in dataloader:
        features = batch["features"].to(device)
        labels   = batch["labels"].to(device)
        mask     = batch["attention_mask"].to(device)

        logits = model(features, mask)        # [B, T, 3]
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
    """
    Singola epoca di training standalone (senza Lightning).

    Returns:
        loss media sull'epoca.
    """
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

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Entry point — training con PyTorch Lightning
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TempAgg Baseline — Mistake Detection")
    # Dati
    p.add_argument("--processed_dir",   default="data/processed")
    p.add_argument("--annots_dir",
                   default="data/annotations/assembly101-mistake-detection/annots")
    p.add_argument("--val_split",       type=float, default=0.15)
    p.add_argument("--test_split",      type=float, default=0.15)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--num_workers",     type=int,   default=4)
    # Architettura
    p.add_argument("--hidden_dim",      type=int,   default=512)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--spanning_scales", type=int,   nargs="+", default=[8, 16, 24])
    p.add_argument("--recent_scales",   type=int,   nargs="+", default=[30, 90, 150])
    # Training
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-5)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--max_seq_len",     type=int,   default=None,
                   help="Lunghezza massima delle sequenze (frame). Quelle più lunghe "
                        "vengono troncate ai frame più recenti. Consigliato: 500-1000.")
    # Logging
    p.add_argument("--wandb_project",   default="mistake-detection")
    p.add_argument("--wandb_run_name",  default="tempagg-baseline")
    p.add_argument("--no_wandb",        action="store_true")
    # Output
    p.add_argument("--ckpt_dir",        default="experiments/checkpoints")
    p.add_argument("--resume",          action="store_true", help="Riprende l'addestramento da last.ckpt se presente")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Configurazione Logging ─────────────────────────────────────────────
    log_dir = Path("experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info(f"Avvio addestramento. Log file: {log_file}")

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

    # ── LightningModule ────────────────────────────────────────────────────
    lit_model = TempAggLightningModule(
        hidden_dim      = args.hidden_dim,
        dropout         = args.dropout,
        spanning_scales = args.spanning_scales,
        recent_scales   = args.recent_scales,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
    )

    # ── Logger & Callbacks ─────────────────────────────────────────────────
    loggers = []
    if not args.no_wandb:
        wandb_logger = WandbLogger(
            project = args.wandb_project,
            name    = args.wandb_run_name,
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
        patience  = 15,       # aumentata: il modello ha bisogno di più tempo (warmup + correction)
        min_delta = 0.001,    # abbassata: un miglioramento piccolo conta comunque
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs          = args.epochs,
        accelerator         = "auto",
        devices             = "auto",
        logger              = loggers if loggers else False,
        callbacks           = [ckpt_callback, lr_monitor, early_stop],
        log_every_n_steps   = 10,
        gradient_clip_val   = 1.0,
        deterministic       = "warn",   # True causa crash con roi_pool backward (nessuna impl. deterministica)
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

    # ── Valutazione finale sul test set ────────────────────────────────────
    logging.info("Training completato.")
    logging.info(f"Best checkpoint: {ckpt_callback.best_model_path}")

    if test_loader is not None:
        logging.info("Avvio valutazione sul test set (best checkpoint)...")
        trainer.test(lit_model, dataloaders=test_loader, ckpt_path="best")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
