"""
Lightning Module — Mistake Detection (condiviso tra tutti i modelli)
=====================================================================
Questo modulo contiene SOLO la logica di training condivisa:
  - compute_class_weights()
  - MistakeDetectionLightningModule

Non importa nessun modello specifico (zero dipendenze da mamba_ssm, xlstm, ecc.)
Ogni script di training dedicato importa da qui ciò che gli serve.
"""

import torch
import torch.nn as nn
import lightning as L
from typing import Dict, Optional, Tuple
from torchmetrics import Precision, Recall


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

    Progressione dei run:
      - Run 1: exp=1.0  → [1.0, 4.87,  11.55] → correction ignorata (F1=0)
      - Run 2: exp=2.0  → [1.0, 23.7, 133.5]  → correction appresa (F1=0.077) ma overfitting
      - Run 3: exp=0.5  → [1.0, 2.21,   3.40]  → collasso su 'correct' (recall_correct=0.99)
      - Run 4: exp=1.5  → [1.0, 7.6,   38.5]   → compromesso ottimale tra run 2 e run 3
    """
    inv = torch.tensor([
        1.0 / correct_frac,
        1.0 / mistake_frac,
        1.0 / correction_frac,
    ])
    # Potenza 1.5: tra lineare (1.0) e quadratica (2.0)
    weights = inv ** 1.5
    # Normalizza rispetto alla classe più frequente
    weights = weights / weights.min()
    return weights   # [1.0, ~7.6, ~38.5]


# ---------------------------------------------------------------------------
# LightningModule (generico, model-agnostic)
# ---------------------------------------------------------------------------

class MistakeDetectionLightningModule(L.LightningModule):
    """
    LightningModule generico per Mistake Detection.
    Accetta qualsiasi nn.Module con interfaccia forward(features, attention_mask) → logits.

    Gestisce:
      - Forward pass con attention_mask
      - CrossEntropyLoss con class weighting e ignore_index=-1
      - Masking del padding prima del calcolo della loss
      - Precision e Recall per classe (torchmetrics)
      - Logging su WandB
      - LR Schedule: LinearWarmup (3 epoche) → CosineAnnealing
    """

    def __init__(
        self,
        model:           nn.Module,
        # Ottimizzazione
        lr:              float = 1e-4,
        weight_decay:    float = 1e-5,
        warmup_steps:    int   = 500,
        # Loss
        correct_frac:    float = 0.774,
        mistake_frac:    float = 0.159,
        correction_frac: float = 0.067,
        num_classes:     int   = 3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # ── Modello ────────────────────────────────────────────────────────
        self.model = model

        # ── Loss ──────────────────────────────────────────────────────────────
        class_weights = compute_class_weights(
            correct_frac, mistake_frac, correction_frac
        )
        self.criterion = nn.CrossEntropyLoss(
            weight       = class_weights,
            ignore_index = -1,    # ignora il padding
            reduction    = "mean",
            # NB: label_smoothing RIMOSSO — redistribuisce massa probabilistica
            # uniformemente, contraddicendo i class weights con classe dominante al 77%.
        )

        # ── Metriche (per classe 1=mistake, 2=correction) ──────────────────
        metric_kwargs = dict(
            task="multiclass",
            num_classes=num_classes,
            average=None,        # una metrica per classe
            ignore_index=-1,
        )
        self.val_precision  = Precision(**metric_kwargs)
        self.val_recall     = Recall(**metric_kwargs)
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
        B = batch["features"].shape[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=B)
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

        B = batch["features"].shape[0]
        self.log("val/loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=B)

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
        B = batch["features"].shape[0]
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)

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


# Alias per backward-compatibility
TempAggLightningModule = MistakeDetectionLightningModule
