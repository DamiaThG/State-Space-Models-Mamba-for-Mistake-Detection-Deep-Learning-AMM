"""
Lightning Module — Mistake Detection (condiviso tra tutti i modelli)
=====================================================================
Questo modulo contiene SOLO la logica di training condivisa:
  - compute_class_weights()
  - FocalLoss
  - MistakeDetectionLightningModule

Non importa nessun modello specifico (zero dipendenze da mamba_ssm, xlstm, ecc.)
Ogni script di training dedicato importa da qui ciò che gli serve.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Focal Loss (class-weighted, con ignore_index)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss per classificazione multi-classe sbilanciata.

    Formula:  FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Rispetto alla CrossEntropyLoss pesata standard:
      - Quando il modello è molto "sicuro" sulla classe corretta (p_t → 1),
        il fattore (1-p_t)^gamma → 0: la loss si annulla per gli esempi facili.
      - Quando il modello sbaglia (p_t → 0), il fattore → 1: la loss resta inalterata.

    Effetto pratico: i gradienti vengono concentrati sulle classi rare/difficili
    (mistake, correction) anziché essere dominati dai frame 'correct' facili.

    Args:
        weight:       pesi per classe [C], come in CrossEntropyLoss.
        gamma:        esponente di focalizzazione (default=2.0).
                      gamma=0 → equivalente alla CrossEntropyLoss pesata.
        ignore_index: indice da ignorare nel calcolo della loss (padding).
        reduction:    'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        # Registriamo i pesi come buffer (si spostano automaticamente su GPU)
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [N, C]  logits grezzi (pre-softmax).
            targets: [N]     indici di classe.
        """
        # Maschera dei token validi (esclude ignore_index)
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return logits.sum() * 0.0  # nessun token valido, loss = 0

        logits_valid  = logits[valid_mask]    # [N_valid, C]
        targets_valid = targets[valid_mask]   # [N_valid]

        # Log-probabilità e probabilità della classe corretta
        log_probs = F.log_softmax(logits_valid, dim=-1)             # [N_valid, C]
        probs     = log_probs.exp()                                 # [N_valid, C]

        # Seleziona p_t e log(p_t) per la classe target
        targets_one_hot = F.one_hot(targets_valid, num_classes=logits_valid.shape[-1])  # [N_valid, C]
        p_t     = (probs * targets_one_hot).sum(dim=-1)             # [N_valid]
        log_p_t = (log_probs * targets_one_hot).sum(dim=-1)         # [N_valid]

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma                    # [N_valid]

        # Class weights (alpha_t)
        if self.weight is not None:
            alpha_t = self.weight[targets_valid]                     # [N_valid]
        else:
            alpha_t = 1.0

        # Loss per-sample: -alpha_t * (1-p_t)^gamma * log(p_t)
        loss = -alpha_t * focal_weight * log_p_t                    # [N_valid]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# LightningModule (generico, model-agnostic)
# ---------------------------------------------------------------------------

class MistakeDetectionLightningModule(L.LightningModule):
    """
    LightningModule generico per Mistake Detection.
    Accetta qualsiasi nn.Module con interfaccia forward(features, attention_mask) → logits.

    Gestisce:
      - Forward pass con attention_mask
      - FocalLoss (default, gamma=2.0) o CrossEntropyLoss (gamma=0) con class weighting
      - Masking del padding prima del calcolo della loss
      - Precision e Recall per classe (torchmetrics)
      - Logging metriche per-epoch tramite MetricsCallback
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
        focal_gamma:     float = 2.0,
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
        if focal_gamma > 0:
            self.criterion = FocalLoss(
                weight       = class_weights,
                gamma        = focal_gamma,
                ignore_index = -1,
                reduction    = "mean",
            )
        else:
            # gamma=0 → Focal Loss degenera in CrossEntropyLoss pesata
            self.criterion = nn.CrossEntropyLoss(
                weight       = class_weights,
                ignore_index = -1,
                reduction    = "mean",
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
    ) -> Dict[str, torch.Tensor]:
        loss, preds, labels = self._shared_step(batch, "test")
        valid = labels != -1
        self.test_precision.update(preds[valid], labels[valid])
        self.test_recall.update(preds[valid], labels[valid])
        B = batch["features"].shape[0]
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        # Ritorna preds e labels per MetricsCallback (timeline plot)
        return {"preds": preds.cpu(), "labels": labels.cpu()}

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

        # F1 per tutte le classi
        for i in range(3):
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
