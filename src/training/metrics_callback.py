"""
MetricsCallback — Logging per-epoch e generazione grafici automatici
=====================================================================
Callback riutilizzabile per tutti i modelli (Mamba, xLSTM, Baseline).

Per ogni training viene creato automaticamente un run_id incrementale:
  {model_name}_{NNN}  →  es. xlstm_001, mamba_003, baseline_001

Output per ogni run:
  experiments/logs/{run_id}.log   — parametri + per-epoch + risultati test
  figures/{run_id}/               — tutti i grafici

Grafici a fine training:
  loss_curves.png, precision_per_class.png, recall_per_class.png,
  f1_per_class.png, lr_schedule.png

Grafici a fine test:
  test_confusion_matrix.png, test_metrics_bar.png, test_timeline.png
"""

import logging
import numpy as np
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import lightning as L


class MetricsCallback(L.Callback):
    """
    Lightning Callback che gestisce logging strutturato e generazione grafici.

    Uso:
        cb = MetricsCallback(model_name="xlstm", args=args)
        trainer = L.Trainer(..., callbacks=[..., cb])
    """

    CLASS_NAMES  = ["correct", "mistake", "correction"]
    CLASS_COLORS = ["#4CAF50", "#F44336", "#FF9800"]   # verde, rosso, arancione

    # ──────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        model_name:  str,
        args:        Namespace,
        figures_dir: str = "figures",
        logs_dir:    str = "experiments/logs",
    ):
        super().__init__()
        self.model_name  = model_name
        self.args        = args
        self.figures_dir = Path(figures_dir)
        self.logs_dir    = Path(logs_dir)

        # Determina run_id auto-incrementale
        self.run_id  = self._next_run_id()
        self.fig_dir = self.figures_dir / self.run_id
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        # Aggiunge FileHandler al root logger (stdout già configurato nel training script)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_dir / f"{self.run_id}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(fh)

        logging.info(
            f"[MetricsCallback] Run ID: {self.run_id} | "
            f"Log: {log_file} | Figures: {self.fig_dir}"
        )

        # ── Storia metriche training ───────────────────────────────────────────
        self.train_losses:    List[float]       = []
        self.val_losses:      List[float]       = []
        self.val_prec:        List[List[float]] = []   # [epoch][class]
        self.val_rec:         List[List[float]] = []
        self.val_f1:          List[List[float]] = []
        self.learning_rates:  List[float]       = []

        # ── Dati test per grafici ──────────────────────────────────────────────
        self.test_preds:  List[torch.Tensor] = []   # un tensore per video
        self.test_labels: List[torch.Tensor] = []

    # ──────────────────────────────────────────────────────────────────────────
    def _next_run_id(self) -> str:
        """Trova il prossimo numero disponibile per il run_id."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        existing = list(self.logs_dir.glob(f"{self.model_name}_*.log"))
        nums = []
        for f in existing:
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                nums.append(int(parts[1]))
        return f"{self.model_name}_{max(nums, default=0) + 1:03d}"

    # ══════════════════════════════════════════════════════════════════════════
    # Lightning hooks — Training
    # ══════════════════════════════════════════════════════════════════════════

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        sep = "=" * 64
        logging.info(sep)
        logging.info(f"MODELLO : {self.model_name.upper()}  |  RUN : {self.run_id}")
        logging.info(f"AVVIO   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(sep)
        logging.info("PARAMETRI TRAINING:")
        for k, v in sorted(vars(self.args).items()):
            logging.info(f"  {k:<35}: {v}")
        logging.info(sep)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        epoch   = trainer.current_epoch

        # 1. Recupera train_loss e lr dell'epoca corrente
        train_loss = float(metrics.get("train/loss_epoch", metrics.get("train/loss", float("nan"))))
        self.train_losses.append(train_loss)

        try:
            lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        except Exception:
            lr = float("nan")
        self.learning_rates.append(lr)

        # 2. Recupera metriche val dell'epoca corrente
        val_loss = float(metrics.get("val/loss", float("nan")))
        self.val_losses.append(val_loss)

        prec = [float(metrics.get(f"val/precision_{c}", float("nan"))) for c in self.CLASS_NAMES]
        rec  = [float(metrics.get(f"val/recall_{c}",    float("nan"))) for c in self.CLASS_NAMES]
        f1   = [float(metrics.get(f"val/f1_{c}",        float("nan"))) for c in self.CLASS_NAMES]

        self.val_prec.append(prec)
        self.val_rec.append(rec)
        self.val_f1.append(f1)

        # 3. Stampa log su file/console
        logging.info(
            f"Epoch {epoch:3d} | "
            f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
            f"mistake    → P:{prec[1]:.3f} R:{rec[1]:.3f} F1:{f1[1]:.3f} | "
            f"correction → P:{prec[2]:.3f} R:{rec[2]:.3f} F1:{f1[2]:.3f} | "
            f"lr: {lr:.2e}"
        )

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # No-op per evitare logging anticipato o disallineato (tutto delegato a on_train_epoch_end)
        pass

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        sep = "=" * 64
        logging.info(sep)
        logging.info("TRAINING COMPLETATO")
        if self.val_losses:
            best_ep = int(np.argmin(self.val_losses))
            logging.info(f"Best epoch    : {best_ep}")
            logging.info(f"Best val_loss : {self.val_losses[best_ep]:.4f}")
        logging.info(sep)
        self._generate_training_plots()
        logging.info(f"Grafici training salvati in: {self.fig_dir}")

    # ══════════════════════════════════════════════════════════════════════════
    # Lightning hooks — Test
    # ══════════════════════════════════════════════════════════════════════════

    def on_test_batch_end(
        self,
        trainer:        L.Trainer,
        pl_module:      L.LightningModule,
        outputs,
        batch,
        batch_idx:      int,
        dataloader_idx: int = 0,
    ) -> None:
        """Raccoglie predizioni e GT per il timeline plot (da test_step output)."""
        if not isinstance(outputs, dict):
            return
        preds  = outputs.get("preds",  None)
        labels = outputs.get("labels", None)
        if preds is None or labels is None:
            return

        B = batch["features"].shape[0]
        T = batch["labels"].shape[1]
        preds_2d  = preds.view(B, T)
        labels_2d = labels.view(B, T)

        for b in range(B):
            valid = labels_2d[b] != -1
            self.test_preds.append(preds_2d[b][valid].cpu())
            self.test_labels.append(labels_2d[b][valid].cpu())

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics = trainer.callback_metrics
        sep = "=" * 64
        logging.info(sep)
        logging.info("RISULTATI TEST SET:")
        for k in sorted(metrics.keys()):
            if k.startswith("test/"):
                logging.info(f"  {k}: {float(metrics[k]):.4f}")
        
        # Calcola e logga le metriche per classe aggregando le predizioni test raccolte
        if self.test_preds and self.test_labels:
            all_preds  = torch.cat(self.test_preds).numpy()
            all_labels = torch.cat(self.test_labels).numpy()
            from sklearn.metrics import precision_recall_fscore_support
            prec, rec, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, labels=[0, 1, 2], zero_division=0
            )
            for i, name in enumerate(self.CLASS_NAMES):
                logging.info(
                    f"  test/{name:<10} → P:{prec[i]:.4f} R:{rec[i]:.4f} F1:{f1[i]:.4f}"
                )
        logging.info(sep)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self.test_preds:
            logging.warning("[MetricsCallback] Nessun dato test raccolto — grafici test saltati.")
            return
        all_preds  = torch.cat(self.test_preds).numpy()
        all_labels = torch.cat(self.test_labels).numpy()
        self._generate_test_plots(all_preds, all_labels)
        logging.info(f"Grafici test salvati in: {self.fig_dir}")

    # ══════════════════════════════════════════════════════════════════════════
    # Plot helpers
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _setup_mpl():
        """Inizializza matplotlib con backend non-interattivo per cluster headless."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            try:
                plt.style.use("seaborn-whitegrid")
            except Exception:
                pass
        return plt

    def _generate_training_plots(self) -> None:
        plt    = self._setup_mpl()
        epochs = list(range(len(self.val_losses)))

        # 1. Loss curves ───────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        if self.train_losses:
            ax.plot(range(len(self.train_losses)), self.train_losses,
                    label="Train Loss", color="#2196F3", linewidth=2)
        ax.plot(epochs, self.val_losses,
                label="Val Loss", color="#F44336", linewidth=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(f"{self.run_id} — Loss Curves")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.fig_dir / "loss_curves.png", dpi=150); plt.close(fig)

        # 2. Precision per class ───────────────────────────────────────────────
        self._plot_metric_per_class(plt, self.val_prec, "Precision", "precision_per_class.png")

        # 3. Recall per class ──────────────────────────────────────────────────
        self._plot_metric_per_class(plt, self.val_rec, "Recall", "recall_per_class.png")

        # 4. F1 per class ──────────────────────────────────────────────────────
        self._plot_metric_per_class(plt, self.val_f1, "F1 Score", "f1_per_class.png")

        # 5. LR schedule ───────────────────────────────────────────────────────
        if self.learning_rates:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(len(self.learning_rates)), self.learning_rates,
                    color="#9C27B0", linewidth=2)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
            ax.set_title(f"{self.run_id} — Learning Rate Schedule")
            ax.set_yscale("log"); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.fig_dir / "lr_schedule.png", dpi=150); plt.close(fig)

    def _plot_metric_per_class(self, plt, data: list, metric_name: str, filename: str) -> None:
        if not data:
            return
        arr    = np.array(data)          # [epochs, 3]
        epochs = list(range(len(data)))
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (name, color) in enumerate(zip(self.CLASS_NAMES, self.CLASS_COLORS)):
            ax.plot(epochs, arr[:, i], label=name, color=color, linewidth=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel(metric_name)
        ax.set_title(f"{self.run_id} — Val {metric_name} per Class")
        ax.legend(); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.fig_dir / filename, dpi=150); plt.close(fig)

    def _generate_test_plots(self, all_preds: np.ndarray, all_labels: np.ndarray) -> None:
        plt = self._setup_mpl()
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

        # 6. Confusion Matrix ──────────────────────────────────────────────────
        cm      = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(self.CLASS_NAMES, fontsize=11)
        ax.set_yticklabels(self.CLASS_NAMES, fontsize=11)
        for i in range(3):
            for j in range(3):
                val = cm_norm[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val > 0.5 else "black", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
        ax.set_title(f"{self.run_id} — Test Confusion Matrix")
        fig.tight_layout()
        fig.savefig(self.fig_dir / "test_confusion_matrix.png", dpi=150); plt.close(fig)

        # 7. Bar chart P / R / F1 per classe ───────────────────────────────────
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=[0, 1, 2], zero_division=0
        )
        x = np.arange(3); w = 0.25
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - w, prec, w, label="Precision", color="#2196F3", alpha=0.88)
        ax.bar(x,     rec,  w, label="Recall",    color="#4CAF50", alpha=0.88)
        ax.bar(x + w, f1,   w, label="F1",        color="#FF9800", alpha=0.88)
        for xi, (p, r, f) in enumerate(zip(prec, rec, f1)):
            ax.text(xi - w, p + 0.01, f"{p:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(xi,     r + 0.01, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(xi + w, f + 0.01, f"{f:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(self.CLASS_NAMES, fontsize=11)
        ax.set_ylabel("Score"); ax.set_ylim(0, 1.12)
        ax.set_title(f"{self.run_id} — Test Metrics per Class")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(self.fig_dir / "test_metrics_bar.png", dpi=150); plt.close(fig)

        # 8. Timeline Ribbon ───────────────────────────────────────────────────
        self._plot_timeline(plt)

    def _plot_timeline(self, plt, max_videos: int = 5) -> None:
        """
        Grafico a nastri temporali: per ogni video mostra due strisce orizzontali
        (Ground Truth sopra, Predizioni sotto) colorate per classe.
        """
        import matplotlib.patches as mpatches

        n = min(max_videos, len(self.test_preds))
        if n == 0:
            return

        def hex_to_rgb(h: str):
            h = h.lstrip("#")
            return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]

        rgb_colors = [hex_to_rgb(c) for c in self.CLASS_COLORS]

        fig, axes = plt.subplots(n, 1, figsize=(18, n * 1.8 + 1.5))
        if n == 1:
            axes = [axes]

        for i in range(n):
            preds  = self.test_preds[i].numpy()
            labels = self.test_labels[i].numpy()
            T      = len(preds)
            ax     = axes[i]

            # Costruisce immagini RGB raster (molto più veloce di barh per T>1000)
            gt_img   = np.array([[rgb_colors[int(c)] for c in labels]])   # [1, T, 3]
            pred_img = np.array([[rgb_colors[int(c)] for c in preds]])    # [1, T, 3]

            ax.imshow(gt_img,   aspect="auto", extent=[0, T, 1, 2], interpolation="nearest")
            ax.imshow(pred_img, aspect="auto", extent=[0, T, 0, 1], interpolation="nearest")

            ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels(["Pred", "GT"], fontsize=8)
            ax.set_xlim(0, T); ax.set_ylim(0, 2)
            ax.set_title(f"Video {i + 1}  (T = {T} frames)", fontsize=9, pad=2)
            if i < n - 1:
                ax.set_xticklabels([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[-1].set_xlabel("Frame", fontsize=10)

        patches = [
            mpatches.Patch(color=self.CLASS_COLORS[j], label=self.CLASS_NAMES[j])
            for j in range(3)
        ]
        fig.legend(handles=patches, loc="upper right", fontsize=9, framealpha=0.9)
        fig.suptitle(
            f"{self.run_id} — Prediction vs Ground Truth Timeline",
            fontsize=12, y=1.005
        )
        fig.tight_layout()
        fig.savefig(self.fig_dir / "test_timeline.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
