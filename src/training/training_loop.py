"""
training_loop.py — Shim di compatibilità
=========================================
Questo file esiste solo per non rompere import esistenti in notebook o script
che referenziano ancora 'src.training.training_loop'.

La logica vera è stata spostata in:
  - src/training/lightning_module.py  → MistakeDetectionLightningModule
  - src/training/train_baseline.py    → training TempAgg (ex main() di qui)
  - src/training/train_mamba_whole_video.py
  - src/training/train_xlstm_whole_video.py

NON aggiungere nuova logica qui.
"""

# Re-export tutto ciò che era pubblico
from src.training.lightning_module import (  # noqa: F401
    compute_class_weights,
    MistakeDetectionLightningModule,
    TempAggLightningModule,
)
