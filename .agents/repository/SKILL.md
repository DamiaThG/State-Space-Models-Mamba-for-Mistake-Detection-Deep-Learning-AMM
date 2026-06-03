# 📁 Repository Structure

> State-Space Models (Mamba) for Mistake Detection — Deep Learning AMM

---

```
.
├── .agents/                        # Agent skill definitions (contesto compresso per AI)
│   ├── cluster/
│   │   └── SKILL.md                # Ambiente GCluster, SLURM, librerie, convenzioni
│   ├── dataset-form/
│   │   └── SKILL.MD                # Formato dataset, struttura .pt, output DataLoader
│   ├── mamba/
│   │   └── SKILL.MD                # Architettura Mamba (SSM)
│   ├── repository/
│   │   └── SKILL.md                # (questo file) Vista directory del progetto
│   ├── tempAgg/
│   │   └── SKILL.MD                # Architettura Temporal Aggregation baseline
│   └── xLSTM/
│       └── SKILL.MD                # Architettura xLSTM
│
├── data/
│   └── README.md                   # Istruzioni dataset
│
├── docs/
│   └── README.md                   # Relazione e presentazione progetto
│
├── experiments/
│   ├── checkpoints/
│   │   └── README.md               # Checkpoint salvati
│   ├── configs/
│   │   └── README.md               # File di configurazione esperimenti
│   └── logs/
│       └── README.md               # Log di training/valutazione
│
├── figures/
│   └── README.md                   # Figure e grafici
│
├── notebooks/
│   └── README.md                   # Jupyter notebooks
│
├── scripts/                        # Script di utilità e analisi
│   ├── TSM_features/
│   │   └── read_lmdb.py            # Lettura feature TSM da LMDB
│   ├── explore_dataset.py          # Esplorazione dataset
│   ├── explore_dataset.sh
│   ├── extract_tsm.sh              # Estrazione feature TSM
│   ├── get_read_lmdb.py
│   ├── get_read_lmdb.sh
│   ├── inspect_lmdb.py             # Ispezione file LMDB
│   ├── inspect_lmdb.sh
│   ├── test_dataloader.py          # Test del dataloader
│   └── test_video_lenght.py        # Test lunghezza video
│
├── src/                            # Codice sorgente principale
│   ├── __init__.py
│   ├── datasets/                   # Gestione dati
│   │   ├── __init__.py
│   │   ├── build_dataset.py        # Costruzione dataset
│   │   ├── dataloader.py           # DataLoader
│   │   ├── download_tsm.py         # Download feature TSM
│   │   └── transforms.py           # Trasformazioni dati
│   │
│   ├── evaluation/                 # Valutazione modelli
│   │   ├── __init__.py
│   │   ├── evaluate.py             # Script di evaluation
│   │   └── metrics.py              # Metriche (F1, AUC, ecc.)
│   │
│   ├── models/                     # Architetture dei modelli e componenti (classi nn.Module)
│   │   ├── __init__.py
│   │   ├── baseline.py             # Modello baseline (TempAgg)
│   │   ├── mamba_model.py          # Modello Mamba (SSM)
│   │   ├── testra_model.py         # Modello TeSTra
│   │   └── xlstm_model.py          # Modello xLSTM
│   │
│   ├── training/                   # Training loop
│   │   └── training_loop.py        # Ciclo di training principale, loss function custom ed optimizers custom
│   │
│   └── utils/                      # funzioni ausiliari
│       ├── __init__.py
│       └── visualization.py        # Visualizzazione risultati
│
├── .gitignore
└── README.md                       # README principale del progetto
```

---

## 📌 Componenti Principali

| Componente | Percorso | Descrizione |
|---|---|---|
| **Modelli** | `src/models/` | Baseline, Mamba, xLSTM, TeSTra |
| **Training** | `src/training/training_loop.py` | Loop di addestramento |
| **Dataset** | `src/datasets/` | Caricamento e preprocessing |
| **Evaluation** | `src/evaluation/` | Metriche e valutazione |
| **Script** | `scripts/` | Utility per LMDB e TSM features |
| **Agent Skills** | `.agents/` | Skill files per agenti AI (cluster, dataset, modelli) |
