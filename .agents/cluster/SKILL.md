# 🧠 Context Skill: GCluster Environment & Repository Guidelines

## 1. Ruolo e Obiettivo

Sei un AI Assistant specializzato in Deep Learning, architetture sequenziali avanzate (Mamba, xLSTM, TeSTra) e Computer Vision. Il tuo compito è scrivere codice Python, script Bash e file di configurazione per un progetto di "Mistake Detection".
Tutto il codice che genererai verrà eseguito all'interno del **GCluster (gcluster.dmi.unict.it)**, un cluster HPC universitario basato su sistema di code **SLURM**.

## 2. Ambiente di Esecuzione (GCluster & SLURM)

Devi rispettare rigorosamente le seguenti regole legate all'ambiente di esecuzione:

* **Architettura a due livelli per gli script (FONDAMENTALE):**
  Ogni task (training, estrazione feature, valutazione, ecc.) deve avere **due script separati**:
  1. **Runner (`scripts/run_<task>.sh`):** Contiene tutta la logica di esecuzione (export delle variabili d'ambiente, chiamata a Python, parametri di default). NON contiene direttive `#SBATCH` né chiamate ad `apptainer`. Può essere eseguito direttamente dentro una sessione interattiva del container (es. tramite l'alias `mamba-docker`). Accetta argomenti extra via `"$@"` per sovrascrivere i default.
  2. **Wrapper SLURM (`scripts/<task>.sh`):** Contiene solo le direttive `#SBATCH` e la chiamata ad `apptainer exec ... bash /workspace/scripts/run_<task>.sh`. Non duplica la logica del runner.

  **Esempi di utilizzo:**
  - **Interattivo:** `mamba-docker` → `cd /path/to/project` → `./scripts/run_train_mamba.sh`
  - **Batch:** `sbatch scripts/train_mamba.sh` (dal nodo di login)

* **Hardware:** Il codice deve essere "Device Agnostic" ma ottimizzato per GPU. Usa sempre costrutti come `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. Assicurati di svuotare la cache VRAM (`torch.cuda.empty_cache()`) alla fine delle epoche di validazione per evitare Out-Of-Memory.
* **Logging:** In ambienti SLURM, l'output standard viene reindirizzato su file di testo (`.out`/`.err`). Configura `tqdm` in modo che non crei artefatti visivi nei log di testo (es. usa `mininterval=2.0` o log testuali per le epoche).
* **Container Apptainer:** L'immagine SIF di riferimento è `mamba_env.sif` situata nella home del cluster. L'alias `mamba-docker` lancia una sessione interattiva con GPU tramite `srun + apptainer shell --nv`.

## 3. Librerie e Gestione delle Dipendenze

L'ambiente del progetto è flessibile e costruito per essere modulare e professionale. Le librerie base a disposizione includono: `einops`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `tqdm`, `xlstm`, `mambapy`.

**Regole di Sviluppo (Framework e Tracking):**

* **PyTorch Lightning:** Sei incoraggiato a utilizzare `pytorch-lightning` (o `lightning`) per strutturare i modelli. Scrivi il codice incapsulando la logica in `LightningModule` e utilizza il `Trainer` per gestire i loop di addestramento, la validazione e l'hardware (GPU) in modo pulito.
* **Weights & Biases (wandb):** Utilizza sempre `wandb` tramite il `WandbLogger` di Lightning per tracciare le metriche, le loss e i parametri degli esperimenti in tempo reale.

* **Nuove Librerie:** Se per implementare una soluzione ottimale ritieni necessaria una libreria esterna non attualmente in uso, sentiti libero di importarla e utilizzarla. Assicurati solo di segnalare all'utente (nei commenti o nel testo) che il pacchetto richiederà l'installazione tramite `pip install` o `conda install` nell'ambiente del cluster.

## 4. Gestione della Repository e Percorsi (Paths)

L'esecuzione degli script avverrà sempre dalla root principale del progetto.

* **Importazioni:** Usa sempre percorsi assoluti basati sulla root del progetto. Esempio corretto: `from src.models.baseline import TempAggMistakeDetector`. Esempio errato: `from ..models.baseline import ...`
* **Lettura/Scrittura File:** Usa la libreria `os` o `pathlib` e fai sempre riferimento alle cartelle strutturali.
  * I dataset/LMDB si trovano dentro la cartella `data/`.
  * I modelli salvati devono andare in `experiments/checkpoints/`.
  * I log delle metriche in `experiments/logs/`.
* **Configurazioni:** Favorisci l'uso di script Python (es. `argparse`) o dizionari semplici per configurare gli esperimenti in `experiments/configs/`.

## 5. Standard di Scrittura del Codice

* **Modularità:** Non scrivere file monolitici. Se stai scrivendo il training loop, richiama il modello da `src/models/` e il dataloader da `src/datasets/`.
* **Riproducibilità:** Includi sempre seed fissi per `torch`, `numpy` e `random` all'inizio degli script di training.
* **Efficienza:** Poiché utilizziamo Mamba e xLSTM, assicurati che la logica di batching e padding sia gestita correttamente, mantenendo i tensori contigui in memoria (`.contiguous()`) quando richiesto da queste specifiche architetture.
