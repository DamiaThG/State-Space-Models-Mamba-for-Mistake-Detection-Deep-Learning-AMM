# State-Space Models (Mamba) for Mistake Detection
### Progetto per il corso di *Deep Learning: Advanced Models and Methods*

---

## 📌 Descrizione del Progetto

Con l'allungarsi delle sequenze temporali nei task di computer vision e modellazione sequenziale — che possono raggiungere o superare l'ora di video continuo — i modelli tradizionali mostrano limiti strutturali evidenti:
* **Transformer:** Soffrono di una complessità computazionale e di memoria quadratica $O(N^2)$ rispetto alla lunghezza della sequenza.
* **LSTM classiche:** Sono soggette al fenomeno del *memory decay* (decadimento della memoria), faticando a mantenere dipendenze a lunghissimo termine.

Questo progetto si propone di superare tali limitazioni esplorando la matematica dei **State-Space Models (SSM) continui**, concentrandosi in particolare sulla recente architettura **Mamba**. L'obiettivo principale è elaborare sequenze temporali estremamente lunghe in modo efficiente dal punto di vista della memoria, al fine di individuare interazioni temporali altamente anomale: i **punti di errore (mistakes)** all'interno di task procedurali.

---

## 📊 Dataset

Il progetto utilizza il dataset **Assembly101**, un benchmark su larga scala per il riconoscimento e l'analisi di attività procedurali (assemblaggio/disassemblaggio di giocattoli complexes).

* **Feature utilizzate:** Feature architetturali pre-estratte.
* **Setup:** Utilizzo di una singola vista (singular view) mappata con annotazioni esplicite per-frame, mirate a individuare gli esatti confini temporali degli errori (*error bounds*).

---

## 🎯 Obiettivi del Progetto

### Obiettivi Minimi (Milestones)
1.  **Baseline C2F:** Implementazione e replica della baseline di *error-detection* Coarse-to-Fine (C2F) stabilita nel paper originale del dataset.
2.  **Mamba Protocol:** Integrazione delle librerie SSM native (`mamba-ssm`) per sostituire l'elaborazione dei blocchi temporali classici con matrici dello spazio di stato continuo.
3.  **Variante xLSTM:** Costruzione della logica procedurale utilizzando il framework **xLSTM** (Extended LSTM) come ulteriore termine di paragone avanzato.
4.  **Verifica e Benchmarking:** Calcolo e confronto delle matrici di benchmark per dimostrare la reale capacità di ritenzione su lunghe sequenze (*long-sequence retention*) confrontando i tre framework: **Mamba vs xLSTM vs Baseline C2F**.

### Obiettivi Extra (Estensioni)
* **Asynchronous Testing Variant:** Implementazione di una variante di test asincrona basata su Transformer temporali online (es. **TeSTra**) per mappare i vantaggi architetturali relativi in scenari real-time.
* **Horizon Manipulation Analysis:** Studio di come la compressione o l'espansione artificiale dell'orizzonte delle sequenze di input manipoli drasticamente le dipendenze prestazionali tra LSTM e State-Space Models.

---

## 🛠️ Tecnologie e Librerie Utilizzate

* **Deep Learning Framework:** PyTorch
* **State-Space Models:** `mamba-ssm` (causal Conv1d e Linear Attention selettiva)
* **Baselines Sequenziali:** Framework xLSTM, Standard LSTM, Transformer (TeSTra)
* **Data Processing:** NumPy, Pandas, SciPy (per la gestione delle feature di Assembly101)

---

## 📁 Struttura della Repository

```text
├── data/                  # Cartella per i vettori di feature di Assembly101 (o istruzioni per il download)
├── src/
│   ├── baselines/         # Implementazione C2F baseline e modelli LSTM
│   ├── models/            # Implementazione dei blocchi Mamba e xLSTM
│   ├── utils/             # Data loader, metriche di valutazione e script di processing
│   └── train.py           # Script principale per il training dei modelli
├── notebooks/             # Jupyter Notebooks per l'analisi esplorativa e i grafici dei benchmark
├── requirements.txt       # Dipendenze del progetto
└── README.md              # Questo file
```

## 📈 Risultati Attesi e Metriche
I modelli verranno valutati sulla precisione nel localizzare i confini temporali dei mistake. Le metriche principali includeranno:

* F1-Score / Mean Average Precision (mAP) sulla rilevazione dei bound di errore.

* Memory Footprint (GPU VRAM) al variare della lunghezza della sequenza.

* Inference Throughput (frames/sec) per misurare l'efficienza temporale dei modelli a confronto.

🎓 Contesto Accademico
Progetto sviluppato per il modulo di Advanced Sequential Modeling all'interno del corso d'esame di Deep Learning: Advanced Models and Methods.
