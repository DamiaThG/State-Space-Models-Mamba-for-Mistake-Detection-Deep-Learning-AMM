# Proposta Metodologica: Dataloader a Video Intero per Mamba e xLSTM vs Baseline TempAgg

---

## 📌 Sintesi della Proposta

Questa proposta mira a risolvere il grave problema di **overfitting** e **instabilità della loss** riscontrato durante l'addestramento del modello Mamba per il task di *Mistake Detection* su *Assembly101*. 

La proposta prevede la **differenziazione del Dataloader tra i modelli**:
1.  **Baseline TempAgg:** Continuerà ad utilizzare il dataloader standard basato su campioni di azioni troncate/prefissi (max 500 frame) per via dei limiti fisici di memoria del ROI Pooling e della Cross-Attention.
2.  **Mamba & xLSTM:** Utilizzeranno un nuovo dataloader basato sull'**intero video come singola sequenza di frame etichettati**, sfruttando la loro complessità lineare $O(L)$ per contesti a lungo termine.

Questa scelta non è solo una soluzione tecnica, ma rappresenta la **dimostrazione empirica del vantaggio architetturale** dei modelli allo spazio di stato (SSM) e delle Extended LSTM (xLSTM) rispetto alle architetture basate su pooling temporale e attenzione.

---

## 🔍 1. Analisi del Problema Attuale (Perché Mamba overfitta?)

Nel setup corrente, il dataset genera un campione per ciascuna azione annotata in un video (righe del CSV). Per ogni azione che termina al frame $F_i$, il dataloader estrae la storia completa dal frame $0$ al frame $F_i$.

Questo approccio causa tre problemi principali:
1.  **Ridondanza dei Dati e Bias dei Prefissi:** Se un video ha $N$ azioni, la porzione iniziale del video $[0, F_1]$ viene processata e inclusa nella loss $N$ volte in una singola epoca. Questo forte sbilanciamento fa sì che Mamba (che è un modello causale) impari a memorizzare la "firma iniziale" di ogni video di train per indovinare a memoria dove avvengono i mistake, azzerando la loss di train (0.017) ma fallendo completamente su video non visti in validazione (loss a 4.062).
2.  **Moltiplicazione Inutile dei Calcoli:** Eseguire $N$ passi in avanti su prefissi sovrapposti dello stesso video comporta un costo computazionale enorme e ridondante sul cluster.
3.  **Instabilità dei Gradienti:** Lo sbilanciamento delle classi (con la classe `correction` pesata a ~38.5 nella loss per compensare la sua rarità al 6.7%) amplifica a dismisura l'effetto di ogni minimo errore di validazione del modello su video non visti, facendo esplodere la validation loss.

---

## 💡 2. La Soluzione: Dataloader Differenziati


### A. TempAgg (Baseline) — Perché soffre di OOM (Limiti di Memoria 4D)
*   **Funzionamento:** Il modulo `SpanningPastBlock` in [baseline.py](file:///c:/Users/ClaudioNunci/Documents/State-Space-Models-Mamba-for-Mistake-Detection-Deep-Learning-AMM/src/models/baseline.py) esegue un **ROI Pooling** della storia da $0$ a $t$ per ciascun istante temporale $t$. Questo genera tensori a 4 dimensioni di forma:
    $$\text{Shape} = [B, T, \text{scale}, D]$$
    dove $B$ è la batch size, $T$ la lunghezza temporale, $\text{scale}$ è il numero di snippet estratti (scale temporali) e $D$ la dimensione nascosta.
*   **Calcolo della Memoria per $T = 5000$ (Video Intero):**
    Se proviamo ad addestrare TempAgg su un video intero con parametri standard ($B=2$ video, $T=5000$ frame, $\text{scale}=24$, $D=512$):
    $$\text{Elementi del tensore} = 2 \times 5000 \times 24 \times 512 = 122.880.000 \text{ float32}$$
    $$\text{Memoria occupata} = 122.880.000 \times 4 \text{ byte} \approx \mathbf{491.5\text{ MB}}$$
    Poiché TempAgg calcola questo processo per 3 scale distinte ($8, 16, 24$) e deve memorizzare i grafi di autograd per il backward pass, il consumo di VRAM sale rapidamente a **10-15 GB**, portando a crash per memoria esaurita (OOM).
*   **Configurazione Proposta:** Manteniamo per la baseline il limite originale di `--max_seq_len 500` con campioni ritagliati per azione.

### B. Mamba & xLSTM — Perché hanno un consumo di memoria minimo (Complessità Lineare 3D)
*   **Funzionamento:** Mamba ([mamba_model.py](file:///c:/Users/ClaudioNunci/Documents/State-Space-Models-Mamba-for-Mistake-Detection-Deep-Learning-AMM/src/models/mamba_model.py)) non genera tensori intermedi a 4 dimensioni. Le sue attivazioni memorizzate rimangono sempre in formato 3D:
    $$\text{Shape} = [B, T, d_{\text{model}}]$$
*   **Calcolo della Memoria per $T = 5000$ (Video Intero):**
    Con gli stessi parametri ($B=2, T=5000, d_{\text{model}}=512$):
    $$\text{Elementi del tensore} = 2 \times 5000 \times 512 = 5.120.000 \text{ float32}$$
    $$\text{Memoria occupata} = 5.120.000 \times 4 \text{ byte} \approx \mathbf{20.48\text{ MB}}$$
    *   **Precisione Mista (FP16/BF16):** Nel nostro trainer (`precision="16-mixed"`), i dati occupano la metà dello spazio, ossia appena **10.24 MB** per tensore.
*   **L'ottimizzazione Hardware (Selective Scan Kernel):** A livello matematico, l'aggiornamento dello stato nascosto dello spazio degli stati richiederebbe una matrice enorme di dimensione $[B, T, d_{\text{model}}, d_{\text{state}}]$. Mamba risolve questo collo di bottiglia a livello di codice CUDA: **non scrive mai questa matrice nella memoria globale della GPU (VRAM)**, ma la ricalcola al volo direttamente nei registri interni della GPU (SRAM) durante la fase di retropropagazione.
*   **Risultato:** Mamba a 6 layer su sequenze di 5000 frame richiede solo circa **1-2 GB** di memoria di attivazione totale, risultando pienamente compatibile con le risorse del cluster.
*   **Configurazione Proposta:** Caricamento del video completo con un limite di sicurezza `--max_seq_len` fissato a **5000 o 7000 frame**.


---

## 📈 3. Vantaggi Scientifici e Presentazione al Docente

Presentare questo sdoppiamento della pipeline offre un forte valore accademico:

*   **Evidenza dei Vantaggi Teorici:** Dimostra concretamente che SSM e xLSTM superano il collo di bottiglia della memoria dei Transformer e delle baselines temporali a pooling. I modelli non hanno più bisogno di "finestre d'azione" artificiali ma possono analizzare il flusso video continuo.
*   **Eliminazione del Bias temporale:** Rimuovendo i campioni sovrapposti, eliminiamo il problema del "fingerprinting" dei video. Ogni frame viene visto una sola volta per epoca, costringendo il modello a trovare pattern semantici reali dei *mistakes* per abbassare la loss.
*   **Efficienza Energetica e Temporale:** Il numero di iterazioni per epoca sui modelli Mamba e xLSTM diminuirà drasticamente (pari al numero di video invece del numero totale di azioni). Questo si traduce in addestramenti fino a **15-20 volte più veloci** sul cluster.
*   **Allineamento con la fase di Inference:** Durante il test, il modello deve valutare il video in modalità "online" dall'inizio alla fine. Addestrarlo sull'intero video garantisce che lo stato nascosto $h(t)$ impari a mantenere le informazioni utili lungo tutta la timeline del video reale.

---

## 🛠️ 4. Dettagli di Implementazione ed Iperparametri

Per implementare con successo questa scelta sul cluster, adotteremo le seguenti accortezze tecniche:

1.  **Dataloader Dedicato:** Implementeremo una classe `WholeVideoDataset` che mappa `sequence_name` direttamente sul file `.pt` e legge le etichette cumulative per tutti i frame dal CSV, anziché iterare sulle singole righe delle azioni.
2.  **Limite di Sicurezza Temporale (`--max_seq_len` a 5000/7000):** Nonostante la complessità lineare, per evitare che singoli video "outlier" eccezionalmente lunghi rallentino il training o costringano a un padding eccessivo nei batch, applicheremo un tetto massimo a **5000 o 7000 frame**. La logica di troncamento (già integrata in [dataloader.py](file:///c:/Users/ClaudioNunci/Documents/State-Space-Models-Mamba-for-Mistake-Detection-Deep-Learning-AMM/src/datasets/dataloader.py#L127)) conserverà i frame più recenti, preservando il contesto causale dell'azione.
3.  **Bucketing / Grouped Batching:** Poiché i video hanno durate variabili, per evitare che video corti ricevano troppo padding (inutilizzato) all'interno di batch contenenti video molto lunghi, utilizzeremo un sampler che raggruppa nei batch i video con lunghezze simili.
4.  **Gradient Checkpointing:** Per Mamba e xLSTM, se la VRAM del cluster dovesse saturare su video estremamente lunghi, attiveremo l'opzione di *Gradient Checkpointing* presente in PyTorch, che ricalcola le attivazioni del forward pass durante il backward pass salvando fino all'80% di memoria GPU.
5.  **Regolarizzazione delle Feature:** Aggiungeremo uno strato di `nn.Dropout(0.2)` all'ingresso della proiezione lineare (`input_proj`) di Mamba per introdurre del rumore costruttivo sui vettori TSM pre-estratti, rendendo la memorizzazione statica ancora più difficile.

