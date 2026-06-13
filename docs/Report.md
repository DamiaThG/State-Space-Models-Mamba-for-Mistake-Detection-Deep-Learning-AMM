# State-Space Models (Mamba) per la Rilevazione di Errori (Mistake Detection)
**Progetto per il corso di *Deep Learning: Advanced Models and Methods (DL26)***  
**Università degli Studi di Catania — Dipartimento di Matematica e Informatica**  

---

## 1. Introduzione e Motivazione Scientifica

Nell'ambito del Deep Learning applicato alla Computer Vision sequenziale, la modellazione di contesti temporali a lungo termine rappresenta una delle sfide aperte più rilevanti. Con l'avvento di dataset moderni caratterizzati da sequenze video molto lunghe — che possono superare agevolmente l'ora di riproduzione continua — le architetture sequenziali tradizionali manifestano limiti strutturali intrinseci:

1. **Transformer (Self-Attention):** Nonostante l'eccezionale capacità di catturare relazioni a lungo termine, i Transformer presentano una complessità computazionale e spaziale (memoria GPU) quadratica, pari a $O(N^2)$ rispetto alla lunghezza della sequenza $N$. Questo rende proibitivo il processamento di interi video senza pesanti operazioni di sottocampionamento o partizionamento in finestre temporali, che frammentano il contesto globale.
2. **LSTM Classiche (Recurrent Neural Networks):** I modelli ricorrenti tradizionali hanno una complessità computazionale lineare $O(N)$ e uno stato di memoria di dimensione costante. Tuttavia, soffrono del fenomeno noto come *memory decay* (decadimento della memoria) e del gradiente svanente, faticando a mantenere informazioni utili quando le distanze temporali tra eventi correlati diventano molto grandi.

Questo progetto esplora soluzioni emergenti basate sulla teoria dei **State-Space Models (SSM) continui**, focalizzandosi in particolare sull'architettura **Mamba** e sul framework **xLSTM** (Extended LSTM). L'obiettivo principale è analizzare le performance, l'efficienza di memoria e la capacità di ritenzione del contesto a lungo termine nel task di **Mistake Detection** (rilevazione di errori procedurali frame-by-frame) su sequenze video estese.

---

## 2. Dataset e Definizione del Task

Il benchmark sperimentale utilizzato è **Assembly101**, un dataset su larga scala progettato per l'analisi di attività procedurali complesse (assemblaggio e disassemblaggio di giocattoli). 

### 2.1 Caratteristiche dei Dati
* **Input Feature:** Vettori di feature pre-estratte tramite architetture Temporal Shift Module (TSM) a 2048 dimensioni, che catturano informazioni spazio-temporali locali per ciascun frame.
* **Setup Video:** Vista singola (single view) con annotazioni dense frame-by-frame.
* **Classi del Task (3 classi):**
  1. `correct`: Frame in cui l'utente sta eseguendo la procedura correttamente.
  2. `mistake`: Frame corrispondenti all'esatto confine temporale in cui si verifica un errore procedurale.
  3. `correction`: Frame in cui l'utente esegue un'azione riparatrice per correggere l'errore.

### 2.2 Sfide Principali
* **Sbilanciamento Estremo:** La stragrande maggioranza dei frame appartiene alla classe `correct`. Gli eventi di errore (`mistake`) e le relative correzioni (`correction`) sono estremamente rari e sparsi nel tempo, rendendo le metriche standard (es. accuratezza globale) fuorvianti. Le metriche di riferimento obbligatorie sono la **Precision (P)**, la **Recall (R)** e l'**F1-Score** calcolati specificamente per le classi minoritarie, oltre al **Mean Average Precision (mAP)**.
* **Orizzonte Temporale Variabile:** Le sequenze presentano lunghezze che variano da poche centinaia di frame fino a oltre 20.000 frame per i video interi.

---

## 3. Descrizione delle Architetture ed Implementazione

Il lavoro mette a confronto tre paradigmi architetturali differenti, uniti da un framework di addestramento solido e condiviso per gestire le specifiche sfide del dataset.

### 3.1 Dettagli Architetturali dei Modelli

#### A. Baseline Coarse-to-Fine (TempAGG)
La baseline temporale riproduce il framework di temporal aggregation stabilito nel paper originale di Assembly101. Essa impiega blocchi di convoluzione temporale gerarchica a scale temporali distinte:
* **Recent Scales:** Finestre temporali strette (es. $[30, 90, 150]$ frame) per estrarre variazioni visive ad alta frequenza e dettagli immediati del movimento.
* **Spanning Scales:** Finestre temporali dilatate (es. $[8, 16, 24]$ passaggi) per catturare un contesto semantico più esteso attorno all'azione corrente.

#### B. Modello Mamba (State-Space Model)
Mamba sostituisce i meccanismi di attenzione globale (tipici dei Transformer) proiettando le feature video in uno spazio di stato continuo linearizzato. Grazie alla parametrizzazione tempo-variante delle matrici di transizione dello stato ($A, B, C$) e all'algoritmo di **Selective Scan** (ottimizzato tramite kernel C++ della libreria nativa `mamba_ssm`), Mamba mantiene una complessità $O(N)$ garantendo al contempo un flusso informativo non lineare in grado di "selezionare" quali pattern ricordare e quali ignorare.
L'architettura del nostro `MambaMistakeDetector` si sviluppa in tre fasi:
1. **Input Projection:** Trasforma le feature TSM in ingresso ($D_{in} = 2048$) nella dimensione interna ($d_{\text{model}}$), applicando LayerNorm, GELU e Dropout.
2. **Backbone Mamba:** Una sequenza impilata di $N$ blocchi causali Mamba, ciascuno dotato di pre-LayerNorm e Residual Connection.
3. **Classification Head:** Una rete MLP che riduce le feature a 3 classi finali.
$$\text{Input } [B, T, 2048] \rightarrow \text{Linear Projection } [B, T, d_{\text{model}}] \rightarrow \text{Mamba Backbone } (N \text{ layer}) \rightarrow \text{Classification Head} \rightarrow \text{Logits } [B, T, 3]$$

#### C. Variante xLSTM
Il framework xLSTM estende il concetto tradizionale di LSTM introducendo due varianti: sLSTM (con gating esponenziale) e mLSTM (con memoria a matrice). Nel nostro `xLSTMMistakeDetector` abbiamo adottato una struttura composta interamente da blocchi **mLSTM** puri. Sostituendo lo scalare/vettore nascosto classico con una vera e propria matrice di memoria, il modello raggiunge una capacità di stoccaggio dell'informazione nettamente superiore e permette un'elaborazione parallelizzata (stile self-attention) estremamente più efficiente rispetto alle classiche Reti Ricorrenti.

### 3.2 Gestione delle Sequenze a Lungo Termine

Processare interi video procedurali richiede accorgimenti specifici per la gestione della dimensione temporale $T$:
* **Troncamento e Orizzonte Temporale (`max_seq_len`):** Sebbene le architetture SSM permettano in teoria lunghezze di sequenza illimitate (Whole Video), all'atto pratico le dinamiche di ottimizzazione risultano più stabili limitando la lunghezza della sequenza (es. a 8.000 o 12.000 frame).
* **Padding e Masking:** I video di diversa durata all'interno dello stesso batch vengono allineati tramite un'apposita `attention_mask`. I logit in corrispondenza del padding vengono sistematicamente esclusi dal calcolo della loss (usando un `ignore_index = -1`), evitando di inquinare i gradienti.
* **Gradient Checkpointing:** Per scongiurare crash per esaurimento della VRAM su GPU (Out Of Memory) durante il training di sequenze che superano i 15.000 frame, abbiamo integrato il gradient checkpointing "per-block". Questa tecnica ricalcola gli attivatori intermedi al volo durante la backpropagation, riducendo drasticamente il *memory footprint* architetturale a scapito di un lieve aumento del tempo di calcolo.

### 3.3 Funzione di Costo: Focal Loss e Class Weighting

Per contrastare il massiccio sbilanciamento delle classi (dove i frame `correct` superano il 77%, contro solo il 15% di `mistake` e <7% di `correction`), abbiamo integrato una **Focal Loss multi-classe**:
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
Questa funzione agisce su due livelli:
1. **Fattore Focalizzante $\gamma$:** Il termine $(1 - p_t)^\gamma$ abbatte dinamicamente il peso degli esempi "facili". Se il modello classifica correttamente con elevata confidenza (es. i frame `correct` ripetitivi), il gradiente prodotto è trascurabile, costringendo la rete a focalizzarsi sugli errori difficili.
2. **Pesi di Classe Dinamici ($\alpha_t$):** Tramite il parametro `class_weight_exp`, i pesi attribuiti a `mistake` e `correction` vengono amplificati in proporzione inversa alla loro frequenza, controbilanciando numericamente il predominio della classe corretta nel dataset.

---

## 4. Analisi dei Risultati Sperimentali

I risultati di tutti gli 11 addestramenti registrati nella cartella `experiments/logs` sono stati estratti e riassunti nella seguente tabella comparativa:

| ID Esperimento | Modello | Configurazione Run (W&B) | Parametri Trainabili | Lunghezza Max Sequenza (`max_seq_len`) | Learning Rate | Weight Decay | Pesi Classi (`class_weight_exp`) | Dropout | Miglior Epoca | Miglior Val Loss | Test Loss | Test Correct F1 | Test Mistake F1 | Test Correction F1 | Test Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **baseline_001** | BASELINE | `tempagg-baseline-5300` | 9.06M | 8000 | 2.0e-04 | 0.0001 | 1.0 | 0.2 | Epoch 0 | 0.4863 | 0.7087 | 93.35% | **25.13%** | 2.91% | **40.46%** |
| **baseline_002** | BASELINE | `tempagg-baseline-5312` | 2.53M | 8000 | 2.0e-04 | 0.0001 | 1.0 | 0.2 | Epoch 0 | 0.3611 | 0.6847 | 94.16% | **25.62%** | 1.61% | **40.46%** |
| **mamba_001** | MAMBA | `mamba-ssm-wholevid-5318` | 11.36M | None (Whole Vid) | 5.0e-05 | 0.0100 | 1.5 | 0.4 | Epoch 2 | 0.5371 | 1.7684 | 95.62% | 13.99% | **4.38%** | 37.99% |
| **mamba_002** | MAMBA | `mamba-ssm-wholevid-5318` | 11.36M | 20000 | 5.0e-05 | 0.0010 | 1.0 | 0.2 | Epoch 5 | 0.2842 | 0.6491 | 95.58% | 13.61% | 3.42% | 37.53% |
| **mamba_003** | MAMBA | `mamba-ssm-wholevid-5324` | 11.36M | 12000 | 2.0e-05 | 0.0010 | 1.0 | 0.2 | Epoch 6 | 0.2959 | 0.4673 | 93.75% | 15.18% | 1.84% | 36.92% |
| **mamba_004** | MAMBA | `mamba-ssm-wholevid-5324` | 11.36M | 8000 | 1.0e-05 | 0.0010 | 1.0 | 0.2 | Epoch 8 | **0.2261** | **0.2605** | 93.76% | 10.88% | 1.61% | 35.41% |
| **mamba_005** | MAMBA | `mamba-ssm-wholevid-5324` | 21.53M | 12000 | 5.0e-05 | 0.0010 | 1.0 | 0.2 | Epoch 4 | 0.3044 | 0.8288 | 95.10% | 10.97% | 2.29% | 36.12% |
| **mamba_006** | MAMBA | `mamba-ssm-wholevid-5324` | **3.19M** | 12000 | 5.0e-05 | 0.0010 | 1.0 | 0.2 | Epoch 6 | 0.3002 | 0.5340 | 95.05% | 10.45% | 2.12% | 35.87% |
| **mamba_007** | MAMBA | `mamba-ssm-wholevid-5324` | 42.64M | 12000 | 5.0e-05 | 0.0010 | 1.0 | 0.2 | Epoch 5 | 0.3191 | 0.8731 | 95.49% | 11.84% | 3.40% | 36.91% |
| **xlstm_001** | XLSTM | `xlstm-wholevid-5292` | 10.89M | 8000 | 2.0e-05 | 0.0100 | 1.0 | 0.2 | Epoch 2 | 0.3303 | 0.9828 | **95.84%** | 10.85% | 1.25% | 35.98% |
| **xlstm_002** | XLSTM | `xlstm-wholevid-5299` | 4.54M | 8000 | 1.5e-05 | 0.0500 | **1.5** | 0.4 | Epoch 1 | 0.6906 | 0.9915 | 94.28% | 13.69% | 3.07% | 37.01% |

---

## 5. Discussione e Analisi Critica dei Risultati

### 5.1 Il Paradosso delle Performance: Baseline vs Mamba/xLSTM
Dall'analisi qualitativa delle metriche si osserva un comportamento all'apparenza controintuitivo: la baseline convoluzionale locale (C2F) ottiene l'F1-Score più elevato sulla classe `mistake` (~25.62%), superando sia Mamba (~15.18%) sia xLSTM (~13.69%).
* **Perché accade?** Il modello convoluzionale localizzato (TempAgg) lavora aggregando finestre temporali rigide e corte (da 30 a 150 frame). Poiché gli errori procedurali nel dataset Assembly101 sono caratterizzati da transizioni visive e dinamiche motorie rapide e localizzate, l'approccio convoluzionale agisce come un rilevatore di transizioni ad alta frequenza molto efficiente.
* **Il limite di Mamba e xLSTM:** Mamba e xLSTM proiettano l'intera sequenza video (fino a 12.000 o 20.000 frame) in uno stato nascosto continuo. Sulle sequenze estremamente lunghe, l'informazione locale e millimetrica associata all'istante esatto dell'errore rischia di venire parzialmente "diluita" o smussata all'interno dello stato globale, portando a una precisione temporale inferiore e quindi a un F1-score più basso.

### 5.2 L'Impatto della Lunghezza della Sequenza (`max_seq_len`) in Mamba
Confrontando le run di Mamba da `mamba_001` a `mamba_004` (in cui i parametri del modello sono a 11.36M, ma varia la lunghezza massima della sequenza in addestramento), emerge un trend chiaro:
1. **Whole Video (mamba_001):** Test loss molto elevata (1.7684), ma F1-score sulla classe `correction` migliore in assoluto (4.38%). Il modello fatica ad ottimizzare la loss su lunghezze variabili ed illimitate, ma cattura meglio le relazioni a lunghissimo termine necessarie per identificare le fasi di "correzione" (che avvengono molto dopo l'errore).
2. **Sequenze Troncate a 8000 (mamba_004):** Il modello raggiunge la test loss più bassa in assoluto (0.2605) e la miglior val loss (0.2261). Il training su sequenze più corte e standardizzate semplifica l'ottimizzazione matematica del selective scan di Mamba, sebbene limiti la capacità del modello di contestualizzare eventi separati da lunghi intervalli.

### 5.3 Analisi della Complessità del Modello (Model Capacity) in Mamba
Nelle run `mamba_003` (11.36M), `mamba_005` (21.53M), `mamba_006` (3.19M) e `mamba_007` (42.64M) a parità di lunghezza sequenza (12000) si analizza la variazione dei parametri:
* **Overfitting dei modelli grandi:** Il modello a 42.64M parametri (`mamba_007`) mostra un degrado netto della test loss (0.8731) e un F1-score sul mistake (11.84%) inferiore al modello standard da 11.36M (`mamba_003`, F1 15.18%). Ciò dimostra che un eccessivo aumento dei canali interni ($d_{\text{model}}$ o numero di layer) porta Mamba a memorizzare pattern spuri del rumore di fondo delle feature TSM, a causa della scarsità degli eventi di errore nel dataset.
* **Efficienza del modello compatto:** Il modello compatto a 3.19M parametri (`mamba_006`) ottiene una test loss di 0.5340 e un F1 sul mistake del 10.45%, dimostrandosi estremamente competitivo ed efficiente rispetto a varianti 10 volte più grandi.

### 5.4 Comportamento del Framework xLSTM
I risultati evidenziano che la variante xLSTM (`xlstm_002` con d_model=384, F1 mistake 13.69%) è altamente competitiva con Mamba a parità di lunghezza di sequenza (8000). 
* L'introduzione di pesi più sbilanciati (`class_weight_exp=1.5` nella run `xlstm_002`) ha permesso di aumentare significativamente il richiamo (Recall) sulle classi di errore, a scapito di un leggero incremento della test loss globale, evidenziando l'importanza cruciale del bilanciamento della loss nei task con sbilanciamento estremo.

---

### 5.5 Confronto con le Baseline Originali del Paper (Assembly101)
Al fine di contestualizzare l'F1-score apparentemente basso sulle classi di errore (10-25%), è cruciale confrontare la nostra formulazione del task con i risultati riportati dagli autori originali di Assembly101 (Sener et al., CVPR 2022).

1. **La Differenza del Task (Classificazione vs Segmentazione Densa):** Nel paper originale (Sezione 5.7 - Mistake detection), il task è formulato come *classificazione a livello di segmento*. Ai modelli vengono fornite le feature esattamente fino alla fine di un "coarse segment" già pre-tagliato. Il nostro progetto affronta invece un problema enormemente più complesso: la **segmentazione temporale densa frame-by-frame**. I nostri modelli elaborano il video continuo e devono individuare autonomamente i confini esatti (*error bounds*). Considerando la maggiore difficoltà, il 25.62% ottenuto dalla nostra baseline C2F è altamente competitivo rispetto alla Precision (30.8%) e Recall (46.6%) della baseline TSM nel paper sul task pre-segmentato.
2. **Il "Tetto" dell'Oracolo (Oracle Ceiling):** Nel paper, gli autori effettuano un esperimento fornendo in input l'etichetta testuale corretta dell'azione in corso (*Ground Truth coarse label*). Incredibilmente, pur conoscendo l'azione perfetta, l'oracolo ottiene solo una Precision del 48.6% e un Recall del 62.7% sui mistake. Questo parallelismo convalida la nostra tesi: l'individuazione degli errori in Assembly101 è un problema visivamente e semanticamente ambiguo. Esiste un "tetto massimo" di apprendimento causato dal dataset stesso, che i nostri modelli si stanno sforzando di approssimare senza l'ausilio di segmentazioni oracolo.
3. **Gestione dello Sbilanciamento:** Gli autori del paper confermano che su 60.000 segmenti di assemblaggio, solo il 15.9% sono mistake. Avendo esteso la valutazione a livello di frame, la rarità dell'evento nei nostri esperimenti è esacerbata. Anche gli autori originali ammettono la necessità di penalizzare la misclassificazione, confermando la correttezza metodologica della nostra scelta di adottare la `class_weight_exp` nella loss.

---

## 6. Analisi Qualitativa e Failure Cases

L'analisi dei log evidenzia che tutti i modelli presentano una precisione molto elevata sulla classe maggioritaria `correct` (F1 costantemente superiore al 93-95%). Tuttavia, l'F1-score sulle classi `mistake` e `correction` rimane basso in valore assoluto.

### Tipologia di Errori Comuni (Failure Cases)
1. **Ritardo Sistematico nei Boundary:** I modelli ricorrenti e SSM tendono a predire l'inizio di un errore con qualche frame di ritardo rispetto alle annotazioni umane, a causa dell'inerzia dello stato nascosto che richiede alcuni frame di feature "anomale" prima di commutare la propria classificazione.
2. **Confusione tra Mistake e Correction:** In attività procedurali veloci, i frame in cui si compie l'errore e i frame in cui si ripara l'errore condividono feature visive molto simili (es. mani che afferrano lo stesso componente del giocattolo). I modelli globali faticano a distinguere la semantica temporale delle due classi senza un segnale di contesto macroscopico.

---

## 7. Conclusioni e Sviluppi Futuri

Questo studio dimostra empiricamente le potenzialità e le sfide delle nuove architetture lineari (Mamba e xLSTM) applicate a task di video understanding su larga scala:
* **Efficienza di VRAM e Scalabilità:** Grazie al gradient checkpointing e alla complessità lineare, è stato possibile processare sequenze fino a 20.000 frame, traguardo proibitivo per architetture basate su Transformer standard.
* **Ottimizzazione delle Prestazioni:** La baseline C2F convoluzionale rimane superiore nella cattura di boundary locali ad alta frequenza, suggerendo che le sole architetture ricorrenti/SSM globali non siano sufficienti.

### Sviluppi Futuri Consigliati
1. **Architetture Ibride (Mamba + Convoluzioni):** Integrare una baseline locale convoluzionale (stile TempAgg) all'inizio del modello come estrattore di feature locali temporali, alimentando successivamente i blocchi Mamba con le feature così aggregate per la modellazione del contesto a lungo termine.
2. **Integrazione Multimodale:** Utilizzare feature audio o di tracciamento delle mani in combinazione con le feature video TSM per risolvere le ambiguità visive nei casi di fallimento.
3. **Analisi Asincrona (TeSTra):** Approfondire l'addestramento in modalità online asincrona per valutare il throughput in frame al secondo (FPS) in scenari di inferenza real-time.