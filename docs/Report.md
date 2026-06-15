# State-Space Models (Mamba) per la Rilevazione di Errori (Mistake Detection)
**Progetto per il corso di *Deep Learning: Advanced Models and Methods (DL26)***  
**Università degli Studi di Catania — Dipartimento di Matematica e Informatica**

**Gruppo LeMeCla: Messina Damiano, Barbagallo Emanuele, Nuncibello Claudio** 

---

## 1. Introduzione e Motivazione Scientifica

Nell'ambito del Deep Learning applicato alla Computer Vision sequenziale e al Video Understanding, la modellazione rigorosa e coerente di contesti temporali a lunghissimo termine rappresenta una delle sfide di ricerca aperte più critiche. Con l'avvento di dataset moderni caratterizzati da sequenze video egocentriche (First-Person Vision) o procedurali estremamente lunghe (che possono superare agevolmente l'ora di riproduzione continua) le architetture sequenziali tradizionali manifestano limiti strutturali intrinseci, sia dal punto di vista matematico che computazionale.

L'individuazione di errori (Mistake Detection) all'interno di complesse attività procedurali è un compito particolarmente ostico: un errore non è quasi mai definito da un singolo frame isolato, bensì dalla violazione di una specifica grammatica procedurale o di una dipendenza causale con azioni compiute potenzialmente migliaia di frame prima. In questo scenario:
1. **Transformer (Self-Attention):** Nonostante l'eccezionale capacità di catturare relazioni a lungo termine e di mitigare il problema del *vanishing gradient*, i modelli basati sull'attenzione globale presentano una complessità computazionale e spaziale (memoria GPU) quadratica, pari a $\mathcal{O}(N^2)$ rispetto alla lunghezza della sequenza $N$. Questo rende fisicamente proibitivo il processamento di interi video senza pesanti operazioni di sottocampionamento o partizionamento in finestre temporali locali, le quali frammentano inevitabilmente il contesto globale dell'azione.
2. **LSTM Classiche (Recurrent Neural Networks):** I modelli ricorrenti tradizionali mantengono il pregio di una complessità computazionale lineare $\mathcal{O}(N)$ e di uno stato di memoria di dimensione costante durante l'inferenza. Tuttavia, soffrono intrinsecamente del fenomeno noto come *memory decay* (decadimento della memoria) e fatica di ottimizzazione dovuta al *vanishing gradient*. Questo porta la rete a sovrascrivere o perdere informazioni cruciali quando le distanze temporali tra eventi causalmente correlati si dilatano eccessivamente.

Il presente progetto esplora soluzioni emergenti e paradigmi matematici alternativi basati sulla teoria dei **State-Space Models (SSM) continui**, focalizzandosi in modo particolare sulla recente e promettente architettura **Mamba**, e mettendola a confronto con il framework **xLSTM** (Extended LSTM) e con una robusta baseline convoluzionale spaziotemporale (*TempAgg*). L'obiettivo primario è valutare empiricamente le performance, l'efficienza di memoria (VRAM footprint) e la reale capacità di ritenzione del contesto a lungo termine nel task di **Mistake Detection** frame-by-frame su sequenze video non partizionate.

---

## 2. Dataset e Definizione del Task

Il benchmark sperimentale adottato per questo progetto è **Assembly101**, un dataset videocentrico su larghissima scala specificamente progettato per il riconoscimento e l'analisi di attività procedurali complesse, quali l'assemblaggio e disassemblaggio strutturato di veicoli giocattolo. 

### 2.1 Caratteristiche dei Dati
* **Input Feature:** L'elaborazione non avviene direttamente sui tensori RGB grezzi, al fine di ridurre il carico computazionale. Il sistema ingerisce vettori di feature pre-estratte tramite architetture *Temporal Shift Module* (TSM) a 2048 dimensioni spaziali. Queste feature catturano una rappresentazione densa e compatta delle informazioni spazio-temporali locali per ciascun frame della sequenza.
* **Setup Video:** Il progetto fa affidamento su una vista singola (*single view*), accompagnata da annotazioni dense di *ground truth* frame-by-frame fornite dagli annotatori tramite csv di annotazione, le quali sono state trasferite sui singoli frame.
* **Classi del Task (Classificazione a 3 vie):** Il problema è formulato come un task di classificazione densa a livello di frame, partizionato in tre classi mutuamente esclusive:
  1. `correct` (0): Frame in cui l'operatore sta eseguendo la procedura correttamente.
  2. `mistake` (1): Frame corrispondenti all'esatto intervallo temporale in cui si verifica e si concretizza un errore procedurale (es. inserimento di un componente errato o in una posizione scorretta).
  3. `correction` (2): Frame in cui l'operatore riconosce l'errore commesso ed esegue un'azione riparatrice volta a ripristinare lo stato corretto dell'assemblaggio.

### 2.2 Sfide Intrinseche
L'approccio scelto espone a due ostacoli di natura statistica e computazionale:
* **Sbilanciamento Estremo delle Classi:** Negli scenari reali e in Assembly101, la stragrande maggioranza del tempo di riproduzione appartiene alla classe `correct` (oltre il 77%). Gli eventi di errore (`mistake`) e le conseguenti correzioni (`correction`) sono anomalie rare e fortemente sparse nel dominio temporale. L'utilizzo di metriche classiche come l'accuratezza globale risulta perciò fuorviante (una rete che predice costantemente `correct` otterrebbe un'accuratezza altissima). Per garantire validità scientifica, le metriche di valutazione elette a riferimento sono **Precision (P)**, **Recall (R)** e l'**F1-Score**, calcolate specificamente per ciascuna classe, nonchè **MacroF1-Score**.
* **Orizzonte Temporale Fortemente Variabile:** La lunghezza delle sequenze è profondamente disomogenea. Si passa da clip contenenti alcune migliaia di frame a procedure estese che oltrepassano i 50.000 frame, introducendo vincoli severi per il raggruppamento in tensori (*batching*). Nello specifico ***la media*** di frame per video è pari a 12.739 frame.

---

## 3. Ottimizzazione del Data Pipeline e Sampling

Al fine di addestrare architetture ad altissima capacità su sequenze contenenti decine di migliaia di frame senza esaurire la memoria della GPU, l'infrastruttura di dataloading è stata oggetto di un profondo lavoro di ottimizzazione ingegneristica.

### 3.1 Prevenzione del Data Leakage (Sequence-Level Split)
A differenza dei task di classificazione statici, i campioni nel dominio video sono sequenze continue. La partizione dei set di *Train*, *Validation* e *Test* è stata implementata a livello di *intera sequenza* piuttosto che a livello di sample (*frame/snippet*). Splittare il dataset a livello di singolo snippet comporterebbe un grave *data leakage*, in quanto porzioni temporali adiacenti dello stesso video finirebbero in partizioni diverse, falsando radicalmente le metriche di validazione. Il codice nel `dataloader.py` garantisce l'isolamento causale e semantico distribuendo interi video o solo nel train, o solo nel test.

### 3.2 Dualità del Dataloader: MistakeDetection vs WholeVideo
Sono state implementate due classi differenziate per l'estrazione dei dati, in funzione del paradigma architetturale testato:
* **MistakeDetectionDataset (Per le baseline basate su storie causali):** Questa classe estrae i dati in conformità a un paradigma "online" rigoroso. Ogni campione rappresenta una storia causale definita dall'intervallo $[0 \dots \text{end\_frame}]$. Durante l'inferenza, il modello non ha alcun accesso alle feature future, simulando fedelmente le condizioni di deployment real-time.
* **WholeVideoDataset (Per le architetture globali come Mamba e xLSTM):** I modelli Sequence-to-Sequence (Seq2Seq) ad elevata capacità vengono addestrati ingerendo la quasi totalità del video in un singolo passaggio. In presenza di video la cui lunghezza eccede un limite soglia prestabilito (`max_seq_len`, es. 20.000 frame), viene applicata una logica di troncamento direzionale **tail-oriented**. Piuttosto che troncare la fine del video, si scarta la parte iniziale più antica. Questa scelta euristica deriva dall'osservazione statistica che le procedure tendono a deragliare (generando anomalie, errori e correzioni) con maggior frequenza nelle fasi avanzate dell'assemblaggio, quando la stanchezza cognitiva dell'operatore aumenta e la complessità strutturale del manufatto cresce. Conservare gli ultimi frame massimizza la densità informativa delle classi minoritarie nel tensore di input.

### 3.3 Gestione del Padding e Length-Grouped Sampling
Ingerire sequenze di lunghezza fortemente variabile all'interno dello stesso mini-batch causa uno spreco colossale di risorse computazionali e di VRAM (dovute all'inserimento di tensori di zero-padding necessari per squadrare il batch).
Per abbattere drasticamente questo spreco, è stato ingegnerizzato un `LengthGroupedSampler`. Questa componente algoritmica analizza *a priori* la lunghezza totale di ciascun video presente nel dataset, per poi aggregare all'interno del medesimo batch esclusivamente sequenze aventi dimensioni simili. Questo meccanismo abbatte il padding medio del 70-80%, consentendo di allocare una quantità significativamente maggiore di batch e di addestrare modelli molto più complessi a parità di risorse hardware.

---

## 4. Descrizione delle Architetture ed Implementazione

La ricerca valuta e mette a confronto tre paradigmi architetturali intrinsecamente differenti.

### 4.1 Baseline Temporal Aggregate Representation (TempAgg)
Il primo modello funge da baseline locale e trae ispirazione dai framework convoluzionali spaziotemporali proposti nella letteratura originale di Assembly101. L'architettura abbandona la ricorrenza in favore di blocchi di convoluzione e aggregazione temporale gerarchica, articolandosi nei seguenti moduli:
* **SpanningPastBlock:** Divide l'intera storia temporale pregressa (da $0$ a $t$) in un numero fisso e predefinito di segmenti ("scale", es. 8, 16, 24). L'estrazione avviene mediante un algoritmo di *Region of Interest (ROI) Pooling* unidimensionale, in grado di comprimere porzioni di memoria storicamente distanti.
* **RecentPastBlock:** Focalizza l'analisi su finestre temporali recenti molto strette (es. $[30, 90, 150]$ frame), estraendo snippet locali per catturare transizioni visive ad alta frequenza e micromovimenti dell'operatore.
* **DynamicNonLocalBlock:** Attua un sofisticato incrocio relazionale (attention mechanism). Il modulo applica prima una *Self-Attention Causale* puramente sullo Spanning Past, e successivamente una *Cross-Attention* asimmetrica, in cui la rappresentazione recente funge da Query (*cosa sto facendo ora*), e lo Spanning Past funge da Key/Value (*cosa ho fatto in precedenza*).
L'aggregazione finale genera un output in grado di bilanciare reattività immediata e contesto pregresso.

### 4.2 Modello Mamba (State-Space Model)
Il modello centrale dello studio, denominato `MambaMistakeDetector`, si basa sulla discretizzazione differenziale continua proposta dalla famiglia dei *Continuous State-Space Models*. 
Mamba sostituisce integralmente i pesanti meccanismi di Self-Attention quadratica proiettando le feature video all'interno di uno spazio di stato continuo linearizzato, la cui discretizzazione produce una complessità algoritmica garantita $\mathcal{O}(N)$. L'innovazione fondamentale di Mamba risiede nella parametrizzazione *tempo-variante* delle matrici di transizione dello stato ($A, B, C$) e nel conseguente algoritmo hardware-aware di **Selective Scan**, che ottimizza la lettura/scrittura sui registri ad altissima velocità della GPU (SRAM) aggirando i colli di bottiglia della High Bandwidth Memory.

Dal punto di vista dell'implementazione, il flusso elabora il segnale secondo il seguente schema:
1. **Input Projection:** Le feature TSM ($2048$ dimensioni) vengono proiettate ortogonalmente ad una dimensione interna (es. $d_{\text{model}} = 512$), normalizzate via LayerNorm e attivate da una funzione non lineare GELU.
2. **Mamba Backbone:** Il nucleo poggia su una pila di blocchi Mamba causali. Per questa implementazione, il modello istanzia nativamente la libreria ufficiale in C++ e CUDA (`mamba_ssm`) per garantire la massima accelerazione hardware e l'ottimizzazione del footprint in VRAM. I layer sono protetti da connessioni residuali stabili per scongiurare il gradiente svanente nella propagazione all'indietro.
3. **Classification Head:** Una rete MLP che riduce lo spazio $d_{\text{model}}$ nei logit bidimensionali necessari per la classificazione a tre classi mutuamente esclusive.

### 4.3 Variante xLSTM
Come termine di paragone per i modelli sequenziali, è stata introdotta un'architettura basata sull'avanzato framework **xLSTM** (Extended LSTM). xLSTM rivoluziona le RNN introducendo gating esponenziali (sLSTM) e stati di memoria a struttura matriciale (mLSTM).
La scelta architetturale all'interno del progetto è ricaduta esclusivamente su un design **mLSTM puro**. Sostituendo il classico vettore di stato nascosto scalare con una matrice di memoria ad alta capacità e rimuovendo i vincoli di non linearità stretti tra passi temporali, il blocco mLSTM consente la parallelizzazione totale dei calcoli computazionali sul versante hardware (in maniera pressoché identica ai meccanismi *Key-Value* dei Transformer). Questo fornisce al modello una dinamica di *long-sequence retention* nettamente superiore alle LSTM tradizionali.

### 4.4 Ingegneria di Training e Ottimizzazione
Per addestrare modelli che processano tensori di dimensioni nell'ordine dei gigabyte senza incorrere in crash catastrofici della memoria (Out Of Memory Error), la pipeline si è dotata di metodologie all'avanguardia:
* **PyTorch Lightning:** Utilizzato per garantire una struttura a blocchi modulari del *training loop*, gestire l'accelerazione asincrona multi-GPU, il clipping automatico del gradiente e standardizzare il *logging* su interfacce remote.
* **Gradient Checkpointing Per-Block:** Si tratta della soluzione chiave che ha reso possibile la convergenza del progetto. Invece di conservare staticamente tutti i tensori intermedi delle attivazioni neurali calcolate durante il *forward pass* (necessari per calcolare le derivate durante la *backpropagation*), questi vengono scartati e ricalcolati dinamicamente, "on-the-fly", blocco per blocco. Questa strategia baratta un moderato incremento nel tempo di elaborazione computazionale con un risparmio colossale della VRAM, abilitando finestre temporali da 20.000+ frame.
* **Class Weighting Esponenziale:** Affrontare la rarità intrinseca delle classi `mistake` e `correction` richiede forzature matematiche. La `Cross-Entropy Loss` è stata irrobustita attribuendo pesi differenziali alle classi, calcolati in modo inversamente proporzionale alla loro frequenza cardinale nel dataset. Un iperparametro custom, `class_weight_exp`, regola la severità dell'esponenziazione: applicando pesi più aggressivi, il modello è matematicamente costretto a penalizzare drasticamente gli errori sulle classi minoritarie.

---

## 5. Analisi dei Risultati Sperimentali

I risultati di tutti gli addestramenti, le esplorazioni iperparametriche e i *run* registrati nella directory `experiments/logs` sono stati aggregati, estratti e sintetizzati nella seguente tabella riepilogativa, focalizzando l'analisi sull'F1-Score (la metrica più sensibile alla performance sulle anomalie).

| ID Esperimento | Modello | Configurazione Run | Parametri Trainabili | Lunghezza Max Sequenza (`max_seq_len`) | Learning Rate | Weight Decay | Pesi Classi (`class_weight_exp`) | Dropout | Miglior Epoca | Miglior Val Loss | Test Loss | Test Correct F1 | Test Mistake F1 | Test Correction F1 | Test Macro F1 |
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

## 6. Discussione e Analisi Critica dei Risultati

### 6.1 Il Paradosso delle Performance (Spazio Locale vs Spazio Globale)
L'osservazione dei risultati restituisce un quadro apparentemente antitetico rispetto alla capacità attesa dei modelli. L'architettura TempAgg (Baseline Convoluzionale Gerarchica) si distingue stabilendo i record empirici sull'individuazione degli errori, sfiorando il ~25.62% di F1-Score sulla classe `mistake`, e sovrastando le performance assolute sia di Mamba (~15.18%) sia di xLSTM (~13.69%).
Tale scarto si giustifica alla luce del bias induttivo intrinseco alle convoluzioni: la baseline TempAgg aggrega in modo rigido e meccanico snippet locali (da 30 a 150 frame). Poiché gli errori di tipo procedurale all'interno del dataset si manifestano visivamente come transizioni motorie o cambiamenti di pattern visivi ad altissima frequenza e dalla durata limitata, la convoluzione agisce come un rilevatore di spigoli temporali altamente specializzato e reattivo. Al contrario, i modelli globali a spazio continuo (Mamba, xLSTM) ingeriscono e comprimono tutto il video in uno stato di memoria latente denso. All'interno di tale astrazione globale prolungata per decine di migliaia di step temporali, il minuscolo e transitorio segnale associato ad una mano che compie l'azione scorretta rischia di venire matematicamente "diluito", producendo contorni di classificazione temporale sfocati e conseguente abbassamento della metrica di precisione frame-by-frame.

### 6.2 L'Impatto dell'Orizzonte Temporale (`max_seq_len`) nei Modelli Mamba
Esaminando le iterazioni del modello Mamba da `mamba_001` a `mamba_004` — in cui i parametri strutturali restano inalterati a 11.36M, ma subisce variazione l'orizzonte temporale limite concesso — emerge una solida tendenza matematica:
1. **Apprendimento su Interi Video Senza Troncatura (mamba_001):** Esporre la rete a video non manipolati di qualsiasi lunghezza ha generato fortissima instabilità, palesata da una Test Loss altamente divergente (1.7684). Sorprendentemente, però, proprio tale modello garantisce il vertice assoluto nel riconoscimento della fase di correzione (`correction`, 4.38%). Questa anomalia suggerisce che l'assenza di limitazioni temporali permetta al modello di strutturare le relazioni di causa-effetto necessarie ad identificare una correzione, la quale si manifesta tipicamente come un ripristino di uno stato iniziale visivamente assente da svariati minuti.
2. **Standardizzazione Orizzontale a 8000 Frame (mamba_004):** Operando su sequenze più ridotte, regolari e prevedibili, l'algoritmo di Selective Scan di Mamba giunge ad un'ottimizzazione ottimale delle distribuzioni di probabilità, stabilendo il minimo storico sia per la Test Loss (0.2605) che per la Val Loss (0.2261), pur sacrificando marginalmente il riconoscimento degli eventi storicamente remoti.

### 6.3 Analisi della Model Capacity (Complessità Parametrica vs Generalizzazione)
Per disaccoppiare la performance della rete dalla sua ampiezza, sono state valutate variazioni della dimensionalità del tensore interno di Mamba. Analizzando a lunghezza costante le run `mamba_003` (11.36M), `mamba_005` (21.53M), `mamba_006` (3.19M) e `mamba_007` (42.64M) si nota che:
* Esiste una spiccata propensione all'**overfitting causato dall'alta dimensionalità**. Il modello Mamba mastodontico a 42.64M di parametri esibisce un netto deterioramento prestazionale (F1 mistake che precipita all'11.84%, a fronte del 15.18% del modello da 11.36M). Allargare eccessivamente i canali dello spazio latente in presenza di anomalie rarissime spinge i gradienti ad aggrapparsi al rumore ad alta frequenza proprio dei feature vector TSM, memorizzando artefatti di background e fallendo il test di generalizzazione.
* In direzione opposta, l'estrema efficienza dell'architettura si conferma nella configurazione minimale a soli 3.19M parametri, che difende una Test Loss solida di 0.5340 ed un F1 mistake del 10.45%, dimostrando come l'efficienza predittiva dell'SSM poggi sull'ingegneria del routing dell'attenzione temporale piuttosto che sulla mera forza bruta parametrica.

### 6.4 Parallelo con la Baseline del Paper Originale (Il Tetto dell'Oracolo)
Per non fraintendere e decontestualizzare il massimale dell'F1-Score stazionante sul 10-25%, occorre richiamare il quadro normativo stabilito dagli autori di Assembly101 (Sener et al., CVPR 2022).
L'articolo originale non approccia mai la Mistake Detection come segmentazione densa continua, ma come classificazione discreta di segmenti pre-tagliati "off-line" (al modello vengono fomite esclusivamente feature fino al confine dell'azione esatta). Eseguendo il salto qualitativo verso un modello in grado di inferire attivamente al volo (*on the fly*) l'inizio e la fine dell'errore temporale in maniera densa (frame-by-frame), la complessità statistica del task esplode in maniera incommensurabile.
In maniera ancor più significativa, il team originario di Assembly101 evidenzia i risultati di un sistema teorico perfetto definito **Oracle**: fornendo in input alla rete la stringa di testo esatta che rivela infallibilmente l'azione corrente, il modello oracolo arranca raggiungendo una Precision del 48.6% e un Recall del 62.7% per la classe mistake. L'esistenza di questo "tetto di cristallo dell'oracolo" convalida a livello teorico la difficoltà ontologica del dataset. Identificare la comparsa di un errore procedurale è per sua natura un problema ambiguo, in cui il confine tra un errore e l'esecuzione grossolana di una procedura corretta giace all'interno di sfumature percettive sottilissime. I modelli sperimentali stanno correntemente approssimando i limiti strutturali massimi di intelligibilità del dataset stesso.

---

## 7. Analisi Qualitativa e Failure Cases

Dall'ispezione empirica e dall'analisi della matrice di predizione log-scalare dei modelli scaturisce che l'accuratezza sulla classe di background `correct` rimane quasi stazionaria e invulnerabile (F1 costantemente superiore al 93-95%). Le inefficienze sui target anomali si aggregano intorno a due pattern prevalenti di *Failure Cases*:
1. **Latenza del Confine Temporale (Boundary Delay):** I modelli a ricorrenza fissa (xLSTM) e a memoria spaziale continua (Mamba), per design intrinseco, palesano un'inerzia dello stato vettoriale nascosto. Quando l'azione muta da un istante all'altro, la rete necessita di assimilare una finestra variabile di molteplici frame "anomali" prima che la probabilità logistica superi la soglia del *threshold*, provocando un costante slittamento in avanti (ritardo di inferenza) rispetto ai confini cronologici definiti dall'annotatore umano.
2. **Cortocircuito Semantico tra Errore e Correzione:** In attività che prevedono assemblaggi fisici veloci, le riprese video che inquadrano le mani intente a disfare un errore per correggerlo presentano vettori TSM quasi indistinguibili dai vettori di quando le medesime mani compivano per la prima volta l'infrazione. I modelli che non dispongono di un ancoraggio contestuale forte faticano a dirimere questa sovrapposizione semantica temporale.

---

## 8. Conclusioni e Sviluppi Futuri

La ricerca consolida e dimostra tangibilmente le potenzialità insite nel cambio di paradigma proposto dai modelli sequenziali lineari continui (Mamba) e matriciali (xLSTM), misurandoli su frontiere complesse e inesplorate del *Video Understanding* su larga scala:
* **Efficienza Costo-VRAM ed Esplosione della Scalabilità:** Le contromisure algoritmiche adottate, unite al *gradient checkpointing* e alla complessità teorica lineare matematica delle nuove architetture, hanno scardinato il muro del partizionamento video, consentendo in scioltezza inferenza e backpropagation stabili su flussi titanici pari a 20.000 frame — imprese assolutamente irrealizzabili con framework vincolati dall'astrazione dell'Attenzione Quadratica convenzionale.
* **Specializzazione Architetturale:** L'evidenza della superiore adeguatezza della baseline convoluzionale spaziotemporale (TempAgg) nell'estrarre transizioni millimetriche ad altissima frequenza sancisce che la mera modellazione globale del contesto asintotico non è la chiave risolutiva per problematiche a manifestazione spiccatamente locale (come le anomalie rapide).

### Indirizzi e Sviluppi Futuri
Alla luce dell'analisi critica esposta, i vettori ottimali per gli studi successivi indirizzano lo sviluppo metodologico verso le seguenti implementazioni:
1. **Ingegnerizzazione di Architetture Ibride Simbiotiche (Conv-Mamba / Conv-xLSTM):** Il futuro della ricerca risiede nella confluenza metodologica. Instanziare un modulo di *Feature Extraction* puramente basato su pattern convoluzionali locali (come il *RecentPastBlock*) ai vertici iniziali della rete, in grado di codificare le transizioni visive sfuggenti, delegando successivamente l'astrazione e il tracciamento mnemonico alle profondità inesauribili del nucleo Mamba.
2. **Integrazione di Pipeline Multimodali:** Arginare le ambiguità percettive puramente visive (che intrappolano perfino gli oracoli) fondendo in ingresso flussi ancillari espliciti, come array descrittivi delle posizioni e del tracking delle mani nello spazio 3D, o flussi testuali generativi per ancorare le transizioni d'azione ad una logica simbolica.