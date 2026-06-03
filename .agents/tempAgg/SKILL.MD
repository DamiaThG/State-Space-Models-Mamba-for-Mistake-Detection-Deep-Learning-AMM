# PROMPT: Implementazione PyTorch di TempAgg per Mistake Detection su Assembly101

## 1. Contesto e Obiettivo del Task

Devi implementare in PyTorch l'architettura **TempAgg** (Temporal Aggregate Representations) e il relativo loop di training/valutazione per il task di Mistake Detection sul dataset video Assembly101.
[cite_start]Il task è una classificazione frame-by-frame (dense prediction) divisa in 3 classi: `"correct"`, `"mistake"`, e `"correction"`[cite: 937].

**Specifiche di Input/Output:**
Il DataLoader (già implementato) restituisce un dizionario contenente:

* `features`: Tensore float di shape `[Batch, T_max, 2048]`.
* `labels`: Tensore intero di shape `[Batch, T_max]` contenente le classi (0, 1, 2) e il valore `-1` usato come padding per le sequenze più corte.
* `attention_mask`: Tensore booleano di shape `[Batch, T_max]` (True per i frame validi, False per il padding).
* L'output finale della rete deve essere un tensore di logits con shape `[Batch, T_max, 3]`.

---

## 2. TASK 1: Implementazione Architettura TempAgg

Scrivi una classe `TempAggMistakeDetector(nn.Module)`. Il forward pass deve rispettare il seguente flusso logico:

* **Riduzione Dimensionale:** Applica un layer Lineare iniziale per proiettare le feature in ingresso da 2048 a una dimensione nascosta (es. `hidden_dim = 512`).
* **Creazione degli Snippet (Temporal Pooling):** * Implementa una logica per raggruppare i frame temporali in "snippet" tramite un'operazione di max-pooling lungo l'asse temporale.
  * Genera il tensore **Spanning Past** (su scale temporali multiple raggruppando gli ultimi N frame) e il tensore **Recent Past** (focalizzato sugli ultimi istanti). Attenzione a mantenere la causalità: non guardare frame futuri rispetto al time-step corrente $t$.
* [cite_start]**Non-Local Blocks (NLB):** Implementa un modulo di attenzione[cite: 158]. [cite_start]Il blocco deve includere layer normalization e dropout[cite: 160].
  * [cite_start]*Self-Attention:* Applica il blocco NLB allo Spanning Past su se stesso[cite: 164].
  * [cite_start]*Cross-Attention:* Usa il Recent Past come "query" per interrogare lo Spanning Past processato[cite: 164].
* [cite_start]**Coupling Block (CB):** Accoppia l'output della cross-attention con le feature recenti originali concatenandole e facendole passare per un layer lineare[cite: 166].
* **Temporal Aggregation Block (TAB):** Aggrega i risultati provenienti dalle diverse scale temporali.
  * [cite_start]Per lo Spanning Past, utilizza un'operazione di max-pooling tra le varie scale[cite: 170, 171].
  * [cite_start]Per il Recent Past, concatena i tensori e applica una trasformazione lineare[cite: 170].
* **Classification Head:** Un Multi-Layer Perceptron finale che mappa il tensore aggregato ai logits finali `[Batch, T_max, 3]`.

---

## 3. TASK 2: Training Loop & Evaluation Condivisa

Scrivi le funzioni `train_epoch` ed `evaluate`. Devono gestire rigorosamente le seguenti criticità:

* [cite_start]**Gestione dello Sbilanciamento delle Classi:** Nel dataset Assembly101, i segmenti di tipo mistake sono solo il 15.9% e i segmenti di correction sono il 6.7%[cite: 940]. Inizializza la `nn.CrossEntropyLoss` passando un tensore al parametro `weight` per penalizzare maggiormente gli errori sulle classi minoritarie (mistake e correction) rispetto alla classe maggioritaria (correct).
* **Masking e Padding:** Usa l'`attention_mask` fornita per escludere il padding dal calcolo della loss. Ricorda di ignorare l'indice `-1` presente nelle labels. Per fare questo correttamente con PyTorch, appiattisci (flatten) le predizioni a `[Batch * T_max, 3]` e le labels a `[Batch * T_max]`, filtrando i valori tramite la maschera prima della loss.
* [cite_start]**Metriche di Valutazione:** Implementa il calcolo di **Precision** e **Recall** (Top-1) separate per singola classe[cite: 1076]. L'accuratezza globale non è una metrica affidabile a causa del forte sbilanciamento; concentrati sulle prestazioni della rete nell'identificare correttamente le classi 1 e 2.
