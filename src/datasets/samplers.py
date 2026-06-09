import random
from typing import List, Iterator
from torch.utils.data import Sampler, Dataset

class LengthGroupedSampler(Sampler[int]):
    """
    Sampler che raggruppa le sequenze con lunghezze simili.
    Ordina il dataset per lunghezza, e poi crea batch di elementi
    consecutive (o quasi), per minimizzare il padding.
    
    Per mantenere la stocasticità durante il training:
    1. Ordina gli indici per lunghezza.
    2. Raggruppa in "macro-bucket" grandi M volte la batch_size.
    3. Mescola internamente ogni macro-bucket.
    4. Crea i batch, e infine mescola l'ordine dei batch stessi.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        mega_batch_mult: int = 50,
        seed: int = 42
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mega_batch_mult = mega_batch_mult
        self.rng = random.Random(seed)

        # Pre-calcola le lunghezze in base all'attributo original_length
        # Attenzione: requires `dataset` to expose original_length or similar
        # Assumiamo che `dataset.samples` abbia info sulla lunghezza, oppure
        # facciamo una query veloce al dataset. Nel nostro WholeVideoDataset, 
        # i sample_info avranno la 'length'.
        
        self.indices_with_lengths = []
        for idx in range(len(dataset)):
            # Per evitare di caricare il tensore (se len(dataset) chiama getitem),
            # WholeVideoDataset.samples[idx] contiene 'length'.
            # Usiamo un approccio duck-typing per sicurezza.
            if hasattr(dataset, "samples") and isinstance(dataset.samples[idx], dict) and "length" in dataset.samples[idx]:
                length = dataset.samples[idx]["length"]
            else:
                # Fallback: carica il tensore se proprio necessario
                item = dataset[idx]
                length = item["original_length"]
            
            self.indices_with_lengths.append((idx, length))

    def __iter__(self) -> Iterator[int]:
        # Sort by length
        sorted_indices = [idx for idx, _ in sorted(self.indices_with_lengths, key=lambda x: x[1])]
        
        if not self.shuffle:
            yield from sorted_indices
            return

        # Raggruppa in mega-batches
        mega_batch_size = self.batch_size * self.mega_batch_mult
        mega_batches = [
            sorted_indices[i : i + mega_batch_size] 
            for i in range(0, len(sorted_indices), mega_batch_size)
        ]
        
        # Shuffle internally in each mega-batch
        for mb in mega_batches:
            self.rng.shuffle(mb)
            
        # Flatten the mega batches back
        shuffled_indices = [idx for mb in mega_batches for idx in mb]
        
        # Split into exact batches
        batches = [
            shuffled_indices[i : i + self.batch_size]
            for i in range(0, len(shuffled_indices), self.batch_size)
        ]
        
        # Shuffle the order of batches
        self.rng.shuffle(batches)
        
        # Yield single elements (DataLoader with batch_sampler=False handles the chunking,
        # but wait, LengthGroupedSampler is passed to `sampler=` argument, so it yields indices
        # one by one, and DataLoader groups them into `batch_size`. 
        # So we just flatten the batches).
        for batch in batches:
            yield from batch

    def __len__(self) -> int:
        return len(self.indices_with_lengths)
