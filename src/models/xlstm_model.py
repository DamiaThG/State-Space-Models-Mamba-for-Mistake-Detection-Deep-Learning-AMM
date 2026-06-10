import torch
import torch.nn as nn
from typing import Optional
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
)

class xLSTMMistakeDetector(nn.Module):
    """
    xLSTM model for mistake detection, using an mLSTM-only architecture.
    Suitable for very long sequences like whole videos.
    """
    def __init__(
        self,
        input_dim: int = 2048,
        d_model: int = 512,
        n_layers: int = 6,
        num_classes: int = 3,
        dropout: float = 0.2,
        max_seq_len: int = 25000,
        use_checkpointing: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.use_checkpointing = use_checkpointing
        self.max_seq_len = max_seq_len

        # 1. Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Backbone xLSTM
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=4,
                )
            ),
            context_length=max_seq_len,
            num_blocks=n_layers,
            embedding_dim=d_model,
            slstm_at=[],  # mLSTM puro
            dropout=dropout,
        )
        _stack = xLSTMBlockStack(cfg)
        
        # Estraiamo i componenti per poter fare gradient checkpointing manuale
        self.blocks = _stack.blocks  # nn.ModuleList
        self.post_norm = _stack.post_blocks_norm  # LayerNorm

        # 3. Classification Head
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, T, input_dim]
            attention_mask: [B, T] boolean tensor (False means padding)
        Returns:
            logits: [B, T, num_classes]
        """
        # 1. Input projection
        x = self.input_proj(features)  # [B, T, d_model]

        # 2. Mask zero-out (xlstm library does not natively support attention_mask)
        if attention_mask is not None:
            # Broadcast mask over the feature dimension
            x = x * attention_mask.unsqueeze(-1).float()

        # 3. Backbone with optional gradient checkpointing per-block
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False
                )
            else:
                x = block(x)

        # 4. Post-norm (if created by xLSTMBlockStack)
        if self.post_norm is not None:
            x = self.post_norm(x)

        # 5. Classification head
        logits = self.cls_head(x)  # [B, T, num_classes]

        return logits

if __name__ == "__main__":
    # Smoke test locale
    print("Inizializzazione modello xLSTM...")
    
    # Parametri ridotti per non andare OOM in locale (se testato su CPU/piccola GPU)
    model = xLSTMMistakeDetector(
        input_dim=2048, 
        d_model=128, 
        n_layers=2, 
        max_seq_len=500,
        use_checkpointing=True
    )
    
    B, T, D = 2, 400, 2048
    print(f"Creazione tensori dummy [Batch={B}, SeqLen={T}, Dim={D}]...")
    
    dummy_features = torch.randn(B, T, D)
    dummy_mask = torch.ones(B, T, dtype=torch.bool)
    # Rendi finto padding negli ultimi frame del secondo elemento
    dummy_mask[1, 350:] = False
    
    print("Esecuzione forward pass...")
    logits = model(dummy_features, dummy_mask)
    
    print(f"Output shape attesa: [2, 400, 3]")
    print(f"Output shape reale:  {list(logits.shape)}")
    
    if list(logits.shape) == [B, T, 3]:
        print("✅ Forward pass completato con successo e shape corretta.")
    else:
        print("❌ ERRORE: Shape di output non corrispondente.")
