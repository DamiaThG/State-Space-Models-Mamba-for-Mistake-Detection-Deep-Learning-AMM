"""
MambaMistakeDetector — State-Space Model per Mistake Detection
===============================================================
Architettura basata su Mamba (SSM) per classificazione dense
frame-by-frame a 3 classi: correct, mistake, correction.

Flusso del forward pass:
    Input [B, T_max, 2048]
        → InputProjection   (Linear 2048 → d_model, LayerNorm, GELU, Dropout)
        → Mamba SSM Backbone (n_layers blocchi Mamba causali)
        → ClassificationHead (LayerNorm → Linear d_model → 3)
        → Output [B, T_max, 3]

Libreria utilizzata: `mambapy` (Pure PyTorch, no CUDA compiler richiesto).
"""

from typing import Optional

import torch
import torch.nn as nn

from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    """Singolo blocco Mamba con pre-LayerNorm e Residual Connection."""
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = Mamba(d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(self.norm(x)) + x

class MambaMistakeDetector(nn.Module):
    """
    Modello Mamba per Mistake Detection su Assembly101.

    Args:
        input_dim:   Dimensione delle feature in ingresso (default: 2048, TSM features).
        d_model:     Dimensione interna del modello Mamba (default: 512).
        n_layers:    Numero di blocchi Mamba impilati (default: 6).
        num_classes: Numero di classi in output (default: 3).
        dropout:     Probabilità di dropout (default: 0.1).
    """

    def __init__(
        self,
        input_dim:   int   = 2048,
        d_model:     int   = 512,
        n_layers:    int   = 6,
        num_classes: int   = 3,
        dropout:     float = 0.2,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.use_checkpointing = use_checkpointing

        # ── 1. Input Projection ───────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── 2. Mamba SSM Backbone ─────────────────────────────────────────
        # Usiamo il pacchetto ufficiale mamba-ssm in C++
        self.backbone = nn.ModuleList([
            MambaBlock(d_model=d_model) for _ in range(n_layers)
        ])

        # ── 3. Classification Head ────────────────────────────────────────
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Inizializzazione Xavier per i layer lineari della projection e head."""
        for module in [self.input_proj, self.cls_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass del modello.
        """
        # 1. Proiezione delle feature: [B, T, 2048] → [B, T, d_model]
        x = self.input_proj(features)

        # 2. Backbone Mamba Layer-by-Layer
        for layer in self.backbone:
            if self.use_checkpointing and self.training:
                # Checkpointing PER SINGOLO LAYER: risparmio VRAM gigantesco
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        # 3. Classification head: [B, T, d_model] → [B, T, 3]
        logits = self.cls_head(x)

        return logits


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, T, D = 2, 400, 2048
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MambaMistakeDetector().to(device)
    x = torch.randn(B, T, D, device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    mask[:, -50:] = False

    logits = model(x, mask)
    print(f"Input  : {x.shape}")
    print(f"Output : {logits.shape}")
    assert logits.shape == (B, T, 3), "Shape mismatch!"
    print("✅  Smoke test passato.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri trainabili: {total_params:,}")
