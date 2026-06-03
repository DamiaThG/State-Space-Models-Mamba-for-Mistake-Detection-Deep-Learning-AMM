"""
TempAgg (Temporal Aggregate Representations) Baseline
======================================================
Architettura baseline per Mistake Detection su Assembly101.

Flusso del forward pass (Aggiornato con Spanning Past a numero di snippet fisso):
    Input [B, T_max, 2048]
        → InputProjection   (Linear 2048 → hidden_dim)
        
        → SpanningPastBlock (per ogni scala S_i)
            - Divide la storia (da 0 a t) in S_i snippet usando ROI Pooling
            - [B, T_max, S_i, D]
            
        → RecentPastBlock (per ogni scala R_i)
            - Finestra R_i divisa in K snippet (es. 5)
            - Raccoglie i K snippet per ogni frame t
            - Concatena e proietta a D → [B, T_max, D]
            
        → DynamicNonLocalBlock (per ogni scala dello Spanning)
            - Self-attention causale: snippet contro snippet
            - Cross-attention causale: Query = Recent, Key/Val = Spanning
            
        → CouplingBlock (per ogni scala dello Spanning)
            - concat(cross_out, recent_query) → Linear
            
        → TAB (Temporal Aggregation)
            - max-pool tra le scale Spanning
            - concat+Linear tra le scale Recent
            - concat(Spanning, Recent) → Linear
            
        → MLPHead → [B, T_max, 3]
"""

import math
from typing import List, Optional, Tuple

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Moduli Past (Spanning e Recent)
# ---------------------------------------------------------------------------

class SpanningPastBlock(nn.Module):
    """
    Raggruppa la storia passata (da 0 a t) in un numero FISSO di snippet (scale).
    Se la scala è S, per ogni t la storia 0...t viene divisa in S snippet usando
    il Region of Interest (ROI) Pooling 1D.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """
        Ritorna:
            spanning_past : [B, T_max, scale, D] 
        """
        B, T, D = x.shape
        # Preparazione tensore in formato 2D per torchvision [B, D, H, W]
        # Poniamo H=1, W=T
        x_2d = x.unsqueeze(2).permute(0, 3, 2, 1).contiguous()  # [B, D, 1, T]
        
        # Generazione vettorializzata delle ROI per ogni batch e per ogni frame t
        # Formato ROI: [batch_index, x1, y1, x2, y2]
        batch_idx = torch.arange(B, device=x.device).view(B, 1).expand(B, T).reshape(-1, 1).float()
        x1 = torch.zeros((B * T, 1), device=x.device)
        y1 = torch.zeros((B * T, 1), device=x.device)
        x2 = torch.arange(1, T + 1, device=x.device).view(1, T).expand(B, T).reshape(-1, 1).float()
        y2 = torch.ones((B * T, 1), device=x.device)
        
        rois = torch.cat([batch_idx, x1, y1, x2, y2], dim=1)  # [B*T, 5]
        
        # ROI Pooling: estrae 'scale' snippet per ogni t
        out = torchvision.ops.roi_pool(x_2d, rois, output_size=(1, scale), spatial_scale=1.0)
        # out: [B*T, D, 1, scale]
        
        out = out.squeeze(2).view(B, T, D, scale).permute(0, 1, 3, 2).contiguous() 
        return out  # [B, T, scale, D]


class RecentPastBlock(nn.Module):
    """
    Finestra di azione recente divisa in K snippet fissi.
    """
    def __init__(self, dim: int, num_snippets: int) -> None:
        super().__init__()
        self.num_snippets = num_snippets
        self.proj = nn.Linear(dim * num_snippets, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        B, T, D = x.shape
        # Dimensione di ogni singolo snippet
        W = max(1, scale // self.num_snippets)
        
        x_t = x.permute(0, 2, 1).contiguous()
        x_padded = F.pad(x_t, (W - 1, 0))
        c_t = F.max_pool1d(x_padded, kernel_size=W, stride=1)  # [B, D, T]
        
        snippets = []
        for k in range(self.num_snippets):
            # L'offset indica quanti frame "indietro" si trova questo snippet rispetto a t
            offset = (self.num_snippets - 1 - k) * W
            if offset > 0:
                shifted_t = F.pad(c_t, (offset, 0))[:, :, :-offset]
                snippets.append(shifted_t.permute(0, 2, 1).contiguous())
            else:
                snippets.append(c_t.permute(0, 2, 1).contiguous())
                
        recent_cat = torch.cat(snippets, dim=-1)   # [B, T, K * D]
        out = self.proj(recent_cat)                # [B, T, D]
        return self.act(out)


# ---------------------------------------------------------------------------
# Attention & Aggregation Blocks
# ---------------------------------------------------------------------------

class DynamicNonLocalBlock(nn.Module):
    """
    Non Local Block ottimizzato per tensori 4D (Spanning Past variabile per ogni t).
    Gestisce automaticamente sia Self-Attention che Cross-Attention.
    """
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.scale = math.sqrt(dim)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # key, value: [B, T, S, D] dove S è la scala (numero di snippet)
        is_self_attn = (query.dim() == 4)
        
        Q = self.W_q(self.norm_q(query)) 
        K = self.W_k(self.norm_k(key))   # [B, T, S, D]
        V = self.W_v(self.norm_k(value)) # [B, T, S, D]
        
        if is_self_attn:
            # Self-Attention: Q è [B, T, S, D]
            B, T, S, D = Q.shape
            Q_flat = Q.view(B * T, S, D)
            K_flat = K.view(B * T, S, D)
            V_flat = V.view(B * T, S, D)
            
            attn = torch.bmm(Q_flat, K_flat.transpose(1, 2)) / self.scale  # [B*T, S, S]
            attn = F.softmax(attn, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.attn_drop(attn)
            
            out_flat = torch.bmm(attn, V_flat)  # [B*T, S, D]
            out = out_flat.view(B, T, S, D)
            out = self.W_o(out)
            return out + query
        else:
            # Cross-Attention: Q è [B, T, D]
            B, T, D = Q.shape
            S = K.shape[2]
            
            Q_exp = Q.unsqueeze(2)  # [B, T, 1, D]
            
            attn = (Q_exp * K).sum(dim=-1) / self.scale  # [B, T, S]
            attn = F.softmax(attn, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.attn_drop(attn)
            
            out = (attn.unsqueeze(-1) * V).sum(dim=2)    # [B, T, D]
            out = self.W_o(out)
            return out + query


class CouplingBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim * 2)
        self.linear = nn.Linear(dim * 2, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, cross_out: torch.Tensor, recent_orig: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cross_out, recent_orig], dim=-1)   # [B, T, 2D]
        x = self.norm(x)
        x = self.linear(x)
        x = self.act(x)
        return self.drop(x)                               


class TemporalAggregationBlock(nn.Module):
    def __init__(self, dim: int, num_recent_scales: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.recent_proj = nn.Linear(dim * num_recent_scales, dim)
        self.combine     = nn.Linear(dim * 2, dim)
        self.norm        = nn.LayerNorm(dim)
        self.act         = nn.GELU()
        self.drop        = nn.Dropout(dropout)

    def forward(self, spanning_list: List[torch.Tensor], recent_list: List[torch.Tensor]) -> torch.Tensor:
        spanning_stack = torch.stack(spanning_list, dim=1)      # [B, S_scales, T, D]
        spanning_agg   = spanning_stack.max(dim=1).values       # [B, T, D]

        recent_cat  = torch.cat(recent_list, dim=-1)            # [B, T, R*D]
        recent_agg  = self.recent_proj(recent_cat)              # [B, T, D]

        combined = torch.cat([spanning_agg, recent_agg], dim=-1)  # [B, T, 2D]
        out = self.combine(combined)
        out = self.norm(out)
        out = self.act(out)
        return self.drop(out)                                     


# ---------------------------------------------------------------------------
# TempAggMistakeDetector — modello principale
# ---------------------------------------------------------------------------

class TempAggMistakeDetector(nn.Module):
    def __init__(
        self,
        input_dim:       int       = 2048,
        hidden_dim:      int       = 512,
        num_classes:     int       = 3,
        spanning_scales: List[int] = [8, 16, 24],
        recent_scales:   List[int] = [30, 90, 150],
        recent_snippets: int       = 5,
        dropout:         float     = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim      = hidden_dim
        self.spanning_scales = spanning_scales
        self.recent_scales   = recent_scales
        self.recent_snippets = recent_snippets

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.spanning_block = SpanningPastBlock(hidden_dim)
        
        self.recent_blocks = nn.ModuleList([
            RecentPastBlock(hidden_dim, recent_snippets)
            for _ in recent_scales
        ])

        self.nlb_self  = DynamicNonLocalBlock(hidden_dim, dropout)
        self.nlb_cross = DynamicNonLocalBlock(hidden_dim, dropout)
        self.coupling  = CouplingBlock(hidden_dim, dropout)

        self.tab = TemporalAggregationBlock(
            dim=hidden_dim,
            num_recent_scales=len(recent_scales),
            dropout=dropout,
        )

        self.cls_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(features)                # [B, T, D]

        # ── 1. Recent Past ────────────────────────────────────────────────
        recent_pasts: List[torch.Tensor] = []
        for i, scale in enumerate(self.recent_scales):
            recent_pasts.append(self.recent_blocks[i](x, scale))  # [B, T, D]

        recent_query = recent_pasts[-1]  # La finestra più grande per la Query
        
        # ── 2. Spanning Past & Attention ──────────────────────────────────
        coupling_outputs: List[torch.Tensor] = []

        for scale in self.spanning_scales:
            sp = self.spanning_block(x, scale)  # [B, T, scale, D]
            
            # Causalità è intrinseca in sp! I valori di sp al frame t contengono solo frames da 0 a t.
            # Non servono maschere aggiuntive.
            sp_sa = self.nlb_self(query=sp, key=sp, value=sp)            # [B, T, scale, D]
            cross_out = self.nlb_cross(query=recent_query, key=sp_sa, value=sp_sa)  # [B, T, D]

            cb_out = self.coupling(cross_out, recent_query)             
            coupling_outputs.append(cb_out)

        # ── 3. Aggregation & Classification ───────────────────────────────
        tab_out = self.tab(coupling_outputs, recent_pasts)              
        logits = self.cls_head(tab_out)                                 

        return logits


if __name__ == "__main__":
    B, T, D = 2, 400, 2048
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TempAggMistakeDetector().to(device)
    x     = torch.randn(B, T, D, device=device)
    mask  = torch.ones(B, T, dtype=torch.bool, device=device)
    mask[:, -50:] = False

    logits = model(x, mask)
    print(f"Input  : {x.shape}")
    print(f"Output : {logits.shape}")
    assert logits.shape == (B, T, 3), "Shape mismatch!"
    print("✅  Smoke test passato.")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri trainabili: {total_params:,}")
