import torch
from torch import nn
import math


class DynamicClipAdapter(nn.Module):
    """
    Self-adapting bottleneck adapter that scales based on input dimension.
    Uses compression ratio instead of fixed bottleneck size.
    """
    def __init__(self, c_in: int, compression_ratio: float = 0.25, dropout: float = 0.15):
        super().__init__()
        # Dynamic bottleneck: scales with input dimension
        bottleneck = max(64, int(c_in * compression_ratio))  # Min 64, typically ~192 for 768-dim
        
        # Pre-normalization for stability
        self.ln_pre = nn.LayerNorm(c_in)
        
        # Enhanced bottleneck with residual connections
        self.down = nn.Linear(c_in, bottleneck, bias=False)
        self.ln_mid1 = nn.LayerNorm(bottleneck)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Middle processing layer (new!)
        self.mid = nn.Linear(bottleneck, bottleneck, bias=False)
        self.ln_mid2 = nn.LayerNorm(bottleneck)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Reconstruction with layer norm
        self.up = nn.Linear(bottleneck, c_in, bias=False)
        self.ln_out = nn.LayerNorm(c_in)
        self.dropout3 = nn.Dropout(dropout * 0.5)  # Lighter dropout on output
        
        # Temperature-scaled gating (starts conservative)
        self.gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12
        
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for better gradient flow"""
        nn.init.orthogonal_(self.down.weight, gain=0.8)
        nn.init.orthogonal_(self.mid.weight, gain=1.0)
        nn.init.orthogonal_(self.up.weight, gain=0.5)
        
    def forward(self, x):
        residual = x
        
        # Normalize input
        x = self.ln_pre(x)
        
        # Down-projection to bottleneck
        med = self.down(x)
        med = self.ln_mid1(med)
        med = self.act1(med)
        med = self.dropout1(med)
        
        # Middle layer (enhances representation)
        med = self.mid(med) + med  # Residual within bottleneck
        med = self.ln_mid2(med)
        med = self.act2(med)
        med = self.dropout2(med)
        
        # Up-projection back to full dimension
        full = self.up(med)
        full = self.ln_out(full)
        full = self.dropout3(full)
        
        # Temperature-scaled gating
        gate_weight = torch.sigmoid(self.gate)
        out = residual + gate_weight * full
        
        return med, full, out


class CLIP_Inplanted(nn.Module):
    """
    Enhanced CLIP adapter with:
    - Dynamic bottleneck sizing
    - Improved blending strategy
    - Better parameter initialization
    """
    def __init__(
        self,
        clip_model,
        features,
        compression_ratio=0.25,  # Instead of fixed bottleneck
        dropout=0.15,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.features = sorted(list(features))
        num_adapted_layers = len(self.features)

        # ViT trunk
        self.image_encoder = clip_model.visual.trunk

        # Visual projection (768 -> 512)
        if hasattr(clip_model.visual, "head") and hasattr(clip_model.visual.head, "proj"):
            self.visual_proj = clip_model.visual.head.proj
        else:
            self.visual_proj = getattr(clip_model.visual, "proj", None)

        hidden_dim = getattr(self.image_encoder, "num_features", None)
        if hidden_dim is None:
            hidden_dim = self.image_encoder.embed_dim

        # Build dynamic adapters
        self.seg_adapters = nn.ModuleList(
            [DynamicClipAdapter(hidden_dim, compression_ratio=compression_ratio, dropout=dropout)
             for _ in self.features]
        )
        self.det_adapters = nn.ModuleList(
            [DynamicClipAdapter(hidden_dim, compression_ratio=compression_ratio, dropout=dropout)
             for _ in self.features]
        )

        # Learnable per-layer blending with better initialization
        # Start conservative, let training adjust
        self.alpha_backbone = nn.Parameter(torch.ones(num_adapted_layers) * 0.85)
        self.alpha_seg = nn.Parameter(torch.ones(num_adapted_layers) * 0.075)
        self.alpha_det = nn.Parameter(torch.ones(num_adapted_layers) * 0.075)

        # Freeze backbone
        for p in self.image_encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        B = x.shape[0]

        # ViT input embedding
        x = self.image_encoder.patch_embed(x)
        cls_token = self.image_encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.image_encoder.pos_embed
        x = self.image_encoder.pos_drop(x)

        # Storage for different representations
        seg_patch_tokens_med = []   # Bottleneck (for few-shot)
        seg_patch_tokens_full = []  # Full-dim (for zero-shot)
        det_patch_tokens_med = []
        det_patch_tokens_full = []

        # Transformer blocks
        for layer_idx, block in enumerate(self.image_encoder.blocks, start=1):
            x = block(x)

            if layer_idx in self.features:
                pos = self.features.index(layer_idx)

                # Get adapter outputs (3 values each)
                seg_med, seg_full, seg_out = self.seg_adapters[pos](x)
                det_med, det_full, det_out = self.det_adapters[pos](x)

                # Learnable blending
                x = (
                    self.alpha_backbone[pos] * x
                    + self.alpha_seg[pos] * seg_out
                    + self.alpha_det[pos] * det_out
                )

                # Store both representations
                seg_patch_tokens_med.append(seg_med)
                seg_patch_tokens_full.append(seg_full)
                det_patch_tokens_med.append(det_med)
                det_patch_tokens_full.append(det_full)

        x = self.image_encoder.norm(x)
        pooled = x[:, 0]

        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)

        return (
            pooled,
            seg_patch_tokens_med,
            seg_patch_tokens_full,
            det_patch_tokens_med,
            det_patch_tokens_full
        )