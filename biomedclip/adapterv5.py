import torch
from torch import nn

# OPTIMIZED ADAPTER - Simpler, more stable
class ClipAdapter(nn.Module):
    def __init__(self, c_in: int, bottleneck: int = 192, dropout: float = 0.15):
        super().__init__()
        self.ln_pre = nn.LayerNorm(c_in)
        
        # Single bottleneck path (reduced from 3 layers to 2)
        self.down = nn.Linear(c_in, bottleneck, bias=False)
        self.ln_mid = nn.LayerNorm(bottleneck)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, c_in, bias=False)
        
        # Fixed residual scale (no learnable gate)
        self.scale = 0.1
        
    def forward(self, x):
        residual = x
        x = self.ln_pre(x)
        
        # Bottleneck projection
        med = self.down(x)
        med = self.ln_mid(med)
        med = self.act(med)
        med = self.dropout(med)
        
        # Reconstruction
        out = self.up(med)
        out = residual + self.scale * out
        
        return med, out


class CLIP_Inplanted(nn.Module):
    def __init__(
        self,
        clip_model,
        features,
        bottleneck=192,
        dropout=0.15,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.features = sorted(list(features))
        
        # ViT trunk
        self.image_encoder = clip_model.visual.trunk

        # Final projection (768 -> 512)
        if hasattr(clip_model.visual, "head") and hasattr(clip_model.visual.head, "proj"):
            self.visual_proj = clip_model.visual.head.proj
        else:
            self.visual_proj = getattr(clip_model.visual, "proj", None)

        hidden_dim = getattr(self.image_encoder, "num_features", None)
        if hidden_dim is None:
            hidden_dim = self.image_encoder.embed_dim

        # Build adapters
        self.seg_adapters = nn.ModuleList(
            [ClipAdapter(hidden_dim, bottleneck=bottleneck, dropout=dropout)
             for _ in self.features]
        )
        self.det_adapters = nn.ModuleList(
            [ClipAdapter(hidden_dim, bottleneck=bottleneck, dropout=dropout)
             for _ in self.features]
        )

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

        seg_patch_tokens = []
        det_patch_tokens = []

        # Transformer blocks
        for layer_idx, block in enumerate(self.image_encoder.blocks, start=1):
            x = block(x)

            if layer_idx in self.features:
                pos = self.features.index(layer_idx)

                # Get adapter features
                seg_med, seg_out = self.seg_adapters[pos](x)
                det_med, det_out = self.det_adapters[pos](x)

                # Simple blending (no learnable alphas)
                x = 0.8 * x + 0.1 * seg_out + 0.1 * det_out

                # Store UNPROJECTED tokens (projection happens in train/test)
                seg_patch_tokens.append(seg_med)
                det_patch_tokens.append(det_med)

        x = self.image_encoder.norm(x)
        pooled = x[:, 0]  # CLS token

        # Only project the pooled CLS token
        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)

        # DO NOT project patch tokens here - let train/test handle it
        return pooled, seg_patch_tokens, det_patch_tokens