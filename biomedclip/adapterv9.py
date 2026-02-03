import torch
from torch import nn

class ClipAdapter(nn.Module):
    def __init__(self, c_in: int, bottleneck: int = 384, dropout: float = 0.2):
        super().__init__()
        self.ln = nn.LayerNorm(c_in)
        self.mlp = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(bottleneck, bottleneck, bias=False),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(bottleneck, c_in, bias=False),
        )
        self.gate = nn.Parameter(torch.tensor(0.08))
        
    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = self.mlp(x)
        med = x
        out = residual + self.gate * x
        return med, out


class CLIP_Inplanted(nn.Module):
    def __init__(
        self,
        clip_model,
        features,
        bottleneck=384,
        dropout=0.2,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.features = sorted(list(features))
        
        num_adapted_layers = len(self.features)

        # timm ViT trunk used by BioMedCLIP
        self.image_encoder = clip_model.visual.trunk

        # Final projection: handles dimensionality alignment (768 -> 512)
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

        # Learnable Blending Parameters (Layer-wise)
        self.alpha_backbone = nn.Parameter(torch.ones(num_adapted_layers) * 0.8)
        self.alpha_seg = nn.Parameter(torch.ones(num_adapted_layers) * 0.1)
        self.alpha_det = nn.Parameter(torch.ones(num_adapted_layers) * 0.1)

        # Freeze backbone parameters to ensure few-shot stability
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

        # Transformer blocks loop
        for layer_idx, block in enumerate(self.image_encoder.blocks, start=1):
            x = block(x)

            if layer_idx in self.features:
                pos = self.features.index(layer_idx)

                # Get adapter features
                seg_med, seg_out = self.seg_adapters[pos](x)
                det_med, det_out = self.det_adapters[pos](x)

                # Use Learnable Alphas per Layer
                x = (
                    self.alpha_backbone[pos] * x
                    + self.alpha_seg[pos] * seg_out
                    + self.alpha_det[pos] * det_out
                )

                # Store tokens (stripping CLS token for local maps)
                seg_patch_tokens.append(seg_med)
                det_patch_tokens.append(det_med)

        x = self.image_encoder.norm(x)
        pooled = x[:, 0]  # CLS token

        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)    

        return pooled, seg_patch_tokens, det_patch_tokens