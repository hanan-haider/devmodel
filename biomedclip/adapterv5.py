#from claude code 
#%%writefile /kaggle/working/devmodel/biomedclip/adapter.py
import torch
from torch import nn
import torch.nn.functional as F

class ClipAdapter(nn.Module):
    def __init__(self, c_in: int, bottleneck: int = 256, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(c_in)
        
        # Enhanced bottleneck MLP
        self.mlp = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=True),  # Added bias
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, c_in, bias=True),  # Added bias
            nn.Dropout(dropout),
        )
        
        # Better initialization for gating - start with small but non-zero value
        self.gate = nn.Parameter(torch.ones(1) * 0.1)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = self.mlp(x)
        
        # Memory bank features (high-resolution)
        med = x 
        
        # Gated residual connection
        out = residual + self.gate * x
        return med, out


class CLIP_Inplanted(nn.Module):
    def __init__(
        self,
        clip_model,
        features,
        bottleneck=256,
        dropout=0.1,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.features = sorted(list(features))
        
        num_adapted_layers = len(self.features)
        self.image_encoder = clip_model.visual.trunk

        # Visual projection
        if hasattr(clip_model.visual, "head") and hasattr(clip_model.visual.head, "proj"):
            self.visual_proj = clip_model.visual.head.proj
        else:
            self.visual_proj = getattr(clip_model.visual, "proj", None)

        hidden_dim = getattr(self.image_encoder, "num_features", None)
        if hidden_dim is None:
            hidden_dim = self.image_encoder.embed_dim

        # Build adapters with increased capacity
        self.seg_adapters = nn.ModuleList([
            ClipAdapter(hidden_dim, bottleneck=bottleneck, dropout=dropout)
            for _ in self.features
        ])
        self.det_adapters = nn.ModuleList([
            ClipAdapter(hidden_dim, bottleneck=bottleneck, dropout=dropout)
            for _ in self.features
        ])

        # Learnable blending with better initialization
        # Use smaller initial values to allow gradual learning
        self.alpha_backbone = nn.Parameter(torch.ones(num_adapted_layers) * 0.85)
        self.alpha_seg = nn.Parameter(torch.ones(num_adapted_layers) * 0.075)
        self.alpha_det = nn.Parameter(torch.ones(num_adapted_layers) * 0.075)

        # Freeze backbone
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        
        # Make visual projection trainable for better alignment
        if self.visual_proj is not None:
            for p in self.visual_proj.parameters():
                p.requires_grad = True

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

                # Get adapter outputs
                seg_med, seg_out = self.seg_adapters[pos](x)
                det_med, det_out = self.det_adapters[pos](x)

                # Normalize alphas with softmax for stable training
                alphas = torch.softmax(torch.stack([
                    self.alpha_backbone[pos],
                    self.alpha_seg[pos],
                    self.alpha_det[pos]
                ]), dim=0)

                # Blended features
                x = alphas[0] * x + alphas[1] * seg_out + alphas[2] * det_out

                seg_patch_tokens.append(seg_med)
                det_patch_tokens.append(det_med)

        x = self.image_encoder.norm(x)
        pooled = x[:, 0]

        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)

        return pooled, seg_patch_tokens, det_patch_tokens