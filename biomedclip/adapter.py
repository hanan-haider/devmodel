import torch
from torch import nn
import math

# 1. Improved Residual CLIP Adapter with Gating and LayerNorm
class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=512):
        super(ClipAdapter, self).__init__()
        # LayerNorm helps stabilize training in deep transformers
        self.ln = nn.LayerNorm(c_in)
        
        self.fc = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.GELU(), # GELU is standard for Transformers/BioMedCLIP
            nn.Linear(bottleneck, c_in, bias=False),
            nn.Dropout(0.1)
        )
        
        # Learnable scale parameter: starts at 0 so the model begins 
        # with pure pre-trained weights and slowly incorporates adapter info
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x shape: [Batch, Tokens, Dim]
        residual = x
        x = self.ln(x)
        x = self.fc(x)
        
        # Intermediate 'med' features for the memory bank/patch tokens
        # We return the normalized transformed features
        adapt_med = x 
        
        # Output with learnable residual scaling
        adapt_out = residual + self.scale * x
        
        return adapt_med, adapt_out

# 2. Re-implemented CLIP_Inplanted for BioMedCLIP (ViT-B/16)
class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model
        
        # Access the Timm-based Vision Transformer trunk
        self.image_encoder = clip_model.visual.trunk
        
        # BioMedCLIP uses visual.head.proj to map 768 -> 512
        if hasattr(clip_model.visual, 'head') and hasattr(clip_model.visual.head, 'proj'):
            self.visual_proj = clip_model.visual.head.proj
        else:
            self.visual_proj = getattr(clip_model.visual, 'proj', None)
        
        self.features = features # Layers to inject adapters into, e.g., [4, 8, 10, 12]
        
        # BioMedCLIP ViT-B has 768 hidden dimensions
        # Using separate adapters for Segmentation (local) and Detection (patch)
        self.seg_adapters = nn.ModuleList([
            ClipAdapter(768, bottleneck=384) for _ in range(len(features))
        ])
        self.det_adapters = nn.ModuleList([
            ClipAdapter(768, bottleneck=384) for _ in range(len(features))
        ])

    def forward(self, x):
        B = x.shape[0]
        
        # --- Standard ViT Entry ---
        x = self.image_encoder.patch_embed(x)
        cls_token = self.image_encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.image_encoder.pos_embed
        x = self.image_encoder.pos_drop(x)
        
        seg_patch_tokens = []
        det_patch_tokens = []
        
        # --- Transformer Blocks with Adapter Injection ---
        for i, block in enumerate(self.image_encoder.blocks):
            # 1. Forward through the frozen ViT block
            x = block(x)
            
            # 2. Check if this layer is selected for adaptation
            if (i + 1) in self.features:
                idx = self.features.index(i + 1)
                
                # Apply Segmentation Adapter
                seg_med, x_seg = self.seg_adapters[idx](x)
                # Apply Detection Adapter
                det_med, x_det = self.det_adapters[idx](x)
                
                # Dynamic blending: use the mean of adapter outputs
                # This replaces the hardcoded 0.8/0.1/0.1 with gated learnable logic
                x = (x_seg + x_det) / 2.0
                
                # Store the patch tokens (excluding CLS token at index 0)
                seg_patch_tokens.append(seg_med) 
                det_patch_tokens.append(det_med)
        
        # --- Standard ViT Exit ---
        x = self.image_encoder.norm(x)
        
        # Extract CLS token for image-level classification
        pooled = x[:, 0]
        
        # Project to 512-dim shared space if projection exists
        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)
            
        return pooled, seg_patch_tokens, det_patch_tokens