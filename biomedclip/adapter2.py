import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

# ✅ ENHANCED Residual CLIP Adapter
class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=384, dropout=0.1):  # ✅ Compressed bottleneck
        super(ClipAdapter, self).__init__()
        
        # ✅ Pre-norm (like Transformer blocks)
        self.norm = nn.LayerNorm(c_in)
        
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(dropout)  # ✅ Regularization
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        
        # ✅ Learnable gating instead of fixed weights
        self.gate = nn.Parameter(torch.tensor(0.1))  # Starts small, learns to grow
        
        # ✅ Stochastic depth survival probability
        self.survival_prob = 0.9
        
    def forward(self, x):
        # ✅ Pre-norm
        adapted = self.norm(x)
        
        # ✅ Adapter transformation
        adapted = self.fc1(adapted)
        adapted = self.fc2(adapted)
        
        # ✅ Learnable residual scaling with gating
        adapted = self.gate * adapted
        
        # ✅ Stochastic depth (only during training)
        if self.training and torch.rand(1).item() > self.survival_prob:
            # Skip adapter (identity path)
            return x, x
        
        # ✅ Post-norm on residual path
        return adapted, self.norm(x + adapted)


class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model
        
        # Access visual encoder
        if hasattr(clip_model.visual, 'trunk'):
            self.image_encoder = clip_model.visual.trunk
        else:
            self.image_encoder = clip_model.visual
            
        # Access projection layer
        self.visual_proj = None
        if hasattr(clip_model.visual, 'head') and hasattr(clip_model.visual.head, 'proj'):
            self.visual_proj = clip_model.visual.head.proj
        elif hasattr(clip_model.visual, 'proj'):
            self.visual_proj = clip_model.visual.proj
        
        # Print parameter info
        total_params = sum(p.numel() for p in self.image_encoder.parameters())
        trainable_params = sum(p.numel() for p in self.image_encoder.parameters() if p.requires_grad)
        print(f"Total parameters in vision encoder: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        self.features = features
        
        # ✅ Create adapters with COMPRESSED bottleneck (768→384→768)
        # This reduces parameters by ~50% and forces efficient learning
        adapter_bottleneck = 384  # Can tune: 192 for extreme compression, 512 for lighter compression
        
        self.seg_adapters = nn.ModuleList([
            ClipAdapter(768, bottleneck=adapter_bottleneck, dropout=0.1) 
            for _ in range(len(features))
        ])
        self.det_adapters = nn.ModuleList([
            ClipAdapter(768, bottleneck=adapter_bottleneck, dropout=0.1) 
            for _ in range(len(features))
        ])
        
        # ✅ LayerNorm for final output stabilization
        self.final_norm = nn.LayerNorm(768)

    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.image_encoder.patch_embed(x)
        cls_token = self.image_encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.image_encoder.pos_embed
        x = self.image_encoder.pos_drop(x)
        
        seg_patch_tokens = []
        det_patch_tokens = []
        
        # Process through transformer blocks
        for i, block in enumerate(self.image_encoder.blocks):
            x = block(x)
            
            # Apply adapters at specified layers
            if (i + 1) in self.features:
                idx = self.features.index(i + 1)
                seg_adapt_med, seg_adapt_out = self.seg_adapters[idx](x)
                det_adapt_med, det_adapt_out = self.det_adapters[idx](x)
                
                # ✅ Replace x with adapter-enhanced version
                x = seg_adapt_out  # Both adapters combined in the residual path
                
                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)
        
        # Apply final layer norm
        x = self.image_encoder.norm(x)
        x = self.final_norm(x)  # ✅ Additional stabilization
        
        # Global pooling (CLS token)
        pooled = x[:, 0]
        
        # Apply visual projection
        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)
        
        return pooled, seg_patch_tokens, det_patch_tokens
