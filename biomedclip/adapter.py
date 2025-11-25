import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image


# Residual CLIP Adapter
class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y

        
class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model
        
        # BiomedCLIP uses TimmModel wrapper around VisionTransformer
        # Access: clip_model.visual.trunk (the actual ViT)
        self.image_encoder = clip_model.visual.trunk
        self.visual_proj = clip_model.visual.proj  # Final projection layer
        
        self.features = features
        
        # BiomedCLIP ViT-B has 768 hidden dimensions (not 1024)
        self.seg_adapters = nn.ModuleList([ClipAdapter(768, bottleneck=768) for i in range(len(features))])
        self.det_adapters = nn.ModuleList([ClipAdapter(768, bottleneck=768) for i in range(len(features))])

    def forward(self, x):
        # BiomedCLIP uses timm's ViT which has a different forward structure
        B = x.shape[0]
        
        # Patch embedding
        x = self.image_encoder.patch_embed(x)
        
        # Add cls token
        cls_token = self.image_encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add positional embedding
        x = x + self.image_encoder.pos_embed
        x = self.image_encoder.pos_drop(x)
        
        # Store attention maps and adapter outputs
        attn_out = []
        seg_patch_tokens = []
        det_patch_tokens = []
        
        # Process through transformer blocks
        # BiomedCLIP ViT-B has 12 blocks (layers)
        for i, block in enumerate(self.image_encoder.blocks):
            x = block(x)
            
            # Extract attention at layer 12 (index 11)
            if i + 1 == 12:
                # For timm ViT, attention is inside the block
                # We need to get it from the last block's attention module
                # Note: timm doesn't return attention by default, so we store x
                attn_out.append(x)  # Store the output instead
            
            # Apply adapters at specified feature layers
            if (i + 1) in self.features:
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i+1)](x)
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i+1)](x)
                
                # Residual connection with adapters
                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out
                
                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)
        
        # Apply final layer norm
        x = self.image_encoder.norm(x)
        
        # Global pooling (extract cls token)
        pooled = x[:, 0]  # CLS token
        
        # Apply visual projection
        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)
        
        return pooled, seg_patch_tokens, det_patch_tokens


# Alternative version with attention extraction (if you need actual attention maps)
class CLIP_Inplanted_WithAttention(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual.trunk
        self.visual_proj = clip_model.visual.proj
        self.features = features
        
        self.seg_adapters = nn.ModuleList([ClipAdapter(768, bottleneck=768) for i in range(len(features))])
        self.det_adapters = nn.ModuleList([ClipAdapter(768, bottleneck=768) for i in range(len(features))])
        
        # Hook to capture attention
        self.attention_maps = []
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention maps from the 12th block"""
        def hook_fn(module, input, output):
            # Capture attention weights
            self.attention_maps.append(output)
        
        # Register hook on the 12th block's attention module
        if len(self.image_encoder.blocks) >= 12:
            self.image_encoder.blocks[11].attn.register_forward_hook(hook_fn)
    
    def forward(self, x):
        B = x.shape[0]
        self.attention_maps = []  # Reset attention maps
        
        # Patch embedding
        x = self.image_encoder.patch_embed(x)
        
        # Add cls token
        cls_token = self.image_encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add positional embedding
        x = x + self.image_encoder.pos_embed
        x = self.image_encoder.pos_drop(x)
        
        seg_patch_tokens = []
        det_patch_tokens = []
        
        # Process through transformer blocks
        for i, block in enumerate(self.image_encoder.blocks):
            x = block(x)
            
            # Apply adapters at specified feature layers
            if (i + 1) in self.features:
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i+1)](x)
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i+1)](x)
                
                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out
                
                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)
        
        # Apply final layer norm
        x = self.image_encoder.norm(x)
        
        # Global pooling
        pooled = x[:, 0]
        
        # Apply visual projection
        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)
        
        return pooled, seg_patch_tokens, det_patch_tokens