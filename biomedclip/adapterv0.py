#original adapter present mvfa, and this adapter modified for BioMedCLIP

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
        
        # BioMedCLIP uses TimmModel wrapper around VisionTransformer
        # Access: clip_model.visual.trunk (the actual ViT)
        self.image_encoder = clip_model.visual.trunk
        #print("Using BioMedCLIP visual encoder:", self.image_encoder)


        self.image_encoder = clip_model.visual.trunk
        #print("Using BioMedCLIP visual encoder:", self.image_encoder)
        
        # BioMedCLIP uses visual.head.proj instead of visual.proj
        if hasattr(clip_model.visual, 'head') and hasattr(clip_model.visual.head, 'proj'):
            self.visual_proj = clip_model.visual.head.proj  # Final projection layer
        else:
            # Fallback: check if it's directly accessible
            self.visual_proj = getattr(clip_model.visual, 'proj', None)
        
        self.features = features 
        
        # BioMedCLIP ViT-B has 768 hidden dimensions
        self.seg_adapters = nn.ModuleList([ClipAdapter(768, bottleneck=768) for i in range(len(features))])
        self.det_adapters = nn.ModuleList([ClipAdapter(768, bottleneck=768) for i in range(len(features))])

    def forward(self, x):
        # BioMedCLIP uses timm's ViT which has a different forward structure
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
        # BioMedCLIP ViT-B has 12 blocks (layers)
        for i, block in enumerate(self.image_encoder.blocks):
            x = block(x)
            
            # Extract attention at layer 12 (index 11)
            if i + 1 == 12:
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
        
        # Apply visual projection if it exists
        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)
        
        return pooled, seg_patch_tokens, det_patch_tokens

