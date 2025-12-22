import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

# =========================================================================
# 1. Improved Adapter: Adds Learnable Scaling (Zero-Initialization)
# =========================================================================
class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768, init_scale=1e-4):
        super(ClipAdapter, self).__init__()
        
        # Standard Bottleneck Architecture
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        
        # SOTA IMPROVEMENT: Learnable Scaling Factor
        # Initialized to near-zero. This ensures the model starts 
        # exactly as the pre-trained BioMedCLIP and slowly "fades in" the adapter.
        self.scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x):
        x_in = x
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Return the SCALED output
        return x * self.scale


# =========================================================================
# 2. Improved Wrapper: Fixes Residual Math & Projection Logic
# =========================================================================
class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model
        
        # --- A. Identify Encoder Architecture ---
        # BioMedCLIP uses TimmModel wrapper around VisionTransformer
        if hasattr(clip_model.visual, 'trunk'):
            self.image_encoder = clip_model.visual.trunk
            self.is_timm = True
        else:
            self.image_encoder = clip_model.visual
            self.is_timm = False

        # --- B. Identify Projection Layer ---
        # BioMedCLIP uses visual.head.proj instead of visual.proj
        self.visual_proj = None
        if hasattr(clip_model.visual, 'head') and hasattr(clip_model.visual.head, 'proj'):
            self.visual_proj = clip_model.visual.head.proj  # Final projection layer
        elif hasattr(clip_model.visual, 'proj'):
            self.visual_proj = clip_model.visual.proj
            
        self.features = features 
        
        # --- C. Adapters ---
        # BioMedCLIP ViT-B has 768 hidden dimensions
        self.seg_adapters = nn.ModuleList([ClipAdapter(768, bottleneck=768) for i in range(len(features))])
        self.det_adapters = nn.ModuleList([ClipAdapter(768, bottleneck=768) for i in range(len(features))])

        # Report params
        total_params = sum(p.numel() for p in self.image_encoder.parameters())
        #print(f"Total parameters in vision encoder: {total_params:,}")

    def forward(self, x):
        B = x.shape[0]
        
        # --- 1. Initial Embeddings (Patch + Pos) ---
        if self.is_timm:
            # Timm ViT forward pass logic
            x = self.image_encoder.patch_embed(x)
            cls_token = self.image_encoder.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.image_encoder.pos_embed
            x = self.image_encoder.pos_drop(x)
            blocks = self.image_encoder.blocks
        else:
            # Standard CLIP forward pass logic
            x = self.image_encoder.conv1(x) 
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls_tokens = self.image_encoder.class_embedding.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.image_encoder.positional_embedding
            # OpenAI CLIP often has a pre-norm
            if hasattr(self.image_encoder, 'ln_pre'):
                x = self.image_encoder.ln_pre(x)
            blocks = self.image_encoder.transformer.resblocks
        
        seg_patch_tokens = []
        det_patch_tokens = []
        
        # --- 2. Transformer Blocks + SOTA Adapter Injection ---
        for i, block in enumerate(blocks):
            # Pass through the original block
            x = block(x)
            
            # Check if this layer requires adaptation
            if (i + 1) in self.features:
                idx = self.features.index(i+1)
                
                # Get scaled adapter outputs
                # Note: The scaling (* self.scale) happens inside the adapter class now
                seg_adapt_out = self.seg_adapters[idx](x)
                det_adapt_out = self.det_adapters[idx](x)
                
                # SOTA FIX: Pure Residual Connection
                # Do NOT use 0.8 * x. We want to preserve the original signal 100%
                # and just "add" the anomaly information on top.
                x = x + seg_adapt_out + det_adapt_out
                
                # Store for loss calculation (removing CLS token at index 0)
                seg_patch_tokens.append(seg_adapt_out)
                det_patch_tokens.append(det_adapt_out)
        
        # --- 3. Final Norm & Pooling ---
        if self.is_timm:
            x = self.image_encoder.norm(x)
        elif hasattr(self.image_encoder, 'ln_post'):
            x = self.image_encoder.ln_post(x)
        
        # Extract CLS token
        pooled = x[:, 0]
        
        # Apply visual projection if it exists (768 -> 512)
        if self.visual_proj is not None:
            if isinstance(self.visual_proj, nn.Linear):
                pooled = self.visual_proj(pooled)
            else:
                pooled = pooled @ self.visual_proj
        
        return pooled, seg_patch_tokens, det_patch_tokens

    # --- 4. Helper for Projection (Use this in training loop!) ---
    def project_features(self, tokens):
        """
        Projects tokens (768-dim) to the joint embedding space (512-dim).
        Usage in training: 
           proj_tokens = model.project_features(seg_patch_tokens[layer])
           anomaly_map = proj_tokens @ text_features.T
        """
        if self.visual_proj is None:
            return tokens

        if isinstance(self.visual_proj, nn.Linear):
            return self.visual_proj(tokens)
        else:
            # Handle matrix multiplication manually if it's a Parameter
            return tokens @ self.visual_proj