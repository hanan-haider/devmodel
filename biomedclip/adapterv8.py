import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np



class RegularizedClipAdapter(nn.Module):
    """Regularized adapter to prevent overfitting in few-shot"""
    def __init__(self, c_in: int, bottleneck: int = 256, dropout: float = 0.15):
        super().__init__()
        
        # Layer normalization with learnable affine
        self.ln1 = nn.LayerNorm(c_in, eps=1e-6)
        self.ln2 = nn.LayerNorm(c_in, eps=1e-6)
        
        # Enhanced bottleneck with skip connection
        self.down = nn.Linear(c_in, bottleneck, bias=False)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        
        # Middle layer for added depth
        self.mid = nn.Linear(bottleneck, bottleneck, bias=False)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout * 0.5)
        
        self.up = nn.Linear(bottleneck, c_in, bias=False)
        self.drop3 = nn.Dropout(dropout)
        
        # Squeeze-Excitation for channel recalibration
        self.se = nn.Sequential(
            nn.Linear(c_in, max(c_in // 16, 32)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.3),
            nn.Linear(max(c_in // 16, 32), c_in),
            nn.Sigmoid()
        )
        
        # Gate with better initialization (start conservative)
        self.gate = nn.Parameter(torch.ones(1) * 0.01)
        
        # Learnable scale for memory features
        self.scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        identity = x
        
        # Main adaptation path
        x = self.ln1(x)
        
        # Bottleneck with residual
        h = self.down(x)
        h = self.act1(h)
        h = self.drop1(h)
        
        # Add skip in bottleneck space
        h = h + 0.3 * self.drop2(self.act2(self.mid(h)))
        
        h = self.up(h)
        h = self.drop3(h)
        
        # Channel attention
        attn = self.se(h.mean(dim=1, keepdim=True))
        h = h * attn
        
        # Memory features (for few-shot comparison)
        mem = self.ln2(h) * self.scale
        
        # Gated residual (conservative adaptation)
        out = identity + torch.clamp(self.gate, 0, 0.5) * h
        
        return mem, out


class ImageLevelAggregator(nn.Module):
    """Advanced aggregation from pixel scores to image scores"""
    def __init__(self, hidden_dim=768):
        super().__init__()
        
        # Multi-scale spatial pooling
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(size) for size in [1, 2, 4]
        ])
        
        # Attention-based weighted aggregation
        self.attn_query = nn.Linear(hidden_dim, hidden_dim // 4)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim // 4)
        
        # Score fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(len(self.pools) + 2, 64),  # +2 for max and mean
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
    def forward(self, score_map):
        """
        score_map: [H, W] pixel-level anomaly scores
        returns: scalar image-level score
        """
        if isinstance(score_map, np.ndarray):
            score_map = torch.from_numpy(score_map).float()
        
        # Ensure 4D: [1, 1, H, W]
        if score_map.dim() == 2:
            score_map = score_map.unsqueeze(0).unsqueeze(0)
        elif score_map.dim() == 3:
            score_map = score_map.unsqueeze(0)
        
        features = []
        
        # Multi-scale pooling
        for pool in self.pools:
            pooled = pool(score_map).squeeze()
            features.append(pooled.mean())
        
        # Max score (most anomalous region)
        features.append(score_map.max())
        
        # Mean score (overall anomaly)
        features.append(score_map.mean())
        
        # Fuse features
        features = torch.stack(features).unsqueeze(0)
        score = torch.sigmoid(self.fusion(features))
        
        return score.item()


class CLIP_Inplanted_SOTA(nn.Module):
    """State-of-the-art adapter with anti-overfitting measures"""
    def __init__(
        self,
        clip_model,
        features,
        bottleneck=256,
        dropout=0.15,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.features = sorted(list(features))
        num_adapted_layers = len(self.features)

        # Image encoder
        self.image_encoder = clip_model.visual.trunk

        # Visual projection
        if hasattr(clip_model.visual, "head") and hasattr(clip_model.visual.head, "proj"):
            self.visual_proj = clip_model.visual.head.proj
        else:
            self.visual_proj = getattr(clip_model.visual, "proj", None)

        hidden_dim = getattr(self.image_encoder, "num_features", None)
        if hidden_dim is None:
            hidden_dim = self.image_encoder.embed_dim

        # Regularized adapters
        self.seg_adapters = nn.ModuleList(
            [RegularizedClipAdapter(hidden_dim, bottleneck=bottleneck, dropout=dropout)
             for _ in self.features]
        )
        self.det_adapters = nn.ModuleList(
            [RegularizedClipAdapter(hidden_dim, bottleneck=bottleneck, dropout=dropout)
             for _ in self.features]
        )

        # Learnable blending with L2 regularization
        self.blend_weights = nn.ParameterList([
            nn.Parameter(torch.tensor([0.9, 0.05, 0.05]))  # Strong backbone bias
            for _ in range(num_adapted_layers)
        ])
        
        # Layer importance weights (learnable ensemble)
        self.layer_importance = nn.Parameter(torch.ones(num_adapted_layers))
        
        # Temperature per layer for calibration
        self.temperatures = nn.Parameter(torch.ones(num_adapted_layers) * 0.07)
        
        # Image-level aggregator
        self.image_aggregator = ImageLevelAggregator(hidden_dim)

        # Freeze backbone
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        
        if self.visual_proj is not None:
            for p in self.visual_proj.parameters():
                p.requires_grad = False

    def forward(self, x):
        B = x.shape[0]

        # ViT embedding
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
                seg_mem, seg_out = self.seg_adapters[pos](x)
                det_mem, det_out = self.det_adapters[pos](x)

                # Softmax blending (ensures sum to 1)
                weights = F.softmax(self.blend_weights[pos], dim=0)
                
                x = weights[0] * x + weights[1] * seg_out + weights[2] * det_out

                seg_patch_tokens.append(seg_mem)
                det_patch_tokens.append(det_mem)

        x = self.image_encoder.norm(x)
        pooled = x[:, 0]

        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)

        return pooled, seg_patch_tokens, det_patch_tokens
    
    def get_regularization_loss(self):
        """L2 regularization on blend weights to prevent overfitting"""
        reg_loss = 0
        for weights in self.blend_weights:
            # Penalize deviation from [1, 0, 0] initialization
            target = torch.tensor([1.0, 0.0, 0.0], device=weights.device)
            reg_loss += F.mse_loss(F.softmax(weights, dim=0), target)
        return reg_loss * 0.01
    
    def get_adaptation_weights(self):
        weights = []
        for w in self.blend_weights:
            weights.append(F.softmax(w, dim=0).detach().cpu().numpy())
        return weights