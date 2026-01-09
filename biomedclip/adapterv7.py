import torch
from torch import nn
import torch.nn.functional as F


class EnhancedClipAdapter(nn.Module):
    """Enhanced adapter with task-specific enhancements"""
    def __init__(self, c_in: int, bottleneck: int = 320, dropout: float = 0.1):
        super().__init__()
        
        # Dual normalization for better gradient flow
        self.ln_input = nn.LayerNorm(c_in)
        self.ln_output = nn.LayerNorm(c_in)
        
        # Enhanced bottleneck with residual
        self.down_proj = nn.Linear(c_in, bottleneck, bias=False)
        self.mid_gelu = nn.GELU()
        self.mid_dropout = nn.Dropout(dropout)
        
        # Additional depth for better representation
        self.mid_proj = nn.Sequential(
            nn.Linear(bottleneck, bottleneck, bias=False),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.up_proj = nn.Linear(bottleneck, c_in, bias=False)
        self.out_dropout = nn.Dropout(dropout)
        
        # Channel attention for feature refinement
        self.channel_attn = nn.Sequential(
            nn.Linear(c_in, c_in // 8),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // 8, c_in),
            nn.Sigmoid()
        )
        
        # Learnable gating and scaling
        self.gate = nn.Parameter(torch.ones(1) * 0.05)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        residual = x
        
        # Main adaptation path
        x = self.ln_input(x)
        x = self.down_proj(x)
        x = self.mid_gelu(x)
        x = self.mid_dropout(x)
        
        # Add residual in bottleneck space
        x = x + 0.5 * self.mid_proj(x)
        
        x = self.up_proj(x)
        x = self.out_dropout(x)
        
        # Apply channel attention
        attn = self.channel_attn(x.mean(dim=1, keepdim=True))
        x = x * attn
        
        # Memory features (normalized separately)
        med = self.ln_output(x) * self.scale
        
        # Gated residual output
        out = residual + self.gate * x
        
        return med, out


class SpatialAttentionPooling(nn.Module):
    """Advanced spatial attention for detection"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim // 4)
        self.key = nn.Linear(hidden_dim, hidden_dim // 4)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = (hidden_dim // 4) ** -0.5
        
    def forward(self, x):
        """x: [B, N, C]"""
        # Compute attention weights
        q = self.query(x.mean(dim=1, keepdim=True))  # [B, 1, C/4]
        k = self.key(x)  # [B, N, C/4]
        v = self.value(x)  # [B, N, C]
        
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)  # [B, 1, N]
        pooled = (attn @ v).squeeze(1)  # [B, C]
        
        return pooled


class MultiScaleDetectionHead(nn.Module):
    """Enhanced detection head with multi-scale reasoning"""
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        
        # Spatial attention pooling for each layer
        self.spatial_pools = nn.ModuleList([
            SpatialAttentionPooling(hidden_dim) for _ in range(num_layers)
        ])
        
        # Layer-wise importance weights
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, token_list):
        """token_list: list of [B, N, C] tensors"""
        # Apply spatial attention pooling to each layer
        pooled_features = []
        for tokens, pool in zip(token_list, self.spatial_pools):
            pooled = pool(tokens.unsqueeze(0) if tokens.dim() == 2 else tokens)
            pooled_features.append(pooled)
        
        # Weighted aggregation
        weights = F.softmax(self.layer_weights, dim=0)
        aggregated = sum(w * feat for w, feat in zip(weights, pooled_features))
        
        # Final fusion
        enhanced = self.fusion(aggregated)
        
        return enhanced


class CLIP_Inplanted_Balanced(nn.Module):
    """Balanced adapter for both segmentation and detection"""
    def __init__(
        self,
        clip_model,
        features,
        bottleneck=320,
        dropout=0.12,
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

        # Enhanced adapters
        self.seg_adapters = nn.ModuleList(
            [EnhancedClipAdapter(hidden_dim, bottleneck=bottleneck, dropout=dropout)
             for _ in self.features]
        )
        self.det_adapters = nn.ModuleList(
            [EnhancedClipAdapter(hidden_dim, bottleneck=bottleneck, dropout=dropout)
             for _ in self.features]
        )

        # Multi-scale detection head
        self.detection_head = MultiScaleDetectionHead(hidden_dim, num_adapted_layers)

        # Learnable blending with normalization
        self.blend_logits = nn.ParameterList([
            nn.Parameter(torch.tensor([2.5, 0.0, 0.0]))  # Backbone bias
            for _ in range(num_adapted_layers)
        ])
        
        # Per-layer temperature for calibration
        self.temperatures = nn.Parameter(torch.ones(num_adapted_layers) * 0.07)
        
        # Feature enhancement
        self.feature_enhance = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5)
            ) for _ in range(num_adapted_layers)
        ])

        # Freeze backbone
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        
        if self.visual_proj is not None:
            for p in self.visual_proj.parameters():
                p.requires_grad = False

    def forward(self, x, return_detection_features=False):
        B = x.shape[0]

        # ViT input
        x = self.image_encoder.patch_embed(x)
        cls_token = self.image_encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.image_encoder.pos_embed
        x = x.image_encoder.pos_drop(x)

        seg_patch_tokens = []
        det_patch_tokens = []
        det_features_for_head = []

        # Transformer blocks
        for layer_idx, block in enumerate(self.image_encoder.blocks, start=1):
            x = block(x)

            if layer_idx in self.features:
                pos = self.features.index(layer_idx)

                # Get adapter features
                seg_med, seg_out = self.seg_adapters[pos](x)
                det_med, det_out = self.det_adapters[pos](x)
                
                # Enhanced blending
                x_enhanced = self.feature_enhance[pos](x)
                blend_weights = F.softmax(self.blend_logits[pos], dim=0)
                
                x = (
                    blend_weights[0] * x_enhanced
                    + blend_weights[1] * seg_out
                    + blend_weights[2] * det_out
                )

                seg_patch_tokens.append(seg_med)
                det_patch_tokens.append(det_med)
                
                # Store for detection head
                det_features_for_head.append(det_med)

        x = self.image_encoder.norm(x)
        pooled = x[:, 0]

        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)

        if return_detection_features:
            # Use enhanced detection head
            det_enhanced = self.detection_head(det_features_for_head)
            return pooled, seg_patch_tokens, det_patch_tokens, det_enhanced
        
        return pooled, seg_patch_tokens, det_patch_tokens
    
    def get_adaptation_weights(self):
        weights = []
        for logits in self.blend_logits:
            weights.append(F.softmax(logits, dim=0).detach().cpu().numpy())
        return weights
    
    def get_temperatures(self):
        return self.temperatures.detach().cpu().numpy()