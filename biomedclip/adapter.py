import torch
from torch import nn
import torch.nn.functional as F

class ImprovedClipAdapter(nn.Module):
    """Enhanced adapter with multi-scale feature processing and attention"""
    def __init__(self, c_in: int, bottleneck: int = 256, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(c_in)
        self.ln2 = nn.LayerNorm(c_in)
        
        # Multi-scale bottleneck with different compression ratios
        self.branch1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, c_in, bias=False),
        )
        
        self.branch2 = nn.Sequential(
            nn.Linear(c_in, bottleneck // 2, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck // 2, c_in, bias=False),
        )
        
        # Channel attention mechanism
        self.channel_attn = nn.Sequential(
            nn.Linear(c_in, c_in // 16),
            nn.GELU(),
            nn.Linear(c_in // 16, c_in),
            nn.Sigmoid()
        )
        
        # Spatial attention for patch tokens
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable gating with better initialization
        self.gate = nn.Parameter(torch.zeros(1))
        self.branch_weight = nn.Parameter(torch.ones(2) * 0.5)
        
    def forward(self, x):
        residual = x
        
        # Multi-branch processing
        x_norm = self.ln1(x)
        feat1 = self.branch1(x_norm)
        feat2 = self.branch2(x_norm)
        
        # Weighted branch fusion
        weights = F.softmax(self.branch_weight, dim=0)
        multi_scale_feat = weights[0] * feat1 + weights[1] * feat2
        
        # Channel attention
        B, N, C = x.shape
        channel_attn = self.channel_attn(multi_scale_feat.mean(dim=1, keepdim=True))
        multi_scale_feat = multi_scale_feat * channel_attn
        
        # Apply spatial attention if we have patch tokens (N > 1)
        if N > 1:
            # Reshape to spatial format for attention
            H = int((N - 1) ** 0.5)  # Assuming square patches, -1 for CLS token
            if H * H == N - 1:
                spatial_feat = multi_scale_feat[:, 1:, :].reshape(B, H, H, C).permute(0, 3, 1, 2)
                avg_pool = torch.mean(spatial_feat, dim=1, keepdim=True)
                max_pool = torch.max(spatial_feat, dim=1, keepdim=True)[0]
                spatial_concat = torch.cat([avg_pool, max_pool], dim=1)
                spatial_attn = self.spatial_attn(spatial_concat)
                spatial_feat = spatial_feat * spatial_attn
                spatial_feat = spatial_feat.permute(0, 2, 3, 1).reshape(B, H * H, C)
                multi_scale_feat = torch.cat([multi_scale_feat[:, :1, :], spatial_feat], dim=1)
        
        multi_scale_feat = self.dropout(multi_scale_feat)
        
        # High-resolution features for memory bank
        med = self.ln2(multi_scale_feat)
        
        # Gated residual connection
        out = residual + self.gate * multi_scale_feat
        
        return med, out


class CLIP_Inplanted(nn.Module):
    """Enhanced CLIP with improved adapters and feature fusion"""
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

        # Enhanced adapters
        self.seg_adapters = nn.ModuleList([
            ImprovedClipAdapter(hidden_dim, bottleneck=bottleneck, dropout=dropout)
            for _ in self.features
        ])
        self.det_adapters = nn.ModuleList([
            ImprovedClipAdapter(hidden_dim, bottleneck=bottleneck, dropout=dropout)
            for _ in self.features
        ])

        # Learnable layer-wise blending with better initialization
        self.alpha_backbone = nn.Parameter(torch.ones(num_adapted_layers) * 0.7)
        self.alpha_seg = nn.Parameter(torch.ones(num_adapted_layers) * 0.15)
        self.alpha_det = nn.Parameter(torch.ones(num_adapted_layers) * 0.15)
        
        # Cross-adapter attention for better task interaction
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
            for _ in self.features
        ])
        
        # Task-specific projection heads for better feature discrimination
        self.seg_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.det_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Freeze backbone
        for p in self.image_encoder.parameters():
            p.requires_grad = False
            
        # Keep visual projection trainable for better alignment
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

        # Transformer blocks with enhanced adapter integration
        for layer_idx, block in enumerate(self.image_encoder.blocks, start=1):
            x = block(x)

            if layer_idx in self.features:
                pos = self.features.index(layer_idx)

                # Get adapter outputs
                seg_med, seg_out = self.seg_adapters[pos](x)
                det_med, det_out = self.det_adapters[pos](x)
                
                # Cross-adapter interaction for task synergy
                seg_enhanced, _ = self.cross_attn[pos](seg_out, det_out, det_out)
                det_enhanced, _ = self.cross_attn[pos](det_out, seg_out, seg_out)
                
                # Normalize alphas to ensure stable training
                alphas = torch.softmax(torch.stack([
                    self.alpha_backbone[pos],
                    self.alpha_seg[pos],
                    self.alpha_det[pos]
                ]), dim=0)
                
                # Enhanced feature fusion
                x = (
                    alphas[0] * x +
                    alphas[1] * (0.7 * seg_out + 0.3 * seg_enhanced) +
                    alphas[2] * (0.7 * det_out + 0.3 * det_enhanced)
                )
                
                # Apply task-specific projections to memory features
                seg_med_proj = self.seg_proj(seg_med)
                det_med_proj = self.det_proj(det_med)

                seg_patch_tokens.append(seg_med_proj)
                det_patch_tokens.append(det_med_proj)

        x = self.image_encoder.norm(x)
        pooled = x[:, 0]

        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)

        return pooled, seg_patch_tokens, det_patch_tokens