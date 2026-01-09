import torch
from torch import nn
import torch.nn.functional as F


class ImprovedClipAdapterBiomed(nn.Module):
    """
    Enhanced adapter for BiomedCLIP visual trunk:
      - Multi-branch MLP (two bottlenecks)
      - Channel + spatial attention on patch tokens
      - Returns:
          med: high-res features for memory bank [B, N, C]
          out: updated transformer tokens [B, N, C]
    """
    def __init__(self, c_in: int, bottleneck: int = 256, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(c_in)
        self.ln2 = nn.LayerNorm(c_in)

        # Multi-scale bottlenecks
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

        # Channel attention (SE-style)
        self.channel_attn = nn.Sequential(
            nn.Linear(c_in, max(c_in // 16, 1)),
            nn.GELU(),
            nn.Linear(max(c_in // 16, 1), c_in),
            nn.Sigmoid()
        )

        # Spatial attention over patch tokens (N-1 -> H×W)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

        # Gating for residual fusion
        self.gate = nn.Parameter(torch.zeros(1))
        self.branch_weight = nn.Parameter(torch.ones(2) * 0.5)

    def forward(self, x):
        """
        x: [B, N, C] where N = 1 + H*W (CLS + patches)
        """
        residual = x

        # Multi-branch processing
        x_norm = self.ln1(x)
        feat1 = self.branch1(x_norm)
        feat2 = self.branch2(x_norm)

        # Weighted fusion of branches
        weights = F.softmax(self.branch_weight, dim=0)
        multi_scale_feat = weights[0] * feat1 + weights[1] * feat2  # [B, N, C]

        # Channel attention (global over tokens)
        B, N, C = multi_scale_feat.shape
        # global context over tokens
        ch_context = multi_scale_feat.mean(dim=1)            # [B, C]
        ch_weight = self.channel_attn(ch_context).unsqueeze(1)  # [B, 1, C]
        multi_scale_feat = multi_scale_feat * ch_weight      # [B, N, C]

        # Spatial attention over patch tokens if N>1
        if N > 1:
            # N = 1 + H*W (assume square grid)
            patch_tokens = multi_scale_feat[:, 1:, :]        # [B, H*W, C]
            H = int((patch_tokens.shape[1]) ** 0.5)
            if H * H == patch_tokens.shape[1]:
                spatial_feat = patch_tokens.view(B, H, H, C).permute(0, 3, 1, 2)  # [B, C, H, W]
                avg_pool = spatial_feat.mean(dim=1, keepdim=True)
                max_pool, _ = spatial_feat.max(dim=1, keepdim=True)
                spatial_concat = torch.cat([avg_pool, max_pool], dim=1)          # [B, 2, H, W]
                attn = self.spatial_attn(spatial_concat)                         # [B, 1, H, W]
                spatial_feat = spatial_feat * attn                               # [B, C, H, W]
                spatial_feat = spatial_feat.permute(0, 2, 3, 1).reshape(B, H * H, C)
                multi_scale_feat = torch.cat(
                    [multi_scale_feat[:, :1, :], spatial_feat], dim=1
                )

        multi_scale_feat = self.dropout(multi_scale_feat)

        # High-res features for memory bank (normalized)
        med = self.ln2(multi_scale_feat)

        # Gated residual
        out = residual + self.gate * multi_scale_feat

        return med, out


class CLIP_Inplanted(nn.Module):
    """
    BiomedCLIP visual trunk + improved adapters:
      - Uses clip_model.visual.trunk as frozen ViT-B/16
      - Adds seg/det adapters at selected layers
      - Blends backbone/seg/det features with learnable alphas
      - visual_proj stays as in BiomedCLIP (768->512) for alignment with text.
    """
    def __init__(
        self,
        clip_model,
        features,          # e.g. [3, 6, 9, 12]
        bottleneck=256,
        dropout=0.1,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.features = sorted(list(features))
        num_adapted_layers = len(self.features)

        # BiomedCLIP visual trunk (timm ViT)
        self.image_encoder = clip_model.visual.trunk

        # 768->512 projection from BiomedCLIP
        if hasattr(clip_model.visual, "head") and hasattr(clip_model.visual.head, "proj"):
            self.visual_proj = clip_model.visual.head.proj
        else:
            self.visual_proj = getattr(clip_model.visual, "proj", None)

        hidden_dim = getattr(self.image_encoder, "num_features", None)
        if hidden_dim is None:
            hidden_dim = self.image_encoder.embed_dim

        # Task-specific adapters
        self.seg_adapters = nn.ModuleList([
            ImprovedClipAdapterBiomed(hidden_dim, bottleneck=bottleneck, dropout=dropout)
            for _ in self.features
        ])
        self.det_adapters = nn.ModuleList([
            ImprovedClipAdapterBiomed(hidden_dim, bottleneck=bottleneck, dropout=dropout)
            for _ in self.features
        ])

        # Learnable layer-wise blending (backbone / seg / det)
        self.alpha_backbone = nn.Parameter(torch.ones(num_adapted_layers) * 0.7)
        self.alpha_seg = nn.Parameter(torch.ones(num_adapted_layers) * 0.15)
        self.alpha_det = nn.Parameter(torch.ones(num_adapted_layers) * 0.15)

        # Cross-adapter interaction
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
            for _ in self.features
        ])

        # Optional task-specific projection of memory features
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

        # Freeze backbone for few-shot stability
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        # You can choose: keep visual_proj frozen (pure adapter) or trainable.
        if self.visual_proj is not None:
            for p in self.visual_proj.parameters():
                p.requires_grad = False  # set True if you want to adapt projection

    def forward(self, x):
        """
        x: [B, 3, H, W] (224×224 for BiomedCLIP)
        Returns:
          pooled: [B, 512]
          seg_patch_tokens: list of [B, N, C]
          det_patch_tokens: list of [B, N, C]
        """
        B = x.shape[0]

        # Standard BiomedCLIP ViT input pipeline
        x = self.image_encoder.patch_embed(x)                        # [B, N, C]
        cls_token = self.image_encoder.cls_token.expand(B, -1, -1)   # [B,1,C]
        x = torch.cat((cls_token, x), dim=1)                         # [B,1+N,C]
        x = x + self.image_encoder.pos_embed
        x = self.image_encoder.pos_drop(x)

        seg_patch_tokens = []
        det_patch_tokens = []

        for layer_idx, block in enumerate(self.image_encoder.blocks, start=1):
            x = block(x)  # transformer block

            if layer_idx in self.features:
                pos = self.features.index(layer_idx)

                # Adapter outputs
                seg_med, seg_out = self.seg_adapters[pos](x)  # [B,N,C]
                det_med, det_out = self.det_adapters[pos](x)  # [B,N,C]

                # Cross-attention between seg & det branches
                seg_enh, _ = self.cross_attn[pos](seg_out, det_out, det_out)
                det_enh, _ = self.cross_attn[pos](det_out, seg_out, seg_out)

                # Normalize alphas
                alpha_vec = torch.stack([
                    self.alpha_backbone[pos],
                    self.alpha_seg[pos],
                    self.alpha_det[pos]
                ], dim=0)
                alpha_vec = torch.softmax(alpha_vec, dim=0)

                # Blend backbone + enhanced adapters
                x = (
                    alpha_vec[0] * x +
                    alpha_vec[1] * (0.7 * seg_out + 0.3 * seg_enh) +
                    alpha_vec[2] * (0.7 * det_out + 0.3 * det_enh)
                )

                # Project memory features for seg/det heads
                seg_med_proj = self.seg_proj(seg_med)
                det_med_proj = self.det_proj(det_med)

                seg_patch_tokens.append(seg_med_proj)
                det_patch_tokens.append(det_med_proj)

        # CLS token for image-level embedding
        x = self.image_encoder.norm(x)
        pooled = x[:, 0]   # [B, C=768]

        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)  # [B, 512] aligned with text tower

        return pooled, seg_patch_tokens, det_patch_tokens
