import math
import torch
from torch import nn


class ClipAdapterBiomed(nn.Module):
    """Residual adapter for BiomedCLIP visual tokens (C=768)."""
    def __init__(self, c_in: int, bottleneck: int = 768):
        super().__init__()
        self.ln = nn.LayerNorm(c_in)
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        # small gate so we start close to backbone
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: [B, N, C] (CLS + patch tokens) from ViT trunk.
        Returns:
          med: adapter internal features for memory bank [B, N, C]
          out: updated tokens [B, N, C]
        """
        residual = x
        x = self.ln(x)
        x = self.fc1(x)
        x = self.fc2(x)
        med = x
        out = residual + self.gate * x
        return med, out


class CLIP_Inplanted(nn.Module):
    """
    BiomedCLIP visual trunk + residual adapters (seg/det) at selected layers.
    This mirrors your original standard CLIP adapter but uses
    clip_model.visual.trunk and 768-d tokens.
    """
    def __init__(self, clip_model, features, bottleneck=768):
        super().__init__()
        self.clipmodel = clip_model
        self.features = sorted(list(features))          # e.g. [3, 6, 9, 12]
        self.image_encoder = clip_model.visual.trunk    # ViT-B/16 trunk

        # BiomedCLIP visual projection 768->512
        if hasattr(clip_model.visual, "head") and hasattr(clip_model.visual.head, "proj"):
            self.visual_proj = clip_model.visual.head.proj
        else:
            self.visual_proj = getattr(clip_model.visual, "proj", None)

        hidden_dim = getattr(self.image_encoder, "num_features", None)
        if hidden_dim is None:
            hidden_dim = self.image_encoder.embed_dim  # 768 for ViT-B/16

        self.seg_adapters = nn.ModuleList(
            [ClipAdapterBiomed(hidden_dim, bottleneck=bottleneck) for _ in self.features]
        )
        self.det_adapters = nn.ModuleList(
            [ClipAdapterBiomed(hidden_dim, bottleneck=bottleneck) for _ in self.features]
        )

        # learnable blending per layer
        L = len(self.features)
        self.alpha_backbone = nn.Parameter(torch.ones(L) * 0.8)
        self.alpha_seg = nn.Parameter(torch.ones(L) * 0.1)
        self.alpha_det = nn.Parameter(torch.ones(L) * 0.1)

        # freeze backbone
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        if self.visual_proj is not None:
            for p in self.visual_proj.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: [B, 3, 224, 224]
        Returns:
          pooled: [B, 512]
          seg_patch_tokens: list of [B, N, C]
          det_patch_tokens: list of [B, N, C]
        """
        B = x.shape[0]

        # patch embedding + CLS + pos (BiomedCLIP ViT)
        x = self.image_encoder.patch_embed(x)                        # [B, N, C]
        cls_token = self.image_encoder.cls_token.expand(B, -1, -1)   # [B, 1, C]
        x = torch.cat((cls_token, x), dim=1)                         # [B, 1+N, C]
        x = x + self.image_encoder.pos_embed
        x = self.image_encoder.pos_drop(x)

        seg_patch_tokens = []
        det_patch_tokens = []

        for layer_idx, block in enumerate(self.image_encoder.blocks, start=1):
            x = block(x)

            if layer_idx in self.features:
                pos = self.features.index(layer_idx)

                seg_med, seg_out = self.seg_adapters[pos](x)   # [B, N, C]
                det_med, det_out = self.det_adapters[pos](x)   # [B, N, C]

                # normalize alphas
                alphas = torch.stack([
                    self.alpha_backbone[pos],
                    self.alpha_seg[pos],
                    self.alpha_det[pos]
                ], dim=0)
                alphas = torch.softmax(alphas, dim=0)

                x = (
                    alphas[0] * x +
                    alphas[1] * seg_out +
                    alphas[2] * det_out
                )

                seg_patch_tokens.append(seg_med)
                det_patch_tokens.append(det_med)

        x = self.image_encoder.norm(x)   # [B, 1+N, C]
        pooled = x[:, 0]                 # CLS

        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)  # [B, 512]

        return pooled, seg_patch_tokens, det_patch_tokens
