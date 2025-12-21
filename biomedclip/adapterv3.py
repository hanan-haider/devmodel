import torch
from torch import nn


# ----------------------------
# Residual CLIP Adapter (improved)
# ----------------------------
class ClipAdapter(nn.Module):
    """
    Lightweight residual adapter:
    - LayerNorm for stability
    - Bottleneck MLP with GELU
    - Learnable residual scale (starts at 0)
    """

    def __init__(self, c_in: int, bottleneck: int = 256, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(c_in)

        self.mlp = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.GELU(),
            nn.Linear(bottleneck, c_in, bias=False),
            nn.Dropout(dropout),
        )

        # start from frozen backbone, gradually turn on adapter
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: (B, N, C)
        returns:
            med: intermediate adapter features (B, N, C)
            out: residual-enhanced features (B, N, C)
        """
        residual = x
        x = self.ln(x)
        x = self.mlp(x)

        med = x
        out = residual + self.scale * x
        return med, out


# ----------------------------
# CLIP_Inplanted for BioMedCLIP
# ----------------------------
class CLIP_Inplanted(nn.Module):
    """
    Adapter-augmented BioMedCLIP vision encoder.

    - Uses clip_model.visual.trunk (timm ViT-B/16).
    - Injects seg/det adapters at selected transformer layers.
    - Keeps ViT backbone frozen; only adapters + projection are trained.
    """

    def __init__(
        self,
        clip_model,
        features,           # e.g. [4, 8, 10, 12]
        bottleneck=256,
        dropout=0.1,
        alpha_backbone=0.8, # residual blending coefficients
        alpha_seg=0.1,
        alpha_det=0.1,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.features = sorted(list(features))

        # timm ViT trunk used by BioMedCLIP
        self.image_encoder = clip_model.visual.trunk

        # final projection: visual.head.proj (BioMedCLIP) or visual.proj
        if hasattr(clip_model.visual, "head") and hasattr(clip_model.visual.head, "proj"):
            self.visual_proj = clip_model.visual.head.proj
        else:
            self.visual_proj = getattr(clip_model.visual, "proj", None)

        # hidden dim of ViT-B (usually 768)
        hidden_dim = getattr(self.image_encoder, "num_features", None)
        if hidden_dim is None:
            hidden_dim = self.image_encoder.embed_dim

        # build adapters for each selected layer
        self.seg_adapters = nn.ModuleList(
            [ClipAdapter(hidden_dim, bottleneck=bottleneck, dropout=dropout)
             for _ in self.features]
        )
        self.det_adapters = nn.ModuleList(
            [ClipAdapter(hidden_dim, bottleneck=bottleneck, dropout=dropout)
             for _ in self.features]
        )

        # residual mixing
        self.alpha_backbone = alpha_backbone
        self.alpha_seg = alpha_seg
        self.alpha_det = alpha_det

        # freeze backbone (recommended for few-shot AD)
        for p in self.image_encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        x: (B, 3, H, W), preprocessed for BioMedCLIP
        returns:
            pooled: (B, D) image embedding (D = 512 for BioMedCLIP)
            seg_patch_tokens: list of (B, N, C) adapter features for seg branch
            det_patch_tokens: list of (B, N, C) adapter features for det branch
        """
        B = x.shape[0]

        # ----- ViT input embedding -----
        x = self.image_encoder.patch_embed(x)              # (B, N, C)
        cls_token = self.image_encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)               # (B, 1+N, C)
        x = x + self.image_encoder.pos_embed
        x = self.image_encoder.pos_drop(x)

        seg_patch_tokens = []
        det_patch_tokens = []

        # ----- transformer blocks with adapters -----
        for layer_idx, block in enumerate(self.image_encoder.blocks, start=1):
            x = block(x)

            if layer_idx in self.features:
                pos = self.features.index(layer_idx)

                seg_med, seg_out = self.seg_adapters[pos](x)
                det_med, det_out = self.det_adapters[pos](x)

                # residual blending: keep strong backbone signal
                x = (
                    self.alpha_backbone * x
                    + self.alpha_seg * seg_out
                    + self.alpha_det * det_out
                )

                # store patch tokens (without CLS) for downstream seg/det heads
                seg_patch_tokens.append(seg_med)
                det_patch_tokens.append(det_med)

        # ----- ViT output head -----
        x = self.image_encoder.norm(x)
        pooled = x[:, 0]  # CLS token

        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)

        return pooled, seg_patch_tokens, det_patch_tokens
