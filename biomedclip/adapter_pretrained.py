import torch
import torch.nn as nn

class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x_mid = self.fc1(x)
        x_out = self.fc2(x_mid)
        return x_mid, x_out
class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model
        assert hasattr(clip_model, "visual")
        assert hasattr(clip_model.visual, "trunk")
        self.image_encoder = clip_model.visual.trunk   # timm ViT
        self.features = features

        embed_dim = self.image_encoder.num_features
        self.seg_adapters = nn.ModuleList(
            [ClipAdapter(embed_dim, bottleneck=embed_dim) for _ in features]
        )
        self.det_adapters = nn.ModuleList(
            [ClipAdapter(embed_dim, bottleneck=embed_dim) for _ in features]
        )

        if hasattr(clip_model.visual, "head") and hasattr(clip_model.visual.head, "proj"):
            self.visual_proj = clip_model.visual.head.proj
        else:
            self.visual_proj = getattr(clip_model.visual, "proj", None)

    def forward(self, x):
        tokens = self.image_encoder.forward_features(x)  # (B, N, C)

        seg_patch_tokens = []
        det_patch_tokens = []

        for i in range(len(self.features)):
            seg_mid, seg_out = self.seg_adapters[i](tokens)
            det_mid, det_out = self.det_adapters[i](tokens)
            tokens = 0.8 * tokens + 0.1 * seg_out + 0.1 * det_out
            seg_patch_tokens.append(seg_mid)
            det_patch_tokens.append(det_mid)

        pooled = tokens[:, 0]
        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)

        # ensure not None
        assert isinstance(seg_patch_tokens, list) and len(seg_patch_tokens) > 0
        assert isinstance(det_patch_tokens, list) and len(det_patch_tokens) > 0

        return pooled, seg_patch_tokens, det_patch_tokens

