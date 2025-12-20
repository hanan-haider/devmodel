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

        # BiomedCLIP uses a TimmModel wrapper
        assert hasattr(clip_model, "visual"), "clip_model.visual not found"
        assert hasattr(clip_model.visual, "trunk"), "clip_model.visual.trunk not found"

        self.image_encoder = clip_model.visual.trunk  # timm ViT
        self.features = features

        embed_dim = self.image_encoder.num_features  # 768 for ViT-B
        self.seg_adapters = nn.ModuleList(
            [ClipAdapter(embed_dim, bottleneck=embed_dim) for _ in features]
        )
        self.det_adapters = nn.ModuleList(
            [ClipAdapter(embed_dim, bottleneck=embed_dim) for _ in features]
        )

        # projection used by BiomedCLIP
        if hasattr(clip_model.visual, "head") and hasattr(clip_model.visual.head, "proj"):
            self.visual_proj = clip_model.visual.head.proj
        else:
            self.visual_proj = getattr(clip_model.visual, "proj", None)

    def forward(self, x):
        B = x.shape[0]

        # timm ViT tokens: (B, N, C) including cls token
        # forward_features returns tokens BEFORE classifier head
        tokens = self.image_encoder.forward_features(x)  # shape (B, N, C)

        seg_patch_tokens = []
        det_patch_tokens = []

        # timm exposes blocks as self.blocks, we iterate manually to tap intermediate layers
        # but forward_features already ran all blocks; for layer-wise features you need custom loop:
        # here we assume we only want adapter inputs on final tokens => simply treat
        # "tokens" as all-layer-aggregated; if you really need per-block tokens,
        # you must reimplement timm forward.
        #
        # To keep your pipeline working, just use the final tokens once:
        for i, layer_id in enumerate(self.features):
            seg_mid, seg_out = self.seg_adapters[i](tokens)
            det_mid, det_out = self.det_adapters[i](tokens)
            # residual update
            tokens = 0.8 * tokens + 0.1 * seg_out + 0.1 * det_out
            seg_patch_tokens.append(seg_mid)   # (B, N, C)
            det_patch_tokens.append(det_mid)   # (B, N, C)

        # global pooling: cls token
        pooled = tokens[:, 0]  # (B, C)

        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)

        return pooled, seg_patch_tokens, det_patch_tokens
