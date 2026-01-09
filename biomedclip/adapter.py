import math
import torch
from torch import nn


# Residual Adapter (unchanged logic)
class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=512):
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
        mid = self.fc1(x)
        out = self.fc2(mid)
        return mid, out


class BioMedCLIP_Implanted(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()

        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features

        embed_dim = self.image_encoder.width   # 768 for ViT-B/16

        self.seg_adapters = nn.ModuleList([
            ClipAdapter(embed_dim, bottleneck=512)
            for _ in range(len(features))
        ])

        self.det_adapters = nn.ModuleList([
            ClipAdapter(embed_dim, bottleneck=512)
            for _ in range(len(features))
        ])

    def forward(self, x):
        # ---- Stem ----
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls_token = self.image_encoder.class_embedding.to(x.dtype)
        cls_token = cls_token.unsqueeze(0).repeat(x.shape[0], 1, 1)

        x = torch.cat([cls_token, x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)

        seg_patch_tokens = []
        det_patch_tokens = []

        # ---- Transformer (12 layers for BiomedCLIP ViT-B) ----
        for i, block in enumerate(self.image_encoder.transformer.resblocks):
            x, attn = block(x, attn_mask=None)

            if (i + 1) in self.features:
                idx = self.features.index(i + 1)

                seg_mid, seg_out = self.seg_adapters[idx](x)
                det_mid, det_out = self.det_adapters[idx](x)

                x = 0.8 * x + 0.1 * seg_out + 0.1 * det_out

                seg_patch_tokens.append(seg_mid)
                det_patch_tokens.append(det_mid)

        x = x.permute(1, 0, 2)

        seg_patch_tokens = [t.permute(1, 0, 2) for t in seg_patch_tokens]
        det_patch_tokens = [t.permute(1, 0, 2) for t in det_patch_tokens]

        # ---- Global Pool ----
        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        return pooled, seg_patch_tokens, det_patch_tokens
