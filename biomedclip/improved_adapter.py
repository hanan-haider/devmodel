import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# 1. Strong LoRA-style Adapter (GELU + Dropout + Scaling)
# =====================================================
class LoRAAdapter(nn.Module):
    def __init__(self, dim=768, r=64, alpha=16.0, dropout=0.1):
        super().__init__()
        self.r = r
        self.scaling = alpha / r

        self.down = nn.Linear(dim, r, bias=False)
        self.up = nn.Linear(r, dim, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Zero-init up → adapter starts as identity
        nn.init.zeros_(self.up.weight)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, x):
        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up(x)
        return residual + x * self.scaling


# =====================================================
# 2. Tip-Adapter Head (Non-parametric Few-Shot Boost)
# =====================================================
class TipAdapterHead(nn.Module):
    def __init__(self, alpha=4.5, beta=6.8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.cache_keys = None  # Will be filled during training

    def build_cache(self, support_features_list):
        """
        support_features_list: list of tensors (N_support, D) from normal images
        """
        self.cache_keys = [f.detach().cpu() for f in support_features_list]  # list of (N, D)
        self.register_buffer('cache_keys_norm',
                             torch.stack([F.normalize(k, dim=-1) for k in self.cache_keys]), persistent=False)

    def forward(self, query_features_list):
        """
        query_features_list: list of (B, N_patch, D)
        Returns: fused residual features (B, D)
        """
        if self.cache_keys is None:
            raise RuntimeError("TipAdapter cache not built! Call build_cache() first.")

        residual = 0
        for q, k_norm in zip(query_features_list, self.cache_keys_norm):
            q = F.normalize(q, dim=-1)
            affinity = q @ k_norm.T  # (B, N_patch, N_support)
            affinity = affinity * self.beta
            weight = F.softmax(affinity, dim=-1)
            residual = residual + weight @ k_norm.to(q.device)  # (B, N_patch, D) → sum over layers

        return residual  # (B, N_patch, D) or you can mean-pool if needed


# =====================================================
# 3. Final Unified Adapter Model (Drop-in Replacement)
# =====================================================
class BioMedCLIP_Adapter(nn.Module):
    def __init__(self, clip_model, feature_layers=[3, 6, 9, 12], adapter_rank=64, use_tip_adapter=True):
        super().__init__()
        self.clip_model = clip_model
        self.feature_layers = feature_layers
        self.use_tip_adapter = use_tip_adapter

        # === Visual backbone ===
        self.visual = clip_model.visual.trunk if hasattr(clip_model.visual, 'trunk') else clip_model.visual

        # === Projection head ===
        if hasattr(clip_model.visual, 'head') and hasattr(clip_model.visual.head, 'proj'):
            self.proj = clip_model.visual.head.proj
        elif hasattr(clip_model.visual, 'proj'):
            self.proj = clip_model.visual.proj
        else:
            self.proj = None

        # === Adapters ===
        dim = 768  # BioMedCLIP ViT-B/16
        self.seg_adapters = nn.ModuleList([LoRAAdapter(dim, r=adapter_rank) for _ in feature_layers])
        self.det_adapters = nn.ModuleList([LoRAAdapter(dim, r=adapter_rank) for _ in feature_layers])

        # Learnable layer fusion weights
        self.register_parameter('seg_fusion_weights', nn.Parameter(torch.ones(len(feature_layers))))
        self.register_parameter('det_fusion_weights', nn.Parameter(torch.ones(len(feature_layers))))

        # Tip-Adapter head
        if use_tip_adapter:
            self.tip_adapter = TipAdapterHead(alpha=4.5, beta=6.8)
        else:
            self.tip_adapter = None

        # Freeze backbone
        for p in self.visual.parameters():
            p.requires_grad = False

    def forward(self, x, apply_tip=False):
        B = x.shape[0]
        seg_patch_tokens = []
        det_patch_tokens = []

        # === Patch embedding + cls token + pos embed ===
        if hasattr(self.visual, 'patch_embed'):
            x = self.visual.patch_embed(x)
            cls_token = self.visual.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.visual.pos_embed
            x = self.visual.pos_drop(x)
        else:
            # Standard OpenAI CLIP path
            x = self.visual.conv1(x)
            x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)
            cls_tokens = self.visual.class_embedding.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.visual.positional_embedding

        # === Transformer blocks ===
        for idx, block in enumerate(self.visual.blocks):
            x = block(x)

            layer_id = idx + 1
            if layer_id in self.feature_layers:
                i = self.feature_layers.index(layer_id)

                # Apply adapters
                seg_feat = self.seg_adapters[i](x)
                det_feat = self.det_adapters[i](x)

                # Light residual update
                x = x + 0.2 * seg_feat + 0.2 * det_feat

                # Save patch tokens (exclude CLS)
                seg_patch_tokens.append(seg_feat[:, 1:, :])
                det_patch_tokens.append(det_feat[:, 1:, :])

        # Final norm
        if hasattr(self.visual, 'norm'):
            x = self.visual.norm(x)

        pooled = x[:, 0]
        if self.proj is not None:
            pooled = self.proj(pooled)

        # === Optional: Apply Tip-Adapter residual (only during inference or few-shot phase) ===
        if apply_tip and self.tip_adapter is not None and self.tip_adapter.cache_keys is not None:
            tip_residual = self.tip_adapter(seg_patch_tokens)  # (B, N, D)
            tip_residual = tip_residual.mean(dim=1)  # Global average pool → (B, D)
            pooled = pooled + tip_residual.to(pooled.device)

        return pooled, seg_patch_tokens, det_patch_tokens

    # Helper to build Tip-Adapter cache from normal support set
    def build_tip_cache(self, support_loader, device):
        if not self.use_tip_adapter:
            return
        support_feats = []
        self.eval()
        with torch.no_grad():
            for img, in support_loader:
                img = img.to(device)
                _, seg_tokens, _ = self(img, apply_tip=False)
                # Use fused or last layer — here we use all layers
                fused = sum(
                    w * t for w, t in zip(F.softmax(self.seg_fusion_weights, dim=0), seg_tokens)
                )
                support_feats.append(fused.mean(dim=1))  # (B, D) → collapse patches
        # Concat all support features
        all_support = torch.cat(support_feats, dim=0)  # (N_support, D)
        layer_wise = [all_support for _ in self.feature_layers]  # repeat for each layer
        self.tip_adapter.build_cache(layer_wise)
        self.train()