import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# 1. SOTA LoRA Adapter (Zero-Initialized Gating)
# =====================================================
class LoRAAdapter(nn.Module):
    def __init__(self, dim=768, r=64, alpha=16.0, dropout=0.1):
        super().__init__()
        self.r = r
        self.scaling = alpha / r

        # Low-rank matrices
        self.down = nn.Linear(dim, r, bias=False)
        self.up = nn.Linear(r, dim, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # SOTA FIX: Learnable Gate initialized to ZERO
        # This ensures the model starts exactly as the pretrained BioMedCLIP
        # and slowly learns to inject anomaly signals.
        self.gate = nn.Parameter(torch.zeros(1))

        # Weight Initialization
        nn.init.zeros_(self.up.weight) # Zero-init up projection
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))

    def forward(self, x):
        # We do NOT add residual here, we return the delta
        # The parent class handles the residual connection
        z = self.down(x)
        z = self.act(z)
        z = self.dropout(z)
        z = self.up(z)
        
        # Apply scaling and the learnable gate
        return z * self.scaling * self.gate

# =====================================================
# 2. Tip-Adapter Head (Cache Retrieval)
# =====================================================
class TipAdapterHead(nn.Module):
    def __init__(self, beta=5.5):
        super().__init__()
        self.beta = beta
        self.cache_keys = None 
        self.cache_values = None

    def build_cache(self, support_keys, support_values=None):
        """
        support_keys: (N_shots, Dim) - Feature vectors from normal images
        support_values: Optional one-hot labels (not needed for pure anomaly detection)
        """
        # Normalize keys for Cosine Similarity
        self.cache_keys = F.normalize(support_keys, dim=-1).detach()
        if support_values is not None:
            self.cache_values = support_values.detach()
        
    def forward(self, query_features):
        """
        query_features: (B, Dim)
        """
        if self.cache_keys is None:
            return query_features # Fallback if cache not built

        # 1. Normalize Query
        q = F.normalize(query_features, dim=-1)
        
        # 2. Calculate Affinity (Cosine Similarity)
        # (B, Dim) @ (N_shots, Dim)^T -> (B, N_shots)
        affinity = q @ self.cache_keys.T
        
        # 3. Sharpening
        affinity = (-self.beta * (1 - affinity)).exp() 
        
        # 4. Weighted Sum (Retrieval)
        # We retrieve the "Normal" features most similar to our query
        retrieved_features = affinity @ self.cache_keys
        
        # 5. Residual Injection
        # We add the retrieved "normal" knowledge to the query
        # alpha can be learnable or fixed
        alpha = 0.5 
        return query_features + alpha * retrieved_features


# =====================================================
# 3. BioMedCLIP Unified Model
# =====================================================
class BioMedCLIP_Adapter(nn.Module):
    def __init__(self, clip_model, feature_layers=[3, 6, 9, 12], adapter_rank=64, use_tip_adapter=True):
        super().__init__()
        self.clip_model = clip_model
        self.feature_layers = feature_layers
        self.use_tip_adapter = use_tip_adapter

        # --- A. Identify Backbone Components ---
        if hasattr(clip_model.visual, 'trunk'):
            self.visual = clip_model.visual.trunk # Timm wrapper
            self.is_timm = True
        else:
            self.visual = clip_model.visual # OpenAI
            self.is_timm = False

        # --- B. Identify Projection Head (Critical for Anomaly Space) ---
        if hasattr(clip_model.visual, 'head') and hasattr(clip_model.visual.head, 'proj'):
            self.proj = clip_model.visual.head.proj
        elif hasattr(clip_model.visual, 'proj'):
            self.proj = clip_model.visual.proj
        else:
            self.proj = None

        # --- C. LoRA Adapters ---
        dim = 768 # BioMedCLIP ViT-B
        # We create independent adapters for Segmentation (Pixel) and Detection (Image)
        self.seg_adapters = nn.ModuleList([LoRAAdapter(dim, r=adapter_rank) for _ in feature_layers])
        self.det_adapters = nn.ModuleList([LoRAAdapter(dim, r=adapter_rank) for _ in feature_layers])

        # --- D. Tip Adapter ---
        if use_tip_adapter:
            self.tip_adapter = TipAdapterHead()
        
        # --- E. Freeze Backbone ---
        for p in self.visual.parameters():
            p.requires_grad = False
        if self.proj is not None:
            if isinstance(self.proj, nn.Parameter):
                self.proj.requires_grad = False
            else:
                for p in self.proj.parameters(): p.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        
        # 1. Patch / Positional Embedding
        if self.is_timm:
            x = self.visual.patch_embed(x)
            cls_token = self.visual.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.visual.pos_embed
            x = self.visual.pos_drop(x)
            blocks = self.visual.blocks
        else:
            x = self.visual.conv1(x)
            x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)
            cls_tokens = self.visual.class_embedding.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.visual.positional_embedding
            x = self.visual.ln_pre(x)
            blocks = self.visual.transformer.resblocks

        seg_patch_tokens = []
        det_patch_tokens = []

        # 2. Pass through Blocks
        for i, block in enumerate(blocks):
            x = block(x)
            
            # Inject Adapters at specific layers
            if (i + 1) in self.feature_layers:
                idx = self.feature_layers.index(i + 1)
                
                # Get Adapter Deltas (Gate is inside LoRAAdapter)
                delta_seg = self.seg_adapters[idx](x)
                delta_det = self.det_adapters[idx](x)
                
                # SOTA FIX: Pure Residual (x + delta)
                x = x + delta_seg + delta_det
                
                # Store features for loss calculation
                seg_patch_tokens.append(delta_seg[:, 1:, :]) # Skip CLS
                det_patch_tokens.append(delta_det[:, 1:, :]) # Skip CLS

        # 3. Final Norm & Pool
        if self.is_timm:
            x = self.visual.norm(x)
        else:
            x = self.visual.ln_post(x)
            
        pooled = x[:, 0]
        
        # 4. Projection to Embedding Space (768 -> 512)
        if self.proj is not None:
            if isinstance(self.proj, nn.Linear):
                pooled = self.proj(pooled)
            else:
                pooled = pooled @ self.proj

        # 5. Apply Tip-Adapter (Image Level)
        if self.use_tip_adapter and self.tip_adapter.cache_keys is not None:
             pooled = self.tip_adapter(pooled)

        return pooled, seg_patch_tokens, det_patch_tokens

    def project_tokens(self, tokens):
        """
        Helper function to project patch tokens (768) to embedding space (512).
        Crucial for calculating anomaly maps correctly.
        """
        if self.proj is None: 
            return tokens
            
        if isinstance(self.proj, nn.Linear):
            return self.proj(tokens)
        else:
            # self.proj is (768, 512) or (512, 768) depending on implementation
            # BioMedCLIP usually expects tokens @ proj
            return tokens @ self.proj