import torch
from torch import nn
import torch.nn.functional as F


class ImprovedClipAdapter(nn.Module):
    """Enhanced adapter with multi-scale feature fusion and attention"""
    def __init__(self, c_in: int, bottleneck: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # Dual normalization for stability
        self.ln1 = nn.LayerNorm(c_in)
        self.ln2 = nn.LayerNorm(c_in)
        
        # Multi-scale bottleneck with residual connections
        self.down_proj = nn.Linear(c_in, bottleneck, bias=False)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Enhanced middle layer with grouped convolution concept
        self.mid_proj = nn.Sequential(
            nn.Linear(bottleneck, bottleneck, bias=False),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.up_proj = nn.Linear(bottleneck, c_in, bias=False)
        self.dropout2 = nn.Dropout(dropout)
        
        # Squeeze-and-Excitation style channel attention
        self.se = nn.Sequential(
            nn.Linear(c_in, c_in // 4),
            nn.GELU(),
            nn.Linear(c_in // 4, c_in),
            nn.Sigmoid()
        )
        
        # Learnable gating with better initialization
        self.gate = nn.Parameter(torch.ones(1) * 0.1)  # Start with small adaptation
        
        # Feature scaling for better gradient flow
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        residual = x
        
        # Main adaptation path
        x_norm = self.ln1(x)
        adapted = self.down_proj(x_norm)
        adapted = self.activation(adapted)
        adapted = self.dropout1(adapted)
        
        # Middle enhancement
        adapted = adapted + self.mid_proj(adapted)  # Residual in bottleneck
        
        adapted = self.up_proj(adapted)
        adapted = self.dropout2(adapted)
        
        # Apply channel attention
        x_pooled = x.mean(dim=1, keepdim=True)  # Global context
        attention = self.se(x_pooled)
        adapted = adapted * attention
        
        # High-resolution feature for memory bank (normalized separately)
        med = self.ln2(adapted) * self.scale
        
        # Gated residual connection for transformer feature
        out = residual + self.gate * adapted
        
        return med, out


class CLIP_Inplanted_V2(nn.Module):
    """Enhanced BioMedCLIP with improved adaptation strategy"""
    def __init__(
        self,
        clip_model,
        features,           # layer indices e.g. [3, 6, 9, 12]
        bottleneck=384,     # Increased from 256
        dropout=0.15,       # Slightly increased for regularization
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.features = sorted(list(features))
        num_adapted_layers = len(self.features)

        # Image encoder from BioMedCLIP
        self.image_encoder = clip_model.visual.trunk

        # Visual projection layer
        if hasattr(clip_model.visual, "head") and hasattr(clip_model.visual.head, "proj"):
            self.visual_proj = clip_model.visual.head.proj
        else:
            self.visual_proj = getattr(clip_model.visual, "proj", None)

        hidden_dim = getattr(self.image_encoder, "num_features", None)
        if hidden_dim is None:
            hidden_dim = self.image_encoder.embed_dim

        # Enhanced adapters
        self.seg_adapters = nn.ModuleList(
            [ImprovedClipAdapter(hidden_dim, bottleneck=bottleneck, dropout=dropout)
             for _ in self.features]
        )
        self.det_adapters = nn.ModuleList(
            [ImprovedClipAdapter(hidden_dim, bottleneck=bottleneck, dropout=dropout)
             for _ in self.features]
        )

        # Layer-wise learnable blending with better initialization
        # Use softmax-based blending for guaranteed sum-to-1
        self.blend_logits = nn.ParameterList([
            nn.Parameter(torch.tensor([2.0, 0.0, 0.0]))  # Bias toward backbone initially
            for _ in range(num_adapted_layers)
        ])
        
        # Cross-layer feature fusion
        self.layer_weights = nn.Parameter(torch.ones(num_adapted_layers) / num_adapted_layers)
        
        # Learnable temperature for better calibration
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        # Feature aggregation module
        self.feature_fusion = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU()
            ) for _ in range(num_adapted_layers)
        ])

        # Freeze backbone
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        
        # Freeze visual projection initially (can be unfrozen later if needed)
        if self.visual_proj is not None:
            for p in self.visual_proj.parameters():
                p.requires_grad = False

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

        # Transformer blocks with adaptive blending
        for layer_idx, block in enumerate(self.image_encoder.blocks, start=1):
            x = block(x)

            if layer_idx in self.features:
                pos = self.features.index(layer_idx)

                # Get adapter features
                seg_med, seg_out = self.seg_adapters[pos](x)
                det_med, det_out = self.det_adapters[pos](x)
                
                # Softmax-based adaptive blending (guarantees sum to 1)
                blend_weights = F.softmax(self.blend_logits[pos], dim=0)
                
                # Enhanced feature fusion with learned aggregation
                x_fused = self.feature_fusion[pos](x)
                
                # Blend: backbone + seg + det
                x = (
                    blend_weights[0] * x_fused
                    + blend_weights[1] * seg_out
                    + blend_weights[2] * det_out
                )

                seg_patch_tokens.append(seg_med)
                det_patch_tokens.append(det_med)

        x = self.image_encoder.norm(x)
        pooled = x[:, 0]  # CLS token

        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)

        return pooled, seg_patch_tokens, det_patch_tokens
    
    def get_adaptation_weights(self):
        """Helper to visualize learned blending weights"""
        weights = []
        for logits in self.blend_logits:
            weights.append(F.softmax(logits, dim=0).detach().cpu().numpy())
        return weights