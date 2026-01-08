# adapter_improved.py
import torch
from torch import nn
from torch.nn import functional as F

class ClipAdapter(nn.Module):
    """
    Improved adapter with better regularization and initialization.
    """
    def __init__(self, c_in: int, bottleneck: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # Layer normalization before adaptation
        self.ln = nn.LayerNorm(c_in)
        
        # Bottleneck MLP with residual scaling
        self.down_proj = nn.Linear(c_in, bottleneck, bias=False)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck, c_in, bias=False)
        self.dropout2 = nn.Dropout(dropout)
        
        # Learnable gating parameter (starts at 0 for stability)
        self.gate = nn.Parameter(torch.zeros(1))
        
        # Initialize weights with smaller values for stability
        nn.init.kaiming_normal_(self.down_proj.weight, a=0, mode='fan_in')
        nn.init.zeros_(self.up_proj.weight)
    
    def forward(self, x):
        """
        Args:
            x: Input features [batch, seq_len, dim]
        
        Returns:
            med: Adapted features for memory bank
            out: Residual-connected output for transformer
        """
        residual = x
        
        # Normalize then adapt
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.up_proj(x)
        x = self.dropout2(x)
        
        # 'med' is the adapted feature for memory bank
        med = x
        
        # 'out' adds gated residual connection
        out = residual + self.gate.tanh() * x  # tanh bounds the gate
        
        return med, out

class CLIP_Inplanted(nn.Module):
    """
    Improved CLIP with adapters, learnable blending, and multi-scale aggregation.
    """
    def __init__(
        self,
        clip_model,
        features,           # layer indices e.g. [3, 6, 9, 12]
        bottleneck=256,
        dropout=0.1,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.features = sorted(list(features))
        num_adapted_layers = len(self.features)
        
        # Learnable temperature (initialized at CLIP's default)
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        # timm ViT trunk used by BioMedCLIP
        self.image_encoder = clip_model.visual.trunk
        
        # Final projection: handles dimensionality alignment (768 -> 512)
        if hasattr(clip_model.visual, "head") and hasattr(clip_model.visual.head, "proj"):
            self.visual_proj = clip_model.visual.head.proj
        else:
            self.visual_proj = getattr(clip_model.visual, "proj", None)
        
        hidden_dim = getattr(self.image_encoder, "num_features", None)
        if hidden_dim is None:
            hidden_dim = self.image_encoder.embed_dim
        
        # Build adapters
        self.seg_adapters = nn.ModuleList(
            [ClipAdapter(hidden_dim, bottleneck=bottleneck, dropout=dropout)
             for _ in self.features]
        )
        self.det_adapters = nn.ModuleList(
            [ClipAdapter(hidden_dim, bottleneck=bottleneck, dropout=dropout)
             for _ in self.features]
        )
        
        # Learnable blending parameters (layer-wise)
        # Use softmax-normalized weights for better stability
        self.alpha_backbone = nn.Parameter(torch.ones(num_adapted_layers) * 0.8)
        self.alpha_seg = nn.Parameter(torch.ones(num_adapted_layers) * 0.1)
        self.alpha_det = nn.Parameter(torch.ones(num_adapted_layers) * 0.1)
        
        # Learnable layer weights for multi-scale aggregation
        self.seg_layer_weights = nn.Parameter(torch.ones(num_adapted_layers) / num_adapted_layers)
        self.det_layer_weights = nn.Parameter(torch.ones(num_adapted_layers) / num_adapted_layers)
        
        # Freeze backbone parameters
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        if self.visual_proj is not None:
            for p in self.visual_proj.parameters():
                p.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass with adapter injection.
        
        Args:
            x: Input images [batch, 3, 224, 224]
        
        Returns:
            pooled: Global image features [batch, embed_dim]
            seg_patch_tokens: List of segmentation adapter features
            det_patch_tokens: List of detection adapter features
        """
        B = x.shape[0]
        
        # ViT input embedding
        x = self.image_encoder.patch_embed(x)
        cls_token = self.image_encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.image_encoder.pos_embed
        x = self.image_encoder.pos_drop(x)
        
        seg_patch_tokens = []
        det_patch_tokens = []
        
        # Transformer blocks loop
        for layer_idx, block in enumerate(self.image_encoder.blocks, start=1):
            x = block(x)
            
            if layer_idx in self.features:
                pos = self.features.index(layer_idx)
                
                # Get adapter features
                seg_med, seg_out = self.seg_adapters[pos](x)
                det_med, det_out = self.det_adapters[pos](x)
                
                # Normalized blending (ensures weights sum to ~1)
                alpha_sum = (
                    self.alpha_backbone[pos].abs() +
                    self.alpha_seg[pos].abs() +
                    self.alpha_det[pos].abs() + 1e-6
                )
                
                x = (
                    (self.alpha_backbone[pos].abs() / alpha_sum) * x +
                    (self.alpha_seg[pos].abs() / alpha_sum) * seg_out +
                    (self.alpha_det[pos].abs() / alpha_sum) * det_out
                )
                
                # Store adapter outputs (without CLS token for local features)
                seg_patch_tokens.append(seg_med)
                det_patch_tokens.append(det_med)
        
        # Final layer norm and pooling
        x = self.image_encoder.norm(x)
        pooled = x[:, 0]  # CLS token
        
        # Project to embedding space
        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)
        
        return pooled, seg_patch_tokens, det_patch_tokens
