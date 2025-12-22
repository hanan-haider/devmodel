import torch
from torch import nn
import torch.nn.functional as F


# ----------------------------
# Improved CLIP Adapter
# ----------------------------
class ClipAdapter(nn.Module):
    """
    Enhanced residual adapter with:
    - Better initialization for few-shot stability
    - Separate dropout for down/up projections
    - Learnable residual gating (starts near zero)
    """
    def __init__(
        self, 
        c_in: int, 
        bottleneck: int = 128,  # Reduced from 256 for less overfitting
        dropout: float = 0.2,    # Increased from 0.1
        init_scale: float = 1e-3,
    ):
        super().__init__()
        
        # Pre-normalization
        self.ln = nn.LayerNorm(c_in)
        
        # Down projection
        self.down_proj = nn.Linear(c_in, bottleneck, bias=False)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Up projection
        self.up_proj = nn.Linear(bottleneck, c_in, bias=False)
        self.dropout2 = nn.Dropout(dropout)
        
        # Learnable gating (small init prevents disrupting frozen backbone)
        self.gate = nn.Parameter(torch.ones(1) * init_scale)
        
        # ✅ Better initialization
        self._init_weights(init_scale)
    
    def _init_weights(self, init_scale):
        """Initialize weights for stable few-shot learning"""
        # Kaiming init for down projection (good for GELU)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=0, mode='fan_in', nonlinearity='linear')
        
        # Small random init for up projection (start near identity)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=init_scale)
    
    def forward(self, x):
        """
        Args:
            x: [B, N, C] input tokens
        Returns:
            med: [B, N, bottleneck] intermediate features for memory bank
            out: [B, N, C] residual output
        """
        residual = x
        
        # Pre-norm + down projection
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.act(x)
        x = self.dropout1(x)
        
        med = x  # Store bottleneck features for memory bank
        
        # Up projection
        x = self.up_proj(x)
        x = self.dropout2(x)
        
        # Gated residual (gate is learned)
        out = residual + self.gate * x
        return med, out


# ----------------------------
# Enhanced CLIP_Inplanted
# ----------------------------
class CLIP_Inplanted(nn.Module):
    """
    Improved BioMedCLIP adapter with:
    - Better initialization
    - Optional alpha normalization
    - Alpha regularization
    - Gradient clipping support
    - Task-specific layer selection
    """
    def __init__(
        self,
        clip_model,
        features,                # layer indices e.g. [3, 6, 9, 12]
        bottleneck=128,          # Reduced for less overfitting
        dropout=0.2,             # Increased regularization
        init_scale=1e-3,
        alpha_init=(0.8, 0.1, 0.1),  # (backbone, seg, det)
        normalize_alphas=False,       # Force alphas to sum to 1 via softmax
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.features = sorted(list(features))
        self.num_adapted_layers = len(self.features)
        self.normalize_alphas = normalize_alphas
        
        # timm ViT trunk used by BioMedCLIP
        self.image_encoder = clip_model.visual.trunk
        
        # Final projection: handles dimensionality alignment (768 -> 512)
        if hasattr(clip_model.visual, "head") and hasattr(clip_model.visual.head, "proj"):
            self.visual_proj = clip_model.visual.head.proj
        else:
            self.visual_proj = getattr(clip_model.visual, "proj", None)
        
        # Get hidden dimension
        hidden_dim = getattr(self.image_encoder, "num_features", None)
        if hidden_dim is None:
            hidden_dim = self.image_encoder.embed_dim
        
        # ✅ Build improved adapters
        self.seg_adapters = nn.ModuleList([
            ClipAdapter(
                hidden_dim, 
                bottleneck=bottleneck, 
                dropout=dropout,
                init_scale=init_scale,
            )
            for _ in self.features
        ])
        self.det_adapters = nn.ModuleList([
            ClipAdapter(
                hidden_dim, 
                bottleneck=bottleneck, 
                dropout=dropout,
                init_scale=init_scale,
            )
            for _ in self.features
        ])
        
        # ✅ Learnable blending weights
        alpha_bb, alpha_seg, alpha_det = alpha_init
        
        if normalize_alphas:
            # If using softmax, init in logit space for desired ratios
            logit_bb = torch.log(torch.tensor(alpha_bb) + 1e-8)
            logit_seg = torch.log(torch.tensor(alpha_seg) + 1e-8)
            logit_det = torch.log(torch.tensor(alpha_det) + 1e-8)
            
            self.alpha_backbone = nn.Parameter(logit_bb.repeat(self.num_adapted_layers))
            self.alpha_seg = nn.Parameter(logit_seg.repeat(self.num_adapted_layers))
            self.alpha_det = nn.Parameter(logit_det.repeat(self.num_adapted_layers))
        else:
            # Direct initialization
            self.alpha_backbone = nn.Parameter(torch.ones(self.num_adapted_layers) * alpha_bb)
            self.alpha_seg = nn.Parameter(torch.ones(self.num_adapted_layers) * alpha_seg)
            self.alpha_det = nn.Parameter(torch.ones(self.num_adapted_layers) * alpha_det)
        
        # Freeze backbone parameters
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        
        # ✅ Also freeze visual projection
        if self.visual_proj is not None:
            for p in self.visual_proj.parameters():
                p.requires_grad = False
    
    def get_blend_weights(self, layer_pos):
        """
        Get blending weights for a specific layer.
        Optionally applies softmax normalization.
        """
        alphas = torch.stack([
            self.alpha_backbone[layer_pos],
            self.alpha_seg[layer_pos],
            self.alpha_det[layer_pos]
        ])
        
        if self.normalize_alphas:
            # Softmax ensures weights sum to 1
            alphas = F.softmax(alphas, dim=0)
        
        return alphas[0], alphas[1], alphas[2]
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input images
        Returns:
            pooled: [B, 512] CLS token embedding
            seg_patch_tokens: List[Tensor] seg features per adapted layer
            det_patch_tokens: List[Tensor] det features per adapted layer
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
        
        # Transformer blocks loop with adapter injection
        for layer_idx, block in enumerate(self.image_encoder.blocks, start=1):
            x = block(x)
            
            if layer_idx in self.features:
                pos = self.features.index(layer_idx)
                
                # Get adapter features
                seg_med, seg_out = self.seg_adapters[pos](x)
                det_med, det_out = self.det_adapters[pos](x)
                
                # ✅ Learnable blending with optional normalization
                a_bb, a_seg, a_det = self.get_blend_weights(pos)
                x = a_bb * x + a_seg * seg_out + a_det * det_out
                
                # Store intermediate features (bottleneck features for memory bank)
                seg_patch_tokens.append(seg_med)
                det_patch_tokens.append(det_med)
        
        # Final normalization
        x = self.image_encoder.norm(x)
        pooled = x[:, 0]  # CLS token
        
        # Project to CLIP embedding space
        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)
        
        return pooled, seg_patch_tokens, det_patch_tokens
    
    def get_alpha_regularization(self, reg_type='l2', lambda_reg=1e-4):
        """
        ✅ Compute regularization loss on alpha parameters.
        Encourages alphas to stay close to initialization.
        """
        if reg_type == 'l2':
            reg_loss = (
                torch.sum((self.alpha_backbone - 0.8) ** 2) +
                torch.sum((self.alpha_seg - 0.1) ** 2) +
                torch.sum((self.alpha_det - 0.1) ** 2)
            )
        elif reg_type == 'l1':
            reg_loss = (
                torch.sum(torch.abs(self.alpha_backbone - 0.8)) +
                torch.sum(torch.abs(self.alpha_seg - 0.1)) +
                torch.sum(torch.abs(self.alpha_det - 0.1))
            )
        else:
            reg_loss = 0.0
        
        return lambda_reg * reg_loss
    
    def get_alpha_summary(self):
        """
        ✅ Print learned alpha values per layer (debugging utility)
        """
        print("\n" + "="*60)
        print("LEARNED ALPHA BLENDING WEIGHTS")
        print("="*60)
        for i in range(self.num_adapted_layers):
            a_bb, a_seg, a_det = self.get_blend_weights(i)
            total = a_bb + a_seg + a_det
            print(f"Layer {self.features[i]:2d}: "
                  f"backbone={a_bb.item():.4f} ({a_bb/total*100:.1f}%), "
                  f"seg={a_seg.item():.4f} ({a_seg/total*100:.1f}%), "
                  f"det={a_det.item():.4f} ({a_det/total*100:.1f}%)")
        print("="*60 + "\n")
