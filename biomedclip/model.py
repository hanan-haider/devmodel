
from dataclasses import dataclass
import logging
import math
from typing import Optional, Tuple, Union
from itertools import repeat
import collections.abc
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing import Final





@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.2  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256 # n_queries for attentional pooler
    attn_pooler_heads: int = 8 # n heads for attentional_pooling

    # hugging face timm model integration
    
    timm_model_name: Optional[str] = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models




def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype

    


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            block_type=vision_cfg.block_type,
            qk_norm=vision_cfg.qk_norm,
            scaled_cosine_attn=vision_cfg.scaled_cosine_attn,
            scale_heads=vision_cfg.scale_heads,
            scale_attn_inner=vision_cfg.scale_attn_inner,
            scale_attn=vision_cfg.scale_attn,
            scale_fc=vision_cfg.scale_fc,
        )

    return visual




    
def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            eos_id=text_cfg.eos_id,
            pool_type=text_cfg.pool_type,
            proj_type=text_cfg.proj_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
            block_type=text_cfg.block_type,
            qk_norm=text_cfg.qk_norm,
            scaled_cosine_attn=text_cfg.scaled_cosine_attn,
            scale_heads=text_cfg.scale_heads,
            scale_attn_inner=text_cfg.scale_attn_inner,
            scale_attn=text_cfg.scale_attn,
            scale_fc=text_cfg.scale_fc,
        )
    return text


class CustomTextCLIP(nn.Module):
    output_dict: Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()










def build_model_from_biomedclip_state_dict(
    state_dict: dict,
    quick_gelu=True,
    cast_dtype=torch.float16,
):
    """
    Build BiomedCLIP model from state dict.
    BiomedCLIP uses:
    - Vision: timm ViT-B/16 (visual.trunk.*)
    - Text: HuggingFace BERT (text.transformer.*)
    """
    
    print("Building BiomedCLIP model from state dict...")
    
    # ========== VISION ENCODER DETECTION ==========
    # BiomedCLIP uses timm's ViT structure (visual.trunk.*)
    vit = any(k.startswith("visual.trunk") for k in state_dict.keys())
    
    if not vit:
        raise ValueError("BiomedCLIP requires ViT backbone (visual.trunk.*)")
    
    print("✓ Detected ViT backbone (timm structure)")
    
    # Vision parameters from BiomedCLIP state dict
    vision_width = state_dict["visual.trunk.patch_embed.proj.weight"].shape[0]  # 768
    vision_patch_size = state_dict["visual.trunk.patch_embed.proj.weight"].shape[-1]  # 16
    
    # Count transformer blocks
    vision_layers = len([k for k in state_dict.keys() 
                        if k.startswith("visual.trunk.blocks") and k.endswith(".attn.qkv.weight")])  # 12
    
    # Calculate image size from positional embedding
    # pos_embed shape: (1, num_patches + 1, dim) = (1, 197, 768)
    # num_patches = 196 = 14x14, so image_size = 14 * 16 = 224
    grid_size = int(np.sqrt(state_dict["visual.trunk.pos_embed"].shape[1] - 1))
    image_size = vision_patch_size * grid_size
    
    print(f"Vision config -> width: {vision_width}, layers: {vision_layers}, "
          f"patch_size: {vision_patch_size}, image_size: {image_size}")
    
    # ========== TEXT ENCODER DETECTION ==========
    # BiomedCLIP uses HuggingFace BERT structure (text.transformer.*)
    bert = any(k.startswith("text.transformer") for k in state_dict.keys())
    
    if not bert:
        raise ValueError("BiomedCLIP requires BERT text encoder (text.transformer.*)")
    
    print("✓ Detected BERT text encoder (HuggingFace structure)")
    
    # Text parameters from BiomedCLIP state dict
    text_width = state_dict["text.transformer.embeddings.word_embeddings.weight"].shape[1]  # 768
    vocab_size = state_dict["text.transformer.embeddings.word_embeddings.weight"].shape[0]  # 30522
    context_length = state_dict["text.transformer.embeddings.position_embeddings.weight"].shape[0]  # 512
    
    # Count BERT layers
    text_layers = len([k for k in state_dict.keys() 
                      if k.startswith("text.transformer.encoder.layer") 
                      and k.endswith(".attention.self.query.weight")])  # 12
    
    # BERT uses 64-dimensional heads
    text_heads = text_width // 64  # 768 // 64 = 12
    
    print(f"Text config -> width: {text_width}, layers: {text_layers}, "
          f"heads: {text_heads}, vocab_size: {vocab_size}, context_length: {context_length}")
    
    # ========== EMBEDDING DIMENSION ==========
    # BiomedCLIP projects both vision and text to 512 dimensions
    # Visual: 768 -> 512 (visual.head.proj.weight)
    # Text: 768 -> 640 -> 512 (text.proj.0.weight, text.proj.2.weight)
    embed_dim = state_dict["visual.head.proj.weight"].shape[0]  # 512
    
    print(f"Embed dim (joint space): {embed_dim}")
    
    # ========== CREATE MODEL CONFIGS ==========
    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        head_width=64,  # Standard for ViT
        patch_size=vision_patch_size,
        image_size=image_size,
        mlp_ratio=4.0,  # 3072 / 768 = 4
        output_tokens=True,  # BiomedCLIP outputs all tokens
    )
    
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=text_width,
        heads=text_heads,
        layers=text_layers,
        mlp_ratio=4.0,  # 3072 / 768 = 4
        proj='mlp',  # BiomedCLIP uses 2-layer MLP projection
        pooler_type='cls_last_hidden_state_pooler',  # Uses [CLS] token
        hf_model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
        output_tokens=False,
    )
    
    print("\nConfigs created:")
    print(f"  Vision: {vision_cfg}")
    print(f"  Text: {text_cfg}")
    
    # ========== BUILD MODEL ==========
    model = CustomTextCLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,
        cast_dtype=cast_dtype,
    )
    
    print("\n✓ Model instance created")
    
    # ========== ADAPT STATE DICT KEYS ==========
    # BiomedCLIP has different key names than standard CLIP
    new_state_dict = {}
    
    print("\nAdapting state dict keys...")
    for key, value in state_dict.items():
        new_key = key
        
        # Map visual.head.proj -> visual.proj (if needed by your model)
        if key == "visual.head.proj.weight":
            new_key = "visual.proj.weight"
            print(f"  Mapped: {key} -> {new_key}")
        
        # Skip position_ids (it's a buffer, not a parameter)
        elif key == "text.transformer.embeddings.position_ids":
            print(f"  Skipping: {key} (buffer, not parameter)")
            continue
        
        # Keep BiomedCLIP text projection as-is (2-layer MLP)
        elif key.startswith("text.proj"):
            print(f"  Keeping: {key} (BiomedCLIP MLP projection)")
        
        new_state_dict[new_key] = value
    
    # Remove unused keys if present
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in new_state_dict:
            new_state_dict.pop(key)
            print(f"  Removed: {key}")
    
    # ========== CONVERT WEIGHTS ==========
    # Note: BiomedCLIP might already be in fp32, so check before converting
    if cast_dtype == torch.float16:
        print("\nConverting weights to fp16...")
        convert_weights_to_fp16(model)
    
    # ========== LOAD STATE DICT ==========
    print("\nLoading state dict...")
    
    # Use strict=False to handle minor mismatches
    incompatible_keys = model.load_state_dict(new_state_dict, strict=False)
    
    if incompatible_keys.missing_keys:
        print(f"\n⚠️ Missing keys: {incompatible_keys.missing_keys}")
    
    if incompatible_keys.unexpected_keys:
        print(f"\n⚠️ Unexpected keys: {incompatible_keys.unexpected_keys}")
    
    print("\n✓ BiomedCLIP model loaded successfully!")
    
    return model.eval()