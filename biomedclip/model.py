
from dataclasses import dataclass
import logging
import math
from typing import Optional, Tuple, Union, List, Dict
from functools import partial

from itertools import repeat
import collections.abc
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing import Final
from .timm_model import TimmModel
from .hf_model import HFTextEncoder




@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value

    patch_dropout: float = 0.05  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results

    input_patchnorm: bool = True # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256 # n_queries for attentional pooler
    attn_pooler_heads: int = 8 # n heads for attentional_pooling

    # hugging face timm model integration
    
    timm_model_name: Optional[str] = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = ''  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.1  # head dropout
    timm_drop_path: Optional[float] = 0.1 # backbone stochastic depth


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
    print("\n Building vision tower...")
    print("Vision config:", vision_cfg,"\n")

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
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual




    
def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,

):
    print("\n Building text tower...")
    print("Here is the text config:", text_cfg,"\n")

    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)
    
    print("\nText config:", text_cfg)

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
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            output_tokens=text_cfg.output_tokens,
            pad_id=text_cfg.pad_id,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = set()
        if hasattr(self.visual, 'no_weight_decay'):
            for n in self.visual.no_weight_decay():
                no_wd.add('visual.' + n)
        if hasattr(self.text, 'no_weight_decay'):
            for n in self.text.no_weight_decay():
                no_wd.add('text.' + n)
        return no_wd

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward_intermediates(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            image_indices: Optional[Union[int, List[int]]] = None,
            text_indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize: bool = True,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            image_output_fmt: str = 'NCHW',
            image_output_extra_tokens: bool = False,
            text_output_fmt: str = 'NLC',
            text_output_extra_tokens: bool = False,
            output_logits: bool = False,
            output_logit_scale_bias: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            image: Input image tensor
            text: Input text tensor
            image_indices: For image tower, Take last n blocks if int, all if None, select matching indices if sequence
            text_indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            normalize: L2 Normalize final image and text features (if present)
            normalize_intermediates: Apply final encoder norm layer to all intermediates (if possible)
            intermediates_only: Only return intermediate features, do not return final features
            image_output_fmt: Shape of intermediate image feature outputs
            image_output_extra_tokens: Return both prefix and spatial intermediate tokens
            text_output_fmt: Shape of intermediate text feature outputs
            text_output_extra_tokens: Return both prefix and spatial intermediate tokens
            output_logits: Include logits in output
            output_logit_scale_bias: Include the logit scale bias in the output
        Returns:

        """
        output = {}
        if intermediates_only:
            # intermediates only disables final feature normalization, and include logits
            normalize = False
            output_logits = False
        if output_logits:
            assert image is not None and text is not None, 'Both image and text inputs are required to compute logits'

        if image is not None:
            image_output = self.visual.forward_intermediates(
                image,
                indices=image_indices,
                stop_early=stop_early,
                normalize_intermediates=normalize_intermediates,
                intermediates_only=intermediates_only,
                output_fmt=image_output_fmt,
                output_extra_tokens=image_output_extra_tokens,
            )
            if normalize and "image_features" in image_output:
                image_output["image_features"] = F.normalize(image_output["image_features"], dim=-1)
            output.update(image_output)

        if text is not None:
            text_output = self.text.forward_intermediates(
                text,
                indices=text_indices,
                stop_early=stop_early,
                normalize_intermediates=normalize_intermediates,
                intermediates_only=intermediates_only,
                output_fmt=text_output_fmt,
                output_extra_tokens=text_output_extra_tokens,
            )
            if normalize and "text_features" in text_output:
                text_output["text_features"] = F.normalize(text_output["text_features"], dim=-1)
            output.update(text_output)

        logit_scale_exp = self.logit_scale.exp() if output_logits or output_logit_scale_bias else None

        if output_logits:
            image_logits = logit_scale_exp * output["image_features"] @ output["text_features"].T
            if self.logit_bias is not None:
                image_logits += self.logit_bias
            text_logits = image_logits.T
            output["image_logits"] = image_logits
            output["text_logits"] = text_logits

        if output_logit_scale_bias:
            output["logit_scale"] = logit_scale_exp
            if self.logit_bias is not None:
                output['logit_bias'] = self.logit_bias

        return output

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()





def build_model_from_biomedclip_state_dict(
    state_dict: dict,
    cast_dtype=torch.float16,
):
    """
    Build BiomedCLIP model from official Microsoft checkpoint state_dict
    (e.g. microsoft/BiomedCLIP-PubMedBERT-ViT-B-16)
    """

    print("Building BiomedCLIP model from state_dict...")
    print(f"   Total parameters in state_dict: {len(state_dict)}")

    # === Vision Tower (ViT-B/16) ===
    print("Parsing vision tower (ViT-B/16)...")
    vision_width = state_dict["visual.trunk.patch_embed.proj.weight"].shape[0]  # 768
    vision_layers = len([
        k for k in state_dict.keys()
        if k.startswith("visual.trunk.blocks.") and k.endswith(".attn.qkv.weight")
    ])
    vision_patch_size = state_dict["visual.trunk.patch_embed.proj.weight"].shape[-1]
    pos_embed_shape = state_dict["visual.trunk.pos_embed"].shape
    grid_size = int((pos_embed_shape[1] - 1) ** 0.5)
    image_size = vision_patch_size * grid_size

    print(f"   Vision: width={vision_width}, layers={vision_layers}, patch_size={vision_patch_size}, image_size={image_size}")

    # === Text Tower (PubMedBERT) ===
    print("Parsing text tower (PubMedBERT)...")
    transformer_width = state_dict["text.transformer.embeddings.word_embeddings.weight"].shape[1]
    vocab_size = state_dict["text.transformer.embeddings.word_embeddings.weight"].shape[0]
    context_length = state_dict["text.transformer.embeddings.position_embeddings.weight"].shape[0]

    transformer_layers = len([
        k for k in state_dict.keys()
        if k.startswith("text.transformer.encoder.layer.") and k.endswith(".attention.self.query.weight")
    ])
    transformer_heads = transformer_width // 64

    embed_dim = state_dict["visual.head.proj.weight"].shape[0]  # 512

    print(f"   Text: width={transformer_width}, heads={transformer_heads}, layers={transformer_layers}, "
          f"vocab={vocab_size}, ctx_len={context_length}, embed_dim={embed_dim}")

    # === Configs ===
    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
        timm_model_name="vit_base_patch16_224",
        timm_model_pretrained=False,
        timm_pool='',
        timm_proj='linear',
    )

    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
        hf_model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
        hf_tokenizer_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
        hf_proj_type='mlp',
        hf_pooler_type='cls_last_hidden_state_pooler',
    )

    print("Creating CustomTextCLIP model instance...")
    model = CustomTextCLIP(
        embed_dim=embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=False,
        cast_dtype=cast_dtype,
    )



    print("   Loading state_dict into model (strict=False)...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"   Missing keys ({len(missing_keys)}): {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
    if unexpected_keys:
        print(f"   Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")

    critical_missing = [k for k in missing_keys if not k.startswith("text.proj.")]
    if critical_missing:
        print(f"   WARNING: {len(critical_missing)} critical keys missing (not just text.proj)!")
    else:
        print("   All critical weights loaded successfully!")

    # === Finalize ===
    if cast_dtype == torch.float16:
        print("   Converting model to float16 (half precision)")
        model = model.half()
    else:
        print(f"   Keeping model in {cast_dtype}")

    print("BiomedCLIP model successfully built and loaded!")
    return model.eval()
    


import math
import torch
import torch.nn.functional as F
from timm.layers import to_2tuple
import logging

def resize_pos_embed_biomedclip(state_dict, model,
                                 interpolation: str = 'bicubic',
                                 antialias: bool = True):
    """
    Resize BiomedCLIP visual positional embeddings (visual.trunk.pos_embed)
    to match model.visual.trunk.grid_size.
    """

    # ---- 1. GET ORIGINAL POS EMB ----
    old_pos_embed = state_dict.get('visual.trunk.pos_embed', None)
    if old_pos_embed is None:
        return

    # Expected shape: (1, 197, 768)
    B, old_len, C = old_pos_embed.shape

    if not hasattr(model.visual.trunk, 'grid_size'):
        return 

    grid_size = to_2tuple(model.visual.trunk.grid_size)   # e.g. (14, 14)

    # ---- 2. DEFINE NEW SEQ LEN ----
    num_patches = grid_size[0] * grid_size[1]
    extra_tokens = 1     # class token
    new_len = num_patches + extra_tokens

    if new_len == old_len:
        # Already correct size
        return

    # ---- 3. SPLIT CLASS TOKEN + PATCH TOKENS ----
    cls_token = old_pos_embed[:, :1, :]         # (1,1,768)
    patch_tokens = old_pos_embed[:, 1:, :]      # (1,196,768)

    # ---- 4. COMPUTE OLD GRID SIZE ----
    old_size = int(math.sqrt(patch_tokens.shape[1]))  # 14 for ViT-B/16
    old_grid = (old_size, old_size)

    logging.info(f"Resizing BiomedCLIP pos_embed from {old_grid} → {grid_size}")

    # ---- 5. RESHAPE TO IMAGE GRID ----
    patch_tokens = patch_tokens.reshape(
        1, old_grid[0], old_grid[1], C
    ).permute(0, 3, 1, 2)  # (1, 768, 14, 14)

    # ---- 6. INTERPOLATE ----
    patch_tokens = F.interpolate(
        patch_tokens,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )

    # ---- 7. FLATTEN BACK ----
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(
        1, num_patches, C
    )  # (1, 196, 768)

    # ---- 8. CONCAT CLASS TOKEN + NEW PATCH TOKENS ----
    new_pos_embed = torch.cat([cls_token, patch_tokens], dim=1)

    # ---- 9. SAVE BACK TO STATE DICT ----
    state_dict['visual.trunk.pos_embed'] = new_pos_embed



