""" OpenAI pretrained model functions

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import os
import warnings
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn

from .model import get_cast_dtype, build_model_from_biomedclip_state_dict

__all__ = ["load_biomedclip_model"]


def load_biomedclip_model(
        name: str,
        precision: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        jit: bool = True,
        cache_dir: Optional[str] = None,
        return_vision_proj: bool = True,  # ✅ Default True to always return projection
) -> Union[nn.Module, Tuple[nn.Module, nn.Linear]]:
    """Model loader for BiomedCLIP (.bin checkpoints).
    Supports ONLY non-JIT PyTorch state_dict files.

    Parameters
    ----------
    name : str
        Path to .bin checkpoint
    precision : str
        'fp16', 'fp32', or 'bf16'
    device : device string or torch.device
    jit : bool
        Whether to load as JIT model (not supported for BioMedCLIP .bin files)
    cache_dir : Optional[str]
        Cache directory (not used)
    return_vision_proj : bool
        If True, returns (model, vision_proj) tuple. If False, returns only model.
        Default: True

    Returns
    -------
    model : nn.Module
        Fully initialized BiomedCLIP model
    vision_proj : nn.Linear (if return_vision_proj=True)
        Vision projection layer (768 -> 512), frozen
    """
    print("Loading BiomedCLIP model from:", name)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision is None:
        precision = 'fp32' if device == 'cpu' else 'fp16'

    if os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found")

    # BioMedCLIP .bin files are state_dicts, not JIT archives
    # Force jit=False for .bin files
    jit = False
    
    try:
        # loading JIT archive (unlikely to work with .bin files)
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
        print("Loaded as JIT model")
    except RuntimeError:
        # loading saved state dict (expected path for .bin files)
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        # Build model from state dict
        cast_dtype = get_cast_dtype(precision)
        
        # Use the standard OpenAI state dict builder for BioMedCLIP
        # BioMedCLIP uses the same architecture as CLIP but with different weights
        try:
            model = build_model_from_biomedclip_state_dict(state_dict, cast_dtype=cast_dtype)
        except Exception as e:
            print(f"Error building with standard state dict: {e}")
            # Try alternative state dict format (common in trained models)
            if "state_dict" in state_dict:
                sd = state_dict["state_dict"]
                # Remove "module." prefix if present (from DataParallel/DistributedDataParallel)
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
                model = build_model_from_biomedclip_state_dict(sd, cast_dtype=cast_dtype)
            else:
                raise e

        # Handle precision conversion
        model = model.to(device)
        if precision.startswith('amp') or precision == 'fp32':
            model.float()
        elif precision == 'bf16':
            from .model import convert_weights_to_lp
            convert_weights_to_lp(model, dtype=torch.bfloat16)
        elif precision == 'fp16':
            from .model import convert_weights_to_lp
            convert_weights_to_lp(model, dtype=torch.float16)

        # ✅ Extract vision projection layer if requested
        if return_vision_proj:
            print("\n" + "="*60)
            print("Extracting Vision Projection Layer from checkpoint")
            print("="*60)
            
            vision_proj = _extract_vision_projection_from_state_dict(
                state_dict, 
                device=device, 
                precision=precision
            )
            
            print(f"✅ Vision projection ready: {vision_proj.weight.shape}")
            print(f"   Device: {vision_proj.weight.device}")
            print(f"   Dtype: {vision_proj.weight.dtype}")
            print(f"   Frozen: {not vision_proj.weight.requires_grad}")
            print("="*60 + "\n")
            
            return model, vision_proj
        else:
            return model


def _extract_vision_projection_from_state_dict(
    state_dict: dict,
    device: Union[str, torch.device] = 'cuda',
    precision: str = 'fp16',
) -> nn.Linear:
    """
    Internal function to extract vision projection layer from state dict.
    
    Args:
        state_dict: Loaded checkpoint state dict
        device: Device to load onto ('cuda' or 'cpu')
        precision: 'fp16', 'fp32', or 'bf16'
    
    Returns:
        nn.Linear: Vision projection layer (768 -> 512), frozen
    """
    # Extract vision projection weight
    proj_key = 'visual.head.proj.weight'
    
    if proj_key not in state_dict:
        raise KeyError(
            f"'{proj_key}' not found in checkpoint! "
            f"Available keys (first 10): {list(state_dict.keys())[:10]}"
        )
    
    vision_proj_weight = state_dict[proj_key]
    print(f"   Found vision projection weight: {vision_proj_weight.shape}")  # [512, 768]
    
    # Create projection layer (BiomedCLIP has no bias in projection)
    out_dim, in_dim = vision_proj_weight.shape
    vision_proj = nn.Linear(in_dim, out_dim, bias=False)
    
    # Load weights
    vision_proj.weight.data = vision_proj_weight.clone()
    
    # Move to device
    vision_proj = vision_proj.to(device)
    
    # Handle precision conversion
    if precision == 'fp16':
        vision_proj = vision_proj.half()
    elif precision == 'bf16':
        vision_proj = vision_proj.bfloat16()
    # fp32 is default, no conversion needed
    
    # Set to eval mode and freeze
    vision_proj.eval()
    for param in vision_proj.parameters():
        param.requires_grad = False
    
    return vision_proj
