""" OpenAI pretrained model functions

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import os
import warnings
from typing import List, Optional, Union

import torch

from .model import get_cast_dtype,  build_model_from_biomedclip_state_dict
#convert_weights_to_lp,

__all__ = ["load_biomedclip_model"]


def load_biomedclip_model(
        name: str,
        precision: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        jit: bool = True,
        cache_dir: Optional[str] = None,
):
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

    Returns
    -------
    model : nn.Module
        Fully initialized BiomedCLIP model
    """
    print("name inside biomedclip model loader:", name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision is None:
        precision = 'fp32' if device == 'cpu' else 'fp16'

    if os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models")

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
        #print(f"Loaded state dict with keys: {list(state_dict.keys())}")

    if not jit:
        # Build model from state dict
        cast_dtype = get_cast_dtype(precision)
        
        # Use the standard OpenAI state dict builder for BioMedCLIP
        # BioMedCLIP uses the same architecture as CLIP but with different weights
        try:
            model = build_model_from_openai_state_dict(state_dict, cast_dtype=cast_dtype)
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
            convert_weights_to_lp(model, dtype=torch.bfloat16)
        elif precision == 'fp16':
            convert_weights_to_lp(model, dtype=torch.float16)

        return model