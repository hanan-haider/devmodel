""" OpenAI pretrained model functions — Adapted for BiomedCLIP """

import os
import warnings
from typing import Optional, Union

import torch

from .model import get_cast_dtype , build_model_from_biomedclip_state_dict
#, build_model_from_biomedclip_state_dict, convert_weights_to_lp


__all__ = ["load_biomedclip_model"]


def load_biomedclip_model(
    name: str,
    precision: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    jit: bool = False,  # BiomedCLIP is NOT jitted → default False
    cache_dir: Optional[str] = None,
):
    """
    Load BiomedCLIP model from local .pth/.bin or HuggingFace-style state_dict.
    Fully compatible with: microsoft/BiomedCLIP-PubMedBERT-ViT-B-16

    Parameters
    ----------
    name : str
        Local path to .pth/.bin file OR pretrained identifier (e.g. 'microsoft')
    precision : str, optional
        'fp16', 'fp32', 'bf16', or 'amp'
    device : str or torch.device, optional
        Target device
    jit : bool
        Set False — BiomedCLIP checkpoints are regular state_dicts, not JIT

    Returns
    -------
    model : torch.nn.Module
        Loaded and ready-to-use BiomedCLIP model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision is None:
        precision = 'fp16' if device == 'cuda' else 'fp32'

    device = torch.device(device)

    # Resolve path if it's a local file
    if os.path.isfile(name):
        model_path = name
    else:
        # For HF-style loading (e.g., 'microsoft'), assume it's handled upstream
        # Or fallback: assume name is path
        model_path = name

    print(f"Loading BiomedCLIP model from: {model_path}")

    # Load state dict (BiomedCLIP is never JIT)
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        #print("Loaded state dict keys", list(state_dict.keys())[:5], "...")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    # Handle nested state_dict (common in HF caches)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]



    # Determine cast dtype
    cast_dtype = get_cast_dtype(precision)
    #print("building BiomedCLIP model...")

    # Build model from cleaned state dict
    try:
        model = build_model_from_biomedclip_state_dict(state_dict, cast_dtype=cast_dtype)
    except Exception as e:
        raise RuntimeError(f"Failed to build BiomedCLIP model from state dict. Key mismatch? Error: {e}")

    # Move to device and cast
    model = model.to(device)

    if precision == "fp32":
        model.float()
    elif precision == "bf16":
        convert_weights_to_lp(model, dtype=torch.bfloat16)
    # fp16 / amp: keep as-is (already in half if loaded that way)

    model.eval()
    return model