
import collections.abc
from itertools import repeat
from typing import List, Optional, Tuple, Union

import torch
from torch import nn as nn
from torch import _assert
from torchvision.ops.misc import FrozenBatchNorm2d



def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res






import numpy as np
import torch
import torch.nn.functional as F
import kornia as K
from PIL import Image
from biomedclip.tokenizer import tokenize



def encode_text_with_biomedclip_prompt_ensemble(model, obj, device):
    # obj will be like "Brain", "Liver,as", etc. → we map to proper medical terms
    medical_names = {
        'Brain': 'brain MRI',
        'Liver': 'liver CT',
        'Retina_RESC': 'retinal color fundus photograph',
        'Retina_OCT2017': 'retinal OCT image',
        'Chest': 'chest X-ray',
        'Histopathology': 'histopathology slide'
    }
    base_term = medical_names[obj]

    # === Normal / Healthy prompts ===
    prompt_normal = [
        f"normal {base_term}",
        f"healthy {base_term}",
        f"{base_term} without abnormality",
        f"{base_term} with no lesion",
        f"{base_term} showing normal anatomy",
        f"{base_term} of a healthy individual",
        f"unremarkable {base_term}",
        f"negative {base_term}",
        f"{base_term} without pathology",
        f"normal-appearing {base_term}",
        f"{base_term} in a patient with no disease",
        f"control {base_term}"
    ]

    # === Abnormal / Diseased prompts ===
    prompt_abnormal = [
        f"abnormal {base_term}",
        f"pathological {base_term}",
        f"{base_term} with lesion",
        f"{base_term} showing abnormality",
        f"{base_term} with disease",
        f"{base_term} demonstrating pathology",
        f"{base_term} with tumor",
        f"{base_term} with mass",
        f"{base_term} with hemorrhage",
        f"{base_term} with edema",
        f"{base_term} with infarction",
        f"{base_term} with pneumonia",
        f"{base_term} with metastasis",
        f"{base_term} with inflammatory changes",
        f"{base_term} in a patient with known disease",
        f"{base_term} positive for malignancy",
        f"affected {base_term}"
    ]

    # === Medical report–style templates ===
    templates = [
        "A medical image of {}.",
        "Imaging shows {}.",
        "The scan reveals {}.",
        "Radiology image: {}.",
        "This is {}.",
        "Medical scan depicting {}.",
        "Figure showing {}.",
        "Diagnostic image of {}.",
        "Pathology slide of {}.",
        "Clinical photograph of {}.",
        "{}.",
        "Caption: {}.",
        "Finding: {}.",
        "Observation: {}.",
        "The image demonstrates {}.",
        "Evidence of {} on imaging.",
        "There is {} present.",
        "Image findings consistent with {}.",
        "Representative image of {}.",
        "Example of {}.",
        "A case of {}.",
        "Patient with {}.",
        "Study demonstrating {}.",
    ]

    # Combine everything
    prompted_sentences = []
    text_features = []

    for state_list in [prompt_normal, prompt_abnormal]:
        for phrase in state_list:
            for template in templates:
                prompted_sentences.append(template.format(phrase))

        # FIXED INDENTATION — this part must be OUTSIDE inner loops
        prompted_sentence = tokenize(prompted_sentences).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        """print("class_embeddings (raw):", class_embeddings)
        print("type:", type(class_embeddings))

        if isinstance(class_embeddings, tuple):
            print("Tuple length:", len(class_embeddings))
            for i, item in enumerate(class_embeddings):
                try:
                    print(f"  item[{i}] type:", type(item))
                    print(f"  item[{i}] shape:", item.shape)
                except AttributeError:
                    print(f"  item[{i}] has no shape attribute:", item)"""
        if isinstance(class_embeddings, tuple):
            # usually index 0 is the CLS pooled embedding
            class_embeddings = class_embeddings[0]


        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)

    text_features = torch.stack(text_features, dim=1).to(device)
    return text_features


"""
Improved BioMedCLIP Prompt Ensemble for Medical Anomaly Detection
------------------------------------------------------------------
Based on PMC-15M training data characteristics:
- PubMed abstracts and figure captions
- Clinical terminology and diagnostic language
- Concise, factual descriptions
- Avoids overprompting and redundancy
"""

# ============================================
# PMC-ALIGNED MEDICAL TERMINOLOGY
# ============================================

# Use exact medical imaging terminology from PMC captions
MEDICAL_IMAGING_TERMS = {
    'Brain': 'brain MRI',
    'Liver': 'liver CT',
    'Retina_RESC': 'retinal fundus photograph',
    'Retina_OCT2017': 'retinal OCT',
    'Chest': 'chest radiograph',
    'Histopathology': 'histopathology'
}

# ============================================
# NORMAL STATE PROMPTS (PMC-Style)
# ============================================
# Based on how normal findings are described in PMC captions:
# - Short, factual statements
# - Clinical terminology
# - "Normal", "unremarkable", "no evidence"

NORMAL_PROMPTS = [
    # Core clinical terms (highest priority)
    "normal {}",
    "unremarkable {}",
    "no abnormality",

    # Negative findings (common in radiology reports)
    "no lesion",
    "no pathology",
    "within normal limits",

    # Healthy state descriptors
    "healthy {}",
    "normal appearance",

    # Clinical negatives
    "no findings",
    "negative study",
]

# ============================================
# ABNORMAL STATE PROMPTS (Organ-Specific, PMC-Style)
# ============================================
# Based on PMC figure captions for pathological findings
# Each organ gets 8-10 HIGH-QUALITY, SPECIFIC terms

ABNORMAL_PROMPTS = {
    'Brain': [
        # Neoplastic
        "brain tumor",
        "glioma",
        "brain mass",
        "intracranial lesion",

        # Vascular/Structural
        "stroke",
        "hemorrhage",
        "infarction",

        # General pathology
        "abnormal {}",
        "brain lesion",
        "pathological findings",
    ],

    'Liver': [
        # Neoplastic
        "liver tumor",
        "hepatocellular carcinoma",
        "liver mass",
        "hepatic lesion",

        # Diffuse disease
        "cirrhosis",
        "hepatic steatosis",
        "liver fibrosis",

        # General
        "abnormal {}",
        "liver pathology",
        "hepatic abnormality",
    ],

    'Chest': [
        # Infections
        "pneumonia",
        "pulmonary infiltrate",
        "consolidation",

        # Fluid/Structural
        "pleural effusion",
        "pulmonary edema",

        # Masses/Nodules
        "lung nodule",
        "pulmonary mass",

        # General
        "abnormal {}",
        "chest abnormality",
        "pulmonary lesion",
    ],

    'Retina_RESC': [
        # Diabetic retinopathy
        "diabetic retinopathy",
        "retinal hemorrhage",
        "microaneurysm",

        # Exudative findings
        "hard exudate",
        "cotton wool spot",

        # Macular disease
        "macular edema",
        "retinal lesion",

        # General
        "abnormal {}",
        "retinal pathology",
        "fundus abnormality",
    ],

    'Retina_OCT2017': [
        # AMD/CNV
        "choroidal neovascularization",
        "drusen",
        "age-related macular degeneration",

        # Macular disease
        "macular edema",
        "cystoid macular edema",
        "diabetic macular edema",

        # Structural
        "retinal fluid",
        "subretinal fluid",

        # General
        "abnormal {}",
        "retinal pathology",
    ],

    'Histopathology': [
        # Malignant
        "carcinoma",
        "malignancy",
        "cancer",
        "tumor",

        # Cellular changes
        "dysplasia",
        "neoplasia",

        # General
        "abnormal {}",
        "pathological tissue",
        "histopathologic abnormality",
    ],
}

# ============================================
# PMC-STYLE CAPTION TEMPLATES
# ============================================
# Reduced from 34 to 12 high-quality, diverse templates
# Based on actual PMC figure caption structures

CAPTION_TEMPLATES = [
    # === Direct statements (PMC most common) ===
    "{}",  # Most common: just the finding
    "{}.",  # With period

    # === Figure/Image labels (very common in PMC) ===
    "Image shows {}",
    "Figure showing {}",

    # === Clinical imaging reports ===
    "Imaging demonstrates {}",
    "{} on imaging",

    # === Diagnostic language ===
    "Findings consistent with {}",
    "Evidence of {}",

    # === Case presentation ===
    "Patient with {}",
    "Case of {}",

    # === Radiological descriptions ===
    "Radiograph reveals {}",
    "Scan shows {}",
]

# ============================================
# IMPROVED ENSEMBLE FUNCTION
# ============================================

def encode_text_with_biomedclip_prompt_ensemble1(model, obj, device):
    """
    Generate text embeddings using PMC-aligned prompts.

    Args:
        model: BioMedCLIP model
        obj: Organ name (e.g., 'Liver', 'Brain')
        device: torch device

    Returns:
        text_features: [embed_dim, 2] tensor with [normal, abnormal] embeddings
    """
    model.to(device)
    model.eval()

    # Get medical imaging term
    imaging_term = MEDICAL_IMAGING_TERMS[args.obj]

    # Build normal prompts
    normal_prompts = [p.format(imaging_term) if '{}' in p else p
                      for p in NORMAL_PROMPTS]

    # Build abnormal prompts (organ-specific)
    abnormal_prompts = [p.format(imaging_term) if '{}' in p else p
                        for p in ABNORMAL_PROMPTS[obj]]

    print(f"\n{'='*60}")
    print(f"Generating embeddings for: {obj} ({imaging_term})")
    print(f"{'='*60}")
    print(f"Normal prompts ({len(normal_prompts)}): {normal_prompts[:3]}...")
    print(f"Abnormal prompts ({len(abnormal_prompts)}): {abnormal_prompts[:3]}...")


    text_features = []

    for state_name, prompt_list in [('Normal', normal_prompts),
                                     ('Abnormal', abnormal_prompts)]:

        # Generate all combinations of prompts × templates
        all_sentences = []
        for prompt in prompt_list:
            for template in CAPTION_TEMPLATES:
                sentence = template.format(prompt)
                all_sentences.append(sentence)

        print(f"\n{state_name}: {len(all_sentences)} total sentences")
        print(f"  Examples: {all_sentences[:3]}")

        # Tokenize (handle batch size limits)
        max_batch = 256
        all_embeddings = []

        for i in range(0, len(all_sentences), max_batch):
            batch_sentences = all_sentences[i:i+max_batch]
            tokens = tokenizer(batch_sentences).to(device)

            with torch.no_grad():
                embeddings = model.encode_text(tokens)

                # Handle tuple output (some models return (embeddings, pooled))
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]

                # L2 normalize individual embeddings
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings)

        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Average ensemble across all prompt variations
        class_embedding = all_embeddings.mean(dim=0)

        # Final L2 normalization
        class_embedding = class_embedding / class_embedding.norm()

        text_features.append(class_embedding)
        print(f"  Final embedding norm: {class_embedding.norm().item():.4f}")

    # Stack [normal, abnormal] → shape [embed_dim, 2]
    text_features = torch.stack(text_features, dim=1).to(device)

    print(f"\n✅ Text features shape: {text_features.shape}")
    print(f"   Normal vs Abnormal similarity: {(text_features[:, 0] @ text_features[:, 1]).item():.4f}")
    print(f"{'='*60}\n")
    print(text_features.shape)

    return text_features







def cos_sim(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def get_translation_mat(a, b):
    return torch.tensor([[1, 0, a],
                         [0, 1, b]])

def rot_img(x, theta):
    dtype =  torch.FloatTensor
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def translation_img(x, a, b):
    dtype =  torch.FloatTensor
    rot_mat = get_translation_mat(a, b)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def hflip_img(x):
    x = K.geometry.transform.hflip(x)
    return x

def vflip_img(x):
    x = K.geometry.transform.vflip(x)
    return x

def rot90_img(x,k):
    # k is 0,1,2,3
    degreesarr = [0., 90., 180., 270., 360]
    degrees = torch.tensor(degreesarr[k])
    x = K.geometry.transform.rotate(x, angle = degrees, padding_mode='reflection')
    return x


def augment(fewshot_img, fewshot_mask=None):

    augment_fewshot_img = fewshot_img

    if fewshot_mask is not None:
        augment_fewshot_mask = fewshot_mask

        # rotate img
        for angle in [-np.pi/8, -np.pi/16, np.pi/16, np.pi/8]:
            rotate_img = rot_img(fewshot_img, angle)
            augment_fewshot_img = torch.cat([augment_fewshot_img, rotate_img], dim=0)

            rotate_mask = rot_img(fewshot_mask, angle)
            augment_fewshot_mask = torch.cat([augment_fewshot_mask, rotate_mask], dim=0)
        # translate img
        for a,b in [(0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
            trans_img = translation_img(fewshot_img, a, b)
            augment_fewshot_img = torch.cat([augment_fewshot_img, trans_img], dim=0)

            trans_mask = translation_img(fewshot_mask, a, b)
            augment_fewshot_mask = torch.cat([augment_fewshot_mask, trans_mask], dim=0)

        # hflip img
        flipped_img = hflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)
        flipped_mask = hflip_img(fewshot_mask)
        augment_fewshot_mask = torch.cat([augment_fewshot_mask, flipped_mask], dim=0)

        # vflip img
        flipped_img = vflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        flipped_mask = vflip_img(fewshot_mask)
        augment_fewshot_mask = torch.cat([augment_fewshot_mask, flipped_mask], dim=0)

    else:
        # rotate img
        for angle in [-np.pi/8, -np.pi/16, np.pi/16, np.pi/8]:
            rotate_img = rot_img(fewshot_img, angle)
            augment_fewshot_img = torch.cat([augment_fewshot_img, rotate_img], dim=0)

        # translate img
        for a,b in [(0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
            trans_img = translation_img(fewshot_img, a, b)
            augment_fewshot_img = torch.cat([augment_fewshot_img, trans_img], dim=0)

        # hflip img
        flipped_img = hflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        # vflip img
        flipped_img = vflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        B, _, H, W = augment_fewshot_img.shape
        augment_fewshot_mask = torch.zeros([B, 1, H, W])
    
    return augment_fewshot_img, augment_fewshot_mask
