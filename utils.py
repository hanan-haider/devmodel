
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
        print("Here is the class embedding",class_embeddings, type(class_embeddings))

        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)

    text_features = torch.stack(text_features, dim=1).to(device)
    return text_features

        
"""
    # Optional: add some very direct class names (helps in few-shot/zero-shot)
    direct_terms = [
        f"{base_term} normal finding",
        f"{base_term} abnormal finding",
        f"{base_term} pathology negative",
        f"{base_term} pathology positive",
    ]
    prompted_sentences.extend(direct_terms)

    # Tokenize all at once (much faster)
   texts = tokenize(prompted_sentences).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Average all features for this class (or you can max-pool, etc.)
    text_features = text_features.mean(dim=0, keepdim=True)
    return text_features

    def encode_text_with_prompt_ensemble(model, obj, device):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(obj) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
"""



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
