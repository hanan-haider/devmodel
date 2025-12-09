# train full code 

import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from dataset.full_data import MedDataset
from biomedclip.clip import create_model
from biomedclip.adapter import CLIP_Inplanted
from sklearn.metrics import roc_auc_score
from loss import FocalLoss, BinaryDiceLoss
from utils import cos_sim, encode_text_with_biomedclip_prompt_ensemble
from prompt import REAL_NAME


from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}

# Global variables that will be accessible across cells
global_vars = { }

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main():
    parser = argparse.ArgumentParser(description='BiomedCLIP Testing')
    # General defaults
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16',
                        help="BiomedCLIP model version")    
    parser.add_argument('--text_encoder', type=str, default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
                        help="Text encoder used for BiomedCLIP" )

    parser.add_argument('--pretrain', type=str, default='microsoft',
                            help="pretrained checkpoint source")
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/',
                        help="path to dataset"  )
    #parser.add_argument('--data_path', type=str, default='/kaggle/input/preprocessed/Liver')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--img_size', type=int, default=224, 
                        help="BiomedCLIP trained with 224x224 resolution")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9, 12],
                        help="layer features used for adapters")    
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)

    #args = parser.parse_args()
#printing the arguments 
    args, _ = parser.parse_known_args()

    # Print all arguments
    print("\nParsed Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
        global_vars[arg] = getattr(args, arg)  # Store each argument in global_vars
    
    # Set up seed
    setup_seed(args.seed)
    print("\nSeed set to:", args.seed)
  #for printing at the outset of training

    setup_seed(args.seed)
