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
from utils import cos_sim, encode_text_with_biomedclip_prompt_ensemble1
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
    #parser.add_argument('--text_encoder', type=str, default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
    #                    help="Text encoder used for BiomedCLIP" )

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
    #parser.add_argument('--shot', type=int, default=4)
    #parser.add_argument('--iterate', type=int, default=0)

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

    
    # fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, require_pretrained=True )
    
   
    clip_model.eval()
    


    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
  


    for name, param in model.named_parameters():
        param.requires_grad = True


    # ✅ NEW - ADD THIS:
    seg_optimizer = AdamW(model.seg_adapters.parameters(), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=1e-4)
    det_optimizer = AdamW(model.det_adapters.parameters(), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=1e-4)

    #✅  SCHEDULER (Warmup + Cosine)
    warmup_seg = LinearLR(seg_optimizer, start_factor=0.1, total_iters=5)
    cosine_seg = CosineAnnealingLR(seg_optimizer, T_max=args.epoch-5, eta_min=1e-6)
    seg_scheduler = SequentialLR(seg_optimizer, schedulers=[warmup_seg, cosine_seg], milestones=[5])

    warmup_det = LinearLR(det_optimizer, start_factor=0.1, total_iters=5)
    cosine_det = CosineAnnealingLR(det_optimizer, T_max=args.epoch-5, eta_min=1e-6)
    det_scheduler = SequentialLR(det_optimizer, schedulers=[warmup_det, cosine_det], milestones=[5])

        # load test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        # Similarly for train and valid
    train_dataset = MedDataset(dataset_path=args.data_path, class_name=args.obj, split='train', resize=args.img_size)
    print("\n Training dataset",len(train_dataset))

    #train_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    # ✅ CHECK FIRST BATCH SHAPES
    # Get the first batch directly
    first_batch = next(iter(train_loader))
    images, labels, masks = first_batch

    print(f"\n=== Batch Shapes ===")
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")
    print(f"Masks batch shape: {masks.shape}")
    print(f"Labels values (first 3): {labels[:3]}")
    print(f"Labels dtype: {labels.dtype}")

    valid_dataset = MedDataset(dataset_path=args.data_path, class_name=args.obj, split='valid', resize=args.img_size)
    
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)


        # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()


    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble1(clip_model, REAL_NAME[args.obj], device)
    print("Text features shape:", text_features.shape)  

    best_result = 0


    for epoch in range(args.epoch):
        print('epoch ', epoch, ':')

        loss_list = []
        for (image, gt, label) in train_loader:
            image = image.to(device)
            with torch.cuda.amp.autocast():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]
                    
                 # det loss
                det_loss = 0
                image_label = label.to(device)
                print("\n--- Detection Loss Calculation Start ---")
                print(f"Initial image_label (ground truth): {image_label.shape}, values: {image_label}")
                for layer in range(len(det_patch_tokens)):
                    # Normalize tokens
                    det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    print(f"\nLayer {layer} tokens shape after normalization: {det_patch_tokens[layer].shape}")
                    # Calculate cosine similarity with text features
                    # Note: Assuming 'text_features' is defined and on the correct device
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features).unsqueeze(0)
                    print(f"Anomaly map shape after unsqueeze(0) (pre-softmax): {anomaly_map.shape}")
                    # Apply softmax and select the 'abnormal' probability channel (index 1)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    print(f"Anomaly map shape after softmax/selection: {anomaly_map.shape}")
                    # Calculate the image-level anomaly score (mean across patches)
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    print(f"Anomaly score shape (image-level prediction): {anomaly_score.shape}")
                    print(f"Anomaly score values: {anomaly_score}")
                    # Calculate the loss using Binary Cross-Entropy
                    # Note: Assuming 'loss_bce' is defined (e.g., nn.BCEWithLogitsLoss or nn.BCELoss)
                    loss = loss_bce(anomaly_score, image_label)
                    det_loss += loss
                    print(f"Loss for Layer {layer}: {loss.item()}")
                print(f"\nTotal det_loss for the batch: {det_loss.item()}")
                print("--- Detection Loss Calculation End ---")
        


    
      









if __name__ == '__main__':
    main()
