# main.py

import os
import argparse
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from dataset.full_data import MedDataset
from biomedclip.clip import create_model ,_MODEL_CKPT_PATHS
from biomedclip.adapter import CLIP_Inplanted
from sklearn.metrics import roc_auc_score
from loss import FocalLoss, BinaryDiceLoss
from utils import encode_text_with_biomedclip_prompt_ensemble1
from prompt import REAL_NAME

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}

global_vars = {}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='BiomedCLIP Training')
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument('--seed', type=int, default=111)

    args, _ = parser.parse_known_args()

    print("\nParsed Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
        global_vars[arg] = getattr(args, arg)
    
    setup_seed(args.seed)
    print("\nSeed set to:", args.seed)

    # ✅ Load BiomedCLIP model - now returns both model and vision_proj automatically
    print("\n" + "="*70)
    print("LOADING BIOMEDCLIP MODEL AND VISION PROJECTION")
    print("="*70)
    
    clip_model = create_model(
        model_name=args.model_name, 
        img_size=args.img_size, 
        device=device, 
        pretrained=args.pretrain, 
        require_pretrained=True
    )
    # Note: create_model internally calls load_biomedclip_model with return_vision_proj=True
    
    clip_model.eval()
    

    # ✅ Create adapter model
    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)

    for name, param in model.named_parameters():
        param.requires_grad = True

    # Optimizers
    seg_optimizer = AdamW(model.seg_adapters.parameters(), lr=args.learning_rate, 
                          betas=(0.5, 0.999), weight_decay=1e-4)
    det_optimizer = AdamW(model.det_adapters.parameters(), lr=args.learning_rate, 
                          betas=(0.5, 0.999), weight_decay=1e-4)

    # Schedulers
    warmup_seg = LinearLR(seg_optimizer, start_factor=0.1, total_iters=5)
    cosine_seg = CosineAnnealingLR(seg_optimizer, T_max=args.epoch-5, eta_min=1e-6)
    seg_scheduler = SequentialLR(seg_optimizer, schedulers=[warmup_seg, cosine_seg], milestones=[5])

    warmup_det = LinearLR(det_optimizer, start_factor=0.1, total_iters=5)
    cosine_det = CosineAnnealingLR(det_optimizer, T_max=args.epoch-5, eta_min=1e-6)
    det_scheduler = SequentialLR(det_optimizer, schedulers=[warmup_det, cosine_det], milestones=[5])

    # Load datasets
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = MedDataset(dataset_path=args.data_path, class_name=args.obj, 
                               split='train', resize=args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, 
                                               shuffle=True, **kwargs)
    
    valid_dataset = MedDataset(dataset_path=args.data_path, class_name=args.obj, 
                               split='valid', resize=args.img_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, 
                                               shuffle=False, **kwargs)

    # Losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()



    # Load vision projection from checkpoint
    checkpoint_path = _MODEL_CKPT_PATHS[args.model_name]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create projection layer
    vision_proj = nn.Linear(768, 512, bias=False).to(device)
    vision_proj.weight.data = checkpoint['visual.head.proj.weight'].to(device)
    vision_proj.eval()
    
    # Freeze (optional but recommended)
    for param in vision_proj.parameters():
        param.requires_grad = False
    
    print(f"✅ Vision projection loaded: {vision_proj.weight.shape}")

    # ✅ Text features (pre-computed)
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble1(
            clip_model, REAL_NAME[args.obj], device
        )
    print("Text features shape:", text_features.shape)



    best_result = 0

    # ✅ Training loop
    for epoch in range(args.epoch):
        print(f'\nEpoch {epoch}:')
        model.train()
        loss_list = []
        
        for (image, label, gt) in train_loader:
            image = image.to(device)
            
            with torch.cuda.amp.autocast():
                # Get patch tokens from adapters
                _, seg_patch_tokens, det_patch_tokens = model(image)
                seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

                # ✅ Detection loss with vision projection
                det_loss = 0
                image_label = label.to(device).float()
                
                for layer in range(len(det_patch_tokens)):
                    raw_tokens=det_patch_tokens[layer]  #[196,768]
                    #print(f"  Raw tokens shape: {raw_tokens.shape}")

                    # ✅ CRITICAL: Project from 768 to 512 dimensions
                    with torch.no_grad():  # Don't backprop through frozen projection
                        projected_tokens = vision_proj(raw_tokens)  # [196, 768] -> [196, 512]
                    projected_tokens = projected_tokens / projected_tokens.norm(dim=-1, keepdim=True)
                    #print(f"  After normalization, mean norm: {projected_tokens.norm(dim=-1).mean():.4f}")

                    # ✅ NOW dimensions match: [196, 512] @ [512, 2] = [196, 2]
                    anomaly_map = (100.0 * projected_tokens @ text_features).unsqueeze(0) 
                    #print(f"  Anomaly map shape (pre-softmax): {anomaly_map.shape}")  # [1, 196, 2]

                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    #print(f"  Anomaly map shape (post-softmax): {anomaly_map.shape}")  # [1, 196]

                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    #print(f"  Anomaly score: {anomaly_score.item():.4f}")
                    det_loss += loss_bce(anomaly_score, image_label)


                # Segmentation loss (add your code here)
                if CLASS_INDEX[args.obj] > 0:
                    # pixel level
                    seg_loss = 0
                    mask = gt.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    for layer in range(len(seg_patch_tokens)):
                        seg_patch_tokens[layer] = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features).unsqueeze(0)
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)

                    loss = seg_loss + det_loss
                    loss.requires_grad_(True)
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    loss.backward()
                    seg_optimizer.step()
                    det_optimizer.step()

                else:
                    loss = det_loss
                    loss.requires_grad_(True)
                    det_optimizer.zero_grad()
                    loss.backward()
                    det_optimizer.step()

                loss_list.append(loss.item())

        print("Loss: ", np.mean(loss_list))
        # ✅ ADD THESE LINES AT END OF EPOCH:
        seg_scheduler.step()
        det_scheduler.step()
        


if __name__ == '__main__':
    main()
