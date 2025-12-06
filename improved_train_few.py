import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from dataset.medical_few import MedDataset
from biomedclip.clip import create_model
from biomedclip.model import resize_pos_embed_biomedclip # Import the resize function
from biomedclip.adapter import CLIP_Inplanted
from sklearn.metrics import roc_auc_score
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_biomedclip_prompt_ensemble
from prompt import REAL_NAME

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='BiomedCLIP SOTA Training')
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')    
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    # SOTA TIP: Try running with img_size 448 for better medical defect detection
    parser.add_argument('--img_size', type=int, default=224) 
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9, 12])    
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    
    args, _ = parser.parse_known_args()
    setup_seed(args.seed)

    # 1. Create Model Skeleton
    # We set require_pretrained=False first to handle resizing safely if needed
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, 
                              device=device, pretrained=None, require_pretrained=False)
    
    # 2. SOTA: Resize Positional Embeddings MANUALLY if size != 224
    checkpoint_path = os.path.join(os.path.dirname(__file__), "biomedclip/ckpt/open_clip_pytorch_model.bin")
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        if args.img_size != 224:
            print(f"Resizing positional embeddings to {args.img_size}...")
            resize_pos_embed_biomedclip(state_dict, clip_model)
            
        msg = clip_model.load_state_dict(state_dict, strict=False)
        print("Model Loaded:", msg)
    else:
        print("Warning: Checkpoint not found at local path, relying on default load.")

    clip_model.eval()

    # 3. Initialize Adapter Wrapper
    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.eval()

    # 4. Freeze Backbone, Unfreeze Adapters
    for name, param in model.named_parameters():
        param.requires_grad = False # Default freeze
        
    for param in model.seg_adapters.parameters(): param.requires_grad = True
    for param in model.det_adapters.parameters(): param.requires_grad = True

    # Optimizer
    seg_optimizer = torch.optim.Adam(model.seg_adapters.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(model.det_adapters.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    # Load Data
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # Augmentation & Memory Bank Setup
    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)
    
    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
    augment_fewshot_label = torch.cat([torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0)

    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)

    # Losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    # Text Prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)

    best_result = 0

    # Training Loop
    for epoch in range(args.epoch):
        model.train() # Adapters are trainable
        loss_list = []
        
        for (image, gt, label) in train_loader:
            image = image.to(device)
            label = label.to(device)
            gt = gt.to(device)
            
            with torch.cuda.amp.autocast():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                
                # Strip CLS token (already done in adapter, but double check shape)
                # Adapter output is [B, N_patches, C]
                
                # --- DETECTION HEAD ---
                det_loss = 0
                for layer in range(len(det_patch_tokens)):
                    tokens = det_patch_tokens[layer]
                    
                    # SOTA FIX 1: Project -> Normalize (Correct CLIP Math)
                    # Use the helper we added to CLIP_Inplanted
                    proj_tokens = model.project_features(tokens)
                    proj_tokens = proj_tokens / proj_tokens.norm(dim=-1, keepdim=True)
                    
                    # Compute Anomaly Map
                    # (B, N, 512) @ (512, 1) -> (B, N, 1)
                    anomaly_map = (100.0 * proj_tokens @ text_features.T)
                    
                    # SOTA FIX 2: Max Pooling for Detection
                    # Anomalies are local. Mean pooling dilutes the signal.
                    # We take the top 5% of anomalous patches or just Max
                    top_k_vals, _ = torch.topk(anomaly_map.view(anomaly_map.shape[0], -1), k=5)
                    anomaly_score = top_k_vals.mean(dim=-1)
                    
                    det_loss += loss_bce(anomaly_score, label)

                # --- SEGMENTATION HEAD ---
                if CLASS_INDEX[args.obj] > 0:
                    seg_loss = 0
                    # Prepare Mask
                    mask = gt.squeeze(0) # [1, H, W]
                    mask[mask > 0.5] = 1
                    mask[mask <= 0.5] = 0
                    
                    for layer in range(len(seg_patch_tokens)):
                        tokens = seg_patch_tokens[layer]
                        
                        # SOTA FIX 1: Project -> Normalize
                        proj_tokens = model.project_features(tokens)
                        proj_tokens = proj_tokens / proj_tokens.norm(dim=-1, keepdim=True)
                        
                        # Anomaly Map
                        anomaly_map = (100.0 * proj_tokens @ text_features.T)
                        
                        # Interpolate to Image Size
                        B, L, C = anomaly_map.shape # C=1 usually (text prompt dim)
                        H = int(np.sqrt(L))
                        
                        anomaly_map = anomaly_map.permute(0, 2, 1).view(B, 1, H, H)
                        anomaly_map = F.interpolate(anomaly_map, size=args.img_size, mode='bilinear', align_corners=True)
                        
                        # Apply Losses (Focal + Dice)
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map, mask)
                    
                    loss = seg_loss + det_loss
                    
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    loss.backward()
                    seg_optimizer.step()
                    det_optimizer.step()
                else:
                    loss = det_loss
                    det_optimizer.zero_grad()
                    loss.backward()
                    det_optimizer.step()

                loss_list.append(loss.item())

        print(f"Epoch {epoch}: Loss {np.mean(loss_list):.4f}")

        # --- EVALUATION (Simplified call) ---
        # Note: You need to ensure Memory Bank generation also uses Project->Normalize
        # But for now, we leave the test function call structure as is.
        # Ideally, move the memory bank generation INSIDE test() or update it here similarly.
        
        # (Memory bank generation code omitted for brevity, but apply same logic: Project -> Normalize)
        
        if (epoch + 1) % 5 == 0: # Save every 5 epochs
             if args.save_model == 1:
                if not os.path.exists(args.save_path): os.makedirs(args.save_path)
                ckp_path = os.path.join(args.save_path, f'{args.obj}_last.pth')
                torch.save({'seg_adapters': model.seg_adapters.state_dict(),
                            'det_adapters': model.det_adapters.state_dict()}, ckp_path)

if __name__ == '__main__':
    main()