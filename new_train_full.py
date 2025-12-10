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
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--features_list", type=int, nargs="+", default=[4, 8, 10, 12])
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--memory_bank_size', type=int, default=300)  # ✅ NEW: Limit memory bank size

    args, _ = parser.parse_known_args()

    print("\nParsed Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
        global_vars[arg] = getattr(args, arg)
    
    setup_seed(args.seed)
    print("\nSeed set to:", args.seed)

    # Load BiomedCLIP model
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
    clip_model.eval()

    # Create adapter model
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
    kwargs = {'num_workers': 2, 'pin_memory': False}  # ✅ Reduce workers, disable pin_memory
    
    train_dataset = MedDataset(dataset_path=args.data_path, class_name=args.obj, 
                               split='train', resize=args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                               shuffle=True, **kwargs)
    
    valid_dataset = MedDataset(dataset_path=args.data_path, class_name=args.obj, 
                               split='valid', resize=args.img_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1 , 
                                               shuffle=False, **kwargs)

    # ============================================
    # ✅ BUILD MEMORY BANK (MEMORY-EFFICIENT VERSION)
    # ============================================
    print("\n" + "="*70)
    print("BUILDING MEMORY BANK")
    print("="*70)
    
    # Get normal image indices
    normal_indices = [i for i, (_, label, _) in enumerate(train_dataset) if label == 0]
    print(f"Found {len(normal_indices)} normal images")
    
    # ✅ Sample subset if too many
    if len(normal_indices) > args.memory_bank_size:
        random.seed(args.seed)
        normal_indices = random.sample(normal_indices, args.memory_bank_size)
        print(f"Sampled {len(normal_indices)} images for memory bank")
    
    # Create subset
    from torch.utils.data import Subset
    support_dataset = Subset(train_dataset, normal_indices)
    
    # ✅ Use batches for memory bank
    support_loader = torch.utils.data.DataLoader(
        support_dataset,
        batch_size=16,  # Process 16 images at once
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )

    # Losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    # Load vision projection
    checkpoint_path = _MODEL_CKPT_PATHS[args.model_name]
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # ✅ Load to CPU first
    
    vision_proj = nn.Linear(768, 512, bias=False).to(device)
    vision_proj.weight.data = checkpoint['visual.head.proj.weight'].to(device)
    vision_proj.eval()
    
    for param in vision_proj.parameters():
        param.requires_grad = False
    
    print(f"✅ Vision projection loaded: {vision_proj.weight.shape}")
    
    # Clear checkpoint from memory
    del checkpoint
    torch.cuda.empty_cache()

    # Text features (pre-computed)
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble1(
            clip_model, REAL_NAME[args.obj], device
        )
    print("Text features shape:", text_features.shape)

    # ============================================
    # ✅ EXTRACT MEMORY BANK FEATURES (EFFICIENT)
    # ============================================
    seg_features_list = []
    det_features_list = []

    model.eval()
    print("\nExtracting memory bank features...")
    
    with torch.no_grad():
        for batch_idx, (images, _, _) in enumerate(tqdm(support_loader, desc="Memory bank")):
            images = images.to(device)
            
            # Forward pass
            _, seg_tokens, det_tokens = model(images)
            
            # Extract features for each image in batch
            batch_size = images.shape[0]
            for i in range(batch_size):
                # ✅ Move to CPU immediately
                seg_feat = [p[i, 1:, :].detach().cpu() for p in seg_tokens]
                det_feat = [p[i, 1:, :].detach().cpu() for p in det_tokens]
                
                seg_features_list.append(seg_feat)
                det_features_list.append(det_feat)
            
            # ✅ Clear GPU every 5 batches
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()

    # ✅ Concatenate features (on CPU)
    print("Concatenating memory bank features...")
    num_layers = len(seg_features_list[0])
    
    seg_mem_features = []
    det_mem_features = []
    
    for layer_idx in range(num_layers):
        seg_layer = torch.cat([seg_features_list[j][layer_idx] for j in range(len(seg_features_list))], dim=0)
        det_layer = torch.cat([det_features_list[j][layer_idx] for j in range(len(det_features_list))], dim=0)
        
        # ✅ Keep on CPU
        seg_mem_features.append(seg_layer)
        det_mem_features.append(det_layer)
        
        print(f"  Layer {layer_idx}: {seg_layer.shape}")

    print(f"✅ Memory bank complete: {len(seg_features_list)} images")
    print("="*70 + "\n")
    
    # ✅ Clear temporary lists
    del seg_features_list, det_features_list
    torch.cuda.empty_cache()

    best_result = 0
    # ============================================
    # TRAINING LOOP WITH PROPER BATCH PROCESSING
    # ============================================
    
    for epoch in range(args.epoch):
        model.train()
        loss_list = []
        
        for (image, label, gt) in tqdm(train_loader, desc=f"Epoch {epoch}"):
            # ✅ image is now [batch_size, 3, 224, 224]
            # ✅ label is now [batch_size]
            # ✅ gt is now [batch_size, 1, 224, 224]
            
            image = image.to(device)
            batch_size = image.shape[0]
            
            # Forward pass
            _, seg_patch_tokens, det_patch_tokens = model(image)
            
            # ✅ FIXED: Remove CLS token for ALL images in batch
            # Old: [p[0, 1:, :] for p in det_patch_tokens]  # Only first image
            # New: Keep batch dimension
            seg_patch_tokens = [p[:, 1:, :] for p in seg_patch_tokens]  # [B, 196, 768]
            det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]  # [B, 196, 768]
    
            # ============================================
            # DETECTION LOSS (BATCHED)
            # ============================================
            det_loss = 0
            image_label = label.to(device).float()  # [batch_size]
            
            for layer in range(len(det_patch_tokens)):
                raw_tokens = det_patch_tokens[layer]  # [batch_size, 196, 768]
                
                # ✅ Project all images at once
                # Flatten: [B, 196, 768] -> [B*196, 768]
                flat_tokens = raw_tokens.reshape(-1, 768)
                projected_flat = vision_proj(flat_tokens)  # [B*196, 512]
                
                # Reshape back: [B*196, 512] -> [B, 196, 512]
                projected_tokens = projected_flat.reshape(batch_size, -1, 512)
                
                # Normalize
                projected_tokens = projected_tokens / projected_tokens.norm(dim=-1, keepdim=True)
                
                # Compute anomaly maps for all images
                # [B, 196, 512] @ [512, 2] -> [B, 196, 2]
                anomaly_maps = 100.0 * torch.matmul(projected_tokens, text_features)
                
                # Softmax: [B, 196, 2] -> [B, 196]
                anomaly_maps = torch.softmax(anomaly_maps, dim=-1)[:, :, 1]
                
                # Average over patches: [B, 196] -> [B]
                anomaly_scores = torch.mean(anomaly_maps, dim=-1)
                
                # BCE loss (now works because both are [B])
                det_loss += loss_bce(anomaly_scores, image_label)
    
            # ============================================
            # SEGMENTATION LOSS (BATCHED)
            # ============================================
            if CLASS_INDEX[args.obj] > 0:
                seg_loss = 0
                mask = gt.to(device)  # [batch_size, 1, 224, 224]
                mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                
                for layer in range(len(seg_patch_tokens)):
                    raw_tokens = seg_patch_tokens[layer]  # [B, 196, 768]
                    
                    # Project
                    flat_tokens = raw_tokens.reshape(-1, 768)
                    projected_flat = vision_proj(flat_tokens)
                    projected_tokens = projected_flat.reshape(batch_size, -1, 512)
                    
                    # Normalize
                    projected_tokens = projected_tokens / projected_tokens.norm(dim=-1, keepdim=True)
                    
                    # Anomaly map: [B, 196, 512] @ [512, 2] -> [B, 196, 2]
                    anomaly_map = 100.0 * torch.matmul(projected_tokens, text_features)
                    
                    # Reshape to spatial: [B, 196, 2] -> [B, 2, 14, 14]
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = anomaly_map.permute(0, 2, 1).view(B, 2, H, H)
                    
                    # Upsample: [B, 2, 14, 14] -> [B, 2, 224, 224]
                    anomaly_map = F.interpolate(
                        anomaly_map,
                        size=args.img_size,
                        mode='bilinear',
                        align_corners=True
                    )
                    
                    # Softmax
                    anomaly_map = torch.softmax(anomaly_map, dim=1)
                    
                    # Losses
                    seg_loss += loss_focal(anomaly_map, mask)
                    seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)
    
                # Total loss
                loss = seg_loss + det_loss
                
                # Backward
                seg_optimizer.zero_grad()
                det_optimizer.zero_grad()
                loss.backward()
                seg_optimizer.step()
                det_optimizer.step()
            
            else:
                # Detection only
                loss = det_loss
                det_optimizer.zero_grad()
                loss.backward()
                det_optimizer.step()
    
            loss_list.append(loss.item())
    
        print(f"Epoch {epoch} - Loss: {np.mean(loss_list):.4f}")
    
        # Update schedulers
        seg_scheduler.step()
        det_scheduler.step()
        
    
        # ✅ Test (memory bank features on CPU, moved to GPU in test function)
        result = test(args, model, valid_loader, text_features, vision_proj,
                     seg_mem_features, det_mem_features)
        
        if result > best_result:
            best_result = result
            print("Best result!")
            if args.save_model == 1:
                os.makedirs(args.save_path, exist_ok=True)
                ckp_path = os.path.join(args.save_path, f'{args.obj}.pth')
                torch.save({'seg_adapters': model.seg_adapters.state_dict(),
                            'det_adapters': model.det_adapters.state_dict()}, 
                            ckp_path)


          
def test(args, model, valid_loader, text_features, vision_proj, seg_mem_features, det_mem_features):
    gt_list = []
    gt_mask_list = []
    det_image_scores_zero = []
    det_image_scores_few = []
    seg_score_map_zero = []
    seg_score_map_few = []

    model.eval()

    for (image, y, mask) in tqdm(valid_loader, desc="Testing"):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

            if CLASS_INDEX[args.obj] > 0:
                # ============================================
                # FEW-SHOT SEGMENTATION (MEMORY-EFFICIENT)
                # ============================================
                anomaly_maps_few_shot = []
                for idx, p in enumerate(seg_patch_tokens):
                    # Load memory bank for this layer to GPU
                    seg_mem_layer = seg_mem_features[idx].to(device)  # [58800, 768]
                    
                    # ✅ EFFICIENT COSINE SIMILARITY using matrix multiplication
                    # Normalize both tensors
                    p_norm = p / p.norm(dim=-1, keepdim=True)  # [196, 768]
                    seg_mem_norm = seg_mem_layer / seg_mem_layer.norm(dim=-1, keepdim=True)  # [58800, 768]
                    
                    # Matrix multiplication: [58800, 768] @ [768, 196] = [58800, 196]
                    cos = seg_mem_norm @ p_norm.T  # Cosine similarity matrix
                    
                    # Compute anomaly scores (1 - similarity)
                    anomaly_scores = 1 - cos  # [58800, 196]
                    
                    # Take minimum across all memory patches
                    anomaly_patch = torch.min(anomaly_scores, dim=0)[0]  # [196]
                    
                    # Reshape to spatial map
                    height = int(np.sqrt(anomaly_patch.shape[0]))
                    anomaly_map_few_shot = anomaly_patch.reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(
                        anomaly_map_few_shot,
                        size=args.img_size,
                        mode='bilinear',
                        align_corners=True
                    )
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                    
                    # ✅ Clear GPU memory immediately
                    del seg_mem_layer, seg_mem_norm, cos, anomaly_scores, anomaly_patch
                    torch.cuda.empty_cache()
                
                score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                seg_score_map_few.append(score_map_few)

                # ============================================
                # ZERO-SHOT SEGMENTATION
                # ============================================
                anomaly_maps = []
                for layer in range(len(seg_patch_tokens)):
                    raw_tokens = seg_patch_tokens[layer]
                    projected_tokens = vision_proj(raw_tokens)
                    projected_tokens = projected_tokens / projected_tokens.norm(dim=-1, keepdim=True)
                    
                    anomaly_map = (100.0 * projected_tokens @ text_features).unsqueeze(0)
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(
                        anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                        size=args.img_size,
                        mode='bilinear',
                        align_corners=True
                    )
                    anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                    anomaly_maps.append(anomaly_map.cpu().numpy())
                
                score_map_zero = np.sum(anomaly_maps, axis=0)
                seg_score_map_zero.append(score_map_zero)

            else:
                # ============================================
                # FEW-SHOT DETECTION (MEMORY-EFFICIENT)
                # ============================================
                anomaly_maps_few_shot = []
                for idx, p in enumerate(det_patch_tokens):
                    # Load memory bank for this layer to GPU
                    det_mem_layer = det_mem_features[idx].to(device)  # [58800, 768]
                    
                    # ✅ EFFICIENT COSINE SIMILARITY
                    p_norm = p / p.norm(dim=-1, keepdim=True)  # [196, 768]
                    det_mem_norm = det_mem_layer / det_mem_layer.norm(dim=-1, keepdim=True)  # [58800, 768]
                    
                    # Matrix multiplication
                    cos = det_mem_norm @ p_norm.T  # [58800, 196]
                    anomaly_scores = 1 - cos
                    anomaly_patch = torch.min(anomaly_scores, dim=0)[0]  # [196]
                    
                    # Reshape to spatial map
                    height = int(np.sqrt(anomaly_patch.shape[0]))
                    anomaly_map_few_shot = anomaly_patch.reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(
                        anomaly_map_few_shot,
                        size=args.img_size,
                        mode='bilinear',
                        align_corners=True
                    )
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                    
                    # ✅ Clear GPU
                    del det_mem_layer, det_mem_norm, cos, anomaly_scores, anomaly_patch
                    torch.cuda.empty_cache()
                
                anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                score_few_det = anomaly_map_few_shot.mean()
                det_image_scores_few.append(score_few_det)

                # ============================================
                # ZERO-SHOT DETECTION
                # ============================================
                anomaly_score = 0
                for layer in range(len(det_patch_tokens)):
                    raw_tokens = det_patch_tokens[layer]
                    projected_tokens = vision_proj(raw_tokens)
                    projected_tokens = projected_tokens / projected_tokens.norm(dim=-1, keepdim=True)
                    
                    anomaly_map = (100.0 * projected_tokens @ text_features).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score += anomaly_map.mean()
                
                det_image_scores_zero.append(anomaly_score.cpu().numpy())

            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())

    # ============================================
    # EVALUATION
    # ============================================
    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)

    if CLASS_INDEX[args.obj] > 0:
        # Segmentation metrics
        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)
        
        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min())
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min())
        
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc, 4)}')
        
        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'{args.obj} AUC : {round(roc_auc_im, 4)}')
        
        return seg_roc_auc + roc_auc_im

    else:
        # Detection metrics
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)
        
        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min())
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min())
        
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'{args.obj} AUC : {round(img_roc_auc_det, 4)}')
        
        return img_roc_auc_det

if __name__ == '__main__':
    main()
