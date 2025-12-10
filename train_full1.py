"""
OPTIMAL TRAINING CONFIGURATION FOR BIOMEDCLIP MEDICAL ANOMALY DETECTION
========================================================================
Based on your results analysis:
- Liver: pAUC 0.972 (good), AUC 0.57 (BAD) → Detection head failing
- Retina: AUC 0.91 → Better but still suboptimal
- Loss plateau at 2.77 → Learning rate too high

KEY FIXES:
1. Reduce learning rate from 0.001 → 0.00001
2. Separate LRs for detection vs segmentation
3. Add gradient accumulation
4. Improve loss weighting
5. Better optimizer settings
"""

import os
import argparse
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from dataset.full_data import MedDataset
from biomedclip.clip import create_model, _MODEL_CKPT_PATHS
from biomedclip.adapter import CLIP_Inplanted
from sklearn.metrics import roc_auc_score
from loss import FocalLoss, BinaryDiceLoss
from utils import encode_text_with_biomedclip_prompt_ensemble1
from prompt import REAL_NAME
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================
# ✅ IMPROVED LOSS WEIGHTING
# ============================================
class AdaptiveLossWeighting(nn.Module):
    """Learnable task weighting using uncertainty"""
    def __init__(self):
        super().__init__()
        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_det = nn.Parameter(torch.zeros(1))
    
    def forward(self, seg_loss, det_loss):
        precision_seg = torch.exp(-self.log_var_seg)
        loss_seg = precision_seg * seg_loss + self.log_var_seg
        
        precision_det = torch.exp(-self.log_var_det)
        loss_det = precision_det * det_loss + self.log_var_det
        
        return loss_seg + loss_det

# ============================================
# MAIN TRAINING FUNCTION
# ============================================
def main():
    parser = argparse.ArgumentParser(description='BiomedCLIP Optimal Training')
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epoch', type=int, default=50)
    
    # ✅ OPTIMAL LEARNING RATES (CRITICAL FIX)
    parser.add_argument('--seg_lr', type=float, default=1e-5, 
                       help='Segmentation adapter LR (1e-5 for pixel-level)')
    parser.add_argument('--det_lr', type=float, default=5e-6, 
                       help='Detection adapter LR (5e-6, more sensitive)')
    
    parser.add_argument('--features_list', type=int, nargs="+", default=[4, 8, 10, 12])
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--memory_bank_size', type=int, default=300)
    
    # ✅ NEW: Advanced training parameters
    parser.add_argument('--accumulation_steps', type=int, default=4,
                       help='Gradient accumulation (effective batch = 12*4 = 48)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma')

    args, _ = parser.parse_known_args()
    
    print("\n" + "="*70)
    print("OPTIMAL BIOMEDCLIP TRAINING CONFIGURATION")
    print("="*70)
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("="*70 + "\n")
    
    setup_seed(args.seed)

    # ============================================
    # LOAD MODEL
    # ============================================
    print("Loading BioMedCLIP...")
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained=args.pretrain,
        require_pretrained=True
    )
    clip_model.eval()
    
    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    
    # Freeze everything except adapters
    for name, param in model.named_parameters():
        param.requires_grad = 'adapter' in name

    # ✅ OPTIMAL OPTIMIZER CONFIGURATION
    seg_optimizer = AdamW(
        model.seg_adapters.parameters(),
        lr=args.seg_lr,
        betas=(0.9, 0.999),  # Changed from (0.5, 0.999)
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    
    det_optimizer = AdamW(
        model.det_adapters.parameters(),
        lr=args.det_lr,  # Lower LR for detection
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    # ✅ OPTIMAL SCHEDULER: OneCycleLR (better than Cosine)
    seg_scheduler = OneCycleLR(
        seg_optimizer,
        max_lr=args.seg_lr * 10,  # Peak at 10x base LR
        epochs=args.epoch,
        steps_per_epoch=1,  # We'll step once per epoch
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=10.0,  # Start at max_lr/10
        final_div_factor=1000.0  # End at max_lr/1000
    )
    
    det_scheduler = OneCycleLR(
        det_optimizer,
        max_lr=args.det_lr * 10,
        epochs=args.epoch,
        steps_per_epoch=1,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=1000.0
    )

    # ============================================
    # DATA LOADERS
    # ============================================
    kwargs = {'num_workers': 2, 'pin_memory': False}
    
    train_dataset = MedDataset(
        dataset_path=args.data_path,
        class_name=args.obj,
        split='train',
        resize=args.img_size
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    
    valid_dataset = MedDataset(
        dataset_path=args.data_path,
        class_name=args.obj,
        split='valid',
        resize=args.img_size
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        **kwargs
    )

    # ============================================
    # BUILD MEMORY BANK
    # ============================================
    print("Building memory bank...")
    normal_indices = [i for i, (_, label, _) in enumerate(train_dataset) if label == 0]
    
    if len(normal_indices) > args.memory_bank_size:
        random.seed(args.seed)
        normal_indices = random.sample(normal_indices, args.memory_bank_size)
    
    from torch.utils.data import Subset
    support_dataset = Subset(train_dataset, normal_indices)
    support_loader = torch.utils.data.DataLoader(
        support_dataset,
        batch_size=16,
        shuffle=False,
        **kwargs
    )

    # ✅ IMPROVED LOSSES
    loss_focal = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    loss_dice = BinaryDiceLoss()
    loss_bce = nn.BCEWithLogitsLoss()
    adaptive_loss = AdaptiveLossWeighting().to(device)

    # Load vision projection
    checkpoint_path = _MODEL_CKPT_PATHS[args.model_name]
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    vision_proj = nn.Linear(768, 512, bias=False).to(device)
    vision_proj.weight.data = checkpoint['visual.head.proj.weight'].to(device)
    vision_proj.eval()
    
    for param in vision_proj.parameters():
        param.requires_grad = False
    
    del checkpoint
    torch.cuda.empty_cache()

    # Text features
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble1(
            clip_model, REAL_NAME[args.obj], device
        )
    print(f"Text features: {text_features.shape}\n")

    # Extract memory bank features
    seg_mem_features, det_mem_features = build_memory_bank(
        model, support_loader, args
    )

    # ============================================
    # TRAINING LOOP WITH GRADIENT ACCUMULATION
    # ============================================
    best_result = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(args.epoch):
        model.train()
        loss_list = []
        seg_loss_list = []
        det_loss_list = []
        
        # Reset gradients at epoch start
        seg_optimizer.zero_grad()
        det_optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (image, label, gt) in enumerate(pbar):
            image = image.to(device)
            batch_size = image.shape[0]
            
            with torch.cuda.amp.autocast():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                
                seg_patch_tokens = [p[:, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]
                
                # ============================================
                # DETECTION LOSS
                # ============================================
                det_loss = 0
                image_label = label.to(device).float()
                
                for layer in range(len(det_patch_tokens)):
                    raw_tokens = det_patch_tokens[layer]
                    flat_tokens = raw_tokens.reshape(-1, 768)
                    projected_flat = vision_proj(flat_tokens)
                    projected_tokens = projected_flat.reshape(batch_size, -1, 512)
                    projected_tokens = projected_tokens / projected_tokens.norm(dim=-1, keepdim=True)
                    
                    anomaly_maps = 100.0 * torch.matmul(projected_tokens, text_features)
                    anomaly_maps = torch.softmax(anomaly_maps, dim=-1)[:, :, 1]
                    anomaly_scores = torch.mean(anomaly_maps, dim=-1)
                    
                    det_loss += loss_bce(anomaly_scores, image_label)
                
                det_loss = det_loss / len(det_patch_tokens)
                
                # ============================================
                # SEGMENTATION LOSS
                # ============================================
                seg_loss = 0
                if CLASS_INDEX[args.obj] > 0:
                    mask = gt.to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    
                    for layer in range(len(seg_patch_tokens)):
                        raw_tokens = seg_patch_tokens[layer]
                        flat_tokens = raw_tokens.reshape(-1, 768)
                        projected_flat = vision_proj(flat_tokens)
                        projected_tokens = projected_flat.reshape(batch_size, -1, 512)
                        projected_tokens = projected_tokens / projected_tokens.norm(dim=-1, keepdim=True)
                        
                        anomaly_map = 100.0 * torch.matmul(projected_tokens, text_features)
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = anomaly_map.permute(0, 2, 1).view(B, 2, H, H)
                        anomaly_map = F.interpolate(
                            anomaly_map,
                            size=args.img_size,
                            mode='bilinear',
                            align_corners=True
                        )
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += 0.5 * loss_dice(anomaly_map[:, 1, :, :], mask)
                    
                    seg_loss = seg_loss / len(seg_patch_tokens)
                    
                    # ✅ Use adaptive loss weighting
                    total_loss = adaptive_loss(seg_loss, det_loss)
                else:
                    total_loss = det_loss
                
                # ✅ Scale loss by accumulation steps
                total_loss = total_loss / args.accumulation_steps
            
            # Backward
            total_loss.backward()
            
            # ✅ Gradient accumulation
            if (batch_idx + 1) % args.accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.seg_adapters.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.det_adapters.parameters(), 1.0)
                
                # Optimizer step
                seg_optimizer.step()
                det_optimizer.step()
                
                # Zero gradients
                seg_optimizer.zero_grad()
                det_optimizer.zero_grad()
            
            # Track losses
            loss_list.append(total_loss.item() * args.accumulation_steps)
            if CLASS_INDEX[args.obj] > 0:
                seg_loss_list.append(seg_loss.item())
            det_loss_list.append(det_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{np.mean(loss_list):.4f}',
                'det': f'{np.mean(det_loss_list):.4f}',
                'seg': f'{np.mean(seg_loss_list):.4f}' if seg_loss_list else 'N/A'
            })
        
        # Step schedulers
        seg_scheduler.step()
        det_scheduler.step()
        
        print(f"\nEpoch {epoch}:")
        print(f"  Loss: {np.mean(loss_list):.4f}")
        print(f"  Det Loss: {np.mean(det_loss_list):.4f}")
        if seg_loss_list:
            print(f"  Seg Loss: {np.mean(seg_loss_list):.4f}")
        print(f"  Seg LR: {seg_optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Det LR: {det_optimizer.param_groups[0]['lr']:.2e}")
        
        # Validation
        result = test(
            args, model, valid_loader, text_features,
            vision_proj, seg_mem_features, det_mem_features
        )
        
        if result > best_result:
            best_result = result
            patience_counter = 0
            print(f"✅ BEST MODEL: {result:.4f}")
            
            if args.save_model:
                os.makedirs(args.save_path, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'seg_adapters': model.seg_adapters.state_dict(),
                    'det_adapters': model.det_adapters.state_dict(),
                    'adaptive_loss': adaptive_loss.state_dict(),
                    'best_result': best_result
                }, f"{args.save_path}/{args.obj}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break

def build_memory_bank(model, support_loader, args):
    """Extract memory bank features efficiently"""
    seg_features_list = []
    det_features_list = []
    
    model.eval()
    
    with torch.no_grad():
        for images, _, _ in tqdm(support_loader, desc="Memory bank"):
            images = images.to(device)
            _, seg_tokens, det_tokens = model(images)
            
            batch_size = images.shape[0]
            for i in range(batch_size):
                seg_feat = [p[i, 1:, :].detach().cpu() for p in seg_tokens]
                det_feat = [p[i, 1:, :].detach().cpu() for p in det_tokens]
                seg_features_list.append(seg_feat)
                det_features_list.append(det_feat)
    
    # Concatenate
    num_layers = len(seg_features_list[0])
    seg_mem = [torch.cat([seg_features_list[j][i] for j in range(len(seg_features_list))], dim=0)
               for i in range(num_layers)]
    det_mem = [torch.cat([det_features_list[j][i] for j in range(len(det_features_list))], dim=0)
               for i in range(num_layers)]
    
    print(f"✅ Memory bank: {len(seg_features_list)} images × {num_layers} layers")
    return seg_mem, det_mem

def test(args, model, valid_loader, text_features, vision_proj, seg_mem_features, det_mem_features):
    """Validation function (same as before but with progress tracking)"""
    gt_list = []
    gt_mask_list = []
    det_scores_zero = []
    det_scores_few = []
    seg_scores_zero = []
    seg_scores_few = []
    
    model.eval()
    
    for (image, y, mask) in tqdm(valid_loader, desc="Validation"):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
        
        with torch.no_grad():
            _, seg_tokens, det_tokens = model(image)
            seg_tokens = [p[0, 1:, :] for p in seg_tokens]
            det_tokens = [p[0, 1:, :] for p in det_tokens]
            
            if CLASS_INDEX[args.obj] > 0:
                # Segmentation evaluation
                few_maps = []
                for idx, p in enumerate(seg_tokens):
                    mem_layer = seg_mem_features[idx].to(device)
                    p_norm = p / p.norm(dim=-1, keepdim=True)
                    mem_norm = mem_layer / mem_layer.norm(dim=-1, keepdim=True)
                    cos = mem_norm @ p_norm.T
                    anomaly = 1 - cos
                    anomaly_patch = torch.min(anomaly, dim=0)[0]
                    h = int(np.sqrt(anomaly_patch.shape[0]))
                    map_few = anomaly_patch.reshape(1, 1, h, h)
                    map_few = F.interpolate(map_few, size=args.img_size, mode='bilinear')[0].cpu().numpy()
                    few_maps.append(map_few)
                    del mem_layer, mem_norm, cos, anomaly
                    torch.cuda.empty_cache()
                
                seg_scores_few.append(np.sum(few_maps, axis=0))
                
                zero_maps = []
                for raw_tokens in seg_tokens:
                    proj = vision_proj(raw_tokens)
                    proj = proj / proj.norm(dim=-1, keepdim=True)
                    logits = (100.0 * proj @ text_features).unsqueeze(0)
                    B, L, C = logits.shape
                    H = int(np.sqrt(L))
                    logits = F.interpolate(logits.permute(0, 2, 1).view(1, 2, H, H),
                                          size=args.img_size, mode='bilinear')
                    map_zero = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    zero_maps.append(map_zero)
                
                seg_scores_zero.append(np.sum(zero_maps, axis=0))
            
            else:
                # Detection evaluation
                few_maps = []
                for idx, p in enumerate(det_tokens):
                    mem_layer = det_mem_features[idx].to(device)
                    p_norm = p / p.norm(dim=-1, keepdim=True)
                    mem_norm = mem_layer / mem_layer.norm(dim=-1, keepdim=True)
                    cos = mem_norm @ p_norm.T
                    anomaly = 1 - cos
                    anomaly_patch = torch.min(anomaly, dim=0)[0]
                    h = int(np.sqrt(anomaly_patch.shape[0]))
                    map_few = anomaly_patch.reshape(1, 1, h, h)
                    map_few = F.interpolate(map_few, size=args.img_size, mode='bilinear')[0].cpu().numpy()
                    few_maps.append(map_few)
                    del mem_layer, mem_norm, cos, anomaly
                    torch.cuda.empty_cache()
                
                det_scores_few.append(np.sum(few_maps, axis=0).mean())
                
                score_zero = 0
                for raw_tokens in det_tokens:
                    proj = vision_proj(raw_tokens)
                    proj = proj / proj.norm(dim=-1, keepdim=True)
                    logits = 100.0 * proj @ text_features
                    score_zero += torch.softmax(logits, dim=-1)[:, 1].mean()
                
                det_scores_zero.append(score_zero.cpu().item())
        
        gt_mask_list.append(mask.squeeze().cpu().numpy())
        gt_list.extend(y.cpu().numpy())
    
    gt_list = np.array(gt_list)
    gt_mask_list = np.array(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)
    
    if CLASS_INDEX[args.obj] > 0:
        seg_scores_zero = np.array(seg_scores_zero)
        seg_scores_few = np.array(seg_scores_few)
        
        seg_scores_zero = (seg_scores_zero - seg_scores_zero.min()) / (seg_scores_zero.max() - seg_scores_zero.min() + 1e-8)
        seg_scores_few = (seg_scores_few - seg_scores_few.min()) / (seg_scores_few.max() - seg_scores_few.min() + 1e-8)
        
        final_scores = 0.5 * seg_scores_zero + 0.5 * seg_scores_few
        pauc = roc_auc_score(gt_mask_list.flatten(), final_scores.flatten())
        auc = roc_auc_score(gt_list, final_scores.reshape(len(gt_list), -1).max(1))
        
        print(f"  {args.obj} pAUC: {pauc:.4f} | AUC: {auc:.4f}")
        return pauc + auc
    else:
        det_scores_zero = np.array(det_scores_zero)
        det_scores_few = np.array(det_scores_few)
        
        det_scores_zero = (det_scores_zero - det_scores_zero.min()) / (det_scores_zero.max() - det_scores_zero.min() + 1e-8)
        det_scores_few = (det_scores_few - det_scores_few.min()) / (det_scores_few.max() - det_scores_few.min() + 1e-8)
        
        final_scores = 0.5 * det_scores_zero + 0.5 * det_scores_few
        auc = roc_auc_score(gt_list, final_scores)
        
        print(f"  {args.obj} AUC: {auc:.4f}")
        return auc

if __name__ == '__main__':
    main()