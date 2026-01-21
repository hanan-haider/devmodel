#%%writefile /kaggle/working/devmodel/train_fewori.py
import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from dataset.medical_few import MedDataset
from biomedclip.clip import create_model 
from biomedclip.tokenizer import tokenize
from biomedclip.adapterv4 import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_biomedclip_prompt_ensemble1
from prompt import REAL_NAME
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
    parser = argparse.ArgumentParser(description='Improved BiomedCLIP Few-Shot Training')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16',
                        help="BiomedCLIP model version")
    parser.add_argument('--pretrain', type=str, default='microsoft',
                        help="pretrained checkpoint source")
    
    # Dataset arguments
    parser.add_argument('--obj', type=str, default='Liver',
                        help="Dataset name")
    parser.add_argument('--data_path', type=str, default='./data/',
                        help="path to dataset")
    parser.add_argument('--img_size', type=int, default=224, 
                        help="BiomedCLIP trained with 224x224 resolution")
    parser.add_argument('--shot', type=int, default=4,
                        help="Number of few-shot examples")
    parser.add_argument('--iterate', type=int, default=0,
                        help="Iteration number for multiple runs")
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for testing (use 1 for training)")
    parser.add_argument('--epoch', type=int, default=25,
                        help="Maximum training epochs")
    parser.add_argument('--learning_rate', type=float, default=0.00005,
                        help="Base learning rate (OPTIMIZED: 0.00005)")
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help="L2 regularization weight decay")
    
    # Scheduler arguments
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help="Number of warmup epochs")
    parser.add_argument('--min_lr', type=float, default=1e-7,
                        help="Minimum learning rate for cosine annealing")
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=10,
                        help="Early stopping patience (epochs without improvement)")
    
    # Model architecture
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9, 12],
                        help="Transformer layer indices for adapters")
    parser.add_argument('--bottleneck', type=int, default=256,
                        help="Adapter bottleneck dimension")
    parser.add_argument('--dropout', type=float, default=0.15,
                        help="Dropout rate for adapters")
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=111,
                        help="Random seed")
    
    # Checkpointing
    parser.add_argument('--save_model', type=int, default=1,
                        help="Whether to save model checkpoints")
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/',
                        help="Path to save checkpoints")
    
    args, _ = parser.parse_known_args()
    
    # Print configuration
    print("\n" + "="*70)
    print("🚀 IMPROVED FEW-SHOT BIOMEDCLIP TRAINING")
    print("="*70)
    print("\n📋 Configuration:")
    for arg in vars(args):
        print(f"  {arg:20s}: {getattr(args, arg)}")
        global_vars[arg] = getattr(args, arg)
    
    # Set seed
    setup_seed(args.seed)
    print(f"\n🎲 Seed set to: {args.seed}")
    
    # Load BiomedCLIP model
    print("\n🔧 Loading BiomedCLIP model...")
    clip_model = create_model(
        model_name=args.model_name, 
        img_size=args.img_size, 
        device=device, 
        pretrained=args.pretrain, 
        require_pretrained=True
    )
    clip_model.eval()
    
    # Build adapter model
    print("🔧 Building adapter model...")
    model = CLIP_Inplanted(
        clip_model=clip_model, 
        features=args.features_list,
        bottleneck=args.bottleneck,
        dropout=args.dropout
    ).to(device)
    
    # Freeze backbone, unfreeze adapters
    for p in model.parameters():
        p.requires_grad = False
    
    for p in model.seg_adapters.parameters():
        p.requires_grad = True
    for p in model.det_adapters.parameters():
        p.requires_grad = True
    
    for p in [model.alpha_backbone, model.alpha_seg, model.alpha_det]:
        p.requires_grad = True
    
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"✅ Trainable parameters: {len(trainable)}")
    print(f"   Examples: {trainable[:8]}")
    
    # ============================================================================
    # IMPROVED OPTIMIZER: Single optimizer with parameter groups
    # ============================================================================
    # REPLACE WITH:
    optimizer = AdamW([
        {'params': model.seg_adapters.parameters(), 'lr': args.learning_rate, 'name': 'seg_adapters'},
        {'params': model.det_adapters.parameters(), 'lr': args.learning_rate, 'name': 'det_adapters'},
        {'params': [model.alpha_backbone, model.alpha_seg, model.alpha_det], 
         'lr': args.learning_rate * 1.0, 'name': 'alphas'},  # ← SAME AS BASE LR!
], weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    # ============================================================================
    # IMPROVED SCHEDULER: Longer warmup + cosine annealing
    # ============================================================================
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        total_iters=args.warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epoch - args.warmup_epochs, 
        eta_min=args.min_lr
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[args.warmup_epochs]
    )
    
    # Load datasets
    print("\n📂 Loading datasets...")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        **kwargs
    )
    
    # Few-shot augmentation
    print("🔄 Performing few-shot augmentation...")
    augment_abnorm_img, augment_abnorm_mask = augment(
        test_dataset.fewshot_abnorm_img, 
        test_dataset.fewshot_abnorm_mask
    )
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)
    
    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
    augment_fewshot_label = torch.cat([
        torch.Tensor([1] * len(augment_abnorm_img)), 
        torch.Tensor([0] * len(augment_normal_img))
    ], dim=0)
    
    train_dataset = torch.utils.data.TensorDataset(
        augment_fewshot_img, 
        augment_fewshot_mask, 
        augment_fewshot_label
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=1,  # FIXED: batch_size=1 for stability
        shuffle=True, 
        **kwargs
    )
    
    # Memory bank (normal samples only)
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(
        support_dataset, 
        batch_size=1, 
        shuffle=False, 
        **kwargs
    )
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Testing samples: {len(test_dataset)}")
    print(f"✅ Support samples: {len(support_dataset)}")
    
    # Loss functions
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()
    
    # Text features
    print("\n📝 Encoding text prompts...")
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble1(
            clip_model, 
            REAL_NAME[args.obj], 
            device
        )
    print(f"✅ Text features shape: {text_features.shape}")
    
    # Early stopping variables
    best_result = 0
    best_epoch = 0
    patience_counter = 0
    
    print("\n" + "="*70)
    print("🎯 STARTING TRAINING")
    print("="*70)
    
    for epoch in range(args.epoch):
        print(f'\n📍 Epoch {epoch}/{args.epoch-1}:')
        
        model.train()
        loss_list = []
        
        for (image, gt, label) in train_loader:
            image = image.to(device)
            
            with torch.cuda.amp.autocast():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]
                
                # Detection loss
                det_loss = 0
                image_label = label.to(device)
                
                for layer in range(len(det_patch_tokens)):
                    raw_tokens = det_patch_tokens[layer]
                    projected_tokens = model.visual_proj(raw_tokens)
                    projected_tokens = projected_tokens / projected_tokens.norm(dim=-1, keepdim=True)
                    
                    anomaly_map = (60 * projected_tokens @ text_features).unsqueeze(0)   
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    det_loss += loss_bce(anomaly_score, image_label)
                
                # Segmentation loss (if applicable)
                if CLASS_INDEX[args.obj] > 0:
                    seg_loss = 0
                    mask = gt.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    
                    for layer in range(len(seg_patch_tokens)):
                        raw_tokens = seg_patch_tokens[layer]
                        projected_tokens = model.visual_proj(raw_tokens)
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
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)
                    
                    loss = seg_loss + det_loss
                else:
                    loss = det_loss
                
                loss.requires_grad_(True)
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_list.append(loss.item())
        
        # Epoch statistics
        avg_loss = np.mean(loss_list)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"  📉 Loss: {avg_loss:.6f}")
        print(f"  📊 LR: {current_lr:.8f}")
        
        # Print alpha statistics
        with torch.no_grad():
            alpha_backbone_mean = model.alpha_backbone.mean().item()
            alpha_seg_mean = model.alpha_seg.mean().item()
            alpha_det_mean = model.alpha_det.mean().item()
            print(f"  🎚️  Alpha weights (mean): "
                  f"Backbone={alpha_backbone_mean:.3f}, "
                  f"Seg={alpha_seg_mean:.3f}, "
                  f"Det={alpha_det_mean:.3f}")
        
        scheduler.step()
        
        # Build memory bank
        model.eval()
        seg_features = []
        det_features = []
        
        with torch.no_grad():
            for image in support_loader:
                image = image[0].to(device)
                with torch.cuda.amp.autocast():
                    _, seg_patch_tokens, det_patch_tokens = model(image)
                    seg_patch_tokens = [p[0].contiguous() for p in seg_patch_tokens]
                    det_patch_tokens = [p[0].contiguous() for p in det_patch_tokens]
                    seg_features.append(seg_patch_tokens)
                    det_features.append(det_patch_tokens)
        
        seg_mem_features = [
            torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) 
            for i in range(len(seg_features[0]))
        ]
        det_mem_features = [
            torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) 
            for i in range(len(det_features[0]))
        ]
        
        # Validation
        result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)
        
        print(f"  🎯 Validation Score: {result:.4f}")
        
        # Check for improvement
        if result > best_result:
            best_result = result
            best_epoch = epoch
            patience_counter = 0
            print(f"  ✅ Best result! Improved by {result - best_result:.4f}")
            
            # Save checkpoint
            if args.save_model == 1:
                os.makedirs(args.save_path, exist_ok=True)
                ckp_path = os.path.join(args.save_path, f'{args.obj}.pth')
                torch.save({
                    'epoch': epoch,
                    'seg_adapters': model.seg_adapters.state_dict(),
                    'det_adapters': model.det_adapters.state_dict(),
                    'alpha_backbone': model.alpha_backbone.data,
                    'alpha_seg': model.alpha_seg.data,
                    'alpha_det': model.alpha_det.data,
                    'best_result': best_result,
                    'optimizer': optimizer.state_dict(),
                }, ckp_path)
                print(f"  💾 Checkpoint saved: {ckp_path}")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{args.patience})")
            
            # Early stopping check
            if patience_counter >= args.patience:
                print(f"\n🛑 Early stopping triggered!")
                print(f"   Best result: {best_result:.4f} at epoch {best_epoch}")
                break
    
    # Training complete
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"📊 Best Validation Score: {best_result:.4f}")
    print(f"📍 Best Epoch: {best_epoch}")
    print(f"💾 Model saved to: {args.save_path}")
    print("="*70 + "\n")


def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    """Evaluation function with improved score computation"""
    gt_list = []
    gt_mask_list = []
    det_image_scores_zero = []
    det_image_scores_few = []
    seg_score_map_zero = []
    seg_score_map_few = []
    
    model.eval()
    
    with torch.no_grad():
        for (image, y, mask) in tqdm(test_loader, desc="  Evaluating", leave=False):
            image = image.to(device)
            mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
            
            batch_size = image.shape[0]
            
            for i in range(batch_size):
                single_image = image[i:i+1]
                single_y = y[i]
                single_mask = mask[i]
                
                with torch.cuda.amp.autocast():
                    _, seg_patch_tokens, det_patch_tokens = model(single_image)
                    seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                    det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]
                    
                    if CLASS_INDEX[args.obj] > 0:
                        # Few-shot segmentation
                        anomaly_maps_few_shot = []
                        for idx, p in enumerate(seg_patch_tokens):
                            cos = cos_sim(seg_mem_features[idx], p)
                            height = int(np.sqrt(cos.shape[1]))
                            anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                            anomaly_map_few_shot = F.interpolate(
                                torch.tensor(anomaly_map_few_shot),
                                size=args.img_size, 
                                mode='bilinear', 
                                align_corners=True
                            )
                            anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                        score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                        seg_score_map_few.append(score_map_few)
                        
                        # Zero-shot segmentation
                        anomaly_maps = []
                        for layer in range(len(seg_patch_tokens)):
                            raw_tokens = seg_patch_tokens[layer]
                            projected_tokens = model.visual_proj(raw_tokens)
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
                        # Few-shot detection
                        anomaly_maps_few_shot = []
                        for idx, p in enumerate(det_patch_tokens):
                            cos = cos_sim(det_mem_features[idx], p)
                            height = int(np.sqrt(cos.shape[1]))
                            anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                            anomaly_map_few_shot = F.interpolate(
                                torch.tensor(anomaly_map_few_shot),
                                size=args.img_size, 
                                mode='bilinear', 
                                align_corners=True
                            )
                            anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                        anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                        score_few_det = anomaly_map_few_shot.mean()
                        det_image_scores_few.append(score_few_det)
                        
                        # Zero-shot detection
                        anomaly_score = 0
                        for layer in range(len(det_patch_tokens)):
                            raw_tokens = det_patch_tokens[layer]
                            projected_tokens = model.visual_proj(raw_tokens)
                            projected_tokens = projected_tokens / projected_tokens.norm(dim=-1, keepdim=True)
                            anomaly_map = (100.0 * projected_tokens @ text_features).unsqueeze(0)
                            anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                            anomaly_score += anomaly_map.mean()
                        det_image_scores_zero.append(anomaly_score.cpu().numpy())
                    
                    gt_mask_list.append(single_mask.cpu().detach().numpy())
                    gt_list.append(single_y.cpu().detach().numpy())
    
    # Aggregate results
    gt_list = np.array(gt_list)
    gt_mask_list = np.array(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)
    
    if CLASS_INDEX[args.obj] > 0:
        # Segmentation metrics
        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)
        
        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min() + 1e-8)
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min() + 1e-8)
        
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        
        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        
        print(f'    📊 {args.obj} pAUC: {seg_roc_auc:.4f}, Image-AUC: {roc_auc_im:.4f}')
        
        return (seg_roc_auc + roc_auc_im) / 2.0
    
    else:
        # Detection metrics
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)
        
        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min() + 1e-8)
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min() + 1e-8)
        
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        
        print(f'    📊 {args.obj} AUC: {img_roc_auc_det:.4f}')
        
        return img_roc_auc_det


if __name__ == '__main__':
    main()
