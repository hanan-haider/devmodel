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
from biomedclip.clip import create_model, _MODEL_CKPT_PATHS
from biomedclip.tokenizer import tokenize
from biomedclip.adapterv5_improved import CLIP_Inplanted  # ✅ Use improved adapter
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_biomedclip_prompt_ensemble1
from prompt import REAL_NAME
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {
    'Brain': 3, 'Liver': 2, 'Retina_RESC': 1,
    'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3
}

global_vars = {}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='BiomedCLIP Few-Shot Training')
    
    # Model configs
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    
    # Dataset configs
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--img_size', type=int, default=224)
    
    # Training configs
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=30)  # ✅ Reduced from 50 (overfitting risk)
    parser.add_argument('--learning_rate', type=float, default=0.0005)  # ✅ Reduced from 0.001
    parser.add_argument('--weight_decay', type=float, default=1e-3)  # ✅ Increased regularization
    parser.add_argument('--dropout', type=float, default=0.2)  # ✅ Increased from 0.1
    
    # Adapter configs
    parser.add_argument('--features_list', type=int, nargs="+", default=None,
                        help="Auto-select based on task if None")
    parser.add_argument('--bottleneck', type=int, default=128)  # ✅ Reduced from 256
    parser.add_argument('--init_scale', type=float, default=1e-3)
    parser.add_argument('--normalize_alphas', action='store_true')  # ✅ NEW
    
    # Few-shot configs
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    
    # Save configs
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    
    # ✅ NEW: Early stopping
    parser.add_argument('--patience', type=int, default=10,
                        help="Early stopping patience")
    
    args, _ = parser.parse_known_args()
    
    # Print args
    print("\n" + "="*60)
    print("BIOMEDCLIP FEW-SHOT TRAINING")
    print("="*60)
    for arg in vars(args):
        print(f"  {arg:20s}: {getattr(args, arg)}")
        global_vars[arg] = getattr(args, arg)
    print("="*60 + "\n")
    
    setup_seed(args.seed)
    
    # ✅ Auto-select layers based on task
    if args.features_list is None:
        if CLASS_INDEX[args.obj] > 0:  # Segmentation tasks
            args.features_list = [3, 6, 9, 12]
            print(f"✓ Auto-selected layers for {args.obj} (SEG+DET): {args.features_list}")
        else:  # Detection-only
            args.features_list = [6, 9, 12]
            print(f"✓ Auto-selected layers for {args.obj} (DET-only): {args.features_list}")
    
    # Load CLIP model
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained=args.pretrain,
        require_pretrained=True
    )
    clip_model.eval()
    
    # ✅ Build improved model
    model = CLIP_Inplanted(
        clip_model=clip_model,
        features=args.features_list,
        bottleneck=args.bottleneck,
        dropout=args.dropout,
        init_scale=args.init_scale,
        normalize_alphas=args.normalize_alphas,
    ).to(device)
    
    # Freeze/unfreeze
    for p in model.parameters():
        p.requires_grad = False
    
    for p in model.seg_adapters.parameters():
        p.requires_grad = True
    for p in model.det_adapters.parameters():
        p.requires_grad = True
    for p in [model.alpha_backbone, model.alpha_seg, model.alpha_det]:
        p.requires_grad = True
    
    # ✅ Uncertainty weights
    log_var_seg = torch.zeros(1, requires_grad=True, device=device)
    log_var_det = torch.zeros(1, requires_grad=True, device=device)
    
    # ✅ SINGLE optimizer (cleaner than two separate ones)
    optimizer = AdamW(
        [
            {"params": model.seg_adapters.parameters()},
            {"params": model.det_adapters.parameters()},
            {"params": [model.alpha_backbone, model.alpha_seg, model.alpha_det]},
            {"params": [log_var_seg, log_var_det]},
        ],
        lr=args.learning_rate,
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay,
    )
    
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"\n✓ Trainable params: {len(trainable)}")
    print(f"  Examples: {trainable[:5]}\n")
    
    # ✅ Scheduler (warmup + cosine)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epoch-5, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
    
    # Load dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )
    
    # Few-shot augmentation
    augment_abnorm_img, augment_abnorm_mask = augment(
        test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask
    )
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)
    
    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
    augment_fewshot_label = torch.cat([
        torch.ones(len(augment_abnorm_img)),
        torch.zeros(len(augment_normal_img))
    ], dim=0)
    
    train_dataset = torch.utils.data.TensorDataset(
        augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, **kwargs
    )
    
    # Memory bank (normal samples only)
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(
        support_dataset, batch_size=1, shuffle=False, **kwargs  # ✅ No shuffle for consistency
    )
    
    # Losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()
    
    # Text features
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble1(
            clip_model, REAL_NAME[args.obj], device
        )
    
    print(f"✓ Text features shape: {text_features.shape}\n")
    
    # ✅ Early stopping
    best_result = 0
    patience_counter = 0
    
    # Training loop
    print("="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(args.epoch):
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
                    
                    anomaly_map = (100.0 * projected_tokens @ text_features).unsqueeze(0)
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
                            size=args.img_size, mode='bilinear', align_corners=True
                        )
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)
                    
                    # ✅ Uncertainty weighting
                    weighted_seg_loss = torch.exp(-log_var_seg) * seg_loss + log_var_seg
                    weighted_det_loss = torch.exp(-log_var_det) * det_loss + log_var_det
                    loss = weighted_seg_loss + weighted_det_loss
                    
                    # ✅ Alpha regularization (keep alphas close to init)
                    alpha_reg = model.get_alpha_regularization(reg_type='l2', lambda_reg=1e-4)
                    loss = loss + alpha_reg
                    
                else:
                    loss = torch.exp(-log_var_det) * det_loss + log_var_det
                
                optimizer.zero_grad()
                loss.backward()
                
                # ✅ Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                loss_list.append(loss.item())
        
        avg_loss = np.mean(loss_list)
        scheduler.step()
        
        print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # ✅ Build memory bank (each epoch for consistency)
        model.eval()
        seg_features = []
        det_features = []
        for image in support_loader:
            image = image[0].to(device)
            with torch.no_grad():
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
        
        # Test
        result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)
        
        # ✅ Save best + early stopping
        if result > best_result:
            best_result = result
            patience_counter = 0
            print(f"  ✓ New best: {best_result:.4f}")
            
            if args.save_model == 1:
                os.makedirs(args.save_path, exist_ok=True)
                ckp_path = os.path.join(args.save_path, f'{args.obj}_best.pth')
                torch.save({
                    "seg_adapters": model.seg_adapters.state_dict(),
                    "det_adapters": model.det_adapters.state_dict(),
                    "alpha_backbone": model.alpha_backbone.detach().cpu(),
                    "alpha_seg": model.alpha_seg.detach().cpu(),
                    "alpha_det": model.alpha_det.detach().cpu(),
                    "log_var_seg": log_var_seg.detach().cpu(),
                    "log_var_det": log_var_det.detach().cpu(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_result": best_result,
                }, ckp_path)
                print(f"  ✓ Saved to {ckp_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
        
        # ✅ Early stopping
        if patience_counter >= args.patience:
            print(f"\n✗ Early stopping at epoch {epoch}")
            break
    
    print("\n" + "="*60)
    print(f"TRAINING COMPLETE | Best Result: {best_result:.4f}")
    print("="*60)
    
    # ✅ Print learned alphas
    model.get_alpha_summary()


def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    """Same as your original test function"""
    gt_list = []
    gt_mask_list = []
    det_image_scores_zero = []
    det_image_scores_few = []
    seg_score_map_zero = []
    seg_score_map_few = []
    
    for (image, y, mask) in tqdm(test_loader, desc="Testing"):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
        
        batch_size = image.shape[0]
        for i in range(batch_size):
            single_image = image[i:i+1]
            single_y = y[i]
            single_mask = mask[i]
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                _, seg_patch_tokens, det_patch_tokens = model(single_image)
                seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]
                
                if CLASS_INDEX[args.obj] > 0:
                    # Segmentation head (few-shot)
                    anomaly_maps_few_shot = []
                    for idx, p in enumerate(seg_patch_tokens):
                        cos = cos_sim(seg_mem_features[idx], p)
                        height = int(np.sqrt(cos.shape[1]))
                        anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                        anomaly_map_few_shot = F.interpolate(
                            torch.tensor(anomaly_map_few_shot),
                            size=args.img_size, mode='bilinear', align_corners=True
                        )
                        anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                    score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                    seg_score_map_few.append(score_map_few)
                    
                    # Segmentation head (zero-shot)
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
                            size=args.img_size, mode='bilinear', align_corners=True
                        )
                        anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                        anomaly_maps.append(anomaly_map.cpu().numpy())
                    score_map_zero = np.sum(anomaly_maps, axis=0)
                    seg_score_map_zero.append(score_map_zero)
                
                else:
                    # Detection head (few-shot)
                    anomaly_maps_few_shot = []
                    for idx, p in enumerate(det_patch_tokens):
                        cos = cos_sim(det_mem_features[idx], p)
                        height = int(np.sqrt(cos.shape[1]))
                        anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                        anomaly_map_few_shot = F.interpolate(
                            torch.tensor(anomaly_map_few_shot),
                            size=args.img_size, mode='bilinear', align_corners=True
                        )
                        anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                    anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                    score_few_det = anomaly_map_few_shot.mean()
                    det_image_scores_few.append(score_few_det)
                    
                    # Detection head (zero-shot)
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
    
    gt_list = np.array(gt_list)
    gt_mask_list = np.array(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)
    
    if CLASS_INDEX[args.obj] > 0:
        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)
        
        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min())
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min())
        
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'    pAUC: {seg_roc_auc:.4f}', end=' | ')
        
        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'iAUC: {roc_auc_im:.4f}')
        
        return seg_roc_auc + roc_auc_im
    
    else:
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)
        
        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min())
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min())
        
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'    AUC: {img_roc_auc_det:.4f}')
        
        return img_roc_auc_det


if __name__ == '__main__':
    main()
