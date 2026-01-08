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
from biomedclip.adapterv6 import CLIP_Inplanted_V2  # Updated import
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_biomedclip_prompt_ensemble1
from prompt import REAL_NAME
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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


class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def main():
    parser = argparse.ArgumentParser(description='Enhanced BiomedCLIP Training')
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')    
    parser.add_argument('--pretrain', type=str, default='microsoft')
    parser.add_argument('--obj', type=str, default='Histopathology')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--features_list', type=int, nargs="+", default=[3, 6, 9, 12])    
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    parser.add_argument('--bottleneck', type=int, default=384)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    args, _ = parser.parse_known_args()

    print("\n" + "="*60)
    print("ENHANCED BIOMEDCLIP TRAINING CONFIGURATION")
    print("="*60)
    for arg in vars(args):
        print(f"  {arg:20s}: {getattr(args, arg)}")
    print("="*60 + "\n")
    
    setup_seed(args.seed)
    
    # Load BioMedCLIP
    clip_model = create_model(
        model_name=args.model_name, 
        img_size=args.img_size, 
        device=device, 
        pretrained=args.pretrain, 
        require_pretrained=True
    )
    clip_model.eval()

    # Enhanced model
    model = CLIP_Inplanted_V2(
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
    for p in model.blend_logits.parameters():
        p.requires_grad = True
    for p in model.feature_fusion.parameters():
        p.requires_grad = True
    model.layer_weights.requires_grad = True
    model.temperature.requires_grad = True

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Trainable parameters: {trainable_params:,}\n")

    # Separate optimizers with different learning rates
    seg_params = list(model.seg_adapters.parameters()) + \
                 [p for n, p in model.named_parameters() if 'blend_logits' in n or 'layer_weights' in n]
    det_params = list(model.det_adapters.parameters()) + \
                 [p for n, p in model.named_parameters() if 'feature_fusion' in n or 'temperature' in n]
    
    seg_optimizer = AdamW(seg_params, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    det_optimizer = AdamW(det_params, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)

    # OneCycleLR for better convergence
    steps_per_epoch = 100  # Approximate
    seg_scheduler = OneCycleLR(
        seg_optimizer, 
        max_lr=args.learning_rate,
        epochs=args.epoch,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    det_scheduler = OneCycleLR(
        det_optimizer,
        max_lr=args.learning_rate,
        epochs=args.epoch,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    # EMA for stable predictions
    ema = EMA(model, decay=args.ema_decay)

    # Mixed precision training
    scaler = GradScaler()

    # Load datasets
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    # Enhanced data augmentation
    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
    augment_fewshot_label = torch.cat([torch.ones(len(augment_abnorm_img)), torch.zeros(len(augment_normal_img))], dim=0)

    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    # Memory bank
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)

    # Enhanced losses
    loss_focal = FocalLoss(alpha=0.25, gamma=2.0)
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss(label_smoothing=args.label_smoothing)

    # Text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble1(clip_model, REAL_NAME[args.obj], device)

    best_result = 0
    patience = 10
    patience_counter = 0

    # Uncertainty weighting
    log_var_seg = torch.zeros(1, requires_grad=True, device=device)
    log_var_det = torch.zeros(1, requires_grad=True, device=device)
    seg_optimizer.add_param_group({'params': [log_var_seg], 'lr': args.learning_rate * 0.1})
    det_optimizer.add_param_group({'params': [log_var_det], 'lr': args.learning_rate * 0.1})

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")

    for epoch in range(args.epoch):
        model.train()
        loss_list = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epoch}')
        
        for batch_idx, (image, gt, label) in enumerate(progress_bar):
            image = image.to(device)
            
            # Mixup augmentation
            if args.mixup_alpha > 0 and random.random() < 0.5:
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                index = torch.randperm(image.size(0))
                image = lam * image + (1 - lam) * image[index]
                label = lam * label + (1 - lam) * label[index]

            with autocast():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]
                
                # Detection loss with temperature scaling
                det_loss = 0
                image_label = label.to(device)
                for layer in range(len(det_patch_tokens)):
                    raw_tokens = det_patch_tokens[layer]
                    projected_tokens = model.visual_proj(raw_tokens)
                    projected_tokens = projected_tokens / projected_tokens.norm(dim=-1, keepdim=True)
                    
                    # Use learnable temperature
                    anomaly_map = (projected_tokens @ text_features / model.temperature).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    det_loss += loss_bce(anomaly_score, image_label)

                if CLASS_INDEX[args.obj] > 0:
                    # Segmentation loss
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

                    # Uncertainty-weighted multi-task loss
                    precision_seg = torch.exp(-log_var_seg)
                    precision_det = torch.exp(-log_var_det)
                    loss = precision_seg * seg_loss + precision_det * det_loss + log_var_seg + log_var_det
                    
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                else:
                    precision_det = torch.exp(-log_var_det)
                    loss = precision_det * det_loss + log_var_det
                    det_optimizer.zero_grad()

            # Gradient scaling and clipping
            scaler.scale(loss).backward()
            
            if CLASS_INDEX[args.obj] > 0:
                scaler.unscale_(seg_optimizer)
                scaler.unscale_(det_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(seg_optimizer)
                scaler.step(det_optimizer)
                seg_scheduler.step()
                det_scheduler.step()
            else:
                scaler.unscale_(det_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(det_optimizer)
                det_scheduler.step()
            
            scaler.update()
            
            # Update EMA
            ema.update()
            
            loss_list.append(loss.item())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = np.mean(loss_list)
        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        print(f"  Seg σ²: {torch.exp(log_var_seg).item():.4f}, Det σ²: {torch.exp(log_var_det).item():.4f}")
        
        # Show adaptation weights
        if epoch % 5 == 0:
            weights = model.get_adaptation_weights()
            print(f"  Adaptation weights (layer 0): {weights[0]}")

        # Build memory bank
        seg_features, det_features = [], []
        for image in support_loader:
            image = image[0].to(device)
            with torch.no_grad():
                _, seg_tokens, det_tokens = model(image)
                seg_features.append([p[0].contiguous() for p in seg_tokens])
                det_features.append([p[0].contiguous() for p in det_tokens])
        
        seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) 
                           for i in range(len(seg_features[0]))]
        det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) 
                           for i in range(len(det_features[0]))]

        # Evaluate with EMA weights
        ema.apply_shadow()
        result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)
        ema.restore()

        if result > best_result:
            best_result = result
            patience_counter = 0
            print(f"✓ New best result: {best_result:.4f}\n")
            
            if args.save_model == 1:
                ckp_path = os.path.join(args.save_path, f'{args.obj}_best.pth')
                os.makedirs(args.save_path, exist_ok=True)
                torch.save({
                    'seg_adapters': model.seg_adapters.state_dict(),
                    'det_adapters': model.det_adapters.state_dict(),
                    'blend_logits': model.blend_logits.state_dict(),
                    'feature_fusion': model.feature_fusion.state_dict(),
                    'ema_shadow': ema.shadow,
                    'epoch': epoch,
                    'best_result': best_result
                }, ckp_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print("\n" + "="*60)
    print(f"TRAINING COMPLETE - Best Result: {best_result:.4f}")
    print("="*60)


def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    model.eval()
    gt_list, gt_mask_list = [], []
    det_scores_zero, det_scores_few = [], []
    seg_scores_zero, seg_scores_few = [], []

    with torch.no_grad():
        for (image, y, mask) in tqdm(test_loader, desc='Testing'):
            image = image.to(device)
            mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
            batch_size = image.shape[0]
            
            for i in range(batch_size):
                single_image = image[i:i+1]
                
                with autocast():
                    _, seg_tokens, det_tokens = model(single_image)
                    seg_tokens = [p[0, 1:, :] for p in seg_tokens]
                    det_tokens = [p[0, 1:, :] for p in det_tokens]

                    if CLASS_INDEX[args.obj] > 0:
                        # Segmentation evaluation
                        maps_few, maps_zero = [], []
                        
                        for idx, p in enumerate(seg_tokens):
                            # Few-shot
                            cos = cos_sim(seg_mem_features[idx], p)
                            h = int(np.sqrt(cos.shape[1]))
                            map_few = torch.min((1 - cos), 0)[0].reshape(1, 1, h, h)
                            map_few = F.interpolate(map_few, size=args.img_size, mode='bilinear', align_corners=True)
                            maps_few.append(map_few[0].cpu().numpy())
                            
                            # Zero-shot
                            raw = seg_tokens[idx]
                            proj = model.visual_proj(raw)
                            proj = proj / proj.norm(dim=-1, keepdim=True)
                            amap = (100.0 * proj @ text_features).unsqueeze(0)
                            B, L, C = amap.shape
                            H = int(np.sqrt(L))
                            amap = F.interpolate(amap.permute(0, 2, 1).view(B, 2, H, H),
                                               size=args.img_size, mode='bilinear', align_corners=True)
                            amap = torch.softmax(amap, dim=1)[:, 1, :, :]
                            maps_zero.append(amap.cpu().numpy())
                        
                        seg_scores_few.append(np.sum(maps_few, axis=0))
                        seg_scores_zero.append(np.sum(maps_zero, axis=0))
                    else:
                        # Detection evaluation
                        score_zero = 0
                        maps_few = []
                        
                        for idx, p in enumerate(det_tokens):
                            # Few-shot
                            cos = cos_sim(det_mem_features[idx], p)
                            h = int(np.sqrt(cos.shape[1]))
                            map_few = torch.min((1 - cos), 0)[0].reshape(1, 1, h, h)
                            map_few = F.interpolate(map_few, size=args.img_size, mode='bilinear', align_corners=True)
                            maps_few.append(map_few[0].cpu().numpy())
                            
                            # Zero-shot
                            proj = model.visual_proj(p)
                            proj = proj / proj.norm(dim=-1, keepdim=True)
                            amap = (100.0 * proj @ text_features).unsqueeze(0)
                            amap = torch.softmax(amap, dim=-1)[:, :, 1]
                            score_zero += amap.mean()
                        
                        det_scores_few.append(np.sum(maps_few, axis=0).mean())
                        det_scores_zero.append(score_zero.cpu().numpy())

                gt_mask_list.append(mask[i].cpu().numpy())
                gt_list.append(y[i].cpu().numpy())

    gt_list = np.array(gt_list)
    gt_mask_list = np.array(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)

    if CLASS_INDEX[args.obj] > 0:
        seg_scores_zero = np.array(seg_scores_zero)
        seg_scores_few = np.array(seg_scores_few)
        
        # Normalize
        seg_scores_zero = (seg_scores_zero - seg_scores_zero.min()) / (seg_scores_zero.max() - seg_scores_zero.min() + 1e-8)
        seg_scores_few = (seg_scores_few - seg_scores_few.min()) / (seg_scores_few.max() - seg_scores_few.min() + 1e-8)
        
        # Weighted combination (can be tuned)
        segment_scores = 0.6 * seg_scores_zero + 0.4 * seg_scores_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        
        segment_scores_flat = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flat, axis=1))
        
        print(f'{args.obj} | pAUC: {seg_roc_auc:.4f} | iAUC: {roc_auc_im:.4f}')
        return seg_roc_auc + roc_auc_im
    else:
        det_scores_zero = np.array(det_scores_zero)
        det_scores_few = np.array(det_scores_few)
        
        det_scores_zero = (det_scores_zero - det_scores_zero.min()) / (det_scores_zero.max() - det_scores_zero.min() + 1e-8)
        det_scores_few = (det_scores_few - det_scores_few.min()) / (det_scores_few.max() - det_scores_few.min() + 1e-8)
        
        image_scores = 0.6 * det_scores_zero + 0.4 * det_scores_few
        img_roc_auc = roc_auc_score(gt_list, image_scores)
        
        print(f'{args.obj} | AUC: {img_roc_auc:.4f}')
        return img_roc_auc


if __name__ == '__main__':
    main()