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
from biomedclip.adapterv7 import CLIP_Inplanted_Balanced
from PIL import Image
from sklearn.metrics import roc_auc_score
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_biomedclip_prompt_ensemble1
from prompt import REAL_NAME
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 
               'Chest':-2, 'Histopathology':-3}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
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
    parser = argparse.ArgumentParser(description='Balanced BiomedCLIP Training')
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    parser.add_argument('--obj', type=str, default='Liver')
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
    parser.add_argument('--bottleneck', type=int, default=320)
    parser.add_argument('--dropout', type=float, default=0.12)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--gradient_clip', type=float, default=1.0)

    args, _ = parser.parse_known_args()

    print("\n" + "="*60)
    print("BALANCED BIOMEDCLIP TRAINING (SEG + DET)")
    print("="*60)
    for arg in vars(args):
        print(f"  {arg:20s}: {getattr(args, arg)}")
    print("="*60 + "\n")

    setup_seed(args.seed)

    # Load model
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained=args.pretrain,
        require_pretrained=True
    )
    clip_model.eval()

    # Balanced model
    model = CLIP_Inplanted_Balanced(
        clip_model=clip_model,
        features=args.features_list,
        bottleneck=args.bottleneck,
        dropout=args.dropout
    ).to(device)

    # Setup trainable parameters
    for p in model.parameters():
        p.requires_grad = False

    trainable_modules = [
        model.seg_adapters,
        model.det_adapters,
        model.detection_head,
        model.blend_logits,
        model.feature_enhance
    ]

    for module in trainable_modules:
        for p in module.parameters():
            p.requires_grad = True

    model.temperatures.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Trainable parameters: {trainable_params:,}\n")

    # Load datasets
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # Augmentation
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)

    # Uncertainty parameters
    log_var_seg = torch.zeros(1, requires_grad=True, device=device)
    log_var_det = torch.zeros(1, requires_grad=True, device=device)

    # Optimizer with differential learning rates
    seg_params = [
        {'params': list(model.seg_adapters.parameters()), 'lr': args.learning_rate},
        {'params': [p for n, p in model.named_parameters() if 'blend_logits' in n], 
         'lr': args.learning_rate * 0.5},
        {'params': [log_var_seg], 'lr': args.learning_rate * 0.1}
    ]

    det_params = [
        {'params': list(model.det_adapters.parameters()), 'lr': args.learning_rate * 1.2},  # Higher LR for det
        {'params': list(model.detection_head.parameters()), 'lr': args.learning_rate * 1.5},  # Even higher for head
        {'params': list(model.feature_enhance.parameters()), 'lr': args.learning_rate * 0.8},
        {'params': [model.temperatures], 'lr': args.learning_rate * 0.3},
        {'params': [log_var_det], 'lr': args.learning_rate * 0.1}
    ]

    seg_optimizer = AdamW(seg_params, betas=(0.9, 0.999), weight_decay=1e-4)
    det_optimizer = AdamW(det_params, betas=(0.9, 0.999), weight_decay=1e-4)

    # Schedulers
    seg_scheduler = CosineAnnealingWarmRestarts(seg_optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    det_scheduler = CosineAnnealingWarmRestarts(det_optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # EMA
    ema = EMA(model, decay=args.ema_decay)

    # Mixed precision
    scaler = GradScaler()

    # Losses
    loss_focal = FocalLoss(alpha=0.25, gamma=2.0)
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    # Text features
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble1(
            clip_model, REAL_NAME[args.obj], device
        )

    best_result = 0
    patience = 15
    patience_counter = 0

    print("="*60)
    print("TRAINING START")
    print("="*60 + "\n")

    for epoch in range(args.epoch):
        model.train()
        loss_list = []
        seg_loss_list = []
        det_loss_list = []

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1:02d}/{args.epoch}')

        for batch_idx, (image, gt, label) in enumerate(progress_bar):
            image = image.to(device)

            with autocast():
                _, seg_patch_tokens, det_patch_tokens, det_enhanced = model(
                    image, return_detection_features=True
                )
                seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

                # Enhanced detection loss using multi-scale head
                det_loss = 0
                image_label = label.to(device)

                # Method 1: Use enhanced detection features
                if det_enhanced is not None:
                    projected_enhanced = model.visual_proj(det_enhanced)
                    projected_enhanced = projected_enhanced / projected_enhanced.norm(dim=-1, keepdim=True)
                    
                    anomaly_logits = (100.0 * projected_enhanced @ text_features)
                    anomaly_probs = torch.softmax(anomaly_logits, dim=-1)[1]
                    det_loss += 2.0 * loss_bce(anomaly_probs.unsqueeze(0), image_label)

                # Method 2: Layer-wise detection with temperature
                for layer_idx, layer_tokens in enumerate(det_patch_tokens):
                    projected = model.visual_proj(layer_tokens)
                    projected = projected / projected.norm(dim=-1, keepdim=True)

                    temp = model.temperatures[layer_idx].clamp(0.01, 0.2)
                    anomaly_map = (projected @ text_features / temp).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    det_loss += loss_bce(anomaly_score, image_label)

                det_loss = det_loss / (len(det_patch_tokens) + 1)

                if CLASS_INDEX[args.obj] > 0:
                    # Segmentation loss
                    seg_loss = 0
                    mask = gt.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

                    for layer_tokens in seg_patch_tokens:
                        projected = model.visual_proj(layer_tokens)
                        projected = projected / projected.norm(dim=-1, keepdim=True)

                        anomaly_map = (100.0 * projected @ text_features).unsqueeze(0)
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(
                            anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                            size=args.img_size, mode='bilinear', align_corners=True
                        )
                        anomaly_map = torch.softmax(anomaly_map, dim=1)

                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)

                    seg_loss = seg_loss / len(seg_patch_tokens)

                    # Balanced multi-task loss with uncertainty weighting
                    precision_seg = torch.exp(-log_var_seg)
                    precision_det = torch.exp(-log_var_det)
                    loss = precision_seg * seg_loss + precision_det * det_loss + 0.5 * (log_var_seg + log_var_det)

                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    
                    seg_loss_list.append(seg_loss.item())
                    det_loss_list.append(det_loss.item())
                else:
                    precision_det = torch.exp(-log_var_det)
                    loss = precision_det * det_loss + 0.5 * log_var_det
                    det_optimizer.zero_grad()
                    det_loss_list.append(det_loss.item())

            # Gradient updates
            scaler.scale(loss).backward()

            if CLASS_INDEX[args.obj] > 0:
                scaler.unscale_(seg_optimizer)
                scaler.unscale_(det_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(seg_optimizer)
                scaler.step(det_optimizer)
                scaler.update()
                seg_scheduler.step(epoch + batch_idx / len(train_loader))
                det_scheduler.step(epoch + batch_idx / len(train_loader))
            else:
                scaler.unscale_(det_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(det_optimizer)
                scaler.update()
                det_scheduler.step(epoch + batch_idx / len(train_loader))

            ema.update()

            loss_list.append(loss.item())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epoch} Summary")
        print(f"{'='*60}")
        print(f"  Total Loss: {np.mean(loss_list):.5f}")
        
        if CLASS_INDEX[args.obj] > 0:
            print(f"  Seg Loss:   {np.mean(seg_loss_list):.5f}  (σ²={torch.exp(log_var_seg).item():.4f})")
            print(f"  Det Loss:   {np.mean(det_loss_list):.5f}  (σ²={torch.exp(log_var_det).item():.4f})")
            print(f"  Seg LR:     {seg_optimizer.param_groups[0]['lr']:.7f}")
            print(f"  Det LR:     {det_optimizer.param_groups[0]['lr']:.7f}")
        else:
            print(f"  Det Loss:   {np.mean(det_loss_list):.5f}  (σ²={torch.exp(log_var_det).item():.4f})")
            print(f"  Det LR:     {det_optimizer.param_groups[0]['lr']:.7f}")

        # Build memory bank
        model.eval()
        seg_features, det_features = [], []
        with torch.no_grad():
            for image in support_loader:
                image = image[0].to(device)
                _, seg_tokens, det_tokens = model(image)
                seg_features.append([p[0].contiguous() for p in seg_tokens])
                det_features.append([p[0].contiguous() for p in det_tokens])

        seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0)
                           for i in range(len(seg_features[0]))]
        det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0)
                           for i in range(len(det_features[0]))]

        # Evaluate with EMA
        ema.apply_shadow()
        result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)
        ema.restore()

        if result > best_result:
            best_result = result
            patience_counter = 0
            print(f"\n🌟 NEW BEST: {best_result:.4f}\n")

            if args.save_model == 1:
                ckp_path = os.path.join(args.save_path, f'{args.obj}_balanced.pth')
                os.makedirs(args.save_path, exist_ok=True)
                torch.save({
                    'model': model.state_dict(),
                    'ema_shadow': ema.shadow,
                    'epoch': epoch,
                    'best_result': best_result
                }, ckp_path)
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping at epoch {epoch+1}")
                break

    print("\n" + "="*60)
    print(f"TRAINING COMPLETE - Best: {best_result:.4f}")
    print("="*60 + "\n")


def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    model.eval()
    gt_list, gt_mask_list = [], []
    det_scores_zero, det_scores_few = [], []
    seg_scores_zero, seg_scores_few = [], []

    with torch.no_grad():
        for (image, y, mask) in tqdm(test_loader, desc='Testing', leave=False):
            image = image.to(device)
            mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
            batch_size = image.shape[0]

            for i in range(batch_size):
                single_image = image[i:i+1]

                with autocast():
                    _, seg_tokens, det_tokens, det_enhanced = model(
                        single_image, return_detection_features=True
                    )
                    seg_tokens = [p[0, 1:, :] for p in seg_tokens]
                    det_tokens = [p[0, 1:, :] for p in det_tokens]

                    if CLASS_INDEX[args.obj] > 0:
                        # Segmentation
                        maps_few, maps_zero = [], []

                        for idx, p in enumerate(seg_tokens):
                            # Few-shot
                            cos = cos_sim(seg_mem_features[idx], p)
                            h = int(np.sqrt(cos.shape[1]))
                            map_few = torch.min((1 - cos), 0)[0].reshape(1, 1, h, h)
                            map_few = F.interpolate(map_few, size=args.img_size, mode='bilinear', align_corners=True)
                            maps_few.append(map_few[0].cpu().numpy())

                            # Zero-shot
                            proj = model.visual_proj(p)
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
                        # Enhanced detection scoring
                        score_zero = 0
                        maps_few = []

                        # Use enhanced detection head
                        if det_enhanced is not None:
                            proj_enh = model.visual_proj(det_enhanced)
                            proj_enh = proj_enh / proj_enh.norm(dim=-1, keepdim=True)
                            score_enh = torch.softmax(100.0 * proj_enh @ text_features, dim=-1)[1]
                            score_zero += 2.0 * score_enh  # Weight enhanced score more

                        # Layer-wise scores
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
                            temp = model.temperatures[idx].clamp(0.01, 0.2)
                            amap = (100.0 / temp * proj @ text_features).unsqueeze(0)
                            amap = torch.softmax(amap, dim=-1)[:, :, 1]
                            score_zero += amap.mean()

                        det_scores_few.append(np.mean([m.mean() for m in maps_few]))
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

        # Emphasize zero-shot more for detection
        image_scores = 0.7 * det_scores_zero + 0.3 * det_scores_few
        img_roc_auc = roc_auc_score(gt_list, image_scores)

        print(f'{args.obj} | AUC: {img_roc_auc:.4f}')
        return img_roc_auc


if __name__ == '__main__':
    main()