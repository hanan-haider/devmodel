import os
import argparse
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from dataset.medical_few import MedDataset
from biomedclip.clip import create_model 
from biomedclip.adapter import CLIP_Inplanted
from sklearn.metrics import roc_auc_score
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_biomedclip_prompt_ensemble1
from prompt import REAL_NAME
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.cuda.amp as amp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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


class ContrastiveLoss(nn.Module):
    """Contrastive loss for better feature discrimination"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        features = F.normalize(features, dim=-1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Mask out self-similarity
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = -mean_log_prob_pos.mean()
        
        return loss


class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
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


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for better generalization"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def main():
    parser = argparse.ArgumentParser(description='Improved BiomedCLIP Training')
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epoch', type=int, default=80)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--features_list', type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--contrastive_weight', type=float, default=0.1)
    
    args, _ = parser.parse_known_args()
    
    print("\nImproved Training Configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
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
    
    model = CLIP_Inplanted(
        clip_model=clip_model,
        features=args.features_list,
        bottleneck=256,
        dropout=0.15
    ).to(device)
    
    # Enhanced optimizers with weight decay
    seg_params = list(model.seg_adapters.parameters()) + list(model.seg_proj.parameters())
    det_params = list(model.det_adapters.parameters()) + list(model.det_proj.parameters())
    alpha_params = [model.alpha_backbone, model.alpha_seg, model.alpha_det]
    
    seg_optimizer = AdamW(seg_params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    det_optimizer = AdamW(det_params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    alpha_optimizer = AdamW(alpha_params, lr=args.learning_rate * 0.5, weight_decay=0)
    
    # Cosine annealing with warm restarts
    seg_scheduler = CosineAnnealingWarmRestarts(seg_optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    det_scheduler = CosineAnnealingWarmRestarts(det_optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    alpha_scheduler = CosineAnnealingWarmRestarts(alpha_optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    
    # EMA for stable predictions
    ema = EMA(model, decay=args.ema_decay)
    
    # Mixed precision training
    scaler = amp.GradScaler()
    
    # Load datasets
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    # Enhanced augmentation
    augment_abnorm_img, augment_abnorm_mask = augment(
        test_dataset.fewshot_abnorm_img,
        test_dataset.fewshot_abnorm_mask
    )
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)
    
    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
    augment_fewshot_label = torch.cat([
        torch.ones(len(augment_abnorm_img)),
        torch.zeros(len(augment_normal_img))
    ], dim=0)
    
    train_dataset = torch.utils.data.TensorDataset(
        augment_fewshot_img,
        augment_fewshot_mask,
        augment_fewshot_label
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, **kwargs)
    
    # Memory bank
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)
    
    # Enhanced losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = nn.BCEWithLogitsLoss()
    loss_contrastive = ContrastiveLoss(temperature=0.07)
    
    # Text features
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble1(
            clip_model,
            REAL_NAME[args.obj],
            device
        )
    
    best_result = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(args.epoch):
        print(f'\nEpoch {epoch + 1}/{args.epoch}:')
        model.train()
        
        loss_list = []
        
        for (image, gt, label) in train_loader:
            image = image.to(device)
            label = label.to(device)
            
            # Mixup augmentation
            if args.mixup_alpha > 0 and random.random() > 0.5:
                image, label_a, label_b, lam = mixup_data(image, label, args.mixup_alpha)
                use_mixup = True
            else:
                use_mixup = False
            
            with amp.autocast():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                
                total_loss = 0
                
                # Detection loss
                det_loss = 0
                det_features_list = []
                
                for layer in range(len(det_patch_tokens)):
                    raw_tokens = det_patch_tokens[layer][:, 1:, :]  # Remove CLS
                    projected_tokens = model.visual_proj(raw_tokens)
                    projected_tokens = F.normalize(projected_tokens, dim=-1)
                    
                    anomaly_map = (100.0 * projected_tokens @ text_features).unsqueeze(1)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1).squeeze()
                    
                    if use_mixup:
                        det_loss += lam * loss_bce(anomaly_score, label_a) + \
                                   (1 - lam) * loss_bce(anomaly_score, label_b)
                    else:
                        det_loss += loss_bce(anomaly_score, label)
                    
                    # Collect features for contrastive loss
                    det_features_list.append(torch.mean(projected_tokens, dim=1))
                
                total_loss += det_loss
                
                # Contrastive loss for better feature discrimination
                if len(det_features_list) > 0 and not use_mixup:
                    det_features_concat = torch.stack(det_features_list, dim=1).mean(dim=1)
                    contrastive_loss = loss_contrastive(det_features_concat, label)
                    total_loss += args.contrastive_weight * contrastive_loss
                
                # Segmentation loss (for pixel-level tasks)
                if CLASS_INDEX[args.obj] > 0:
                    seg_loss = 0
                    mask = gt.to(device)
                    mask = (mask > 0.5).float()
                    
                    for layer in range(len(seg_patch_tokens)):
                        raw_tokens = seg_patch_tokens[layer][:, 1:, :]
                        projected_tokens = model.visual_proj(raw_tokens)
                        projected_tokens = F.normalize(projected_tokens, dim=-1)
                        
                        anomaly_map = (100.0 * projected_tokens @ text_features)
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
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask.squeeze(1))
                    
                    total_loss += seg_loss
            
            # Backward pass with mixed precision
            seg_optimizer.zero_grad()
            det_optimizer.zero_grad()
            alpha_optimizer.zero_grad()
            
            scaler.scale(total_loss).backward()
            
            # Gradient clipping
            scaler.unscale_(seg_optimizer)
            scaler.unscale_(det_optimizer)
            scaler.unscale_(alpha_optimizer)
            
            torch.nn.utils.clip_grad_norm_(seg_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(det_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(alpha_params, max_norm=0.5)
            
            scaler.step(seg_optimizer)
            scaler.step(det_optimizer)
            scaler.step(alpha_optimizer)
            scaler.update()
            
            # Update EMA
            ema.update()
            
            loss_list.append(total_loss.item())
        
        # Update schedulers
        seg_scheduler.step()
        det_scheduler.step()
        alpha_scheduler.step()
        
        avg_loss = np.mean(loss_list)
        print(f"Loss: {avg_loss:.4f} | LR: {seg_optimizer.param_groups[0]['lr']:.6f}")
        
        # Build memory bank
        model.eval()
        seg_features = []
        det_features = []
        
        for image in support_loader:
            image = image[0].to(device)
            with torch.no_grad():
                _, seg_tokens, det_tokens = model(image)
                seg_tokens = [p[0].contiguous() for p in seg_tokens]
                det_tokens = [p[0].contiguous() for p in det_tokens]
                seg_features.append(seg_tokens)
                det_features.append(det_tokens)
        
        seg_mem_features = [
            torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0)
            for i in range(len(seg_features[0]))
        ]
        det_mem_features = [
            torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0)
            for i in range(len(det_features[0]))
        ]
        
        # Evaluate with EMA model
        ema.apply_shadow()
        result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)
        ema.restore()
        
        if result > best_result:
            best_result = result
            patience_counter = 0
            print(f"✓ New Best Result: {best_result:.4f}")
            
            if args.save_model == 1:
                os.makedirs(args.save_path, exist_ok=True)
                ckp_path = os.path.join(args.save_path, f'{args.obj}_improved.pth')
                torch.save({
                    'seg_adapters': model.seg_adapters.state_dict(),
                    'det_adapters': model.det_adapters.state_dict(),
                    'seg_proj': model.seg_proj.state_dict(),
                    'det_proj': model.det_proj.state_dict(),
                    'alpha_backbone': model.alpha_backbone,
                    'alpha_seg': model.alpha_seg,
                    'alpha_det': model.alpha_det,
                    'ema_shadow': ema.shadow,
                    'epoch': epoch,
                    'best_result': best_result
                }, ckp_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best Result: {best_result:.4f}")
    print(f"{'='*50}")


def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    model.eval()
    gt_list = []
    gt_mask_list = []
    det_image_scores_zero = []
    det_image_scores_few = []
    seg_score_map_zero = []
    seg_score_map_few = []
    
    for (image, y, mask) in tqdm(test_loader, desc='Testing'):
        image = image.to(device)
        mask = (mask > 0.5).float()
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [p[:, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]
            
            if CLASS_INDEX[args.obj] > 0:
                # Few-shot segmentation
                anomaly_maps_few = []
                for idx, p in enumerate(seg_patch_tokens):
                    p_flat = p.reshape(-1, p.shape[-1])
                    cos = cos_sim(seg_mem_features[idx], p_flat)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map = F.interpolate(
                        anomaly_map,
                        size=args.img_size,
                        mode='bilinear',
                        align_corners=True
                    )
                    anomaly_maps_few.append(anomaly_map.cpu().numpy())
                
                score_map_few = np.sum(anomaly_maps_few, axis=0)
                seg_score_map_few.append(score_map_few)
                
                # Zero-shot segmentation
                anomaly_maps_zero = []
                for layer in range(len(seg_patch_tokens)):
                    raw_tokens = seg_patch_tokens[layer].reshape(-1, seg_patch_tokens[layer].shape[-1])
                    projected = model.visual_proj(raw_tokens)
                    projected = F.normalize(projected, dim=-1)
                    
                    anomaly_map = (100.0 * projected @ text_features)
                    B, L, C = 1, anomaly_map.shape[0], anomaly_map.shape[1]
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(
                        anomaly_map.unsqueeze(0).permute(0, 2, 1).view(1, 2, H, H),
                        size=args.img_size,
                        mode='bilinear',
                        align_corners=True
                    )
                    anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                    anomaly_maps_zero.append(anomaly_map.cpu().numpy())
                
                score_map_zero = np.sum(anomaly_maps_zero, axis=0)
                seg_score_map_zero.append(score_map_zero)
            
            else:
                # Detection task
                # Few-shot
                anomaly_maps_few = []
                for idx, p in enumerate(det_patch_tokens):
                    p_flat = p.reshape(-1, p.shape[-1])
                    cos = cos_sim(det_mem_features[idx], p_flat)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map = F.interpolate(
                        anomaly_map,
                        size=args.img_size,
                        mode='bilinear',
                        align_corners=True
                    )
                    anomaly_maps_few.append(anomaly_map.cpu().numpy())
                
                score_few = np.sum(anomaly_maps_few, axis=0).mean()
                det_image_scores_few.append(score_few)
                
                # Zero-shot
                anomaly_score = 0
                for layer in range(len(det_patch_tokens)):
                    raw_tokens = det_patch_tokens[layer].reshape(-1, det_patch_tokens[layer].shape[-1])
                    projected = model.visual_proj(raw_tokens)
                    projected = F.normalize(projected, dim=-1)
                    
                    anomaly_map = (100.0 * projected @ text_features)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, 1]
                    anomaly_score += anomaly_map.mean()
                
                det_image_scores_zero.append(anomaly_score.cpu().numpy())
        
        gt_mask_list.append(mask.squeeze().cpu().numpy())
        gt_list.extend(y.cpu().numpy())
    
    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)
    
    if CLASS_INDEX[args.obj] > 0:
        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)
        
        # Normalize
        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / \
                             (seg_score_map_zero.max() - seg_score_map_zero.min() + 1e-8)
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / \
                           (seg_score_map_few.max() - seg_score_map_few.min() + 1e-8)
        
        # Adaptive weighting
        segment_scores = 0.6 * seg_score_map_zero + 0.4 * seg_score_map_few
        
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC: {seg_roc_auc:.4f}')
        
        img_scores = np.max(segment_scores.reshape(segment_scores.shape[0], -1), axis=1)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        print(f'{args.obj} AUC: {img_roc_auc:.4f}')
        
        return seg_roc_auc + img_roc_auc
    
    else:
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)
        
        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / \
                               (det_image_scores_zero.max() - det_image_scores_zero.min() + 1e-8)
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / \
                              (det_image_scores_few.max() - det_image_scores_few.min() + 1e-8)
        
        image_scores = 0.6 * det_image_scores_zero + 0.4 * det_image_scores_few
        img_roc_auc = roc_auc_score(gt_list, image_scores)
        print(f'{args.obj} AUC: {img_roc_auc:.4f}')
        
        return img_roc_auc


if __name__ == '__main__':
    main()