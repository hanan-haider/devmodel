#%%writefile /kaggle/working/devmodel/train_few.py
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
from biomedclip.adapterv5 import CLIP_Inplanted  # Updated import
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_biomedclip_prompt_ensemble1
from prompt import REAL_NAME
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ReduceLROnPlateau

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}
global_vars = {}

def project_tokens(model, tokens):
    """Project 768-dim tokens to 512-dim text space"""
    if model.visual_proj is not None:
        return [model.visual_proj(t) for t in tokens]
    return tokens

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='BiomedCLIP Few-Shot Training')
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    parser.add_argument('--obj', type=str, default='Retina_OCT2017')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/shot4/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    
    # Adaptive architecture params (no fixed bottleneck!)
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument("--compression_ratio", type=float, default=0.25)  # NEW!
    parser.add_argument("--dropout", type=float, default=0.15)
    
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)

    args, _ = parser.parse_known_args()

    print("\nParsed Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
        global_vars[arg] = getattr(args, arg)
    
    setup_seed(args.seed)
    print("\nSeed set to:", args.seed)

    # Load BiomedCLIP
    print("Loading BiomedCLIP model...")
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained=args.pretrain,
        require_pretrained=True
    )
    clip_model.eval()

    print("Initializing adapter model...")
    model = CLIP_Inplanted(
        clip_model=clip_model,
        features=args.features_list,
        compression_ratio=args.compression_ratio,
        dropout=args.dropout
    ).to(device)
    model.train()

    # Set requires_grad properly
    for p in model.parameters():
        p.requires_grad = False
    for p in model.seg_adapters.parameters():
        p.requires_grad = True
    for p in model.det_adapters.parameters():
        p.requires_grad = True
    model.alpha_backbone.requires_grad = True
    model.alpha_seg.requires_grad = True
    model.alpha_det.requires_grad = True

    # Optimizers with differential learning rates
    seg_params = [
        {'params': model.seg_adapters.parameters(), 'lr': args.learning_rate},
        {'params': [model.alpha_seg], 'lr': args.learning_rate * 0.5},
        {'params': [model.alpha_backbone], 'lr': args.learning_rate * 0.5},
    ]
    det_params = [
        {'params': model.det_adapters.parameters(), 'lr': args.learning_rate},
        {'params': [model.alpha_det], 'lr': args.learning_rate * 0.5},
        {'params': [model.alpha_backbone], 'lr': args.learning_rate * 0.5},
    ]

    seg_optimizer = AdamW(seg_params, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    det_optimizer = AdamW(det_params, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # Parameter count
    seg_adapter_params = sum(p.numel() for p in model.seg_adapters.parameters())
    det_adapter_params = sum(p.numel() for p in model.det_adapters.parameters())
    alpha_params = model.alpha_backbone.numel() + model.alpha_seg.numel() + model.alpha_det.numel()
    print(f"\nSegmentation adapter parameters: {seg_adapter_params:,}")
    print(f"Detection adapter parameters:    {det_adapter_params:,}")
    print(f"Alpha blending parameters:       {alpha_params:,}")
    print(f"Total trainable:                 {seg_adapter_params + det_adapter_params + alpha_params:,}")

    # Enhanced scheduler
    warmup_epochs = 8
    total_epochs = args.epoch

    warmup_seg = LinearLR(seg_optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_seg = CosineAnnealingLR(
        seg_optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=args.learning_rate * 0.001
    )
    seg_scheduler = SequentialLR(
        seg_optimizer,
        schedulers=[warmup_seg, cosine_seg],
        milestones=[warmup_epochs]
    )

    warmup_det = LinearLR(det_optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_det = CosineAnnealingLR(
        det_optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=args.learning_rate * 0.001
    )
    det_scheduler = SequentialLR(
        det_optimizer,
        schedulers=[warmup_det, cosine_det],
        milestones=[warmup_epochs]
    )

    # Load datasets
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # Few-shot augmentation
    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
    augment_fewshot_label = torch.cat([torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0)

    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    # Memory bank (normal images only)
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)

    # Losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    # Text features
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble1(clip_model, REAL_NAME[args.obj], device)

    best_result = 0
    patience_counter = 0

    print("="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")

    for epoch in range(args.epoch):
        print(f'epoch {epoch}:')
        model.train()

        loss_list = []
        for (image, gt, label) in train_loader:
            image = image.to(device)
            with torch.cuda.amp.autocast():
                # Model returns 5 outputs
                _, seg_med, seg_full, det_med, det_full = model(image)

                # Project ONLY full tokens for zero-shot
                seg_tokens_proj = project_tokens(model, [p[0, 1:, :] for p in seg_full])
                det_tokens_proj = project_tokens(model, [p[0, 1:, :] for p in det_full])

                # Detection loss
                det_loss = 0
                image_label = label.to(device)
                for layer in range(len(det_tokens_proj)):
                    tokens_norm = det_tokens_proj[layer] / det_tokens_proj[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * tokens_norm @ text_features).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    det_loss += loss_bce(anomaly_score, image_label)

                if CLASS_INDEX[args.obj] > 0:
                    # Segmentation loss
                    seg_loss = 0
                    mask = gt.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

                    for layer in range(len(seg_tokens_proj)):
                        tokens_norm = seg_tokens_proj[layer] / seg_tokens_proj[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * tokens_norm @ text_features).unsqueeze(0)
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)

                    loss = seg_loss + det_loss
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.seg_adapters.parameters(), max_norm=args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(model.det_adapters.parameters(), max_norm=args.grad_clip)
                    seg_optimizer.step()
                    det_optimizer.step()
                else:
                    loss = det_loss
                    det_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.det_adapters.parameters(), max_norm=args.grad_clip)
                    det_optimizer.step()

                loss_list.append(loss.item())

        avg_loss = np.mean(loss_list)
        current_lr = seg_optimizer.param_groups[0]['lr']
        print(f"  Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

        # Build memory bank using MED tokens (bottleneck, unprojected)
        model.eval()
        seg_features_med = []
        det_features_med = []
        for image in support_loader:
            image = image[0].to(device)
            with torch.no_grad():
                _, seg_med, seg_full, det_med, det_full = model(image)
                seg_features_med.append([p[0].contiguous() for p in seg_med])
                det_features_med.append([p[0].contiguous() for p in det_med])

        seg_mem_med = [torch.cat([seg_features_med[j][i] for j in range(len(seg_features_med))], dim=0)
                       for i in range(len(seg_features_med[0]))]
        det_mem_med = [torch.cat([det_features_med[j][i] for j in range(len(det_features_med))], dim=0)
                       for i in range(len(det_features_med[0]))]

        result = test(args, model, test_loader, text_features, seg_mem_med, det_mem_med)

        if result > best_result:
            best_result = result
            patience_counter = 0
            print("  ✓ Best result! Saving checkpoint...")
            if args.save_model == 1:
                ckp_path = os.path.join(args.save_path, f'{args.obj}.pth')
                torch.save({
                    'epoch': epoch,
                    'seg_adapters': model.seg_adapters.state_dict(),
                    'det_adapters': model.det_adapters.state_dict(),
                    'best_result': best_result,
                }, ckp_path)
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{args.patience} epochs")

        if patience_counter >= args.patience:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"Best result: {best_result:.4f}")
            print(f"{'='*60}")
            break

        seg_scheduler.step()
        det_scheduler.step()
        print()

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"Best result: {best_result:.4f}")
    print(f"{'='*60}\n")


def test(args, model, test_loader, text_features, seg_mem_med, det_mem_med):
    gt_list = []
    gt_mask_list = []
    det_image_scores_zero = []
    det_image_scores_few = []
    seg_score_map_zero = []
    seg_score_map_few = []

    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        batch_size = image.shape[0]
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, seg_med, seg_full, det_med, det_full = model(image)

            for i in range(batch_size):
                # Few-shot: med tokens (bottleneck, unprojected)
                seg_med_single = [p[i, 1:, :] for p in seg_med]
                det_med_single = [p[i, 1:, :] for p in det_med]

                # Zero-shot: full tokens (768-dim, project to 512)
                seg_full_single = project_tokens(model, [p[i, 1:, :] for p in seg_full])
                det_full_single = project_tokens(model, [p[i, 1:, :] for p in det_full])

                if CLASS_INDEX[args.obj] > 0:
                    # Few-shot segmentation
                    anomaly_maps_few = []
                    for idx, p in enumerate(seg_med_single):
                        cos = cos_sim(seg_mem_med[idx], p)
                        height = int(np.sqrt(cos.shape[1]))
                        anomaly_map_few = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                        anomaly_map_few = F.interpolate(anomaly_map_few, size=args.img_size, mode='bilinear', align_corners=True)
                        anomaly_maps_few.append(anomaly_map_few[0].cpu().numpy())
                    seg_score_map_few.append(np.sum(anomaly_maps_few, axis=0))

                    # Zero-shot segmentation
                    anomaly_maps = []
                    for t in seg_full_single:
                        t_norm = t / t.norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * t_norm @ text_features).unsqueeze(0)
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                        anomaly_maps.append(anomaly_map.cpu().numpy())
                    seg_score_map_zero.append(np.sum(anomaly_maps, axis=0))

                else:
                    # Few-shot detection
                    anomaly_maps_few = []
                    for idx, p in enumerate(det_med_single):
                        cos = cos_sim(det_mem_med[idx], p)
                        height = int(np.sqrt(cos.shape[1]))
                        anomaly_map_few = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                        anomaly_map_few = F.interpolate(anomaly_map_few, size=args.img_size, mode='bilinear', align_corners=True)
                        anomaly_maps_few.append(anomaly_map_few[0].cpu().numpy())
                    det_image_scores_few.append(np.sum(anomaly_maps_few, axis=0).mean())

                    # Zero-shot detection
                    anomaly_score = 0
                    for t in det_full_single:
                        t_norm = t / t.norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * t_norm @ text_features).unsqueeze(0)
                        anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                        anomaly_score += anomaly_map.mean()
                    det_image_scores_zero.append(anomaly_score.cpu().numpy())

                gt_mask_list.append(mask[i].squeeze().cpu().numpy())

            gt_list.extend(y.cpu().numpy())

    gt_list = np.array(gt_list)
    gt_mask_list = np.array(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)

    if CLASS_INDEX[args.obj] > 0:
        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)

        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min() + 1e-8)
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min() + 1e-8)

        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')

        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'{args.obj} AUC : {round(roc_auc_im, 4)}')

        return seg_roc_auc + roc_auc_im

    else:
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)

        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min() + 1e-8)
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min() + 1e-8)

        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')

        return img_roc_auc_det


if __name__ == '__main__':
    main()