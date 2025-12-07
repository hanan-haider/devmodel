import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from dataset.full_data import MedDataset
from biomedclip.clip import create_model
from biomedclip.adapter import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import roc_auc_score
from loss import FocalLoss, BinaryDiceLoss
from utils import cos_sim, encode_text_with_biomedclip_prompt_ensemble
from prompt import REAL_NAME

os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

def main():
    parser = argparse.ArgumentParser(description='BiomedCLIP Full Data Training - SHAPE FIXED')
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--text_encoder', type=str, default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/full-data/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--features_list', type=int, nargs="+", default=[4, 8, 10, 12])
    parser.add_argument('--seed', type=int, default=111)

    args, _ = parser.parse_known_args()
    
    print("\nParsed Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    setup_seed(args.seed)
    print(f"\nSeed set to: {args.seed}")

    # Load BiomedCLIP
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, 
                            device=device, pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)

    # Enable gradients only for adapters
    for name, param in model.named_parameters():
        if 'adapter' in name:
            param.requires_grad = True

    seg_optimizer = torch.optim.Adam(model.seg_adapters.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(model.det_adapters.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    print("\nLoading datasets...")
    train_dataset = MedDataset(dataset_path=args.data_path, class_name=args.obj, split='train', resize=args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    valid_dataset = MedDataset(dataset_path=args.data_path, class_name=args.obj, split='valid', resize=args.img_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # Losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = nn.BCEWithLogitsLoss()

    # Text features
    print("\nEncoding text prompts...")
    with torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)

    best_auc = 0
    os.makedirs(args.save_path, exist_ok=True)

    for epoch in range(args.epoch):
        print(f'\nEpoch {epoch}/{args.epoch}:')
        
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Train")):
            images, masks, labels = batch
            
            images = images.to(device)
            
            # ✅ FIXED: Extract SCALAR image-level labels [B]
            batch_labels = labels[:, 0, 0].to(device).float() if labels.dim() > 2 else labels.float()
            if batch_labels.dim() > 1:
                batch_labels = batch_labels.squeeze(-1).squeeze(-1).float()
            
            print(f"DEBUG - batch_labels shape: {batch_labels.shape}, values: {batch_labels[:3]}")  # DEBUG
            
            with torch.cuda.amp.autocast():
                _, seg_patch_tokens, det_patch_tokens = model(images)
                
                seg_patch_tokens = [tokens[:, 1:] for tokens in seg_patch_tokens]
                det_patch_tokens = [tokens[:, 1:] for tokens in det_patch_tokens]

                # DETECTION LOSS - ✅ SHAPES NOW MATCH [B] vs [B]
                det_loss = 0
                for layer_idx, layer_tokens in enumerate(det_patch_tokens):
                    layer_tokens = F.normalize(layer_tokens, dim=-1)
                    proj_tokens = F.normalize(layer_tokens @ model.visual_proj.weight.T, dim=-1)
                    logits = 100.0 * proj_tokens @ F.normalize(text_features, dim=-1)
                    anomaly_map = torch.softmax(logits, dim=-1)[:, :, 1]
                    anomaly_score = anomaly_map.mean(dim=1)  # [B]
                    
                    print(f"DEBUG - anomaly_score shape: {anomaly_score.shape}, batch_labels shape: {batch_labels.shape}")  # DEBUG
                    
                    det_loss += loss_bce(anomaly_score, batch_labels)
                
                det_loss /= len(det_patch_tokens)

                # SEGMENTATION LOSS (Liver has masks)
                seg_loss = 0
                for b_idx in range(images.shape[0]):
                    mask = masks[b_idx].to(device)
                    if mask.dim() == 3:  # [1, H, W]
                        mask = mask.squeeze(0)
                    mask = (mask > 0.5).float()

                    for layer_idx, layer_tokens in enumerate(seg_patch_tokens):
                        tokens_b = F.normalize(layer_tokens[b_idx], dim=-1)
                        proj_tokens = F.normalize(tokens_b @ model.visual_proj.weight.T, dim=-1)
                        logits = 100.0 * proj_tokens @ F.normalize(text_features, dim=-1)
                        
                        B, L, C = 1, logits.shape[0], logits.shape[1]
                        H_patch = int(math.sqrt(L))
                        logits_img = logits.unsqueeze(0).permute(0, 2, 1).reshape(1, C, H_patch, H_patch)
                        anomaly_map = F.interpolate(logits_img, size=args.img_size, mode='bilinear', align_corners=False)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, 0]  # [1, H, W]

                        seg_loss += loss_focal(anomaly_map.unsqueeze(0), mask.unsqueeze(0))
                        seg_loss += 0.5 * loss_dice(anomaly_map.unsqueeze(0), mask.unsqueeze(0))
                
                seg_loss /= (len(seg_patch_tokens) * images.shape[0])
                total_loss = det_loss + seg_loss

                seg_optimizer.zero_grad()
                det_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                seg_optimizer.step()
                det_optimizer.step()

                train_losses.append(total_loss.item())

        print(f"Epoch {epoch} Train Loss: {np.mean(train_losses):.4f}")

        # Validation
        model.eval()
        val_auc = simple_validate(model, valid_loader, text_features)
        print(f"Validation AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            if args.save_model:
                torch.save({
                    'seg_adapters': model.seg_adapters.state_dict(),
                    'det_adapters': model.det_adapters.state_dict(),
                }, os.path.join(args.save_path, f'{args.obj}_epoch_{epoch}_auc_{val_auc:.4f}.pth'))
                print(f"✅ SAVED BEST: {val_auc:.4f}")

def simple_validate(model, val_loader, text_features):
    scores, labels = [], []
    model.eval()
    
    with torch.no_grad():
        for batch in val_loader:
            images, masks, lbls = batch
            images = images.to(device)
            
            batch_labels = lbls[:, 0, 0] if lbls.dim() > 2 else lbls
            if batch_labels.dim() > 1:
                batch_labels = batch_labels.squeeze(-1).squeeze(-1)
            
            _, _, det_patch_tokens = model(images)
            det_patch_tokens = [F.normalize(t[:, 1:], dim=-1) for t in det_patch_tokens]
            
            batch_score = 0
            for layer_tokens in det_patch_tokens:
                proj_tokens = F.normalize(layer_tokens @ model.visual_proj.weight.T, dim=-1)
                logits = 100.0 * proj_tokens @ F.normalize(text_features, dim=-1)
                anomaly_score = torch.softmax(logits, dim=-1)[:, :, 1].mean(dim=1)
                batch_score += anomaly_score
            
            scores.extend(batch_score.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    
    return roc_auc_score(labels, scores)

if __name__ == '__main__':
    main()
