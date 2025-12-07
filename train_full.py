import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from dataset.full_data import MedDataset  # Your dataset
from biomedclip.clip import create_model
from biomedclip.tokenizer import tokenize
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
    parser = argparse.ArgumentParser(description='BiomedCLIP Full Data Training - FIXED')
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

    # ✅ Load BiomedCLIP model
    clip_model = create_model(
        model_name=args.model_name, 
        img_size=args.img_size, 
        device=device, 
        pretrained=args.pretrain, 
        require_pretrained=True
    )
    clip_model.eval()

    # ✅ Create adapter model
    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)

    # ✅ Enable gradients only for adapters
    for name, param in model.named_parameters():
        if 'adapter' in name:
            param.requires_grad = True

    # ✅ Separate optimizers for seg/det adapters
    seg_optimizer = torch.optim.Adam(
        list(model.seg_adapters.parameters()), 
        lr=args.learning_rate, 
        betas=(0.5, 0.999)
    )
    det_optimizer = torch.optim.Adam(
        list(model.det_adapters.parameters()), 
        lr=args.learning_rate, 
        betas=(0.5, 0.999)
    )

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    # ✅ Load FULL datasets (train/valid/test)
    print("\nLoading datasets...")
    train_dataset = MedDataset(dataset_path=args.data_path, class_name=args.obj, split='train', resize=args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    valid_dataset = MedDataset(dataset_path=args.data_path, class_name=args.obj, split='valid', resize=args.img_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    test_dataset = MedDataset(dataset_path=args.data_path, class_name=args.obj, split='test', resize=args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # ✅ Losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = nn.BCEWithLogitsLoss()

    # ✅ Text features (anomaly prompt)
    print("\nEncoding text prompts...")
    with torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)

    best_auc = 0
    os.makedirs(args.save_path, exist_ok=True)

    # ✅ TRAINING LOOP
    for epoch in range(args.epoch):
        print(f'\nEpoch {epoch}/{args.epoch}:')
        
        # Training
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Train")):
            images, masks, labels = batch  # ✅ images, masks, labels from dataset
            
            images = images.to(device)
            labels = labels.to(device).float().squeeze(-1).squeeze(-1)  # ✅ [B] image-level labels
            
            with torch.cuda.amp.autocast():
                _, seg_patch_tokens, det_patch_tokens = model(images)
                
                # ✅ Extract patch tokens (exclude CLS token)
                seg_patch_tokens = [tokens[:, 1:] for tokens in seg_patch_tokens]  # [B, 196, D]
                det_patch_tokens = [tokens[:, 1:] for tokens in det_patch_tokens]

                # ✅ DETECTION LOSS (image-level)
                det_loss = 0
                for layer_idx, layer_tokens in enumerate(det_patch_tokens):
                    # Normalize features
                    layer_tokens = F.normalize(layer_tokens, dim=-1)
                    
                    # Project to text embedding space (768 -> 512)
                    proj_tokens = layer_tokens @ model.visual_proj.weight.T  # [B, 196, 512]
                    proj_tokens = F.normalize(proj_tokens, dim=-1)
                    
                    # Similarity with text features
                    logits = 100.0 * proj_tokens @ F.normalize(text_features, dim=-1)  # [B, 196, 2]
                    anomaly_map = torch.softmax(logits, dim=-1)[:, :, 1]  # [B, 196]
                    anomaly_score = anomaly_map.mean(dim=1)  # [B] ✅ Image-level scores
                    
                    det_loss += loss_bce(anomaly_score, labels)  # ✅ SHAPES MATCH: [B] vs [B]
                
                det_loss /= len(det_patch_tokens)

                # ✅ SEGMENTATION LOSS (only for seg datasets)
                if CLASS_INDEX[args.obj] > 0:
                    seg_loss = 0
                    for b_idx in range(images.shape[0]):  # Process each image
                        mask = masks[b_idx].to(device).squeeze()  # [H, W]
                        mask = (mask > 0.5).float()  # Binary mask [H, W]

                        for layer_idx, layer_tokens in enumerate(seg_patch_tokens):
                            tokens_b = layer_tokens[b_idx]  # [196, D]
                            tokens_b = F.normalize(tokens_b, dim=-1)
                            
                            proj_tokens = tokens_b @ model.visual_proj.weight.T  # [196, 512]
                            proj_tokens = F.normalize(proj_tokens, dim=-1)
                            
                            logits = 100.0 * proj_tokens @ F.normalize(text_features, dim=-1)  # [196, 2]
                            logits = logits.unsqueeze(0)  # [1, 196, 2]
                            
                            # Reshape to image
                            B, L, C = logits.shape
                            H_patch = int(math.sqrt(L))
                            logits_img = logits.permute(0, 2, 1).reshape(B, C, H_patch, H_patch)
                            anomaly_map = F.interpolate(logits_img, size=args.img_size, 
                                                      mode='bilinear', align_corners=False)
                            anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1]  # [1, H, W]
                            
                            seg_loss += loss_focal(anomaly_map, mask.unsqueeze(0))
                            seg_loss += loss_dice(anomaly_map, mask.unsqueeze(0))
                    
                    seg_loss /= (len(seg_patch_tokens) * images.shape[0])
                    total_loss = seg_loss + det_loss
                else:
                    total_loss = det_loss

                # Backward pass
                seg_optimizer.zero_grad()
                det_optimizer.zero_grad()
                total_loss.backward()
                seg_optimizer.step()
                det_optimizer.step()

                train_losses.append(total_loss.item())

        print(f"Train Loss: {np.mean(train_losses):.4f}")

        # ✅ Validation
        val_auc = validate(model, valid_loader, text_features, args)
        print(f"Validation AUC: {val_auc:.4f}")

        # ✅ Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            if args.save_model:
                torch.save({
                    'seg_adapters': model.seg_adapters.state_dict(),
                    'det_adapters': model.det_adapters.state_dict(),
                    'epoch': epoch,
                    'auc': val_auc
                }, os.path.join(args.save_path, f'{args.obj}_full_best.pth'))
                print(f"✅ New best model saved! AUC: {val_auc:.4f}")

def validate(model, val_loader, text_features, args):
    """Simple validation function"""
    model.eval()
    scores, labels = [], []
    
    with torch.no_grad():
        for images, masks, lbls in val_loader:
            images = images.to(device)
            lbls = lbls.cpu().numpy()
            
            _, _, det_patch_tokens = model(images)
            det_patch_tokens = [F.normalize(t[:, 1:], dim=-1) for t in det_patch_tokens]
            
            batch_scores = []
            for layer_tokens in det_patch_tokens:
                proj_tokens = F.normalize(layer_tokens @ model.visual_proj.weight.T, dim=-1)
                logits = 100.0 * proj_tokens @ F.normalize(text_features, dim=-1)
                anomaly_map = torch.softmax(logits, dim=-1)[:, :, 1].mean(dim=1)
                batch_scores.append(anomaly_map.cpu())
            
            score = torch.stack(batch_scores).mean(0).numpy()
            scores.extend(score)
            labels.extend(lbls.flatten())
    
    return roc_auc_score(labels, scores)

if __name__ == '__main__':
    main()
