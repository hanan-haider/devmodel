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

def build_memory_bank(model, valid_dataset, args):
    normal_indices = [i for i, lbl in enumerate(valid_dataset.labels) if lbl == 0]
    normal_subset = torch.utils.data.Subset(valid_dataset, normal_indices[:min(50, len(normal_indices))])
    normal_loader = torch.utils.data.DataLoader(normal_subset, batch_size=8, shuffle=False)
    
    seg_features, det_features = [], []
    model.eval()
    
    with torch.no_grad():
        for batch in normal_loader:
            images, _, _ = batch
            images = images.to(device)
            _, seg_tokens, det_tokens = model(images)
            
            seg_tokens = [F.normalize(t[:, 1:], dim=-1) for t in seg_tokens]
            det_tokens = [F.normalize(t[:, 1:], dim=-1) for t in det_tokens]
            
            for b in range(images.shape[0]):
                seg_features.append([t[b] for t in seg_tokens])
                det_features.append([t[b] for t in det_tokens])
    
    n_layers = len(seg_features[0])
    seg_mem = [torch.stack([seg_features[i][l] for i in range(len(seg_features))]).mean(0) for l in range(n_layers)]
    det_mem = [torch.stack([det_features[i][l] for i in range(len(det_features))]).mean(0) for l in range(n_layers)]
    
    return seg_mem, det_mem

def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    gt_list, gt_mask_list = [], []
    det_scores_zero, det_scores_few = [], []
    seg_scores_zero, seg_scores_few = [], []

    for batch in tqdm(test_loader, desc="Testing"):
        images, masks, labels = batch
        images = images.to(device)
        
        batch_labels = labels[:, 0] if labels.dim() > 1 else labels
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            single_img = images[i:i+1]
            single_lbl = batch_labels[i].item()
            single_mask = masks[i].cpu().numpy()
            single_mask = (single_mask > 0.5).astype(float)
            
            with torch.no_grad():
                _, seg_tokens, det_tokens = model(single_img)
                seg_tokens = [t[0, 1:] for t in seg_tokens]
                det_tokens = [t[0, 1:] for t in det_tokens]

                if CLASS_INDEX[args.obj] > 0:
                    maps_few = []
                    for idx, p in enumerate(seg_tokens):
                        p = F.normalize(p, dim=-1)
                        cos = cos_sim(seg_mem_features[idx], p.unsqueeze(0))
                        h = int(np.sqrt(cos.shape[1]))
                        map_few = torch.min(1 - cos, dim=0)[0].reshape(1, 1, h, h)
                        map_few = F.interpolate(map_few, size=args.img_size, mode='bilinear')[0].cpu().numpy()
                        maps_few.append(map_few)
                    seg_scores_few.append(np.sum(maps_few, axis=0))

                    maps_zero = []
                    for t in seg_tokens:
                        t = F.normalize(t, dim=-1)
                        proj = F.normalize(t @ model.visual_proj.weight.T, dim=-1)
                        logits = 100.0 * proj @ F.normalize(text_features, dim=-1)
                        B, L, C = 1, logits.shape[0], 2
                        H = int(np.sqrt(L))
                        logits_img = logits.unsqueeze(0).permute(0, 2, 1).view(1, 2, H, H)
                        map_zero = F.interpolate(logits_img, size=args.img_size, mode='bilinear')
                        map_zero = torch.softmax(map_zero, dim=1)[:, 1].cpu().numpy()
                        maps_zero.append(map_zero)
                    seg_scores_zero.append(np.sum(maps_zero, axis=0))
                else:
                    maps_few = []
                    for idx, p in enumerate(det_tokens):
                        p = F.normalize(p, dim=-1)
                        cos = cos_sim(det_mem_features[idx], p.unsqueeze(0))
                        h = int(np.sqrt(cos.shape[1]))
                        map_few = torch.min(1 - cos, dim=0)[0].reshape(1, 1, h, h)
                        map_few = F.interpolate(map_few, size=args.img_size, mode='bilinear')[0].cpu().numpy()
                        maps_few.append(map_few)
                    det_scores_few.append(np.sum(maps_few, axis=0).mean())

                    score_zero = 0
                    for t in det_tokens:
                        t = F.normalize(t, dim=-1)
                        proj = F.normalize(t @ model.visual_proj.weight.T, dim=-1)
                        logits = 100.0 * proj @ F.normalize(text_features, dim=-1)
                        score_zero += torch.softmax(logits, dim=-1)[:, 1].mean()
                    det_scores_zero.append(score_zero.cpu().item())

            gt_list.append(single_lbl)
            gt_mask_list.append(single_mask)

    gt_list = np.array(gt_list)
    gt_mask_list = np.array(gt_mask_list)

    if CLASS_INDEX[args.obj] > 0:
        seg_scores_zero = np.array(seg_scores_zero)
        seg_scores_few = np.array(seg_scores_few)
        seg_scores_zero = (seg_scores_zero - seg_scores_zero.min()) / (seg_scores_zero.max() - seg_scores_zero.min() + 1e-8)
        seg_scores_few = (seg_scores_few - seg_scores_few.min()) / (seg_scores_few.max() - seg_scores_few.min() + 1e-8)
        
        seg_final = 0.5 * seg_scores_zero + 0.5 * seg_scores_few
        pauc = roc_auc_score(gt_mask_list.flatten(), seg_final.flatten())
        auc = roc_auc_score(gt_list, seg_final.reshape(len(gt_list), -1).max(1))
        print(f'{args.obj} pAUC: {pauc:.4f}, AUC: {auc:.4f}')
        return pauc + auc
    else:
        det_scores_zero = np.array(det_scores_zero)
        det_scores_few = np.array(det_scores_few)
        det_scores_zero = (det_scores_zero - det_scores_zero.min()) / (det_scores_zero.max() - det_scores_zero.min() + 1e-8)
        det_scores_few = (det_scores_few - det_scores_few.min()) / (det_scores_few.max() - det_scores_few.min() + 1e-8)
        
        det_final = 0.5 * det_scores_zero + 0.5 * det_scores_few
        auc = roc_auc_score(gt_list, det_final)
        print(f'{args.obj} AUC: {auc:.4f}')
        return auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--text_encoder', type=str, default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--features_list', type=int, nargs="+", default=[4, 8, 10, 12])
    parser.add_argument('--seed', type=int, default=111)

    args, _ = parser.parse_known_args()
    print("\n=== Arguments ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    setup_seed(args.seed)


    # fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)

    for name, param in model.named_parameters():
        param.requires_grad = 'adapter' in name

    seg_opt = torch.optim.Adam(model.seg_adapters.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    det_opt = torch.optim.Adam(model.det_adapters.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = MedDataset(args.data_path, args.obj, 'train', args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, True, **kwargs)
    
    valid_dataset = MedDataset(args.data_path, args.obj, 'valid', args.img_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, args.batch_size, False, **kwargs)

    loss_focal, loss_dice, loss_bce = FocalLoss(), BinaryDiceLoss(), nn.BCEWithLogitsLoss()

    with torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)

    os.makedirs(args.save_path, exist_ok=True)
    best_auc = 0

    for epoch in range(args.epoch):
        model.train()
        losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images, masks, labels = batch
            images = images.to(device)
            
            # ✅✅✅ THE CRITICAL FIX - EXTRACT FIRST COLUMN ONLY ✅✅✅
            batch_labels = labels[:, 0].to(device).float()
            
            with torch.cuda.amp.autocast():
                _, seg_tokens, det_tokens = model(images)
                seg_tokens = [t[:, 1:] for t in seg_tokens]
                det_tokens = [t[:, 1:] for t in det_tokens]

                det_loss = 0
                for t in det_tokens:
                    t = F.normalize(t, dim=-1)
                    proj = F.normalize(t @ model.visual_proj.weight.T, dim=-1)
                    logits = 100.0 * proj @ F.normalize(text_features, dim=-1)
                    score = torch.softmax(logits, dim=-1)[:, :, 1].mean(1)
                    det_loss += loss_bce(score, batch_labels)
                det_loss /= len(det_tokens)

                seg_loss = 0
                if CLASS_INDEX[args.obj] > 0:
                    for b in range(images.shape[0]):
                        mask = masks[b].squeeze().to(device)
                        mask = (mask > 0.5).float()
                        
                        for t in seg_tokens:
                            tb = F.normalize(t[b], dim=-1)
                            proj = F.normalize(tb @ model.visual_proj.weight.T, dim=-1)
                            logits = 100.0 * proj @ F.normalize(text_features, dim=-1)
                            
                            L, C = logits.shape
                            H = int(np.sqrt(L))
                            logits_img = logits.unsqueeze(0).permute(0, 2, 1).view(1, C, H, H)
                            map_img = F.interpolate(logits_img, args.img_size, mode='bilinear')
                            map_img = torch.softmax(map_img, dim=1)[:, 1]
                            
                            seg_loss += loss_focal(map_img.unsqueeze(0), mask.unsqueeze(0))
                            seg_loss += 0.5 * loss_dice(map_img.unsqueeze(0), mask.unsqueeze(0))
                    seg_loss /= (len(seg_tokens) * images.shape[0])
                
                total_loss = det_loss + seg_loss

            seg_opt.zero_grad()
            det_opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            seg_opt.step()
            det_opt.step()
            losses.append(total_loss.item())

        print(f"Epoch {epoch} Train Loss: {np.mean(losses):.4f}")

        seg_mem, det_mem = build_memory_bank(model, valid_dataset, args)
        val_auc = test(args, model, valid_loader, text_features, seg_mem, det_mem)

        if val_auc > best_auc:
            best_auc = val_auc
            if args.save_model:
                torch.save({'seg_adapters': model.seg_adapters.state_dict(),
                           'det_adapters': model.det_adapters.state_dict()},
                          f"{args.save_path}/{args.obj}_best.pth")
                print(f"✅ BEST: {val_auc:.4f}")

if __name__ == '__main__':
    main()
