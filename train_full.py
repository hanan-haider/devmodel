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
from dataset.full_data import MedDataset  # ✅ Correct import
from biomedclip.clip import create_model
from biomedclip.tokenizer import tokenize
from biomedclip.adapter1 import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_biomedclip_prompt_ensemble
from prompt import REAL_NAME
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

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
    parser = argparse.ArgumentParser(description='BiomedCLIP Full Data Training')
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--text_encoder', type=str, default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/full-data/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-5)  # ✅ Lower for full training
    parser.add_argument("--features_list", type=int, nargs="+", default=[4, 8, 10, 12])  # ✅ Better layers
    parser.add_argument('--seed', type=int, default=111)


    args, _ = parser.parse_known_args()

    print("\nParsed Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
        global_vars[arg] = getattr(args, arg)
    
    setup_seed(args.seed)
    print("\nSeed set to:", args.seed)

    # Load BiomedCLIP
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, 
                            device=device, pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.eval()

    # Make all parameters trainable for full data training
    for name, param in model.named_parameters():
        param.requires_grad = True

    # Optimizers
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    # ✅ FIXED: Load train/valid/test datasets properly
    train_dataset = MedDataset(dataset_path=args.data_path, class_name=args.obj, split='train', resize=args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    valid_dataset = MedDataset(dataset_path=args.data_path, class_name=args.obj, split='valid', resize=args.img_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    #test_dataset = MedDataset(dataset_path=args.data_path, class_name=args.obj, split='test', resize=args.img_size)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # ✅ FIXED: No few-shot augmentation needed for full training
    # Use validation set for memory bank instead
    support_dataset = valid_dataset  # Use normal validation images
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=8, shuffle=True, **kwargs)

    # Losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    # Text features
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)

    best_result = 0

    for epoch in range(args.epoch):
        print('Epoch ', epoch, ':')

        # Training loop
        model.train()
        loss_list = []
        for batch_idx, (image, gt_mask, label) in enumerate(tqdm(train_loader)):
            image = image.to(device)
            label = label.to(device).float()  # ✅ Ensure float for BCE

            with torch.cuda.amp.autocast():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                seg_patch_tokens = [p[:, 1:, :] for p in seg_patch_tokens]  # ✅ Fix batch dim
                det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]

                # Detection loss
                det_loss = 0
                for layer in range(len(det_patch_tokens)):
                    layer_tokens = F.normalize(det_patch_tokens[layer], dim=-1)  # ✅ Normalization
                    proj_tokens = layer_tokens @ model.visual_proj.weight.T
                    proj_tokens = F.normalize(proj_tokens, dim=-1)
                    logits = 100.0 * proj_tokens @ F.normalize(text_features, dim=-1)
                    anomaly_map = torch.softmax(logits, dim=-1)[:, :, 1]
                    anomaly_score = anomaly_map.mean(dim=1)  # ✅ Proper batch dim
                    det_loss += loss_bce(anomaly_score, label)

                det_loss /= len(det_patch_tokens)

                if CLASS_INDEX[args.obj] > 0:  # Segmentation datasets
                    seg_loss = 0
                    for batch in range(image.shape[0]):  # ✅ Process each sample
                        mask = gt_mask[batch].to(device)
                        mask[mask > 0.5] = 1
                        mask[mask <= 0.5] = 0

                        for layer in range(len(seg_patch_tokens)):
                            layer_tokens = F.normalize(seg_patch_tokens[layer][batch], dim=-1)
                            proj_tokens = layer_tokens @ model.visual_proj.weight.T
                            proj_tokens = F.normalize(proj_tokens, dim=-1)
                            logits = 100.0 * proj_tokens @ F.normalize(text_features, dim=-1)
                            anomaly_map = logits.unsqueeze(0)  # [1, L, 2]
                            
                            B, L, C = anomaly_map.shape
                            H = int(np.sqrt(L))
                            anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                      size=args.img_size, mode='bilinear', align_corners=True)
                            anomaly_map = torch.softmax(anomaly_map, dim=1)
                            seg_loss += loss_focal(anomaly_map, mask.unsqueeze(0))
                            seg_loss += loss_dice(anomaly_map[:, 1], mask.unsqueeze(0)[:, 0])
                    
                    seg_loss /= (len(seg_patch_tokens) * image.shape[0])
                    loss = seg_loss + det_loss
                else:
                    loss = det_loss

                # Backprop
                seg_optimizer.zero_grad()
                det_optimizer.zero_grad()
                loss.backward()
                seg_optimizer.step()
                det_optimizer.step()

                loss_list.append(loss.item())

        print("Train Loss: ", np.mean(loss_list))

        # Build memory bank from validation normals
        model.eval()
        seg_features, det_features = [], []
        normal_indices = [i for i, label in enumerate(valid_dataset.labels) if label == 0]
        normal_loader = torch.utils.data.Subset(valid_dataset, normal_indices)
        normal_loader = torch.utils.data.DataLoader(normal_loader, batch_size=8, shuffle=False, **kwargs)

        for image, _, _ in normal_loader:
            image = image.to(device)
            with torch.no_grad():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                seg_patch_tokens = [F.normalize(p[:, 1:], dim=-1) for p in seg_patch_tokens]
                det_patch_tokens = [F.normalize(p[:, 1:], dim=-1) for p in det_patch_tokens]
                seg_features.extend(seg_patch_tokens)
                det_features.extend(det_patch_tokens)

        # Average across samples per layer
        seg_mem_features = [torch.stack([f[layer] for f in seg_features]).mean(0) for layer in range(len(seg_features[0]))]
        det_mem_features = [torch.stack([f[layer] for f in det_features]).mean(0) for layer in range(len(det_features[0]))]

        # Validation
        result = test(args, model, valid_loader, text_features, seg_mem_features, det_mem_features)
        if result > best_result:
            best_result = result
            print(f"New Best Result: {best_result:.4f}")
            if args.save_model == 1:
                os.makedirs(args.save_path, exist_ok=True)
                ckp_path = os.path.join(args.save_path, f'{args.obj}_full.pth')
                torch.save({
                    'seg_adapters': model.seg_adapters.state_dict(),
                    'det_adapters': model.det_adapters.state_dict()
                }, ckp_path)

def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    gt_list = []
    gt_mask_list = []

    det_image_scores_zero = []
    det_image_scores_few = []
    
    seg_score_map_zero = []
    seg_score_map_few= []

    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        # Process each item in the batch separately
        batch_size = image.shape[0]
        
        for i in range(batch_size):
            single_image = image[i:i+1]  # Keep batch dimension
            single_y = y[i]
            single_mask = mask[i]

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

            if CLASS_INDEX[args.obj] > 0:

                # few-shot, seg head
                anomaly_maps_few_shot = []
                for idx, p in enumerate(seg_patch_tokens):
                    cos = cos_sim(seg_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                            size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                seg_score_map_few.append(score_map_few)

                # zero-shot, seg head
                anomaly_maps = []
                for layer in range(len(seg_patch_tokens)):
                    seg_patch_tokens[layer] /= seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    vision_proj = model.visual_proj  # maps 768 -> 512
                    proj_tokens = seg_patch_tokens[layer] @ vision_proj.weight.T
                    anomaly_map = (proj_tokens @ text_features).unsqueeze(0)
                    #anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features).unsqueeze(0)
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                    anomaly_maps.append(anomaly_map.cpu().numpy())
                score_map_zero = np.sum(anomaly_maps, axis=0)
                seg_score_map_zero.append(score_map_zero)
                


            else:
                # few-shot, det head
                anomaly_maps_few_shot = []
                for idx, p in enumerate(det_patch_tokens):
                    cos = cos_sim(det_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                            size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                score_few_det = anomaly_map_few_shot.mean()
                det_image_scores_few.append(score_few_det)

                # zero-shot, det head
                anomaly_score = 0
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] /= det_patch_tokens[layer].norm(dim=-1, keepdim=True)

                    #projection layer 
                    vision_proj = model.visual_proj  # maps 768 -> 512
                    proj_tokens = det_patch_tokens[layer] @ vision_proj.weight.T
                    anomaly_map = (proj_tokens @ text_features).unsqueeze(0)
                    #anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score += anomaly_map.mean()
                det_image_scores_zero.append(anomaly_score.cpu().numpy())

            
            # Append individual items
            gt_mask_list.append(single_mask.cpu().detach().numpy())
            gt_list.append(single_y.cpu().detach().numpy())
            

    # Rest of the function remains the same
    gt_list = np.array(gt_list)
    gt_mask_list = np.array(gt_mask_list)  # Now all masks have same shape
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)


    if CLASS_INDEX[args.obj] > 0:

        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)

        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min())
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min())
    
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

        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min())
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min())
    
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')

        return img_roc_auc_det

if __name__ == '__main__':
    main()

