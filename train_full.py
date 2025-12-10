# main.py

import os
import argparse
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from dataset.full_data import MedDataset
from biomedclip.clip import create_model ,_MODEL_CKPT_PATHS
from biomedclip.adapter import CLIP_Inplanted
from sklearn.metrics import roc_auc_score
from loss import FocalLoss, BinaryDiceLoss
from utils import encode_text_with_biomedclip_prompt_ensemble1
from prompt import REAL_NAME

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

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
    parser = argparse.ArgumentParser(description='BiomedCLIP Training')
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument('--seed', type=int, default=111)

    args, _ = parser.parse_known_args()

    print("\nParsed Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
        global_vars[arg] = getattr(args, arg)
    
    setup_seed(args.seed)
    print("\nSeed set to:", args.seed)

    # ✅ Load BiomedCLIP model - now returns both model and vision_proj automatically
    print("\n" + "="*70)
    print("LOADING BIOMEDCLIP MODEL AND VISION PROJECTION")
    print("="*70)
    
    clip_model = create_model(
        model_name=args.model_name, 
        img_size=args.img_size, 
        device=device, 
        pretrained=args.pretrain, 
        require_pretrained=True
    )
    # Note: create_model internally calls load_biomedclip_model with return_vision_proj=True
    
    clip_model.eval()
    

    # ✅ Create adapter model
    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)

    for name, param in model.named_parameters():
        param.requires_grad = True

    # Optimizers
    seg_optimizer = AdamW(model.seg_adapters.parameters(), lr=args.learning_rate, 
                          betas=(0.5, 0.999), weight_decay=1e-4)
    det_optimizer = AdamW(model.det_adapters.parameters(), lr=args.learning_rate, 
                          betas=(0.5, 0.999), weight_decay=1e-4)

    # Schedulers
    warmup_seg = LinearLR(seg_optimizer, start_factor=0.1, total_iters=5)
    cosine_seg = CosineAnnealingLR(seg_optimizer, T_max=args.epoch-5, eta_min=1e-6)
    seg_scheduler = SequentialLR(seg_optimizer, schedulers=[warmup_seg, cosine_seg], milestones=[5])

    warmup_det = LinearLR(det_optimizer, start_factor=0.1, total_iters=5)
    cosine_det = CosineAnnealingLR(det_optimizer, T_max=args.epoch-5, eta_min=1e-6)
    det_scheduler = SequentialLR(det_optimizer, schedulers=[warmup_det, cosine_det], milestones=[5])

    # Load datasets
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = MedDataset(dataset_path=args.data_path, class_name=args.obj, 
                               split='train', resize=args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, 
                                               shuffle=True, **kwargs)
    
    valid_dataset = MedDataset(dataset_path=args.data_path, class_name=args.obj, 
                               split='valid', resize=args.img_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, 
                                               shuffle=False, **kwargs)



    
    print("Building memory bank from normal training images...")
    
    # Get indices of normal images (label == 0)
    normal_indices = [i for i, (_, label, _) in enumerate(train_dataset) if label == 0]
    
    print(f"Found {len(normal_indices)} normal images out of {len(train_dataset)} total")
    
    # Create subset of normal images
    from torch.utils.data import Subset
    
    support_dataset = Subset(train_dataset, normal_indices)
    support_loader = torch.utils.data.DataLoader(
        support_dataset, 
        batch_size=1, 
        shuffle=False,  # ✅ Don't shuffle memory bank
        **kwargs
    )
    

    # Losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()



    # Load vision projection from checkpoint
    checkpoint_path = _MODEL_CKPT_PATHS[args.model_name]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create projection layer
    vision_proj = nn.Linear(768, 512, bias=False).to(device)
    vision_proj.weight.data = checkpoint['visual.head.proj.weight'].to(device)
    vision_proj.eval()
    
    # Freeze (optional but recommended)
    for param in vision_proj.parameters():
        param.requires_grad = False
    
    print(f"✅ Vision projection loaded: {vision_proj.weight.shape}")

    # ✅ Text features (pre-computed)
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble1(
            clip_model, REAL_NAME[args.obj], device
        )
    print("Text features shape:", text_features.shape)



    best_result = 0

    # ✅ Training loop
    for epoch in range(args.epoch):
        print(f'\nEpoch {epoch}:')
        model.train()
        loss_list = []
        
        for (image, label, gt) in tqdm(train_loader):
            image = image.to(device)
            
            with torch.cuda.amp.autocast():
                # Get patch tokens from adapters
                _, seg_patch_tokens, det_patch_tokens = model(image)
                seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

                # ✅ Detection loss with vision projection
                det_loss = 0
            
                image_label = label.to(device).float()
                
                for layer in range(len(det_patch_tokens)):
        
                    raw_tokens=det_patch_tokens[layer]  #[196,768]
                    #print(f"  Raw tokens shape: {raw_tokens.shape}")

                    # ✅ CRITICAL: Project from 768 to 512 dimensions
                    with torch.no_grad():  # Don't backprop through frozen projection
                        projected_tokens = vision_proj(raw_tokens)  # [196, 768] -> [196, 512]
                    projected_tokens = projected_tokens / projected_tokens.norm(dim=-1, keepdim=True)
                    #print(f"  After normalization, mean norm: {projected_tokens.norm(dim=-1).mean():.4f}")

                    # ✅ NOW dimensions match: [196, 512] @ [512, 2] = [196, 2]
                    anomaly_map = (100.0 * projected_tokens @ text_features).unsqueeze(0) 
                    #print(f"  Anomaly map shape (pre-softmax): {anomaly_map.shape}")  # [1, 196, 2]

                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    #print(f"  Anomaly map shape (post-softmax): {anomaly_map.shape}")  # [1, 196]

                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    #print(f"  Anomaly score: {anomaly_score.item():.4f}")
                    det_loss += loss_bce(anomaly_score, image_label)


                # Segmentation loss (add your code here)
                if CLASS_INDEX[args.obj] > 0:
                    # pixel level
                    seg_loss = 0
                    mask = gt.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    for layer in range(len(seg_patch_tokens)):
                        raw_tokens = seg_patch_tokens[layer]  # [196, 768]

                                            # ✅ CRITICAL: Project from 768 to 512 dimensions
                        with torch.no_grad():  # Don't backprop through frozen projection
                            projected_tokens = vision_proj(raw_tokens)  # [196, 768] -> [196, 512]

                        projected_tokens = projected_tokens / projected_tokens.norm(dim=-1, keepdim=True)
                        #print(f"  After normalization, mean norm: {projected_tokens.norm(dim=-1).mean():.4f}")

                        # ✅ NOW dimensions match: [196, 512] @ [512, 2] = [196, 2]
                        anomaly_map = (100.0 * projected_tokens @ text_features).unsqueeze(0) 

            

                        
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)

                    loss = seg_loss + det_loss
                    loss.requires_grad_(True)
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    loss.backward()
                    seg_optimizer.step()
                    det_optimizer.step()

                else:
                    loss = det_loss
                    loss.requires_grad_(True)
                    det_optimizer.zero_grad()
                    loss.backward()
                    det_optimizer.step()

                loss_list.append(loss.item())

        print("Loss: ", np.mean(loss_list))


        # ✅ ADD THESE LINES AT END OF EPOCH:
        seg_scheduler.step()
        det_scheduler.step()


        
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
        seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) for i in range(len(seg_features[0]))]
        det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in range(len(det_features[0]))]



             

        result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)
        if result > best_result:
            best_result = result
            print("Best result\n")
            if args.save_model == 1:
                ckp_path = os.path.join(args.save_path, f'{args.obj}.pth')
                torch.save({'seg_adapters': model.seg_adapters.state_dict(),
                            'det_adapters': model.det_adapters.state_dict()}, 
                            ckp_path)
          


def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    gt_list = []
    gt_mask_list = []

    det_image_scores_zero = []
    det_image_scores_few = []
    
    seg_score_map_zero = []
    seg_score_map_few= []

    for (image, y, mask) in tqdm(valid_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

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
                    anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features).unsqueeze(0)
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
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score += anomaly_map.mean()
                det_image_scores_zero.append(anomaly_score.cpu().numpy())

            
            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            

    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)


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


if __name__ == '__main__':
    main()
