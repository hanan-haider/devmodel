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
from biomedclip.adapterv5_improved import CLIP_Inplanted  # ✅ Use improved adapter
from sklearn.metrics import roc_auc_score
from utils import augment, cos_sim, encode_text_with_biomedclip_prompt_ensemble1
from prompt import REAL_NAME

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {
    'Brain': 3, 'Liver': 2, 'Retina_RESC': 1,
    'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='BiomedCLIP Few-Shot Testing')
    
    # Model configs
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    
    # Dataset configs
    parser.add_argument('--obj', type=str, default='Liver',
                        help='Organ to test: Brain, Liver, Retina_RESC, Chest, Histopathology')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--img_size', type=int, default=224)
    
    # Test configs
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    
    # ✅ Checkpoint configs (must match training)
    parser.add_argument('--checkpoint_path', type=str, default='./ckpt/few-shot/',
                        help='Path to saved checkpoint')
    parser.add_argument('--features_list', type=int, nargs="+", default=None,
                        help='Layer indices (auto-select if None)')
    parser.add_argument('--bottleneck', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--init_scale', type=float, default=1e-3)
    parser.add_argument('--normalize_alphas', action='store_true')
    
    args, _ = parser.parse_known_args()
    
    # Print config
    print("\n" + "="*60)
    print("BIOMEDCLIP FEW-SHOT TESTING")
    print("="*60)
    for arg in vars(args):
        print(f"  {arg:20s}: {getattr(args, arg)}")
    print("="*60 + "\n")
    
    setup_seed(args.seed)
    
    # ✅ Auto-select layers (must match training)
    if args.features_list is None:
        if CLASS_INDEX[args.obj] > 0:  # Segmentation
            args.features_list = [3, 6, 9, 12]
        else:  # Detection-only
            args.features_list = [6, 9, 12]
        print(f"✓ Auto-selected layers for {args.obj}: {args.features_list}\n")
    
    # Load CLIP model
    print("Loading BioMedCLIP base model...")
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained=args.pretrain,
        require_pretrained=True
    )
    clip_model.eval()
    print("✓ Base model loaded\n")
    
    # ✅ Build model with same config as training
    print("Building adapter model...")
    model = CLIP_Inplanted(
        clip_model=clip_model,
        features=args.features_list,
        bottleneck=args.bottleneck,
        dropout=args.dropout,
        init_scale=args.init_scale,
        normalize_alphas=args.normalize_alphas,
    ).to(device)
    model.eval()
    print("✓ Adapter model built\n")
    
    # ✅ Load trained checkpoint
    checkpoint_file = os.path.join(args.checkpoint_path, f'{args.obj}_best.pth')
    if not os.path.exists(checkpoint_file):
        # Try old naming convention
        checkpoint_file = os.path.join(args.checkpoint_path, f'{args.obj}.pth')
    
    print(f"Loading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Load adapter weights
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])
    
    # ✅ Load alpha parameters if saved
    if "alpha_backbone" in checkpoint:
        model.alpha_backbone.data = checkpoint["alpha_backbone"].to(device)
        model.alpha_seg.data = checkpoint["alpha_seg"].to(device)
        model.alpha_det.data = checkpoint["alpha_det"].to(device)
        print("✓ Loaded learned alpha weights")
    
    print(f"✓ Checkpoint loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"  Training result: {checkpoint.get('best_result', 'N/A'):.4f}\n")
    
    # ✅ Print learned alphas
    model.get_alpha_summary()
    
    # Load test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(
        args.data_path, args.obj, args.img_size, args.shot, args.iterate
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )
    
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples\n")
    
    # Few-shot augmentation (for memory bank)
    print("Building memory bank from few-shot samples...")
    augment_abnorm_img, augment_abnorm_mask = augment(
        test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask
    )
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)
    
    # Memory bank (normal samples only)
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(
        support_dataset, batch_size=1, shuffle=False, **kwargs
    )
    
    # Build memory bank
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
    
    print(f"✓ Memory bank built: {len(seg_features)} normal samples\n")
    
    # Text features
    print("Encoding text prompts...")
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble1(
            clip_model, REAL_NAME[args.obj], device
        )
    print(f"✓ Text features: {text_features.shape}\n")
    
    # Run testing
    print("="*60)
    print("RUNNING EVALUATION")
    print("="*60)
    result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)
    
    print("\n" + "="*60)
    print(f"FINAL RESULT: {result:.4f}")
    print("="*60)


def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    """Evaluation function"""
    model.eval()
    
    gt_list = []
    gt_mask_list = []
    det_image_scores_zero = []
    det_image_scores_few = []
    seg_score_map_zero = []
    seg_score_map_few = []
    
    for (image, y, mask) in tqdm(test_loader, desc="Evaluating"):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
        
        batch_size = image.shape[0]
        
        # Process each image in batch
        for i in range(batch_size):
            single_image = image[i:i+1]
            single_y = y[i]
            single_mask = mask[i]
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                _, seg_patch_tokens, det_patch_tokens = model(single_image)
                seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]
                
                if CLASS_INDEX[args.obj] > 0:
                    # ===== SEGMENTATION TASK =====
                    
                    # Few-shot (memory bank comparison)
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
                    
                    # Zero-shot (text-guided)
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
                    # ===== DETECTION TASK =====
                    
                    # Few-shot
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
                    
                    # Zero-shot
                    anomaly_score = 0
                    for layer in range(len(det_patch_tokens)):
                        raw_tokens = det_patch_tokens[layer]
                        projected_tokens = model.visual_proj(raw_tokens)
                        projected_tokens = projected_tokens / projected_tokens.norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * projected_tokens @ text_features).unsqueeze(0)
                        anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                        anomaly_score += anomaly_map.mean()
                    det_image_scores_zero.append(anomaly_score.cpu().numpy())
                
                gt_mask_list.append(single_mask.cpu().numpy())
                gt_list.append(single_y.cpu().numpy())
    
    gt_list = np.array(gt_list)
    gt_mask_list = np.array(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)
    
    # ===== COMPUTE METRICS =====
    if CLASS_INDEX[args.obj] > 0:
        # Segmentation metrics
        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)
        
        # Normalize
        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / \
                              (seg_score_map_zero.max() - seg_score_map_zero.min() + 1e-8)
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / \
                            (seg_score_map_few.max() - seg_score_map_few.min() + 1e-8)
        
        # Combine zero-shot + few-shot
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        
        # Pixel-level AUC
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'\n  Pixel AUC (pAUC): {seg_roc_auc:.4f}')
        
        # Image-level AUC
        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'  Image AUC (iAUC): {roc_auc_im:.4f}')
        
        return seg_roc_auc + roc_auc_im
    
    else:
        # Detection metrics
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)
        
        # Normalize
        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / \
                                (det_image_scores_zero.max() - det_image_scores_zero.min() + 1e-8)
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / \
                               (det_image_scores_few.max() - det_image_scores_few.min() + 1e-8)
        
        # Combine
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'\n  Image AUC: {img_roc_auc_det:.4f}')
        
        return img_roc_auc_det


if __name__ == '__main__':
    main()
