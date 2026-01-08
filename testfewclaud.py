import os
import argparse
import random
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from dataset.medical_few import MedDataset
from biomedclip.clip import create_model
from biomedclip.adapter import CLIP_Inplanted
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import augment, cos_sim, encode_text_with_biomedclip_prompt_ensemble1
from prompt import REAL_NAME
import matplotlib.pyplot as plt
import cv2

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


def multi_scale_inference(model, image, scales=[0.8, 1.0, 1.2]):
    """Multi-scale inference for better robustness"""
    B, C, H, W = image.shape
    all_seg_tokens = []
    all_det_tokens = []
    
    for scale in scales:
        if scale != 1.0:
            new_h, new_w = int(H * scale), int(W * scale)
            scaled_img = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=True)
        else:
            scaled_img = image
        
        with torch.no_grad():
            _, seg_tokens, det_tokens = model(scaled_img)
            all_seg_tokens.append(seg_tokens)
            all_det_tokens.append(det_tokens)
    
    # Average predictions across scales
    avg_seg_tokens = []
    avg_det_tokens = []
    
    for layer_idx in range(len(all_seg_tokens[0])):
        layer_seg = torch.stack([tokens[layer_idx] for tokens in all_seg_tokens]).mean(dim=0)
        layer_det = torch.stack([tokens[layer_idx] for tokens in all_det_tokens]).mean(dim=0)
        avg_seg_tokens.append(layer_seg)
        avg_det_tokens.append(layer_det)
    
    return avg_seg_tokens, avg_det_tokens


def save_anomaly_visualization(image, mask, pred_mask, save_path, threshold=0.5):
    """Save visualization of predictions"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    img_np = image.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(mask.cpu().numpy(), cmap='hot')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Predicted mask (heatmap)
    axes[2].imshow(pred_mask, cmap='hot')
    axes[2].set_title('Prediction (Heatmap)')
    axes[2].axis('off')
    
    # Overlay
    overlay = img_np.copy()
    pred_binary = (pred_mask > threshold).astype(np.float32)
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + 0.5 * pred_binary, 0, 1)
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Improved BiomedCLIP Testing')
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=1)  # Use batch_size=1 for multi-scale
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--features_list', type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    parser.add_argument('--multi_scale', type=int, default=1, help='Use multi-scale inference')
    parser.add_argument('--tta', type=int, default=1, help='Use test-time augmentation')
    parser.add_argument('--visualize', type=int, default=0, help='Save visualizations')
    parser.add_argument('--vis_path', type=str, default='./visualizations/')
    
    args, _ = parser.parse_known_args()
    
    print("\nImproved Testing Configuration:")
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
    model.eval()
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.save_path, f'{args.obj}_improved.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Trying standard checkpoint...")
        checkpoint_path = os.path.join(args.save_path, f'{args.obj}.pth')
    
    checkpoint = torch.load(checkpoint_path)
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])
    
    # Load additional components if available
    if 'seg_proj' in checkpoint:
        model.seg_proj.load_state_dict(checkpoint["seg_proj"])
    if 'det_proj' in checkpoint:
        model.det_proj.load_state_dict(checkpoint["det_proj"])
    if 'alpha_backbone' in checkpoint:
        model.alpha_backbone.data = checkpoint["alpha_backbone"].data
        model.alpha_seg.data = checkpoint["alpha_seg"].data
        model.alpha_det.data = checkpoint["alpha_det"].data
    
    print(f"\nLoaded checkpoint: {checkpoint_path}")
    if 'best_result' in checkpoint:
        print(f"Checkpoint best result: {checkpoint['best_result']:.4f}")
    
    # Load datasets
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    # Augmentation for memory bank
    augment_abnorm_img, augment_abnorm_mask = augment(
        test_dataset.fewshot_abnorm_img,
        test_dataset.fewshot_abnorm_mask
    )
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)
    
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=False, **kwargs)
    
    # Text features
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble1(
            clip_model,
            REAL_NAME[args.obj],
            device
        )
    
    # Build memory bank
    print("\nBuilding memory bank...")
    seg_features = []
    det_features = []
    
    for image in tqdm(support_loader, desc='Memory bank'):
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
    
    # Test
    print("\nRunning inference...")
    result = test(
        args,
        model,
        test_loader,
        text_features,
        seg_mem_features,
        det_mem_features
    )
    
    print(f"\n{'='*50}")
    print(f"Final Result: {result:.4f}")
    print(f"{'='*50}")


def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    model.eval()
    
    gt_list = []
    gt_mask_list = []
    det_image_scores_zero = []
    det_image_scores_few = []
    seg_score_map_zero = []
    seg_score_map_few = []
    
    if args.visualize:
        os.makedirs(args.vis_path, exist_ok=True)
    
    for idx, (image, y, mask) in enumerate(tqdm(test_loader, desc='Testing')):
        image = image.to(device)
        mask = (mask > 0.5).float()
        
        # Test-time augmentation
        if args.tta:
            # Original + horizontal flip
            predictions = []
            
            for flip in [False, True]:
                test_img = torch.flip(image, dims=[3]) if flip else image
                
                if args.multi_scale:
                    seg_tokens, det_tokens = multi_scale_inference(model, test_img)
                else:
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        _, seg_tokens, det_tokens = model(test_img)
                
                predictions.append((seg_tokens, det_tokens, flip))
            
            # Average predictions
            avg_seg_tokens = []
            avg_det_tokens = []
            
            for layer_idx in range(len(predictions[0][0])):
                layer_preds_seg = []
                layer_preds_det = []
                
                for seg_tokens, det_tokens, flip in predictions:
                    if flip:
                        # Flip back spatial dimensions
                        seg_t = seg_tokens[layer_idx]
                        det_t = det_tokens[layer_idx]
                        # Handle flipping for patch tokens if needed
                        layer_preds_seg.append(seg_t)
                        layer_preds_det.append(det_t)
                    else:
                        layer_preds_seg.append(seg_tokens[layer_idx])
                        layer_preds_det.append(det_tokens[layer_idx])
                
                avg_seg_tokens.append(torch.stack(layer_preds_seg).mean(dim=0))
                avg_det_tokens.append(torch.stack(layer_preds_det).mean(dim=0))
            
            seg_patch_tokens = avg_seg_tokens
            det_patch_tokens = avg_det_tokens
        
        else:
            if args.multi_scale:
                seg_patch_tokens, det_patch_tokens = multi_scale_inference(model, image)
            else:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    _, seg_patch_tokens, det_patch_tokens = model(image)
        
        # Remove CLS token
        seg_patch_tokens = [p[:, 1:, :] for p in seg_patch_tokens]
        det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]
        
        if CLASS_INDEX[args.obj] > 0:
            # Segmentation task
            # Few-shot
            anomaly_maps_few = []
            for p_idx, p in enumerate(seg_patch_tokens):
                p_flat = p.reshape(-1, p.shape[-1])
                cos = cos_sim(seg_mem_features[p_idx], p_flat)
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
            
            # Zero-shot
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
            
            # Visualization
            if args.visualize and idx < 20:
                combined_map = 0.6 * score_map_zero[0] + 0.4 * score_map_few[0, 0]
                vis_path = os.path.join(args.vis_path, f'{args.obj}_sample_{idx}.png')
                save_anomaly_visualization(
                    image[0],
                    mask[0, 0],
                    combined_map,
                    vis_path
                )
        
        else:
            # Detection task
            # Few-shot
            anomaly_maps_few = []
            for p_idx, p in enumerate(det_patch_tokens):
                p_flat = p.reshape(-1, p.shape[-1])
                cos = cos_sim(det_mem_features[p_idx], p_flat)
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
    
    # Calculate metrics
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
        
        # Ensemble with optimal weights
        segment_scores = 0.6 * seg_score_map_zero + 0.4 * seg_score_map_few
        
        # Pixel-level metrics
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        seg_ap = average_precision_score(gt_mask_list.flatten(), segment_scores.flatten())
        
        print(f'\n{args.obj} Pixel-level Metrics:')
        print(f'  pAUC: {seg_roc_auc:.4f}')
        print(f'  AP: {seg_ap:.4f}')
        
        # Image-level metrics
        img_scores = np.max(segment_scores.reshape(segment_scores.shape[0], -1), axis=1)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        img_ap = average_precision_score(gt_list, img_scores)
        
        print(f'{args.obj} Image-level Metrics:')
        print(f'  AUC: {img_roc_auc:.4f}')
        print(f'  AP: {img_ap:.4f}')
        
        return seg_roc_auc + img_roc_auc
    
    else:
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)
        
        # Normalize
        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / \
                               (det_image_scores_zero.max() - det_image_scores_zero.min() + 1e-8)
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / \
                              (det_image_scores_few.max() - det_image_scores_few.min() + 1e-8)
        
        # Ensemble
        image_scores = 0.6 * det_image_scores_zero + 0.4 * det_image_scores_few
        
        img_roc_auc = roc_auc_score(gt_list, image_scores)
        img_ap = average_precision_score(gt_list, image_scores)
        
        print(f'\n{args.obj} Detection Metrics:')
        print(f'  AUC: {img_roc_auc:.4f}')
        print(f'  AP: {img_ap:.4f}')
        
        return img_roc_auc


if __name__ == '__main__':
    main()