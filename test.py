
%%writefile /kaggle/working/devmodel/test.py
#claud code for visualization


import os
import argparse
import random
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, 
    precision_recall_curve, auc
)
from dataset.medical_few import MedDataset
from biomedclip.clip import create_model
from biomedclip.adapterv4 import CLIP_Inplanted
from utils import augment, cos_sim, encode_text_with_biomedclip_prompt_ensemble1
from prompt import REAL_NAME

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {
    'Brain': 3, 'Liver': 2, 'Retina_RESC': 1, 
    'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3
}

def project_tokens(model, tokens):
    """Project tokens from 768-dim to 512-dim (text space)."""
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

def plot_confusion_matrix(y_true, y_pred, save_path, obj_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.title(f'Confusion Matrix - {obj_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{obj_name}_confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Print metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"Confusion Matrix Metrics - {obj_name}")
    print(f"{'='*50}")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
    print(f"-" * 50)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"{'='*50}\n")

def plot_roc_curve(y_true, y_scores, save_path, obj_name, curve_type='detection'):
    """Plot and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {obj_name} ({curve_type})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{obj_name}_roc_{curve_type}.png'), dpi=300)
    plt.close()
    
    return roc_auc

def plot_data_distribution(y_true, y_scores, save_path, obj_name):
    """Plot score distribution for normal vs abnormal samples."""
    normal_scores = y_scores[y_true == 0]
    abnormal_scores = y_scores[y_true == 1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=50, alpha=0.5, label='Normal', color='blue')
    plt.hist(abnormal_scores, bins=50, alpha=0.5, label='Abnormal', color='red')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title(f'Score Distribution - {obj_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{obj_name}_distribution.png'), dpi=300)
    plt.close()
    
    print(f"\n{'='*50}")
    print(f"Score Distribution Statistics - {obj_name}")
    print(f"{'='*50}")
    print(f"Normal samples:   {len(normal_scores)}")
    print(f"  Mean:  {normal_scores.mean():.4f}")
    print(f"  Std:   {normal_scores.std():.4f}")
    print(f"  Min:   {normal_scores.min():.4f}")
    print(f"  Max:   {normal_scores.max():.4f}")
    print(f"-" * 50)
    print(f"Abnormal samples: {len(abnormal_scores)}")
    print(f"  Mean:  {abnormal_scores.mean():.4f}")
    print(f"  Std:   {abnormal_scores.std():.4f}")
    print(f"  Min:   {abnormal_scores.min():.4f}")
    print(f"  Max:   {abnormal_scores.max():.4f}")
    print(f"{'='*50}\n")

def save_predictions_npz(gt_list, predictions, scores, save_path, obj_name, pred_type='detection'):
    """Save predictions to NPZ file."""
    npz_path = os.path.join(save_path, f'{obj_name}_{pred_type}_predictions.npz')
    np.savez(
        npz_path,
        ground_truth=gt_list,
        predictions=predictions,
        scores=scores
    )
    print(f"✓ Saved {pred_type} predictions to: {npz_path}")

def save_segmentation_npz(gt_masks, seg_maps, save_path, obj_name):
    """Save segmentation results to NPZ file."""
    npz_path = os.path.join(save_path, f'{obj_name}_segmentation.npz')
    np.savez_compressed(
        npz_path,
        ground_truth_masks=gt_masks,
        predicted_masks=seg_maps
    )
    print(f"✓ Saved segmentation maps to: {npz_path}")
    print(f"  Ground truth masks shape: {gt_masks.shape}")
    print(f"  Predicted masks shape: {seg_maps.shape}")

def main():
    parser = argparse.ArgumentParser(description='BiomedCLIP Testing with Visualizations')
    parser.add_argument('--model_name', type=str, 
                        default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--results_path', type=str, default='./results/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)

    args, _ = parser.parse_known_args()
    
    # Create results directory
    os.makedirs(args.results_path, exist_ok=True)
    
    print("\n" + "="*60)
    print("BIOMEDCLIP TESTING WITH VISUALIZATIONS")
    print("="*60)
    print(f"Object: {args.obj}")
    print(f"Data path: {args.data_path}")
    print(f"Model path: {args.save_path}")
    print(f"Results path: {args.results_path}")
    print("="*60 + "\n")
    
    setup_seed(args.seed)

    # Load model
    print("Loading BiomedCLIP model...")
    clip_model = create_model(
        model_name=args.model_name, 
        img_size=args.img_size, 
        device=device, 
        pretrained=args.pretrain, 
        require_pretrained=True
    )
    clip_model.eval()

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.eval()

    # Load checkpoint
    checkpoint_path = os.path.join(args.save_path, f'{args.obj}.pth')
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path,weights_only=False)
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])
    print("✓ Model loaded successfully\n")

    # Load dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )

    # Prepare support set
    print("Preparing support set...")
    augment_normal_img, _ = augment(test_dataset.fewshot_norm_img)
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(
        support_dataset, batch_size=1, shuffle=False, **kwargs
    )

    # Text features
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble1(
            clip_model, REAL_NAME[args.obj], device
        )

    # Build memory bank
    print("Building memory bank...")
    seg_features = []
    det_features = []
    for image in tqdm(support_loader, desc="Processing support set"):
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
    print("✓ Memory bank ready\n")

    # Run testing
    test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)


def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    """Enhanced testing with visualizations and NPZ outputs."""
    
    gt_list = []
    gt_mask_list = []
    det_image_scores_zero = []
    det_image_scores_few = []
    seg_score_map_zero = []
    seg_score_map_few = []

    # Project memory bank
    seg_mem_features = project_tokens(model, seg_mem_features)
    det_mem_features = project_tokens(model, det_mem_features)

    print("Running inference on test set...")
    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        batch_size = image.shape[0]
        
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            
            for i in range(batch_size):
                seg_tokens_single = [p[i, 1:, :] for p in seg_patch_tokens]
                det_tokens_single = [p[i, 1:, :] for p in det_patch_tokens]

                seg_tokens_single = project_tokens(model, seg_tokens_single)
                det_tokens_single = project_tokens(model, det_tokens_single)

                if CLASS_INDEX[args.obj] > 0:
                    # Segmentation task
                    anomaly_maps_few_shot = []
                    for idx, p in enumerate(seg_tokens_single):
                        cos = cos_sim(seg_mem_features[idx], p)
                        height = int(np.sqrt(cos.shape[1]))
                        anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                        anomaly_map_few_shot = F.interpolate(
                            anomaly_map_few_shot, size=args.img_size, 
                            mode='bilinear', align_corners=True
                        )
                        anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                    score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                    seg_score_map_few.append(score_map_few)

                    anomaly_maps = []
                    for layer in range(len(seg_tokens_single)):
                        tokens_norm = seg_tokens_single[layer] / seg_tokens_single[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * tokens_norm @ text_features).unsqueeze(0)
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
                    # Detection task
                    anomaly_maps_few_shot = []
                    for idx, p in enumerate(det_tokens_single):
                        cos = cos_sim(det_mem_features[idx], p)
                        height = int(np.sqrt(cos.shape[1]))
                        anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                        anomaly_map_few_shot = F.interpolate(
                            anomaly_map_few_shot, size=args.img_size, 
                            mode='bilinear', align_corners=True
                        )
                        anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                    anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                    score_few_det = anomaly_map_few_shot.mean()
                    det_image_scores_few.append(score_few_det)

                    anomaly_score = 0
                    for layer in range(len(det_tokens_single)):
                        tokens_norm = det_tokens_single[layer] / det_tokens_single[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * tokens_norm @ text_features).unsqueeze(0)
                        anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                        anomaly_score += anomaly_map.mean()
                    det_image_scores_zero.append(anomaly_score.cpu().numpy())

                gt_mask_list.append(mask[i].squeeze().cpu().detach().numpy())
            
            gt_list.extend(y.cpu().detach().numpy())

    # Convert to arrays
    gt_list = np.array(gt_list)
    gt_mask_list = np.array(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)

    print("\n" + "="*60)
    print("RESULTS AND VISUALIZATIONS")
    print("="*60)

    if CLASS_INDEX[args.obj] > 0:
        # Segmentation results
        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)

        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / \
                             (seg_score_map_zero.max() - seg_score_map_zero.min() + 1e-8)
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / \
                            (seg_score_map_few.max() - seg_score_map_few.min() + 1e-8)
        
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        
        # Pixel-level AUC
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'\n{args.obj} Pixel-level AUC: {seg_roc_auc:.4f}')
        
        # Image-level AUC
        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        image_scores = np.max(segment_scores_flatten, axis=1)
        roc_auc_im = roc_auc_score(gt_list, image_scores)
        print(f'{args.obj} Image-level AUC: {roc_auc_im:.4f}')
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # ROC curves
        plot_roc_curve(gt_list, image_scores, args.results_path, args.obj, 'image_level')
        plot_roc_curve(gt_mask_list.flatten(), segment_scores.flatten(), 
                      args.results_path, args.obj, 'pixel_level')
        
        # Confusion matrix (using optimal threshold)
        fpr, tpr, thresholds = roc_curve(gt_list, image_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        predictions = (image_scores >= optimal_threshold).astype(int)
        plot_confusion_matrix(gt_list, predictions, args.results_path, args.obj)
        
        # Data distribution
        plot_data_distribution(gt_list, image_scores, args.results_path, args.obj)
        
        # Save NPZ files
        print("\nSaving results to NPZ files...")
        save_predictions_npz(gt_list, predictions, image_scores, 
                           args.results_path, args.obj, 'image')
        save_segmentation_npz(gt_mask_list, segment_scores, args.results_path, args.obj)
        
        return seg_roc_auc + roc_auc_im

    else:
        # Detection results
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)

        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / \
                                (det_image_scores_zero.max() - det_image_scores_zero.min() + 1e-8)
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / \
                               (det_image_scores_few.max() - det_image_scores_few.min() + 1e-8)
        
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'\n{args.obj} Image-level AUC: {img_roc_auc_det:.4f}')
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # ROC curve
        plot_roc_curve(gt_list, image_scores, args.results_path, args.obj, 'detection')
        
        # Confusion matrix
        fpr, tpr, thresholds = roc_curve(gt_list, image_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        predictions = (image_scores >= optimal_threshold).astype(int)
        plot_confusion_matrix(gt_list, predictions, args.results_path, args.obj)
        
        # Data distribution
        plot_data_distribution(gt_list, image_scores, args.results_path, args.obj)
        
        # Save NPZ file
        print("\nSaving results to NPZ files...")
        save_predictions_npz(gt_list, predictions, image_scores, 
                           args.results_path, args.obj, 'detection')
        
        return img_roc_auc_det


if __name__ == '__main__':
    main()