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
from biomedclip.clip import create_model, _MODEL_CKPT_PATHS
from biomedclip.tokenizer import tokenize
from biomedclip.adapterv6 import CLIP_Inplanted  # Updated adapter
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_biomedclip_prompt_ensemble1
from prompt import REAL_NAME
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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

# ========================
# LEARNABLE PROMPT MODULE (Compatible with Your Ensemble)
# ========================
class LearnablePromptEnsemble(nn.Module):
    """
    Adds learnable context vectors to your existing prompt ensemble.
    Works by learning refinements to static prompts.
    """
    def __init__(self, clip_model, n_ctx=4, init_std=0.02):
        super().__init__()
        self.n_ctx = n_ctx
        
        # Get text encoder
        text_encoder = clip_model.text.transformer
        ctx_dim = text_encoder.config.hidden_size  # 768 for PubMedBERT
        
        # Learnable context vectors for normal and abnormal
        # Smaller n_ctx (4-8) works better to avoid overfitting
        self.ctx_normal = nn.Parameter(torch.empty(n_ctx, ctx_dim))
        self.ctx_abnormal = nn.Parameter(torch.empty(n_ctx, ctx_dim))
        
        # Initialize with small random values
        nn.init.normal_(self.ctx_normal, std=init_std)
        nn.init.normal_(self.ctx_abnormal, std=init_std)
        
        # Optional: learnable scaling factor
        self.prompt_scale = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self):
        """Returns learnable context vectors"""
        return self.ctx_normal, self.ctx_abnormal

def get_text_features_with_learnable_prompts(
    clip_model, 
    obj, 
    device, 
    prompt_learner=None,
    blend_ratio=0.7
):
    """
    Combines your static prompt ensemble with learnable prompts.
    
    Args:
        clip_model: BioMedCLIP model
        obj: Organ name
        device: torch device
        prompt_learner: LearnablePromptEnsemble module (None for static only)
        blend_ratio: Weight for learnable prompts (0.7 = 70% learnable, 30% static)
    
    Returns:
        text_features: [embed_dim, 2] tensor
    """
    # Get static ensemble features
    static_features = encode_text_with_biomedclip_prompt_ensemble1(
        clip_model, obj, device
    )
    
    if prompt_learner is None:
        return static_features
    
    # Get learnable context
    ctx_normal, ctx_abnormal = prompt_learner()
    
    # Process learnable prompts through text encoder
    text_encoder = clip_model.text.transformer
    
    learnable_features = []
    for ctx in [ctx_normal, ctx_abnormal]:
        # Add batch dimension
        ctx = ctx.unsqueeze(0)  # [1, n_ctx, 768]
        
        # Pass through text transformer
        # Note: We're using the context as input embeddings
        outputs = text_encoder(inputs_embeds=ctx)
        
        # Get [CLS] token (first token) from last hidden state
        embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768]
        
        # Project to embedding space if projection exists
        if hasattr(clip_model.text, 'proj') and clip_model.text.proj is not None:
            embedding = clip_model.text.proj(embedding)
        
        # Normalize
        embedding = F.normalize(embedding.squeeze(0), dim=-1)
        learnable_features.append(embedding)
    
    # Stack learnable features: [embed_dim, 2]
    learnable_features = torch.stack(learnable_features, dim=1)
    
    # Blend with static features
    # blend_ratio controls influence of learned vs static prompts
    scale = prompt_learner.prompt_scale.sigmoid()  # 0-1 range
    final_features = (
        scale * blend_ratio * learnable_features + 
        (1 - scale * blend_ratio) * static_features
    )
    
    # Final normalization
    final_features = F.normalize(final_features, dim=0)
    
    return final_features

# ========================
# EMA for Stability
# ========================
class EMA:
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
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

# ========================
# MAIN FUNCTION
# ========================
def main():
    parser = argparse.ArgumentParser(description='BiomedCLIP Training - Improved')
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=8)  # Reduced from 16
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument("--epoch", type=int, default=100)  # Increased from 50
    parser.add_argument("--learning_rate", type=float, default=0.0025)  # Increased
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    
    # New hyperparameters
    parser.add_argument('--use_learnable_prompts', type=int, default=1)
    parser.add_argument('--n_ctx', type=int, default=4, help='Number of learnable context tokens')
    parser.add_argument('--prompt_blend_ratio', type=float, default=0.7)
    parser.add_argument('--use_ema', type=int, default=1)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    
    args, _ = parser.parse_known_args()
    
    print("\n" + "="*60)
    print("IMPROVED BIOMEDCLIP TRAINING WITH LEARNABLE PROMPTS")
    print("="*60)
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
        features=args.features_list
    ).to(device)
    
    # Initialize learnable prompts
    prompt_learner = None
    if args.use_learnable_prompts:
        prompt_learner = LearnablePromptEnsemble(
            clip_model,
            n_ctx=args.n_ctx
        ).to(device)
        print(f"\n✓ Learnable prompts enabled (n_ctx={args.n_ctx})")
    
    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False
    
    # Unfreeze trainable components
    for p in model.seg_adapters.parameters():
        p.requires_grad = True
    for p in model.det_adapters.parameters():
        p.requires_grad = True
    for p in [model.alpha_backbone, model.alpha_seg, model.alpha_det]:
        p.requires_grad = True
    model.temperature.requires_grad = True
    model.seg_layer_weights.requires_grad = True
    model.det_layer_weights.requires_grad = True
    
    # Count parameters
    adapter_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    prompt_params = sum(p.numel() for p in prompt_learner.parameters()) if prompt_learner else 0
    print(f"\nTrainable parameters:")
    print(f"  Adapters: {adapter_params:,}")
    print(f"  Prompts:  {prompt_params:,}")
    print(f"  Total:    {adapter_params + prompt_params:,}")
    
    # Optimizers with improved settings
    seg_optimizer = AdamW(
        model.seg_adapters.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    det_optimizer = AdamW(
        model.det_adapters.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # Add learnable parameters to optimizers
    seg_optimizer.add_param_group({
        "params": [model.alpha_backbone, model.alpha_seg, model.seg_layer_weights],
        "lr": args.learning_rate
    })
    det_optimizer.add_param_group({
        "params": [model.alpha_backbone, model.alpha_det, model.det_layer_weights, model.temperature],
        "lr": args.learning_rate
    })
    
    # Prompt optimizer (higher learning rate)
    prompt_optimizer = None
    if prompt_learner is not None:
        prompt_optimizer = AdamW(
            prompt_learner.parameters(),
            lr=args.learning_rate * 2,  # 2x learning rate for prompts
            weight_decay=0.0
        )
    
    # Schedulers with longer warmup
    warmup_epochs = 10
    warmup_seg = LinearLR(seg_optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_seg = CosineAnnealingLR(seg_optimizer, T_max=args.epoch-warmup_epochs, eta_min=1e-6)
    seg_scheduler = SequentialLR(seg_optimizer, schedulers=[warmup_seg, cosine_seg], milestones=[warmup_epochs])
    
    warmup_det = LinearLR(det_optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_det = CosineAnnealingLR(det_optimizer, T_max=args.epoch-warmup_epochs, eta_min=1e-6)
    det_scheduler = SequentialLR(det_optimizer, schedulers=[warmup_det, cosine_det], milestones=[warmup_epochs])
    
    prompt_scheduler = None
    if prompt_optimizer is not None:
        warmup_prompt = LinearLR(prompt_optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_prompt = CosineAnnealingLR(prompt_optimizer, T_max=args.epoch-warmup_epochs, eta_min=1e-7)
        prompt_scheduler = SequentialLR(prompt_optimizer, schedulers=[warmup_prompt, cosine_prompt], milestones=[warmup_epochs])
    
    # Load dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    # Data augmentation
    print("\nApplying augmentation...")
    augment_abnorm_img, augment_abnorm_mask = augment(
        test_dataset.fewshot_abnorm_img,
        test_dataset.fewshot_abnorm_mask
    )
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)
    
    print(f"  Normal samples: {len(augment_normal_img)}")
    print(f"  Abnormal samples: {len(augment_abnorm_img)}")
    
    # Create training dataset
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
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=False, **kwargs)
    
    # Losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()
    
    # Generate static text features (for initialization and blending)
    print("\nGenerating static text features...")
    with torch.cuda.amp.autocast(), torch.no_grad():
        static_text_features = encode_text_with_biomedclip_prompt_ensemble1(
            clip_model, REAL_NAME[args.obj], device
        )
    print(f"  Static features shape: {static_text_features.shape}")
    
    # Uncertainty weighting
    log_var_seg = torch.zeros(1, requires_grad=True, device=device)
    log_var_det = torch.zeros(1, requires_grad=True, device=device)
    seg_optimizer.add_param_group({'params': [log_var_seg]})
    det_optimizer.add_param_group({'params': [log_var_det]})
    
    # Initialize EMA
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    if ema:
        print(f"✓ EMA enabled (decay={args.ema_decay})")
    
    best_result = 0
    scaler = torch.cuda.amp.GradScaler()
    
    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60 + "\n")
    
    # ==================== TRAINING LOOP ====================
    for epoch in range(args.epoch):
        model.train()
        if prompt_learner is not None:
            prompt_learner.train()
        
        loss_list = []
        seg_loss_list = []
        det_loss_list = []
        
        for batch_idx, (image, gt, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            
            # Generate text features (static + learnable blend)
            with torch.cuda.amp.autocast():
                text_features = get_text_features_with_learnable_prompts(
                    clip_model,
                    REAL_NAME[args.obj],
                    device,
                    prompt_learner,
                    blend_ratio=args.prompt_blend_ratio
                )
            
            with torch.cuda.amp.autocast():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                
                # Remove CLS token
                seg_patch_tokens = [p[:, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]
                
                # Detection loss with learnable temperature and layer weights
                det_loss = 0
                for layer_idx, raw_tokens in enumerate(det_patch_tokens):
                    projected_tokens = model.visual_proj(raw_tokens)
                    projected_tokens = F.normalize(projected_tokens, dim=-1)
                    
                    # Use learnable temperature
                    anomaly_map = (projected_tokens @ text_features / model.temperature).reshape(
                        image.shape[0], -1, 2
                    )
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = anomaly_map.mean(dim=-1)
                    
                    # Apply learnable layer weight
                    layer_weight = F.softmax(model.det_layer_weights, dim=0)[layer_idx]
                    det_loss += layer_weight * loss_bce(anomaly_score, label.float())
                
                # Segmentation loss (if applicable)
                if CLASS_INDEX[args.obj] > 0:
                    seg_loss = 0
                    mask = gt.to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    
                    for layer_idx, raw_tokens in enumerate(seg_patch_tokens):
                        projected_tokens = model.visual_proj(raw_tokens)
                        projected_tokens = F.normalize(projected_tokens, dim=-1)
                        
                        anomaly_map = (projected_tokens @ text_features / model.temperature)
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(
                            anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                            size=args.img_size,
                            mode='bilinear',
                            align_corners=True
                        )
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        
                        # Apply learnable layer weight
                        layer_weight = F.softmax(model.seg_layer_weights, dim=0)[layer_idx]
                        seg_loss += layer_weight * (
                            loss_focal(anomaly_map, mask) +
                            loss_dice(anomaly_map[:, 1, :, :], mask)
                        )
                    
                    # Uncertainty-weighted loss
                    weighted_seg_loss = torch.exp(-log_var_seg) * seg_loss + log_var_seg
                    weighted_det_loss = torch.exp(-log_var_det) * det_loss + log_var_det
                    loss = weighted_seg_loss + weighted_det_loss
                    
                    seg_loss_list.append(seg_loss.item())
                else:
                    loss = torch.exp(-log_var_det) * det_loss + log_var_det
                
                det_loss_list.append(det_loss.item())
            
            # Backward pass
            seg_optimizer.zero_grad()
            det_optimizer.zero_grad()
            if prompt_optimizer is not None:
                prompt_optimizer.zero_grad()
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(seg_optimizer)
            scaler.unscale_(det_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if prompt_optimizer is not None:
                scaler.unscale_(prompt_optimizer)
                torch.nn.utils.clip_grad_norm_(prompt_learner.parameters(), max_norm=1.0)
            
            scaler.step(seg_optimizer)
            scaler.step(det_optimizer)
            if prompt_optimizer is not None:
                scaler.step(prompt_optimizer)
            scaler.update()
            
            # Update EMA
            if ema is not None:
                ema.update()
            
            loss_list.append(loss.item())
        
        # Step schedulers
        seg_scheduler.step()
        det_scheduler.step()
        if prompt_scheduler is not None:
            prompt_scheduler.step()
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epoch}")
        print(f"{'='*60}")
        print(f"  Total Loss: {np.mean(loss_list):.4f}")
        if seg_loss_list:
            print(f"  Seg Loss:   {np.mean(seg_loss_list):.4f}")
        print(f"  Det Loss:   {np.mean(det_loss_list):.4f}")
        print(f"  Seg LR:     {seg_scheduler.get_last_lr()[0]:.8f}")
        print(f"  Det LR:     {det_scheduler.get_last_lr()[0]:.8f}")
        if prompt_scheduler is not None:
            print(f"  Prompt LR:  {prompt_scheduler.get_last_lr()[0]:.8f}")
        print(f"  Temperature: {model.temperature.item():.4f}")
        if prompt_learner is not None:
            print(f"  Prompt Scale: {prompt_learner.prompt_scale.sigmoid().item():.4f}")
        
        # Build memory bank
        model.eval()
        seg_features = []
        det_features = []
        with torch.no_grad():
            for image in support_loader:
                image = image[0].to(device)
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
        
        # Generate test text features
        with torch.cuda.amp.autocast(), torch.no_grad():
            test_text_features = get_text_features_with_learnable_prompts(
                clip_model,
                REAL_NAME[args.obj],
                device,
                prompt_learner,
                blend_ratio=args.prompt_blend_ratio
            )
        
        # Apply EMA for testing
        if ema is not None:
            ema.apply_shadow()
        
        result = test(args, model, test_loader, test_text_features, seg_mem_features, det_mem_features)
        
        # Restore original weights
        if ema is not None:
            ema.restore()
        
        if result > best_result:
            best_result = result
            print(f"\n🎯 NEW BEST RESULT: {best_result:.4f}")
            
            if args.save_model == 1:
                os.makedirs(args.save_path, exist_ok=True)
                ckp_path = os.path.join(args.save_path, f'{args.obj}_best.pth')
                save_dict = {
                    'seg_adapters': model.seg_adapters.state_dict(),
                    'det_adapters': model.det_adapters.state_dict(),
                    'alpha_backbone': model.alpha_backbone,
                    'alpha_seg': model.alpha_seg,
                    'alpha_det': model.alpha_det,
                    'temperature': model.temperature,
                    'seg_layer_weights': model.seg_layer_weights,
                    'det_layer_weights': model.det_layer_weights,
                    'epoch': epoch,
                    'best_result': best_result
                }
                if prompt_learner is not None:
                    save_dict['prompt_learner'] = prompt_learner.state_dict()
                
                torch.save(save_dict, ckp_path)
                print(f"✅ Model saved to {ckp_path}")
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE - Best Result: {best_result:.4f}")
    print(f"{'='*60}\n")

def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    model.eval()
    
    gt_list = []
    gt_mask_list = []
    det_image_scores_zero = []
    det_image_scores_few = []
    seg_score_map_zero = []
    seg_score_map_few = []
    
    with torch.no_grad():
        for (image, y, mask) in tqdm(test_loader, desc="Testing"):
            image = image.to(device)
            mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
            
            batch_size = image.shape[0]
            
            for i in range(batch_size):
                single_image = image[i:i+1]
                single_y = y[i]
                single_mask = mask[i]
                
                with torch.cuda.amp.autocast():
                    _, seg_patch_tokens, det_patch_tokens = model(single_image)
                    seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                    det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]
                    
                    if CLASS_INDEX[args.obj] > 0:
                        # Segmentation evaluation
                        anomaly_maps_few_shot = []
                        for idx, p in enumerate(seg_patch_tokens):
                            cos = cos_sim(seg_mem_features[idx], p)
                            height = int(np.sqrt(cos.shape[1]))
                            anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                            anomaly_map_few_shot = F.interpolate(
                                torch.tensor(anomaly_map_few_shot),
                                size=args.img_size,
                                mode='bilinear',
                                align_corners=True
                            )
                            # Apply layer weight
                            weight = F.softmax(model.seg_layer_weights, dim=0)[idx]
                            anomaly_maps_few_shot.append(weight.item() * anomaly_map_few_shot[0].cpu().numpy())
                        
                        score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                        seg_score_map_few.append(score_map_few)
                        
                        # Zero-shot segmentation
                        anomaly_maps = []
                        for layer_idx, raw_tokens in enumerate(seg_patch_tokens):
                            projected_tokens = model.visual_proj(raw_tokens)
                            projected_tokens = F.normalize(projected_tokens, dim=-1)
                            
                            anomaly_map = (projected_tokens @ text_features / model.temperature).unsqueeze(0)
                            B, L, C = anomaly_map.shape
                            H = int(np.sqrt(L))
                            anomaly_map = F.interpolate(
                                anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                size=args.img_size,
                                mode='bilinear',
                                align_corners=True
                            )
                            anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                            
                            # Apply layer weight
                            weight = F.softmax(model.seg_layer_weights, dim=0)[layer_idx]
                            anomaly_maps.append(weight.item() * anomaly_map.cpu().numpy())
                        
                        score_map_zero = np.sum(anomaly_maps, axis=0)
                        seg_score_map_zero.append(score_map_zero)
                    
                    else:
                        # Detection evaluation
                        anomaly_maps_few_shot = []
                        for idx, p in enumerate(det_patch_tokens):
                            cos = cos_sim(det_mem_features[idx], p)
                            height = int(np.sqrt(cos.shape[1]))
                            anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                            anomaly_map_few_shot = F.interpolate(
                                torch.tensor(anomaly_map_few_shot),
                                size=args.img_size,
                                mode='bilinear',
                                align_corners=True
                            )
                            weight = F.softmax(model.det_layer_weights, dim=0)[idx]
                            anomaly_maps_few_shot.append(weight.item() * anomaly_map_few_shot[0].cpu().numpy())
                        
                        anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                        score_few_det = anomaly_map_few_shot.mean()
                        det_image_scores_few.append(score_few_det)
                        
                        # Zero-shot detection
                        anomaly_score = 0
                        for layer_idx, raw_tokens in enumerate(det_patch_tokens):
                            projected_tokens = model.visual_proj(raw_tokens)
                            projected_tokens = F.normalize(projected_tokens, dim=-1)
                            
                            anomaly_map = (projected_tokens @ text_features / model.temperature).unsqueeze(0)
                            anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                            
                            weight = F.softmax(model.det_layer_weights, dim=0)[layer_idx]
                            anomaly_score += weight * anomaly_map.mean()
                        
                        det_image_scores_zero.append(anomaly_score.cpu().numpy())
                    
                    gt_mask_list.append(single_mask.cpu().numpy())
                    gt_list.append(single_y.cpu().numpy())
    
    gt_list = np.array(gt_list)
    gt_mask_list = np.array(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)
    
    if CLASS_INDEX[args.obj] > 0:
        # Segmentation metrics
        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)
        
        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min() + 1e-8)
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min() + 1e-8)
        
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'  {args.obj} Pixel AUC: {round(seg_roc_auc, 4)}')
        
        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'  {args.obj} Image AUC: {round(roc_auc_im, 4)}')
        
        return seg_roc_auc + roc_auc_im
    
    else:
        # Detection metrics
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)
        
        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min() + 1e-8)
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min() + 1e-8)
        
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'  {args.obj} Image AUC: {round(img_roc_auc_det, 4)}')
        
        return img_roc_auc_det

if __name__ == '__main__':
    main()
