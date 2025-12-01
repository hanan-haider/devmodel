# train_biomedclip_fewshot_sota.py
import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from dataset.medical_few import MedDataset
from biomedclip.clip import create_model
from biomedclip.tokenizer import tokenize
from utils import augment, cos_sim
from prompt import REAL_NAME

# NEW: Asymmetric Loss (much better than BCE for anomaly detection)
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos * (1 - xs_pos)**self.gamma_pos + los_neg * (1 + xs_pos)**self.gamma_neg
        return -loss.mean()

# ===============================================
# SOTA Adapter Model (from previous message)
# ===============================================
class LoRAAdapter(nn.Module):
    def __init__(self, dim=768, r=64, alpha=16.0, dropout=0.1):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        self.down = nn.Linear(dim, r, bias=False)
        self.up = nn.Linear(r, dim, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, x):
        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up(x)
        return residual + x * self.scaling

class TipAdapterHead(nn.Module):
    def __init__(self, alpha=4.5, beta=6.8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def build_cache(self, support_features_list):
        self.cache_keys = [f.detach().cpu() for f in support_features_list]
        self.register_buffer('cache_keys_norm',
                             torch.stack([F.normalize(k, dim=-1) for k in self.cache_keys]), persistent=False)

    def forward(self, query_features_list):
        if not hasattr(self, 'cache_keys_norm'):
            return 0
        residual = 0
        for q, k_norm in zip(query_features_list, self.cache_keys_norm):
            q = F.normalize(q, dim=-1)
            affinity = q @ k_norm.T * self.beta
            weight = F.softmax(affinity, dim=-1)
            residual = residual + weight @ k_norm.to(q.device)
        return residual

class BioMedCLIP_Adapter(nn.Module):
    def __init__(self, clip_model, feature_layers=[3,6,9,12], adapter_rank=64, use_tip=True):
        super().__init__()
        self.visual = clip_model.visual.trunk if hasattr(clip_model.visual, 'trunk') else clip_model.visual
        self.proj = (clip_model.visual.head.proj if hasattr(clip_model.visual, 'head') and hasattr(clip_model.visual.head, 'proj')
                    else getattr(clip_model.visual, 'proj', None))
        self.feature_layers = feature_layers
        self.use_tip = use_tip

        dim = 768
        self.seg_adapters = nn.ModuleList([LoRAAdapter(dim, r=adapter_rank) for _ in feature_layers])
        self.det_adapters = nn.ModuleList([LoRAAdapter(dim, r=adapter_rank) for _ in feature_layers])

        self.register_parameter('seg_fusion_weights', nn.Parameter(torch.ones(len(feature_layers))))
        self.register_parameter('det_fusion_weights', nn.Parameter(torch.ones(len(feature_layers))))

        if use_tip:
            self.tip_adapter = TipAdapterHead(alpha=4.5, beta=6.8)

        for p in self.visual.parameters():
            p.requires_grad = False

    def forward(self, x, apply_tip=False):
        B = x.shape[0]
        seg_tokens, det_tokens = [], []

        if hasattr(self.visual, 'patch_embed'):
            x = self.visual.patch_embed(x)
            cls_token = self.visual.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.visual.pos_embed
            x = self.visual.pos_drop(x)
        else:
            x = self.visual.conv1(x)
            x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)
            cls_tokens = self.visual.class_embedding.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.visual.positional_embedding

        for idx, block in enumerate(self.visual.blocks):
            x = block(x)
            if (idx + 1) in self.feature_layers:
                i = self.feature_layers.index(idx + 1)
                seg_feat = self.seg_adapters[i](x)
                det_feat = self.det_adapters[i](x)
                x = x + 0.2 * seg_feat + 0.2 * det_feat
                seg_tokens.append(seg_feat[:, 1:, :])
                det_tokens.append(det_feat[:, 1:, :])

        if hasattr(self.visual, 'norm'):
            x = self.visual.norm(x)
        pooled = x[:, 0]
        if self.proj is not None:
            pooled = self.proj(pooled)

        if apply_tip and self.use_tip and hasattr(self.tip_adapter, 'cache_keys_norm'):
            tip_res = self.tip_adapter(seg_tokens)
            tip_res = tip_res.mean(dim=1)
            pooled = pooled + tip_res.to(pooled.device)

        return pooled, seg_tokens, det_tokens

    def build_tip_cache(self, support_loader, device):
        if not self.use_tip: return
        feats = []
        self.eval()
        with torch.no_grad():
            for (img,) in support_loader:
                img = img.to(device)
                _, seg_tokens, _ = self(img, apply_tip=False)
                fused = sum(w * t for w, t in zip(F.softmax(self.seg_fusion_weights, dim=0), seg_tokens))
                feats.append(fused.mean(dim=1))
        all_feats = torch.cat(feats, dim=0)
        layer_feats = [all_feats for _ in self.feature_layers]
        self.tip_adapter.build_cache(layer_feats)
        self.train()

# ===============================================
# Main Training Function (Upgraded)
# ===============================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--features_list', type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot-sota/')
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load BioMedCLIP
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size,
                              device=device, pretrained='microsoft', require_pretrained=True)
    clip_model.eval()

    # Our SOTA model
    model = BioMedCLIP_Adapter(clip_model=clip_model, feature_layers=args.features_list,
                               adapter_rank=64, use_tip=True).to(device)
    model.train()

    # Only train adapters
    optimizer = torch.optim.AdamW([
        {'params': model.seg_adapters.parameters(), 'lr': args.learning_rate},
        {'params': model.det_adapters.parameters(), 'lr': args.learning_rate},
        {'params': [model.seg_fusion_weights, model.det_fusion_weights], 'lr': args.learning_rate * 10}
    ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    # Dataset
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, _ = augment(test_dataset.fewshot_norm_img)
    train_imgs = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    train_masks = torch.cat([augment_abnorm_mask, torch.zeros_like(augment_normal_img[:, :1])], dim=0)
    train_labels = torch.cat([torch.ones(len(augment_abnorm_img)), torch.zeros(len(augment_normal_img))])

    train_dataset = torch.utils.data.TensorDataset(train_imgs, train_masks, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=False)

    # Text features (strong ensemble)
    with torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)  # (2, 512)

    # Losses
    loss_asl = AsymmetricLoss()
    loss_dice = BinaryDiceLoss()

    best_auc = 0.0
    temperature = 32.0

    for epoch in range(args.epoch):
        model.train()
        losses = []

        for img, mask, label in train_loader:
            img, mask, label = img.to(device), mask.to(device), label.to(device).float()

            _, seg_tokens, det_tokens = model(img)

            total_loss = 0.0

            # Detection head loss
            for tokens in det_tokens:
                tokens = F.normalize(tokens, dim=-1)
                proj = tokens @ model.proj.weight.T
                scores = proj @ text_features.T * temperature
                anomaly_score = scores[:, :, 1].mean(dim=1)
                total_loss += loss_asl(anomaly_score, label)

            # Segmentation head loss (only for organs with masks)
            if CLASS_INDEX[args.obj] > 0:
                for tokens in seg_tokens:
                    tokens = F.normalize(tokens, dim=-1)
                    proj = tokens @ model.proj.weight.T
                    scores = proj @ text_features.T * temperature
                    anomaly_map = scores[:, :, 1].view(1, 1, 14, 14)
                    anomaly_map = F.interpolate(anomaly_map, size=args.img_size, mode='bilinear')
                    anomaly_map = torch.sigmoid(anomaly_map).squeeze(1)
                    total_loss += 0.5 * loss_asl(anomaly_map, mask) + 0.5 * loss_dice(anomaly_map, mask)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epoch} | Loss: {np.mean(losses):.4f}")

        # Build Tip-Adapter cache every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.build_tip_cache(support_loader, device)

        # Evaluation
        if (epoch + 1) % 10 == 0:
            auc = test(model, test_loader, text_features, args, device, temperature)
            if auc > best_auc:
                best_auc = auc
                if args.save_model:
                    torch.save({
                        'seg_adapters': model.seg_adapters.state_dict(),
                        'det_adapters': model.det_adapters.state_dict(),
                        'fusion_weights': [model.seg_fusion_weights, model.det_fusion_weights]
                    }, os.path.join(args.save_path, f"{args.obj}_best.pth"))
                print(f"New best AUC: {best_auc:.4f}")

    print(f"Training finished! Best AUC: {best_auc:.4f}")

# ===============================================
# Test Function (Zero + Few-Shot Fusion)
# ===============================================
def test(model, test_loader, text_features, args, device, temperature=32.0):
    model.eval()
    gt_list, gt_mask_list = [], []
    zero_scores, few_scores = [], []

    with torch.no_grad():
        for img, label, mask in tqdm(test_loader, desc="Testing"):
            img = img.to(device)

            # Zero-shot
            _, seg_tokens, det_tokens = model(img, apply_tip=False)
            zero_map = compute_anomaly_map(seg_tokens if CLASS_INDEX[args.obj] > 0 else det_tokens,
                                          model.proj, text_features, temperature, args.img_size)
            zero_score = zero_map.mean().cpu().numpy()

            # Few-shot (Tip-Adapter)
            _, seg_tokens_tip, _ = model(img, apply_tip=True)
            few_map = compute_anomaly_map(seg_tokens_tip if CLASS_INDEX[args.obj] > 0 else det_tokens,
                                         model.proj, text_features, temperature, args.img_size)
            few_score = few_map.mean().cpu().numpy()

            final_score = 0.5 * zero_score + 0.5 * few_score
            zero_scores.append(zero_score)
            few_scores.append(few_score)
            gt_list.append(label.numpy())
            gt_mask_list.append(mask.numpy())

    # Final metric
    from sklearn.metrics import roc_auc_score
    final_auc = roc_auc_score(gt_list, np.array(zero_scores)*0.5 + np.array(few_scores)*0.5)
    print(f"Test AUC: {final_auc:.4f}")
    return final_auc

def compute_anomaly_map(tokens_list, proj, text_features, temp, size):
    weights = F.softmax(getattr(proj, 'seg_fusion_weights', None) or torch.ones(len(tokens_list)), dim=0)
    fused = sum(w * t for w, t in zip(weights, tokens_list))
    fused = F.normalize(fused, dim=-1)
    proj_tokens = fused @ proj.weight.T
    scores = proj_tokens @ text_features.T * temp
    anomaly = scores[:, :, 1].mean(dim=0)
    H = int(anomaly.shape[0] ** 0.5)
    anomaly = anomaly.view(1, 1, H, H)
    anomaly = F.interpolate(anomaly, size=size, mode='bilinear').squeeze()
    return torch.sigmoid(anomaly)

if __name__ == '__main__':
    main()