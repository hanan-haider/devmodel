import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter  # if you use it elsewhere

from dataset.medical_few import MedDataset
from biomedclip.tokenizer import tokenize  # if used elsewhere
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_biomedclip_prompt_ensemble1
from prompt import REAL_NAME
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

# suppress TF / HF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {
    "Brain": 3,
    "Liver": 2,
    "Retina_RESC": 1,
    "Retina_OCT2017": -1,
    "Chest": -2,
    "Histopathology": -3,
}

global_vars = {}


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------
class ClipAdapter(nn.Module):
    def __init__(self, c_in: int, bottleneck: int = 768):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False),
        )

    def forward(self, x):
        x_mid = self.fc1(x)
        x_out = self.fc2(x_mid)
        return x_mid, x_out


class CLIP_Inplanted(nn.Module):
    """
    Wrap BiomedCLIP visual trunk (timm ViT) to output:
      - pooled global image embedding
      - seg_patch_tokens: list[Tensor(B, N, C)]
      - det_patch_tokens: list[Tensor(B, N, C)]
    """

    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model

        # BiomedCLIP visual encoder is a TimmModel wrapper
        assert hasattr(clip_model, "visual"), "clip_model.visual not found"
        assert hasattr(clip_model.visual, "trunk"), "clip_model.visual.trunk not found"

        self.image_encoder = clip_model.visual.trunk  # timm ViT
        self.features = features

        embed_dim = self.image_encoder.num_features  # 768 for ViT-B
        self.seg_adapters = nn.ModuleList(
            [ClipAdapter(embed_dim, bottleneck=embed_dim) for _ in features]
        )
        self.det_adapters = nn.ModuleList(
            [ClipAdapter(embed_dim, bottleneck=embed_dim) for _ in features]
        )

        # projection used by BiomedCLIP
        if hasattr(clip_model.visual, "head") and hasattr(clip_model.visual.head, "proj"):
            self.visual_proj = clip_model.visual.head.proj
        else:
            self.visual_proj = getattr(clip_model.visual, "proj", None)

    def forward(self, x):
        # tokens: (B, N, C) including CLS, from timm ViT
        tokens = self.image_encoder.forward_features(x)  # shape (B, N, C)

        seg_patch_tokens = []
        det_patch_tokens = []

        # apply adapters on same tokens multiple times
        for i in range(len(self.features)):
            seg_mid, seg_out = self.seg_adapters[i](tokens)
            det_mid, det_out = self.det_adapters[i](tokens)
            tokens = 0.8 * tokens + 0.1 * seg_out + 0.1 * det_out
            seg_patch_tokens.append(seg_mid)
            det_patch_tokens.append(det_mid)

        # global pooling: CLS token
        pooled = tokens[:, 0]  # (B, C)
        if self.visual_proj is not None:
            pooled = self.visual_proj(pooled)

        return pooled, seg_patch_tokens, det_patch_tokens


# ---------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------
def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    gt_list = []
    gt_mask_list = []

    det_image_scores_zero = []
    det_image_scores_few = []

    seg_score_map_zero = []
    seg_score_map_few = []

    for image, y, mask in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            # remove CLS token
            seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]  # (N, C)
            det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

            if CLASS_INDEX[args.obj] > 0:
                # ----------------- segmentation (pixel-level) -----------------
                # few-shot, seg head
                anomaly_maps_few_shot = []
                for idx, p in enumerate(seg_patch_tokens):
                    cos = cos_sim(seg_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(
                        1, 1, height, height
                    )
                    anomaly_map_few_shot = F.interpolate(
                        torch.tensor(anomaly_map_few_shot),
                        size=args.img_size,
                        mode="bilinear",
                        align_corners=True,
                    )
                    anomaly_maps_few_shot.append(
                        anomaly_map_few_shot[0].cpu().numpy()
                    )
                score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                seg_score_map_few.append(score_map_few)

                # zero-shot, seg head
                anomaly_maps = []
                for layer in range(len(seg_patch_tokens)):
                    seg_patch_tokens[layer] /= seg_patch_tokens[layer].norm(
                        dim=-1, keepdim=True
                    )
                    anomaly_map = (
                        100.0 * seg_patch_tokens[layer] @ text_features
                    ).unsqueeze(0)
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(
                        anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                        size=args.img_size,
                        mode="bilinear",
                        align_corners=True,
                    )
                    anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                    anomaly_maps.append(anomaly_map.cpu().numpy())
                score_map_zero = np.sum(anomaly_maps, axis=0)
                seg_score_map_zero.append(score_map_zero)

            else:
                # ----------------- detection (image-level) -----------------
                # few-shot, det head
                anomaly_maps_few_shot = []
                for idx, p in enumerate(det_patch_tokens):
                    cos = cos_sim(det_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(
                        1, 1, height, height
                    )
                    anomaly_map_few_shot = F.interpolate(
                        torch.tensor(anomaly_map_few_shot),
                        size=args.img_size,
                        mode="bilinear",
                        align_corners=True,
                    )
                    anomaly_maps_few_shot.append(
                        anomaly_map_few_shot[0].cpu().numpy()
                    )
                anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                score_few_det = anomaly_map_few_shot.mean()
                det_image_scores_few.append(score_few_det)

                # zero-shot, det head
                anomaly_score = 0
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] /= det_patch_tokens[layer].norm(
                        dim=-1, keepdim=True
                    )
                    anomaly_map = (
                        100.0 * det_patch_tokens[layer] @ text_features
                    ).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score += anomaly_map.mean()
                det_image_scores_zero.append(anomaly_score.cpu().numpy())

            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())

    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)

    if CLASS_INDEX[args.obj] > 0:
        # segmentation metrics
        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)

        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (
            seg_score_map_zero.max() - seg_score_map_zero.min()
        )
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (
            seg_score_map_few.max() - seg_score_map_few.min()
        )

        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f"{args.obj} pAUC : {round(seg_roc_auc,4)}")

        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f"{args.obj} AUC : {round(roc_auc_im, 4)}")

        return seg_roc_auc + roc_auc_im

    else:
        # detection metrics
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)

        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (
            det_image_scores_zero.max() - det_image_scores_zero.min()
        )
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (
            det_image_scores_few.max() - det_image_scores_few.min()
        )

        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f"{args.obj} AUC : {round(img_roc_auc_det,4)}")

        return img_roc_auc_det


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BiomedCLIP Testing")
    parser.add_argument(
        "--model_name",
        type=str,
        default="BiomedCLIP-PubMedBERT-ViT-B-16",
    )
    parser.add_argument("--pretrain", type=str, default="microsoft")
    parser.add_argument("--obj", type=str, default="Liver")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_model", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="./ckpt/few-shot/")
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
    )
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--features_list",
        type=int,
        nargs="+",
        default=[3, 6, 9, 12],
    )
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--iterate", type=int, default=0)

    args, _ = parser.parse_known_args()

    print("\nParsed Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
        global_vars[arg] = getattr(args, arg)

    setup_seed(args.seed)
    print("\nSeed set to:", args.seed)

    # BiomedCLIP model + tokenizer
    from open_clip import create_model_from_pretrained, get_tokenizer

    clip_model, preprocess = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    tokenizer = get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    global device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # wrapped vision model
    wrapped_model = CLIP_Inplanted(clip_model, args.features_list)
    print("Wrapped model class:", type(wrapped_model))
    wrapped_model.to(device)
    wrapped_model.eval()

    # dataset
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    test_dataset = MedDataset(
        args.data_path, args.obj, args.img_size, args.shot, args.iterate
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )

    # few-shot augmentation
    augment_abnorm_img, augment_abnorm_mask = augment(
        test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask
    )
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    augment_fewshot_img = torch.cat(
        [augment_abnorm_img, augment_normal_img], dim=0
    )
    augment_fewshot_mask = torch.cat(
        [augment_abnorm_mask, augment_normal_mask], dim=0
    )
    augment_fewshot_label = torch.cat(
        [
            torch.Tensor([1] * len(augment_abnorm_img)),
            torch.Tensor([0] * len(augment_normal_img)),
        ],
        dim=0,
    )

    train_dataset = torch.utils.data.TensorDataset(
        augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, **kwargs
    )

    # support set (memory bank)
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(
        support_dataset, batch_size=1, shuffle=True, **kwargs
    )

    # losses (if you use them later)
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    # text prompt embeddings: use ORIGINAL clip_model (has encode_text)
    with torch.amp.autocast("cuda"), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble1(
            clip_model, REAL_NAME[args.obj], device
        )

    print(text_features.shape)

    # build memory bank using wrapped_model
    seg_features = []
    det_features = []
    for image in support_loader:
        image = image[0].to(device)
        with torch.no_grad():
            _, seg_patch_tokens, det_patch_tokens = wrapped_model(image)
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

    result = test(
        args, wrapped_model, test_loader, text_features, seg_mem_features, det_mem_features
    )
    print("Final metric:", result)


if __name__ == "__main__":
    main()
