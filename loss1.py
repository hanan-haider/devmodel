import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (N, C, H, W) or (N, C) - Raw logits (before Sigmoid/Softmax)
        targets: same shape as inputs - Binary labels (0 or 1)
        """
        # 1. BCE with Logits (Stable)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 2. Get probabilities (pt) for the correct class
        pt = torch.exp(-bce_loss) 
        
        # 3. Calculate Focal Component: (1-pt)^gamma
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply Sigmoid to get probabilities [0, 1]
        probs = torch.sigmoid(logits)
        
        # Flatten for calculation
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice