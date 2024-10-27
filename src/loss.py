import math
import torch
from torch import nn


class MarginBCEWithLogitsLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        if not (0 < self.margin < 0.5):
            raise ValueError("Margin must be between 0 and 0.5")
        self.z_margin_pos = math.log(self.margin / (1 - self.margin))
        self.z_margin_neg = math.log((1 - self.margin) / self.margin)
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        bce_loss = self.bce_with_logits(logits, target)
        zeros = target == 0
        ones = target == 1
        margin_loss_zeros = 0.0
        margin_loss_ones = 0.0
        if zeros.any():
            margin_loss_zeros = torch.relu(logits[zeros] - (-self.z_margin_neg)).mean()
        if ones.any():
            margin_loss_ones = torch.relu(self.z_margin_pos - logits[ones]).mean()
        total_loss = bce_loss + 0.1 * (margin_loss_zeros + margin_loss_ones)
        return total_loss
