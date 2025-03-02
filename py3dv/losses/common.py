import torch
import torch.nn as nn


class SquaredFrobeniusLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, a, b):
        loss = torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1))
        # print(a.shape, b.shape, loss.shape)
        return self.loss_weight * torch.mean(loss)
