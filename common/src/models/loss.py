import torch
import torch.nn as nn

# Custom losses
class DistanceLoss(nn.Module):
    def __init__(self, thre):
        super(DistanceLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.thre = thre
        
    def forward(self, pred, label):
        mask = (label < self.thre).float() + (label >= self.thre).float() * torch.exp(-label/10)
        mask_mse = mask
        loss_mse = self.mse(pred, label)
        den_mse = torch.maximum(torch.sum(mask_mse), torch.ones(1, device=pred.device))
        loss = torch.sum(loss_mse * mask_mse) / den_mse 

        return loss

