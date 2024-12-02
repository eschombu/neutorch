import torch
from torch import nn
import numpy as np


def gunpowder_balance(target: torch.Tensor, mask: torch.Tensor=None, thresh: float=0.):
    if not torch.any(target):
        return None

    if mask is not None:
        bmsk = (mask > 0)
        nmsk = bmsk.sum().item()
        assert nmsk > 0
    else:
        bmsk = torch.ones_like(target, dtype=torch.uint8)
        nmsk = np.prod(bmsk.size())
    
    lpos = (torch.gt(target, thresh) * bmsk).type(torch.float)
    lneg = (torch.le(target, thresh) * bmsk).type(torch.float)

    npos = lpos.sum().item()

    fpos = np.clip(npos / nmsk, 0.05, 0.95)
    fneg = (1.0 - fpos)

    wpos = 1. / (2. * fpos)
    wneg = 1. / (2. * fneg)

    return (lpos * wpos + lneg * wneg).type(torch.float32)


class BinomialCrossEntropyWithLogits(nn.Module):
    """
    A version of BCE w/ logits with the ability to mask
    out regions of output.
    """
    def __init__(self, rebalance: bool = True):
        super().__init__()
        self.rebalance = rebalance
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _reduce_loss(self, loss: torch.Tensor, mask: torch.Tensor=None):
        if mask is None:
            cost = loss.sum() #/ np.prod(loss.size())
        else:
            cost = (loss * mask).sum() #/ mask.sum()
        return cost

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask=None):
        loss = self.bce(pred, target)

        if mask is not None:
            rebalance_weight = gunpowder_balance(target, mask=mask)
            loss *= rebalance_weight

        cost = self._reduce_loss(loss, mask=mask)
        return cost


class FocalLoss(BinomialCrossEntropyWithLogits):
    def __init__(self, alpha: float = 0.25, gamma: float=2., rebalance: bool=True):
        """reweight the loss to focus more on the inaccurate rear spots

        Args:
            alpha (float, optional): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. . Defaults to 0.25.
            gamma (float, optional): Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples. Defaults to 2.
            rebalance (bool, optional): rebalance the positive and negative voxels. Defaults to True.
        """
        super().__init__(rebalance=rebalance)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None):
        """
        implementation was partially copied from here.
        https://github.com/pytorch/vision/blob/master/torchvision/ops/focal_loss.py
        Note that the license is BSD 3-Clause License
        """
        loss = self.bce(pred, target)

        p = torch.sigmoid(pred)
        p_t = p * target + (1 - p) * (1 - target)
        loss = loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1. - self.alpha) * (1. - target)
            loss = alpha_t * loss
        
        if mask is not None:
            rebalance_weight = gunpowder_balance(target, mask=mask)
            loss *= rebalance_weight
   
        cost = self._reduce_loss(loss, mask=mask)
        return cost


class MSEMaskedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None):
        loss = self.mse(pred, target)

        if mask is not None:
            loss *= mask

        cost = loss.sum()
        return cost


class AffinitiesLoss(BinomialCrossEntropyWithLogits):
    pass


class LSDsLoss(MSEMaskedLoss):
    pass


class AffinitiesAndLSDsLoss(nn.Module):
    def __init__(self, num_affinities: int, lsds_to_affs_weight_ratio: float):
        super().__init__()
        self.num_affinities = num_affinities
        self.lsds_to_affs_weight_ratio = lsds_to_affs_weight_ratio
        self.loss_modules = nn.ModuleDict({
            'affs': AffinitiesLoss(),
            'lsds': LSDsLoss()
        })

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None):
        aff_loss = self.loss_modules['affs'](
            prediction[:, :self.num_affinities, ...], target[:, :self.num_affinities, ...], mask=mask)
        lsd_loss = self.loss_modules['lsds'](
            torch.sigmoid(prediction[:, self.num_affinities:, ...]), target[:, self.num_affinities:, ...], mask=mask)
        return aff_loss + self.lsds_to_affs_weight_ratio * lsd_loss


# TO-DO
# tversky loss
# https://gitlab.mpcdf.mpg.de/connectomics/codat/-/blob/master/codat/training/losses.py