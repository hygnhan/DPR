import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedCELoss(nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
        
    def forward(self, logits, targets):
        if self.q == 0:
            return F.cross_entropy(logits,targets, reduction='none')

        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = t.gather(p, 1, t.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')
        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight

        return loss


class LogitCorrectionLoss(nn.Module):
    """Define the proposed logit correction loss."""

    def __init__(self, eta: float = 1.):
        super(LogitCorrectionLoss, self).__init__()
        # eta is the hyperparameter for GroupMixUp.
        self.eta = eta

    def forward(self, logits: t.tensor, label: t.tensor, prior=t.tensor(1.)):
        """Calculate Loss."""
        # Calculate the correction.
        correction = t.log((prior ** self.eta) + 1e-4)
        # add correction to the original logit.
        corrected_logits = logits + correction
        loss = F.cross_entropy(corrected_logits, label, reduction='none')
        return loss