import torch

from torch import nn


class LossWrapper(nn.Module):
    def __init__(self, args):
        super(LossWrapper, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.cl1_loss = nn.L1Loss()
        self.use_cl1_loss = args.USE_CL1_LOSS
    def forward(self, outputs, targets, complement=None, complement_target=None):
        l1_loss = self.l1_loss(outputs, targets)

        if self.use_cl1_loss:
            cl1_loss = self.cl1_loss(complement, complement_target)
            loss = l1_loss + cl1_loss
            return {'l1_loss' : l1_loss, 'cl1_loss': cl1_loss, 'loss': loss}
        else:
            loss = l1_loss
            return {'l1_loss': l1_loss,  'loss': loss}

def build_criterion(args):
    return LossWrapper(args)