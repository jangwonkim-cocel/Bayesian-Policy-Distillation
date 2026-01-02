import torch
import torch.nn as nn
import torch.nn.functional as F


class SGVLB(nn.Module):
    def __init__(self, net, train_size, loss_type='cross_entropy', device='cuda'):
        super(SGVLB, self).__init__()
        self.train_size = train_size
        self.net = net
        self.loss_type = loss_type
        self.device = device

    def forward(self, input, target, kl_weight=1.0):
        assert not target.requires_grad
        kl = torch.FloatTensor([0.0]).to(self.device)
        for module in self.net.children():
            if hasattr(module, 'kl_reg'):
                kl = kl + module.kl_reg()

        if self.loss_type == 'cross_entropy':
            SGVLB = F.cross_entropy(input, target) * self.train_size + kl_weight * kl
        elif self.loss_type in ['l2', 'L2']:
            SGVLB = ((input - target) ** 2).mean() * self.train_size + kl_weight * kl
        else:
            raise NotImplementedError
        return SGVLB
