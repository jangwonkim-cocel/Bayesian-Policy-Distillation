import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class LinearSVDO(nn.Module):
    def __init__(self, in_features, out_features, alpha_threshold, theta_threshold, device):
        super(LinearSVDO, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha_threshold = alpha_threshold
        self.theta_threshold = theta_threshold
        self.device = device

        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()
        self.W.data.normal_(0, 0.02)
        self.log_sigma.data.fill_(-5)

    def forward(self, x):
        self.log_alpha = self.log_sigma * 2.0 - 2.0 * torch.log(1e-16 + torch.abs(self.W))
        self.log_alpha = torch.clamp(self.log_alpha, -10, 10)

        if self.training:
            lrt_mean = F.linear(x, self.W) + self.bias
            lrt_std = F.linear(torch.sqrt(x * x), torch.exp(2*self.log_sigma)+ 1e-8)
            eps = torch.randn_like(lrt_std)
            return lrt_mean + lrt_std * eps

        out = self.W * (self.log_alpha < self.alpha_threshold).float()
        out = F.linear(x, out) + self.bias
        return out

    def get_pruned_weights(self):
        W = self.W * (self.log_alpha < self.alpha_threshold).float()
        return W

    def get_num_remained_weights(self):
        num = ((self.log_alpha < self.alpha_threshold) * (torch.abs(self.W) > self.theta_threshold)).sum().item()
        return num

    def kl_reg(self):
        k1, k2, k3 = torch.FloatTensor([0.63576]).to(self.device), torch.FloatTensor([1.8732]).to(self.device), torch.FloatTensor([1.48695]).to(self.device)
        KL = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        KL = - torch.sum(KL)
        return KL

