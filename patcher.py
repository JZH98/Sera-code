import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from torch.nn import Module
from einops.layers.torch import Rearrange


class Patcher(nn.Module):
    def __init__(self, patch_size, stride, in_chan, out_dim):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
        self.to_out = nn.Sequential(
            nn.Linear(int(in_chan*patch_size[0]*patch_size[1]), out_dim, bias=False),
        )
        self.to_patch = Rearrange("b l n -> b n l")

    def forward(self, x):
        # x: b, k, c, l
        x = self.unfold(x)
        x = self.to_patch(x)
        x = self.to_out(x)
        return x
