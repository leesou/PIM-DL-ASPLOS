import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter, UninitializedParameter


class LUTLinear_t(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool=True,
        device=None,
        dtype=None,
        nsharecodebooks: int=1,
        ncentroids: int=16,
        vec_len: int=16,
        fp16=False,
        debug=False,
        distance_p="inf",
        **factory_kwargs,
    ):
        super().__init__()

        self.ncodebooks = in_features // vec_len // nsharecodebooks
        self.in_features = in_features
        self.out_features = out_features
        self.ncentroids = ncentroids

        assert self.in_features % self.ncodebooks == 0
        self.vec_len = vec_len
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.centroids = nn.Embedding(self.ncodebooks, self.ncentroids * self.vec_len, **factory_kwargs)
        self.weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs, requires_grad=False))

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.debug = False

        self.fp16 = fp16
        self.distance_p = distance_p.lower()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5e-2)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def forward(self, x):
        if x.dim() == 2:
            batch, in_features = x.shape
            seq_len = 1
        else:
            batch, seq_len, in_features = x.shape

        print('all shape: ', x.shape, self.centroids.shape, self.weight.shape) if self.debug else None

        x_flat = x.view(batch * seq_len, self.ncodebooks, self.vec_len).permute(1, 0, 2)
        weight_flat = self.weight.view(self.ncodebooks, self.vec_len, self.out_features)
        soft_output = torch.bmm(x_flat, weight_flat).sum(0)

        dist = torch.cdist(x_flat, self.centroids.weight.view(self.ncodebooks, self.ncentroids, self.vec_len), p=float(self.distance_p))
        min_indices = dist.argmin(dim=-1)
        one_hot = F.one_hot(min_indices, num_classes=self.ncentroids).float()

        lut = torch.bmm(self.centroids.weight.view(self.ncodebooks, self.ncentroids, self.vec_len), weight_flat)
        quant_output = torch.bmm(one_hot, lut).sum(0)

        self.lut_loss = (torch.mean((quant_output.detach() - soft_output) ** 2) + torch.mean((quant_output - soft_output.detach()) ** 2))

        quant_output = soft_output + (quant_output - soft_output).detach()
        output = quant_output.view(batch, seq_len, self.out_features) if x.dim() == 3 else quant_output.view(batch, self.out_features)

        if self.bias is not None:
            output = output + self.bias
        return output
