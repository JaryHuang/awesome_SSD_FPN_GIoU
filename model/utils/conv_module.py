import warnings

import torch.nn as nn
from mmcv.cnn import kaiming_init, constant_init

from .norm import build_norm_layer


class ConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 normalize=None,
                 activation='relu',
                 inplace=True,
                 activate_last=True):
        super(ConvModule, self).__init__()
        self.with_norm = normalize is not None
        self.with_activatation = activation is not None
        self.with_bias = bias
        self.activation = activation
        self.activate_last = activate_last

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias)

        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_norm:
            # norm after conv or not
            norm_channels = out_channels if self.activate_last else in_channels
            self.norm_name, norm = build_norm_layer(normalize, norm_channels)
            self.add_module(self.norm_name, norm)

        if self.with_activatation:
            assert activation in ['relu'], 'Only ReLU supported.'
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x, activate=True, norm=True):
        if self.activate_last:
            x = self.conv(x)
            if norm and self.with_norm:
                x = self.norm(x)
            if activate and self.with_activatation:
                x = self.activate(x)
        else:
            if norm and self.with_norm:
                x = self.norm(x)
            if activate and self.with_activatation:
                x = self.activate(x)
            x = self.conv(x)
        return x
