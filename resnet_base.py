"""This module implements the resnet building blocks

author: Louis-Émile Robitaille
date-of-creation: 2019-01-30
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """Defines a convolutional layer for the encoder

    Args:
        in_channels (int): the number of input channels
        out_channels (int): the number of output channels
        kernel_size (int): the size of the square kernel
        stride (int): the stride to use (optional, default: 1)

        use_norm (bool): flag to use instance normalization
                         (optional, default: True)
        affine (bool): flag to use affine transformation on normalization
                       (optional, default: True)
        pre_activate (bool): flag to use pre-activation for resnet
                             (optional, default: False)
        activation (F): the activation to use (optional, default: F.elu)
        use_wscale (bool): flag to use "equalized learning rate"[1]
                           (optional, default: True)

    Notes:
        [1] is described in https://arxiv.org/pdf/1710.10196.pdf
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, use_norm=True,
                 affine=True, pre_activate=False,
                 activation=F.elu, use_wscale=True):
        super().__init__()
        # convolution options
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride = kernel_size, stride
        self.padding = kernel_size // 2

        # layer options
        self.use_norm, self.affine = use_norm, affine
        self.pre_activate = pre_activate
        self.activation = activation
        self.use_wscale = use_wscale

        # compute the He's initialization std
        self.he_std = np.sqrt(2 / (in_channels * kernel_size ** 2))

        # defines the parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels,
                                                kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # initialize the parameters
        if self.use_wscale:
            nn.init.normal_(self.weight)
        else:
            nn.init.normal_(self.weight, mean=0, std=self.std)
        nn.init.constant_(self.bias, 0)

        # create the normalization layer
        if use_norm:
            self.norm = nn.InstanceNorm2d(out_channels, affine=affine)

    def conv(self, x):
        """Computes the convolution
        """
        if self.use_wscale:
            y = F.conv2d(x, self.std * self.weight, self.bias,
                         self.stride, self.padding)
        else:
            y = F.conv2d(x, self.weight, self.bias,
                         self.stride, self.padding)
        return y

    def forward(self, x):
        if self.pre_activate:
            y = self.norm(x) if self.use_norm else x
            y = self.conv(self.activation(y))
        else:
            y = self.norm(self.conv(x)) if self.use_norm else self.conv(x)
            y = self.activation(y)
        return y


class ResLayer(nn.Module):
    """Defines a ResNet Layer

    x --> [ConvLayer] --> [ConvLayer] --> (+) -->
    |                                      ^
    v                                      |
    ----------------------------------------

    Args:
        num_channels (int): the number of the convolutions input/output channels
        kernel_size (int): the size of the convolutions square kernel
    """

    def __init__(self, num_channels, kernel_size):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size

        self.conv1 = ConvLayer(num_channels, num_channels,
                               kernel_size, pre_activate=True)
        self.conv2 = ConvLayer(num_channels, num_channels,
                               kernel_size, pre_activate=True)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y = y2 + x
        return y



class ResBlock(nn.Module):
    """Defines a ResNet Block

    Args:
        num_layers (int): the number of ResNet Layers
        num_channels (int): the number of the convolutions input/output channels
        kernel_size (int): the size of the convolutions square kernel
    """

    def __init__(self, num_layers, num_channels, kernel_size):
        super().__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_size = kernel_size

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = ResLayer(num_channels, kernel_size)
            self.layers.append(layer)

    def forward(self, x):
        y = x
        for i in range(self.num_layers):
            y = self.layers[i](y)
        return y


class ResNetEncoder(nn.Module):
    """Defines a ResNet Encoder

    Args:
        in_channels (int): the number of input channels
        num_levels (int): the number of resblock levels
        num_layers (list of int): the number of layers in each level
        num_channels (list of int): the number of channels in each level
        kernel_sizes (list of int): the number of kernel size in each level

    Raises:
        AttributeError: If one of the list is not the right size
    """

    def __init__(self, in_channels, num_levels, num_layers,
                 num_channels, kernel_sizes):
        super().__init__()
        self.in_channels = in_channels
        self.num_levels = num_levels
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes

        if len(num_layers) != num_levels:
            raise AttributeError("The length of num_layers does not match the"
                                 " number of levels")

        if len(num_channels) != num_levels:
            raise AttributeError("The length of num_channels does not match the"
                                 " number of levels")

        if len(kernel_sizes) != num_levels:
            raise AttributeError("The length of kernel_sizes does not match the"
                                 " number of levels")

        self.init_conv = ConvLayer(in_channels, num_channels[0], kernel_size=1,
                                   pre_activate=False)

        self.resblocks = nn.ModuleList()
        self.downsize_convs = nn.ModuleList()
        for i in range(num_levels):
            self.resblocks.append(ResBlock(num_layers[i],
                                           num_channels[i],
                                           kernel_sizes[i]))
            if i < (num_levels - 1):
                self.downsize_convs.append(ConvLayer(num_channels[i],
                                                     num_channels[i+1],
                                                     kernel_size=1,
                                                     stride=2,
                                                     pre_activate=False))

    def forward(self, x):
        batch_size = x.size(0)

        y = self.init_conv(x)
        for i in range(self.num_levels):
            y = self.resblocks[i](y)
            if i < (self.num_levels - 1):
                y = self.downsize_convs[i](y)

        # global average pooling
        y = y.view(batch_size, self.num_channels[-1], -1).mean(dim=2)

        return y


class ClassHead(nn.Module):
    """Defines a classification head

    Args:
        in_units (int): the number of input units
        num_levels (int): the number of unit levels
        num_units (list of int): the number of units in each level
        num_classes (int): the number of output classes
        use_norm (bool): flag to use instance normalization
                         (optional, default: True)
        dropout (list of float): flag to use dropout (optional, default: None)
    """

    def __init__(self, in_units, num_levels, num_units, num_classes,
                 use_norm=True, dropout=None):
        super().__init__()
        self.in_units = in_units
        self.num_levels = num_levels
        self.num_units = num_units
        self.num_classes = num_classes
        self.use_norm = use_norm
        self.use_dropout = dropout is not None

        if len(num_units) != num_levels:
            raise AttributeError("The length of num_units does not match the"
                                 " number of levels")

        if dropout is not None and len(dropout) != num_levels:
            raise AttributeError("The length of dropout does not match the"
                                 " number of levels")

        self.layers = nn.ModuleList()
        for i in range(num_levels):
            layer = []

            if i == 0:
                layer.append(nn.Linear(in_units, num_units[i]))
            else:
                layer.append(nn.Linear(num_units[i-1], num_units[i]))

            if use_norm:
                layer.append(nn.BatchNorm1d(num_units[i]))

            layer.append(nn.ELU())

            if self.use_dropout:
                layer.append(nn.Dropout(dropout[i]))

            self.layers.append(nn.Sequential(*layer))

        self.out = nn.Linear(num_units[-1], num_classes)

    def forward(self, x):
        for i in range(self.num_levels):
            y = self.layers[i](y)
        return F.softmax(self.out(y))
