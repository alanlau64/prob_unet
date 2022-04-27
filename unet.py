import torch
import torch.nn as nn
import numpy as np
from utils import init_weights


class DownConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, initializer, padding, kernel_size=3, stride_size=1, pool=True):
        super(DownConvBlock, self).__init__()
        layers = []
        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        layers.append(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride_size, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(output_dim, output_dim, kernel_size=kernel_size, stride=stride_size, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(output_dim, output_dim, kernel_size=kernel_size, stride=stride_size, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, patch):
        return self.layers(patch)


class UpConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, initializers, padding, kernel_size=2, stride_size=2, bilinear=False):
        super(UpConvBlock, self).__init__()
        self.bilinear = bilinear

        if not self.bilinear:
            self.upconv_layer = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride_size)
            self.upconv_layer.apply(init_weights)

        self.conv_block = DownConvBlock(2 * output_dim, output_dim, initializers, padding, pool=False)

    def forward(self, x, bridge):
        if self.bilinear:
            up = nn.functional.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)
        else:
            up = self.upconv_layer(x)

        assert up.shape[3] == bridge.shape[3]
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out


class Unet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_classes, num_filters, initializers, apply_last_layer=True, padding=True):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvBlock(input, output, initializers, padding, pool=pool))

        self.upsampling_path = nn.ModuleList()

        for i in range(len(self.num_filters) - 1, 0, -1):
            input = output
            output = self.num_filters[i-1]
            self.upsampling_path.append(UpConvBlock(input, output, initializers, padding))

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output, num_classes, kernel_size=1)
            # nn.init.kaiming_normal_(self.last_layer.weight, mode='fan_in',nonlinearity='relu')
            # nn.init.normal_(self.last_layer.bias)

    def forward(self, x, val):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i - 1])

        del blocks

        # Used for saving the activations and plotting
        if val:
            self.activation_maps.append(x)

        if self.apply_last_layer:
            x = self.last_layer(x)

        return x