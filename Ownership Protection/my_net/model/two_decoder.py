from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
import h5py
from PIL import Image
import os
import time
import math
import random
import hashlib
from spectral import SpectralNorm
class ConvBNReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvBNReLU, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv = SpectralNorm(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class Decoder(nn.Module):
    def __init__(self, in_channels, message_length):  # message_length is default in lua file,the value 30
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.message_length = message_length
        # if grayscale,input dimension batchsize*1*H*W;else,batchsize*3*H*W	note from lua file
        self.conv0 = nn.Conv2d(in_channels, 64, kernel_size=1,stride=1)
        self.conv1 = ConvBNReLU(64, 64)
        self.conv2 = ConvBNReLU(64, 64)
        self.conv3 = ConvBNReLU(64, 64)
        self.conv4 = ConvBNReLU(64, 64)
        self.conv5 = ConvBNReLU(64, 64)
        self.conv6 = ConvBNReLU(64, 64)
        self.conv7 = ConvBNReLU(64, 64)
        self.conv8 = ConvBNReLU(64, message_length)# batchsize*L*H*W
        self.average_pooling = nn.AdaptiveAvgPool2d(
            1)  # nn.SpatialAdaptiveAveragePooling(1, 1)) in lua file; 2D or 3D? batchsize*L*1*1
        self.linear = SpectralNorm(nn.Linear(message_length, message_length))

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.average_pooling(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out
