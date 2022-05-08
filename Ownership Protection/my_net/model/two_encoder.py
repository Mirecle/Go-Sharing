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
class Encoder(nn.Module):
    def __init__(self, in_channels, message_length):
        super(Encoder, self).__init__()
        # if grayscale,input dimension batchsize*1*H*W;else,batchsize*3*H*W	note from lua file
        self.conv0 = nn.Conv2d(in_channels, 64, kernel_size=1,stride=1)
        self.conv1 = ConvBNReLU((64 + message_length ), 64)
        self.conv2 = ConvBNReLU((64 + message_length ), 64)
        self.conv3 = ConvBNReLU((64 + message_length ), 64)
        self.conv4 = ConvBNReLU((64 + message_length ), 64)  # batchsize*64*H*W
        self.conv5 = ConvBNReLU((64 + message_length ), 64)
        self.conv6 = nn.Conv2d(64, in_channels, kernel_size=1, stride=1)
        # x: cover image; y: message_volume
        # for p in self.parameters():
        #     p.requires_grad = False
    def forward(self, x, y, S_factor):
        y = y.unsqueeze(-1)
        y.unsqueeze_(-1)

        y = y.expand(-1,-1, 128, 128)
        z = self.conv0(x)
        z = torch.cat((y, z), dim=1)
        z = self.conv1(z)
        z = torch.cat((y, z), dim=1)
        z = self.conv2(z)
        z = torch.cat((y, z), dim=1)
        z = self.conv3(z)
        z = torch.cat((y, z), dim=1)
        z = self.conv4(z)
        z = torch.cat((y, z), dim=1)
        z = self.conv5(z)
        z = self.conv6(z)
        return z*S_factor + x