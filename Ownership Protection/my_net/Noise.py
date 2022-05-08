# coding=gbk

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms
import cv2
import h5py
import os
import time
import math
import random
import hashlib
import skimage
import JPEG2
from PIL import Image,ImageFilter


#Resize 鎶婂師鍥惧ぇ灏忚浆鍙樹负resize_W,resize_H锛岀劧鍚庡啀娆¤浆鍙樺洖鏉?
def Resize(batch_encoded_image_tensor, args):
    if args=="low":
        rangs=random.randint(70,100)/100
    else:
        rangs=random.randint(50,100)/100
    rangs=0.6
    B,C,W,H = batch_encoded_image_tensor.shape
    resize_W, resize_H =int(W*rangs),int(H*rangs)

    batch_encoded_image = batch_encoded_image_tensor.cpu().detach().numpy()
    batch_encoded_image =batch_encoded_image.transpose((0, 2, 3, 1))
    for idx in range(batch_encoded_image.shape[0]):
        encoded_image = batch_encoded_image[idx]
        resize_image = cv2.resize(encoded_image, (resize_W, resize_H))  #still H*W*C after resize
        resize_image = cv2.resize(resize_image,(H, W))
        resize_image = torch.from_numpy(resize_image.transpose((2,0,1))).type(torch.FloatTensor).cuda()
        if (idx == 0):
            batch_noise_image = resize_image.unsqueeze(0)
        else:
            batch_noise_image = torch.cat((batch_noise_image, resize_image.unsqueeze(0)), 0) #batch*H*W*C
    batch_noise_image = Variable(batch_noise_image,requires_grad=True).cuda()   #batch*C*H*W
    return batch_noise_image,{'resize_W':resize_W,'resize_H':resize_H}


#增加48，49行代码，保证图片的尺寸一致
def Zero_padding_two(tensor,args):
    if args=="low":
        weight=random.randint(1,5)
    else:
        weight=random.randint(0,20)
    weight=5
    (batch_size,channel,H,W)=tensor.shape
    y_zero = torch.Tensor(torch.zeros(batch_size,channel,weight,W)).cuda()
    x_zero = torch.Tensor(torch.zeros(batch_size,channel,H+2*weight,weight)).cuda()
    tensor_x = torch.cat((y_zero,tensor),dim=2)
    tensor_x = torch.cat((tensor_x,y_zero),dim=2)
    tensor_x = torch.cat((tensor_x,x_zero),dim=3)
    tensor_x = torch.cat((x_zero,tensor_x), dim=3)
    torch_resize=torchvision.transforms.Resize([H,W])
    tensor_x=torch_resize(tensor_x)
    return tensor_x,{'weight':weight}

def Zero_padding_one(tensor):
    (batch_size,channel,H,W)=tensor.shape
    y_zero = torch.Tensor(torch.zeros(batch_size,channel,1,W)).cuda()
    x_zero = torch.Tensor(torch.zeros(batch_size,channel,H+1,1)).cuda()
    tensor_x = torch.cat((y_zero,tensor),dim=2)
    tensor_x = torch.cat((tensor_x,x_zero),dim=3)
    torch_resize = torchvision.transforms.Resize([H, W])
    tensor_x = torch_resize(tensor_x)
    return tensor_x

def Padding(tensor,img_H,img_W):
    (batch_size, channel, H, W) = tensor.shape
    if((img_H-H)%2==0):
        for i in range(int((img_H-H)/2)):
            tensor = Zero_padding_two(tensor)
    else:
        for i in range(int((img_H-H)/2)):
            tensor = Zero_padding_two(tensor)
        tensor = Zero_padding_one(tensor)
    torch_resize = torchvision.transforms.Resize([H, W])
    tensor = torch_resize(tensor)
    return tensor


##Standard_deviation为均方差
def Gaussian_noise(batch_encoded_image_tensor,args):
    if args=="low":
        Standard_deviation=random.randint(1,15)
    else:
        Standard_deviation=random.randint(1,30)
    Standard_deviation=15
    batch_encoded_image_tensor = (batch_encoded_image_tensor + 1) / 2
    batch_encoded_image = batch_encoded_image_tensor.cpu().detach().numpy()*255
    batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
    for idx in range(batch_encoded_image.shape[0]):
        encoded_image = batch_encoded_image[idx]
        noise_image = skimage.util.random_noise(encoded_image, mode= 'gaussian',clip = False, var = (Standard_deviation) ** 2 )
        noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
        if (idx == 0):
            batch_noise_image = noise_image.unsqueeze(0)
        else:
            batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
    batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
    batch_noise_image = batch_noise_image/255
    batch_noise_image = batch_noise_image * 2 - 1  # 配合HiDDeN -1到1 的归一化
    return batch_noise_image,{'Standard_deviation':Standard_deviation}
def Salt_Pepper(batch_encoded_image_tensor,Amount):

    batch_encoded_image_tensor = (batch_encoded_image_tensor + 1) / 2
    batch_encoded_image = batch_encoded_image_tensor.cpu().detach().numpy()*255
    batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
    for idx in range(batch_encoded_image.shape[0]):
        encoded_image = batch_encoded_image[idx]
        noise_image = skimage.util.random_noise(encoded_image, mode='s&p', amount = Amount)
        noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
        if (idx == 0):
            batch_noise_image = noise_image.unsqueeze(0)
        else:
            batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
    batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
    batch_noise_image = batch_noise_image/255
    batch_noise_image = batch_noise_image * 2 - 1  # 配合HiDDeN -1到1 的归一化
    return batch_noise_image
def Gussian_blur(batch_encoded_image_tensor,size):
    batch_encoded_image_tensor = (batch_encoded_image_tensor + 1) / 2
    batch_encoded_image = batch_encoded_image_tensor.cpu().detach().numpy() * 255
    batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
    for idx in range(batch_encoded_image.shape[0]):
        encoded_image = batch_encoded_image[idx]
        noise_image = cv2.GaussianBlur(encoded_image,(size,size),0)
        noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
        if (idx == 0):
            batch_noise_image = noise_image.unsqueeze(0)
        else:
            batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
    batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
    batch_noise_image = batch_noise_image / 255
    batch_noise_image = batch_noise_image * 2 - 1  # 配合HiDDeN -1到1 的归一化
    return batch_noise_image

def JPEG_Compression(batch_encoded_image_tensor,args):
    Q=50
    batch_encoded_image_tensor = (batch_encoded_image_tensor + 1) / 2
    # 配合HiDDeN -1到1 的归一化
    batch_encoded_image = batch_encoded_image_tensor.cpu().detach().numpy() * 255
    batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
    for idx in range(batch_encoded_image.shape[0]):
        encoded_image = batch_encoded_image[idx]
        noise_image = JPEG2.JPEG_Mask(encoded_image,Q)
        noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
        if (idx == 0):
            batch_noise_image = noise_image.unsqueeze(0)
        else:
            batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
    batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
    batch_noise_image=batch_noise_image / 255
    batch_noise_image = batch_noise_image * 2 - 1# 配合HiDDeN -1到1 的归一化
    return batch_noise_image,{Q:50}


def DropOut(batch_encoded_image_tensor,batch_coverd_image_tensor,P): #P是mask中1的概率
    mask = np.random.choice([0.0, 1.0], batch_encoded_image_tensor.shape[2:], p=[1 - P, P])
    mask_tensor = torch.tensor(mask, device=batch_encoded_image_tensor.device, dtype=torch.float)
    mask_tensor = mask_tensor.expand_as(batch_encoded_image_tensor)#将mask_tensor 由H x W 扩展为batch_size x 3 x H x W
    noised_image = batch_encoded_image_tensor * mask_tensor + batch_coverd_image_tensor * (1 - mask_tensor)
    return noised_image

def random_float(min, max):
    return np.random.rand() * (max - min) + min

def get_random_rectangle_inside(image, p):
    image_height = image.shape[2]
    image_width = image.shape[3]
    remaining_height = int(np.rint(random_float(p, p) * image_height))
    remaining_width = int(np.rint(random_float(p, p) * image_width))
    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)
    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)
    return height_start, height_start+remaining_height, width_start, width_start+remaining_width

def Crop(batch_encoded_image_tensor,args):
    # if args=='low':
    #     p = random.randint(85,100)/100
    # else:
    #     p = random.randint(65, 100)/100
    p=0.85
    (batch_size, channel, H, W) = batch_encoded_image_tensor.shape
    h_start, h_end, w_start, w_end = get_random_rectangle_inside(batch_encoded_image_tensor, p)
    noised_image = batch_encoded_image_tensor[
                          :,
                          :,
                          h_start: h_end,
                          w_start: w_end].clone()
    torch_resize = torchvision.transforms.Resize([H, W])
    noised_image = torch_resize(noised_image)
    return noised_image,{"p":p}

# def CropOut(batch_encoded_image_tensor,args): #等于篡改攻击，其中p是cover image的占比占了p**2
#     if args=='low':
#         p = random.randint(0,200)/1000
#     else:
#         p = random.randint(0,500)/1000
#     p=0.5
#     cropout_mask = torch.zeros_like(batch_encoded_image_tensor)
#     h_start, h_end, w_start, w_end = get_random_rectangle_inside(batch_encoded_image_tensor, p)
#     cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1
#
#     noised_image = batch_encoded_image_tensor * cropout_mask + batch_coverd_image_tensor * (1 - cropout_mask)
#     return noised_image,{"p":p}

def Contrast(batch_encoded_image_tensor,args):
    #duibi:0~2
    #liangdu:0~100
    # if args=='low':
    #     duibi,liangdu=random.randint(50,150)/100,random.randint(1000,3000)/100
    # else:
    #     duibi, liangdu = random.randint(100, 200)/100, random.randint(1000,6000)/100
    duibi=1.5
    liangdu=20
    batch_encoded_image_tensor = (batch_encoded_image_tensor + 1) / 2
    batch_encoded_image = batch_encoded_image_tensor.cpu().detach().numpy() * 255
    batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
    res2 = np.uint8(np.clip((duibi * batch_encoded_image + liangdu), 0, 255))

    noise_image = torch.from_numpy(res2.transpose((0, 3, 1, 2))).type(torch.FloatTensor).cuda()  # 配合HiDDeN -1到1 的归一化
    batch_noise_image = Variable(noise_image, requires_grad=True).cuda()
    batch_noise_image = batch_noise_image / 255
    batch_noise_image = batch_noise_image * 2 - 1  # 配合HiDDeN -1到1 的归一化
    return batch_noise_image,{"duibi":duibi,"liagndu":liangdu}



def randomNoise(batch_encoded_image_tensor,args):
    #num=random.randint(0,4)

    func_list=[None,Contrast,Crop,Gaussian_noise,Zero_padding_two,Resize]
    for j in range(len(batch_encoded_image_tensor)):
        t=batch_encoded_image_tensor[j].unsqueeze(0)
        choose_fuc = random.sample(func_list, 3)
        choose_fuc=[Resize]
        noise_info={}
        for i in choose_fuc:
            if i != None:
                t,noise_info[i.__name__] = i(t,args)
        batch_encoded_image_tensor[j]=t[0]
    return batch_encoded_image_tensor,noise_info







