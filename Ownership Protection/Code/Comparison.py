import cv2 as cv2
import PIL
from PIL import Image
import genwatermark
import insert
import numpy as np
import random
import imagehash
import skimage
import os
from matplotlib import pyplot as plt  
from blind_watermark import att
import sys
# bwm1 = WaterMark(password_img=1, password_wm=1)
def Acc(img1,img2):
    # 计算准确率

    (b,g,extracts1)=insert.rgb2Gray(img1)
    watermark1=insert.extract(extracts1)
    (b1,g1,extracts2)=insert.rgb2Gray(img1)
    watermark2=insert.extract(extracts2)
    result1,result_ownership1=genwatermark.deWatermark(watermark1,128)
    result2,result_ownership2=genwatermark.deWatermark(watermark1,128)
    accuracy=insert.Accuracy(result_ownership1,result_ownership2)
    return accuracy
def Whash(Oimg,img):
    w1 = imagehash.whash(Oimg)
    w2 = imagehash.whash(img)
    
    return (1-(w1-w2)/len(w1.hash)**2)*100
def remark(image):
    b,g,r = cv2.split(image)
    img = cv2.merge([r,g,b])#由于 cv2.imread读取的图片是bgr形式,转换为rgb格式
    return img
def PepperandSalt(src,percetage):
   
    M,N,Z=src.shape
    NoiseImg=src.copy()
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        if random.randint(0,1)<=0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

def gasuss_noise(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out
def add_gaussian_noise(image_in, noise_sigma):
    temp_image = image_in.copy()
    mean=0
    var=0
 
    h, w, _ = temp_image.shape
    # 标准正态分布*noise_sigma
    noise = np.random.randn(h, w) * noise_sigma
    noise1 = np.random.normal(mean, var ** 0.5, image_in.shape)
    noisy_image = np.zeros(temp_image.shape)

    noisy_image = temp_image + noise1

        
    return noisy_image
def tamper_attack(image_in,tamper_in=100):
    temp_image = image_in.copy()
    a=tamper_in
    test=np.zeros((a,a,3),dtype='float32')
    
    temp_image[0:a,0:a]=test
    return temp_image
# salt=np.random.random(image.shape)>.988

# pepper=np.random.random(image.shape)>.988


# image=np.maximum(salt*170,image)
# image=np.minimum(pepper*30+image*(~pepper),image)

#image=insert.readImage('oimage-out-out.png')



# img = cv.imread(r'./2/2-out-out.jpg', 1)
# rows, cols, channels = img.shape
# rotate = cv.getRotationMatrix2D((rows*0.5, cols*0.5), 360, 1)
# '''
# 第一个参数：旋转中心点
# 第二个参数：旋转角度
# 第三个参数：缩放比例
# '''


# Waddress='black32.jpg'
image1=insert.readImage(sys.argv[1])
#In='01100001011000010110001101100011001100110110000101100001011000010110001101100011011110010110101111010101010101000000001011000001'

# res = cv.warpAffine(image, rotate, (cols, rows))

# (b,g,extracts)=insert.rgb2Gray(aaa)
# watermark=insert.extract(extracts,d1,d2,32)
# result,result_ownership=genwatermark.deWatermark(watermark,len(genwatermark.encode(ownership)))
image2=insert.readImage(sys.argv[2])
plt.subplot(221)

plt.imshow(remark(image1))
plt.title('Photo 2' ,fontdict={'weight':'normal','size': 12},verticalalignment='baseline')
plt.xlabel('Accuracy of our algorithm\n Similarity of Whash' ,y=-1,fontdict={'weight':'normal','size': 10})
plt.xticks([]), plt.yticks([])

plt.subplot(222)



ac1=Whash(Image.fromarray(np.uint8(image1)),Image.fromarray(np.uint8(image2)))
plt.imshow(remark(image2))

accuracy=Acc(image1,image2)

print('准确率:%f ' %accuracy)
plt.title('Photo 1' ,fontdict={'weight':'normal','size': 12})
plt.xlabel('Accuracy:%.2f%% \nWhash:%.2f%%'%(accuracy,ac1) ,y=-1,fontdict={'weight':'normal','size': 10})
plt.xticks([]), plt.yticks([])
 




plt.show()