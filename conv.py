import cv2
import os
import torch
import torchvision.transforms as transforms

def load_image(i):
    #load one img once a time
    filePath = './images/images/%d.jpg' % i
    img = cv2.imread(filePath)
    h, w = img.shape[:2]
    if w > h:
        side = w
        delta = w - h
        pad_top = delta // 2
        pad_bottom = delta - pad_top
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
    if h > w:
        side = h
        delta = h - w
        pad_left = delta // 2
        pad_right = delta - pad_left
        img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    # cv2.imshow('test',img)
    img = torch.from_numpy((img.astype('float32')/255.0).transpose(2,0,1))
    # print(img.shape)
    return img

def resize_img(img,target_size=(224,224)):

# resize image to target size, return tensor
    img = img.unsqueeze(0)
    resized_img = torch.nn.functional.interpolate(img,target_size,mode='bilinear')
    return resized_img

import os
img_path = './images/images/'
tag_path = './images/tags/'
vaild_path = []

def check_dataset_exist():
    for i in range (1, 5072):
        tag_file = f'{tag_path}{i}.txt'
        img_file = f'{img_path}{i}.jpg'
        if os.path.exists(tag_file):
            if os.path.exists(img_file):
                vaild_path.append(i)

    # print(len(vaild_path))
    return vaild_path
# test
# imgs = load_image(1945)
# resized_imgs = resize_img(imgs)

# print (resized_imgs[0].shape)
# imgtest = resized_imgs[0].permute(1,2,0).numpy()
# imgtest = (imgtest * 255).astype('uint8')
# print (imgtest.shape)
# cv2.imshow('test2',imgtest)
# cv2.waitKey(0)