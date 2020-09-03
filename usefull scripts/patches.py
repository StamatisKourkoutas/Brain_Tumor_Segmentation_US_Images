'''
    Code used to obtain patches from a directory of images
'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


dataset_path = "../datasets/trainingnew/"
image_path = dataset_path+"image/"
masks_path = dataset_path+"mask/"
save_path = dataset_path+"patches/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

case_num = 1

#patch size
M=300
N=300
Xstride = 200
Ystride = 200

masks = [masks_path + f for f in os.listdir(masks_path) if f.endswith('.png')]
masks = sorted(masks)

image_num = 0
for mask in masks:
    img = cv2.imread(mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #plt.imshow(img)
    #plt.show()
    tiles = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],Xstride) for y in range(0,img.shape[1],Ystride)]
    patch_num = 0
    for tile_num in range(len(tiles)):
        #plt.imshow(tiles[tile_num])
        #plt.show()
        im = Image.fromarray(tiles[tile_num])
        im.save(save_path+"case"+str(case_num)+"_image"+str(image_num)+"_patch"+str(patch_num)+".jpg")
        patch_num = patch_num + 1
    image_num = image_num + 1
