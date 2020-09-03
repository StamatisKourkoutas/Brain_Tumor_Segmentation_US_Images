'''
    Code for black and white mask extraction from color masks.
    The code filters the green, brown and pink color from images.
'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Define the path of a folder with images.
dataset_path = "../datasets/testingnew/"
masks_path = dataset_path+"mask/"
masks = [masks_path + f for f in os.listdir(masks_path) if f.endswith('.jpeg')]
masks = sorted(masks)
size = len(masks)

for mask in masks:
    img = cv2.imread(mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    l, w = r.shape
    filt = np.zeros((l,w)).astype('uint8')

    for i in range(0,l):
        for j in range(0,w):
            #green collor filter: (red and blue channels almost the same, green bigger channel
            if((abs(int(r[i,j])-int(b[i,j]))<5) and (int(g[i,j])-int(b[i,j])>27)):
                filt[i,j] = 255
            #brown collor filter
            elif((int(r[i,j])-int(g[i,j])>5) and (int(g[i,j])-int(b[i,j])>0)):
                filt[i,j] = 255
            #pink color filter
            elif((abs(int(g[i,j])-int(b[i,j]))<5) and int(r[i,j])-int(b[i,j])>5):
                filt[i,j] = 255
            else:
                filt[i,j] = 0

    im = Image.fromarray(filt)
    
    save_path = dataset_path+"bw_masks/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    im.save(save_path+mask.replace(masks_path, ''))
