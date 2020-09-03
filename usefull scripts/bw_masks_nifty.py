'''
    Code for black and white mask extraction from nifti files.
'''

import os
import nibabel as nib
import numpy as np
from PIL import Image

# Define the path of a dataset with nifti files.
dataset_path = ""

# Load nibabel image object.
img = nib.load(dataset_path+"002.nii")

# Get data from nibabel image object (returns numpy memmap object)
img_data = np.asanyarray(img.dataobj, dtype='uint8')

# Save images and black and white masks
img_save_path = dataset_path+"images/"
if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)

# Load nibabel segmented image object.
segm = nib.load(dataset_path+"S002.nii")
segm_data = np.asanyarray(segm.dataobj, dtype='uint8')

bw_mask_save_path = dataset_path+"bw_masks/"
if not os.path.exists(bw_mask_save_path):
    os.makedirs(bw_mask_save_path)

image_num, _, _ = img.shape
for i in range(image_num):
    final_img = Image.fromarray(img_data[i,:,:], 'L')
    final_img.save(img_save_path+"image"+str(i)+".jpeg")
    final_segm = Image.fromarray(segm_data[i,:,:]*255, 'L')
    final_segm.save(bw_mask_save_path+"image"+str(i)+".jpeg")
