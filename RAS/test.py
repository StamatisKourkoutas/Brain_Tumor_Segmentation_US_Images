import torch
import torch.nn.functional as F
import time
import numpy as np
import pdb, os, argparse
import cv2

from RAS import RAS
from data import test_dataset

#parser = argparse.ArgumentParser()
#parser.add_argument('--testsize', type=int, default=352, help='testing size')
#opt = parser.parse_args()

# Class Args is used instead of parser so that this code can also work in a jupyter notebook enviroment.
class Args:
    testsize = 448

args=Args()
opt = args

dataset_path = '../datasets/'

model = RAS()
model.load_state_dict(torch.load('./models/training2_3_9_15/RAS.v1.30.pth'))

model.cuda()
model.eval()

test_datasets = ['testingnew']

for dataset in test_datasets:
    save_path = './results/'+dataset+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/image/'
    test_loader = test_dataset(image_root, opt.testsize)
    time_t = 0.0
    for i in range(test_loader.size):
        image, img_size, name = test_loader.load_data()
        image = image.cuda()
        time_start = time.time()
        res, _, _, _, _ = model(image)
        torch.cuda.synchronize()
        time_end = time.time()
        time_t = time_t + time_end - time_start
        res = F.interpolate(res, img_size, mode='bilinear', align_corners=True)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = 255 * res
        cv2.imwrite(os.path.join(save_path + name), res)
    fps = test_loader.size / time_t
    print('FPS is %f' %(fps))
