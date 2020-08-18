import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from scipy import misc
import imageio

from CPD_models import CPD_VGG
from CPD_ResNet_models import CPD_ResNet
from data import test_dataset

from crfasrnn import util##
from crfasrnn.crfasrnn_model import CrfRnnNet##

#parser = argparse.ArgumentParser()
#parser.add_argument('--testsize', type=int, default=352, help='testing size')
#parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
#opt = parser.parse_args()

### MyCode
class Args:
    testsize = 448
    is_ResNet = False

args=Args()
opt = args
###

dataset_path = '/home/stamatis/Desktop/Imperial Thesis/Thesis code/datasets/7/'

if opt.is_ResNet:
    model = CPD_ResNet()
    model.load_state_dict(torch.load('CPD-R.pth'))
else:
    model = CPD_VGG()
    model.load_state_dict(torch.load('/home/stamatis/Desktop/Imperial Thesis/Thesis code/models/CPD/6/CPD.pth.29'))

model.cuda()
model.eval()

test_datasets = ['crf']

###import cv2##
###saved_weights_path = "crfasrnn_weights.pth"##
###model_crf = CrfRnnNet()##
###model_crf.load_state_dict(torch.load(saved_weights_path))##
###model_crf.eval()##

for dataset in test_datasets:
    if opt.is_ResNet:
        save_path = './results/ResNet50/' + dataset + '/'
    else:
        save_path = '/home/stamatis/Desktop/Imperial Thesis/Thesis code/models/CPD/6/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + '/images/'
    gt_root = dataset_path + '/full_masks/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        _, res = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        #misc.imsave(save_path+name, res)

        ###res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)##
        imageio.imwrite(save_path+name, res)
        

        ###img_data, img_h, img_w, size = util.get_preprocessed_image(save_path+name)

        ###out = model_crf.forward(torch.from_numpy(img_data))

        ###probs = out.detach().numpy()[0]
        ###label_im = util.get_label_image(probs, img_h, img_w, size)
        ###label_im.save(save_path+"1.png")

