import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime

from CPD_models import CPD_VGG
from CPD_ResNet_models import CPD_ResNet
from data import get_loader
from utils import clip_gradient, adjust_lr


#parser = argparse.ArgumentParser()
#parser.add_argument('--epoch', type=int, default=100, help='epoch number')
#parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
#parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
#parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
#parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
#parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
#parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
#parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
#opt = parser.parse_args()

# Class Args is used instead of parser so that this code can also work in a jupyter notebook enviroment.
class Args:
    epoch = 100
    lr = 1e-4
    batchsize = 10
    trainsize = 448
    clip = 0.5
    is_ResNet = False
    decay_rate = 0.1
    decay_epoch = 25
    transfer_learning = False

args=Args()
opt = args

print('Learning Rate: {} ResNet: {}'.format(opt.lr, opt.is_ResNet))
# build models
if opt.is_ResNet:
    model = CPD_ResNet()
else:
    model = CPD_VGG()
    if opt.transfer_learning:
        print("Transfer learning")
        model.load_state_dict(torch.load('./models/pretrained by authors/CPD.pth'))


model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# dataset path
image_root = '../datasets/trainingnew/image/'
gt_root = '../datasets/trainingnew/mask/'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()


def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        atts, dets = model(images)
        loss1 = CE(atts, gts)
        loss2 = CE(dets, gts)
        loss = loss1 + loss2
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 400 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data))

    if opt.is_ResNet:
        save_path = './models/CPD_ResNet/'
    else:
        save_path = './models/CPD_VGG/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), save_path + 'CPD.pth' + '.%d' % epoch)

print("Let's go!")
for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
