import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random

class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root)]
        self.gts = [gt_root + f for f in os.listdir(gt_root)]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        # Optionally uncomment these lines to perform live data augmentation.
        #image, gt = self.cv_random_crop(image, gt, 340, 340)
        #image, gt = self.cv_random_flip(image, gt)
        #image, gt = self.cv_random_rotate(image, gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def cv_random_flip(self, img, gt):
        #if np.random.randint(2)==0:
        if random.randint(0,1)==0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        return img, gt

    def cv_random_rotate(self, img, gt):
        #rotate_degree = np.random.random()*2 * 10 - 10
        rotate_degree = random.random()*2 * 10 - 10
        img = img.rotate(rotate_degree, Image.BILINEAR)
        gt = gt.rotate(rotate_degree, Image.NEAREST)
        return img, gt

    def cv_random_crop(self, img, gt, width, height):
        wi, hi = img.size
        wg, hg = gt.size
        assert hi >= height
        assert wi >= width
        assert hg >= height
        assert wg >= width

        x = random.randint(0, wi - width)
        y = random.randint(0, hi - height)
        img = img.crop((x, y, x+width, y+height))
        gt = gt.crop((x, y, x+width, y+height))
        return img, gt

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root)]
        self.gts = [gt_root + f for f in os.listdir(gt_root)]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.bmp'):
            name = name.split('.bmp')[0] + '.jpg'

        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


