import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import os
import json
import numpy as np
from PIL import Image
import h5py

import sys

from libs.datasets.augment import to_pil_image, normalize_meanstd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../'))


class AcdcDataset_Isensee(Dataset):
    def __init__(self, data_list, img_size=352,Normalization=normalize_meanstd(),Isensee_augment=None, prob=0.67):

        self.data_list = data_list
        self.img_size = img_size
        self.augment = Isensee_augment
        self.normalization = Normalization
        self.prob = prob

        with open(data_list, 'r') as f:
            self.data_infos = json.load(f)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):

        img = np.array(h5py.File(self.data_infos[index], 'r')['image'])
        gt = np.array(h5py.File(self.data_infos[index], 'r')['label'])
        original_shape = np.array(img.shape[:2])
        pixcel_spacing = np.array(h5py.File(self.data_infos[index], 'r')['pixel_spacing'][()])
        img_as_image = to_pil_image(img[:, :, None].astype(np.float32))
        gt_as_image = to_pil_image(gt[:, :, None].astype(np.float32))
        img_resize = img_as_image.resize((self.img_size, self.img_size), Image.BILINEAR)
        gt_resize = gt_as_image.resize((self.img_size, self.img_size), Image.NEAREST)
        img_numpy = np.array(img_resize)
        gt_numpy = np.array(gt_resize)


        img = img_numpy.squeeze()[None,None,:,:].astype(np.float32)
        gt = gt_numpy.squeeze()[None,None,:,:].astype(np.float32)


        if self.augment is not None:
            rnd_val = np.random.uniform()
            if rnd_val<self.prob:
                img,gt = self.augment(img, gt)

        img = img.squeeze(0)
        gt = gt.squeeze(0)

        img = self.normalization(img)

        img = torch.from_numpy(img)
        gt  = torch.from_numpy(gt)


        return img, gt, original_shape,pixcel_spacing

        #===============================================================================================================
        # pixcel_spacing = np.array(h5py.File(self.data_infos[index], 'r')['pixel_spacing'][()])
        # roi_center = np.array(h5py.File(self.data_infos[index], 'r')['roi_center'][()])
        # roi_radii = np.array(h5py.File(self.data_infos[index], 'r')['roi_radii'][()])
        # original_shape = np.array(img.shape[:2])


if __name__ == '__main__':
    a = np.array([1, 2, 3, 4])
    b = np.zeros(4)
    print(b)
    c = np.array([22, 33])
    b[2:3] = c
    print(b)


