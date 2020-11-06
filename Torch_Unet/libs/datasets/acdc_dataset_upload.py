import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import os
import json
import numpy as np
from PIL import Image
import h5py

import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../'))


class AcdcDataset(Dataset):
    def __init__(self, data_list, joint_augment=None, augment=None, target_augment=None, ):
        self.joint_augment = joint_augment
        self.augment = augment
        self.target_augment = target_augment
        self.data_list = data_list

        with open(data_list, 'r') as f:
            self.data_infos = json.load(f)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):

        img = h5py.File(self.data_infos[index], 'r')['image']
        img = np.array(img)[:, :, None].astype(np.float32)
        original_shape = img.shape[:2]

        if self.joint_augment is not None:
            img, img = self.joint_augment(img, img)
        if self.augment is not None:
            img = self.augment(img)
        pixcel_spacing = np.array(h5py.File(self.data_infos[index], 'r')['pixel_spacing'][()])


        return img, original_shape,pixcel_spacing

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


