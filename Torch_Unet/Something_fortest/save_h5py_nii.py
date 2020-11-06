import json
import h5py
import numpy as np
import SimpleITK as sitk
import os

from tools.train import create_dataloader

root_path = '/home/laisong/github/processed_acdc_dataset/nii'
with open('/home/laisong/github/DirectionalFeature/libs/datasets/acdcjson/test.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
train_loader, test_loader = create_dataloader()

for i in range(len(json_data)):
    name = json_data[i]
    f = h5py.File(json_data[i], 'r')  # 打开h5文件
    label = np.array(f['label'])
    str_test = '/home/laisong/github/processed_acdc_dataset/hdf5_files/test_set/'
    str_train = '/home/laisong/github/processed_acdc_dataset/hdf5_files/train_set/'
    str_val = '/home/laisong/github/processed_acdc_dataset/hdf5_files/validation_set/'

    if str_test in name:
        name = json_data[i].replace(str_test, '')
    elif str_train in name:
        name = json_data[i].replace(str_train, '')
    elif str_val in name:
        name = json_data[i].replace(str_val, '')
    name = name.replace('.hdf5', '')
    save_path = os.path.join(root_path, name + '.nii.gz')
    label = sitk.GetImageFromArray(label)
    sitk.WriteImage(label, save_path)
