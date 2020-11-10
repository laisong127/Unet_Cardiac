import json
import h5py
import numpy as np
import SimpleITK as sitk
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from libs.datasets import joint_augment, AcdcDataset
from libs.datasets import augment as standard_augment
from libs.datasets.acdc_dataset_Isensee import AcdcDataset_Isensee
from libs.datasets.batchgenerators.transforms import SpatialTransform
from libs.datasets.collate_batch import BatchCollator
from tools.train import create_dataloader, args
from libs.configs.config_acdc import cfg
from libs.network.backbone0 import CleanU_Net
from libs.network import U_Net
from skimage.morphology import label
import torch.nn.functional as F
from libs.datasets.augment import to_pil_image
from libs.datasets.acdc_dataset_upload import AcdcDataset as AcdcDataset_Upload

def dice(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()
    smooth = 0.00001

    # intersection = (pred * target).sum(dim=2).sum(dim=2)
    pred_flat = pred.view(1, -1)
    target_flat = target.view(1, -1)

    intersection = (pred_flat * target_flat).sum().item()

    # loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    dice = (2 * intersection + smooth) / (pred_flat.sum().item() + target_flat.sum().item() + smooth)
    return dice

def cal_perfer3d(preds, masks):
    LV_dice = []  # 3
    MYO_dice = []  # 2
    RV_dice = []  # 1

    for i in range(0,381,10):
        LV_dice.append(dice(preds[i:i+10, 3, :, :], masks[i:i+10, 3, :, :]))
        RV_dice.append(dice(preds[i:i+10, 1, :, :], masks[i:i+10, 1, :, :]))
        MYO_dice.append(dice(preds[i:i+10, 2, :, :], masks[i:i+10, 2, :, :]))

    LV_dice = np.array(LV_dice)
    RV_dice = np.array(RV_dice)
    MYO_dice = np.array(MYO_dice)
    return LV_dice.mean(),RV_dice.mean(),MYO_dice.mean()
def cal_perfer(preds, masks):
    LV_dice = []  # 3
    MYO_dice = []  # 2
    RV_dice = []  # 1

    for i in range(preds.shape[0]):
        LV_dice.append(dice(preds[i, 3, :, :], masks[i, 3, :, :]))
        RV_dice.append(dice(preds[i, 1, :, :], masks[i, 1, :, :]))
        MYO_dice.append(dice(preds[i, 2, :, :], masks[i, 2, :, :]))

    LV_dice = np.array(LV_dice)
    RV_dice = np.array(RV_dice)
    MYO_dice = np.array(MYO_dice)
    return LV_dice.mean(),RV_dice.mean(),MYO_dice.mean()



def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).scatter_(1, input.cpu().long(), 1)
    # result_cleanUnet = result_cleanUnet.scatter_(1, input.cpu(), 1)

    return result

def postprocess_prediction(seg):
    # basically look for connected components and choose the largest one, delete everything else
    mask = seg != 0 # change label to {0,1} 0:background 1:mask(many be not one kind)
    lbls = label(mask, 4) # calculate number of connected region
    lbls_sizes = [np.sum(lbls==i) for i in np.unique(lbls)] # calculate every region's number
    largest_region = np.argmax(lbls_sizes[1:]) + 1 # from 1 because need excluding the background
    seg[lbls != largest_region]=0  # only allow one pred region,set others to zero
    return seg

train_joint_transform = joint_augment.Compose([
                            joint_augment.To_PIL_Image(),
                            joint_augment.RandomAffine(0,translate=(0.125, 0.125)),
                            joint_augment.RandomRotate((-180,180)),
                            joint_augment.FixResize(256)
                            ])
transform = standard_augment.Compose([
                    standard_augment.to_Tensor(),
                    standard_augment.normalize_meanstd()])
target_transform = standard_augment.Compose([
                        standard_augment.to_Tensor()])

if cfg.DATASET.NAME == 'acdc':
    train_set = AcdcDataset(data_list=cfg.DATASET.TRAIN_LIST,
                            joint_augment=train_joint_transform,
                            augment=transform, target_augment=target_transform)

train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True,
                          num_workers=1, shuffle=False)
eval_transform = joint_augment.Compose([
    joint_augment.To_PIL_Image(),
    joint_augment.FixResize(256),
    joint_augment.To_Tensor()])

evalImg_transform = standard_augment.Compose([
    # standard_augment.normalize([cfg.DATASET.MEAN], [cfg.DATASET.STD])])
    standard_augment.normalize_meanstd()])

if cfg.DATASET.NAME == 'acdc':
    test_set = AcdcDataset(data_list=cfg.DATASET.TEST_LIST,
                           joint_augment=eval_transform,
                           augment=evalImg_transform)
    test_set_upload = AcdcDataset_Upload(data_list=cfg.DATASET.TEST_UPLOAD,
                                  joint_augment=eval_transform,
                                  augment=evalImg_transform)


test_loader = DataLoader(test_set, batch_size=1, pin_memory=True,
                         num_workers=args.workers, shuffle=False)
test_loader_upload = DataLoader(test_set_upload, batch_size=1, pin_memory=True,
                         num_workers=1, shuffle=False)
do_elastic_transform = True
alpha = (100., 350.)
sigma = (14., 17.)
do_rotation = True
a_x = (0., 2 * np.pi)
a_y = (-0.000001, 0.00001)
a_z = (-0.000001, 0.00001),
do_scale = True
scale_range = (0.7, 1.3)
seeds = 12345

transform = SpatialTransform((352, 352), list(np.array((352, 352)) // 2),do_elastic_transform, alpha,sigma,
                                                   do_rotation, a_x, a_y,a_z,do_scale, scale_range,
                                                   'constant', 0, 3, 'constant',
                                                   0, 0, random_crop=False)


train_set_Isensee = AcdcDataset_Isensee(data_list=cfg.DATASET.TRAIN_LIST, Isensee_augment=transform)

train_loader_Isensee = DataLoader(train_set_Isensee, batch_size=args.batch_size, pin_memory=True,
                          num_workers=1, shuffle=False)


model = CleanU_Net()
nii_numpy = []
nii_numpy_data = []
nii_numpy_lab = []
save_path = '/home/laisong/github/processed_acdc_dataset/nii'
result_path = '/home/laisong/github/Unet_Cardiac/result'
experiment_name = '/result_backbone0_prob0.67'
pth_path = '/ckpt/model_best.pth'
save_npy_path = '/home/laisong/github/Unet_Cardiac/saved_npy'
#=======================================================================================================================
# for i,batch in enumerate(test_loader_upload):
#     data  = batch[0]
#     data_squeeze = data.squeeze()
#     data_as_numpy = data_squeeze.numpy()
#     nii_numpy_data.append(data_as_numpy)
# nii_numpy_data = np.array(nii_numpy_data)
#
# np.save(save_npy_path + '/data' + '_upload.npy',nii_numpy_data)
# nii_numpy_data = np.load(save_npy_path + '/data' + '_upload.npy')
# print(nii_numpy_data.shape)
# nii = sitk.GetImageFromArray(nii_numpy_data)
# sitk.WriteImage(nii, save_path + '/data' + '_upload.nii.gz')
#=======================================================================================================================

#=======================================================================================================================
for batch in tqdm((test_loader)):
    # shape = batch[2]
    # print(shape)
    data = batch[0]
    data = data.to('cuda')
    original_shape = batch[2]
    model_path = result_path + experiment_name + pth_path
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state'])
    model.to('cuda')
    model.eval()
    seg_out = model(data)[0]
    seg_out = F.softmax(seg_out, dim=1)
    _, preds = torch.max(seg_out, 1) # (1,256,256)
    preds = preds.squeeze()
    preds = preds.cpu()
    preds_as_numpy = preds.numpy()
    if np.sum(preds_as_numpy):
        preds_as_numpy = postprocess_prediction(preds_as_numpy)
    nii_numpy.append(preds_as_numpy)
nii_numpy = np.array(nii_numpy)
nii_numpy = nii_numpy.astype(np.int16)
np.save(save_npy_path + experiment_name + '.npy',nii_numpy)
nii_numpy = np.load(save_npy_path + experiment_name + '.npy')
# print(nii_numpy.shape)
nii = sitk.GetImageFromArray(nii_numpy)

sitk.WriteImage(nii, save_path + experiment_name + '.nii.gz')
#=======================================================================================================================

pred = sitk.ReadImage(save_path + experiment_name + '.nii.gz')
pred_numpy = sitk.GetArrayFromImage(pred)
pred_tensor = torch.from_numpy(pred_numpy).unsqueeze(dim=1)
true = sitk.ReadImage(save_path+'/total.nii.gz')
true_numpy = sitk.GetArrayFromImage(true)
true_tensor = torch.from_numpy(true_numpy).unsqueeze(dim=1)
lv,rv,myo = cal_perfer(make_one_hot(pred_tensor,num_classes=4),make_one_hot(true_tensor,num_classes=4))
print(experiment_name,':',lv,rv,myo,(lv+rv+myo)/3)
# cleanUnet_GroupNorm:
#     0.9281110291125365 0.8518843602703697 0.8812323948223711 0.8870759280684258 (no postprocess)
#     0.9272031502565893 0.8581292277517463 0.8801480579810547 0.8884934786631301 (postprocess)
#     0.9567352463530363 0.9249240508760627 0.9047632125532963 0.9288075032607984 (test 3d 10 slice a sample)
# AMTA_Instance:
#     0.9320878348918799 0.8569617372246106 0.885539367675955 0.8915296465974819 (no postprocess)

# cleanUnet_instance:
#     0.9611906490217177 0.9297033111705141 0.9095370377036909 0.9334769992986409 (test 3d 10 slice a sample)
#     0.9321287414997868 0.8582924146998778 0.8887431411372549 0.8930547657789732 (postprocess)
# backbone0:
#     0.9289842022628627 0.8583915506748259 0.8843547571222288 0.890576836686639 (postprocess)

#/result_backbone0 : 0.9317439688483693 0.8536856795646007 0.8826269705092368 0.8893522063074023 (postprocess)
#=======================================================================================================================
# recover size
# nii_numpy = np.load('backbone0_postprocess.npy')
# nii_numpy_recover = []
# for i,batch in enumerate(test_loader1):
#     if i >= 10:
#         break
#     lab = batch[1]
#     original_shape = np.array(lab.shape[2:])
#     pred_as_img = to_pil_image(nii_numpy[i][:,:,None].astype(np.float32))
#     pred_as_recover = pred_as_img.resize((original_shape[1],original_shape[0]), Image.BILINEAR)
#     nii_numpy_recover.append(np.array(pred_as_recover).astype(np.int16))
# print(np.array(nii_numpy_recover).shape)
# np.save('pred_as_recover_p10ED.npy',np.array(nii_numpy_recover))
# nii_numpy_recover = np.load('pred_as_recover_p10ED.npy')
# nii = sitk.GetImageFromArray(nii_numpy_recover)
# sitk.WriteImage(nii, save_path+'pred_as_recover_p10ED.nii.gz')