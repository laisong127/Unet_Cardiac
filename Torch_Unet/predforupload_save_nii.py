import json
import h5py
import numpy as np
import SimpleITK as sitk
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from libs.datasets import joint_augment, AcdcDataset_Upload
from libs.datasets import augment as standard_augment
from libs.datasets.collate_batch import BatchCollator
from tools.train import create_dataloader, args
from libs.configs.config_acdc import cfg
from libs.network.backbone0 import CleanU_Net
from libs.network import U_Net
from skimage.morphology import label
import torch.nn.functional as F
from libs.datasets.augment import to_pil_image


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
    lbls = label(mask, 8) # calculate number of connected region
    lbls_sizes = [np.sum(lbls==i) for i in np.unique(lbls)] # calculate every region's number
    largest_region = np.argmax(lbls_sizes[1:]) + 1 # from 1 because need excluding the background
    print('labls:',np.unique(lbls),'largest_region:',largest_region)
    seg[lbls != largest_region]=0  # only allow one pred region,set others to zero
    return seg

eval_transform = joint_augment.Compose([
    joint_augment.To_PIL_Image(),
    joint_augment.FixResize(256),
    joint_augment.To_Tensor()])

evalImg_transform = standard_augment.Compose([
    standard_augment.normalize_meanstd()])

if cfg.DATASET.NAME == 'acdc':
    test_set = AcdcDataset_Upload(data_list=cfg.DATASET.TEST_UPLOAD,
                           joint_augment=eval_transform,
                           augment=evalImg_transform)


test_loader = DataLoader(test_set, batch_size=1, pin_memory=True,
                         num_workers=args.workers, shuffle=False)

model = CleanU_Net()
nii_numpy_data = []
nii_numpy_lab = []
save_path = '/home/laisong/github/processed_acdc_dataset/nii/upload'
result_path = '/home/laisong/github/Unet_Cardiac/result'
experiment_name = '/result_backbone0'
save_path = save_path + experiment_name + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
pth_path = '/ckpt/model_best.pth'
save_npy_path_ = r'/home/laisong/github/Unet_Cardiac/saved_npy'
save_npy_path = save_npy_path_ + experiment_name
if not os.path.exists(save_npy_path):
    os.makedirs(save_npy_path)
TPE = ['ED','ES']
ID = range(101,151)
spilt_path_ = '/home/laisong/github/Unet_Cardiac/Torch_Unet/libs/datasets/acdcjson/split_json'
split_json = []
spilt_path = os.listdir(spilt_path_)
spilt_path.sort(key= lambda x:str(x[:5]))

for file in spilt_path:
        split_json.append(os.path.join(spilt_path_,file))
mode_index = 0
model_path = result_path + experiment_name + pth_path
print('model_path:',model_path)
for index_id,id in enumerate(ID):
    for index_tye,tye in enumerate(TPE):
        data_list = split_json[mode_index]
        test_set = AcdcDataset_Upload(data_list=data_list,
                                      joint_augment=eval_transform,
                                      augment=evalImg_transform)

        test_loader = DataLoader(test_set, batch_size=1, pin_memory=True,
                                 num_workers=args.workers, shuffle=False)

        nii_numpy = []
        for batch in tqdm(test_loader):
            data = batch[0]
            original_shape = batch[1]
            data = data.to('cuda')

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
            pred_as_img = to_pil_image(preds_as_numpy[:, :, None].astype(np.float32))
            pred_as_recover = pred_as_img.resize((original_shape[1], original_shape[0]), Image.NEAREST)
            pred_as_recover_numpy = np.array(pred_as_recover).astype(np.int16)
            if np.sum(pred_as_recover_numpy):
                print(mode_index, ':', str(id) + tye, 'postprocessing...')
                preds_as_recove_numpy_post = postprocess_prediction(pred_as_recover_numpy)
            else:
                print(mode_index,':',str(id)+tye,'not postprocessed')
                preds_as_recove_numpy_post = pred_as_recover_numpy
            nii_numpy.append(preds_as_recove_numpy_post)
        nii_numpy = np.array(nii_numpy)
        nii_numpy = nii_numpy.astype(np.int16)
        print(mode_index,':',str(id)+tye,nii_numpy.shape)
        np.save(save_npy_path + experiment_name + '_' + str(id)+tye+'_upload.npy',nii_numpy)
        nii_numpy = np.load(save_npy_path + experiment_name + '_' +str(id)+tye+'_upload.npy')
        nii = sitk.GetImageFromArray(nii_numpy)
        sitk.WriteImage(nii, save_path + 'patient%d'%id + '_%s_new.nii.gz'%tye)
        mode_index += 1

# print(split_json)
#
# for i,batch in enumerate(test_loader):
#     data = batch[0]
#     data = data.to('cuda')
#     original_shape = batch[2]
#     model_path = result_path + experiment_name + pth_path
#     checkpoint = torch.load(model_path)
#     model.load_state_dict(checkpoint['model_state'])
#     model.to('cuda')
#     model.eval()
#     seg_out = model(data)
#     seg_out = F.softmax(seg_out, dim=1)
#     _, preds = torch.max(seg_out, 1) # (1,256,256)
#     preds = preds.squeeze()
#     preds = preds.cpu()
#     preds_as_numpy = preds.numpy()
#     if np.sum(preds_as_numpy):
#         preds_as_numpy = postprocess_prediction(preds_as_numpy)
#
#     nii_numpy.append(preds_as_numpy)
# nii_numpy = np.array(nii_numpy)
# nii_numpy = nii_numpy.astype(np.int16)
# np.save(save_npy_path + experiment_name + '_upload.npy',nii_numpy)
# nii_numpy = np.load(save_npy_path + experiment_name + '_upload.npy')
# print(nii_numpy.shape)
# nii = sitk.GetImageFromArray(nii_numpy)
# #
# sitk.WriteImage(nii, save_path + experiment_name + '_upload.nii.gz')
#=======================================================================================================================
# recover size
# name like patient101_ED.nii.gz
# nii_numpy = np.load(save_npy_path + experiment_name + '_upload.npy')
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