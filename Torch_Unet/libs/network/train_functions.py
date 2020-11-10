import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import cv2
from collections import namedtuple
from libs.losses.Cal_Diceloss import make_one_hot, DiceLoss, smooth_dist_loss


def model_backbone0_decorator():
    ModelReturn = namedtuple("ModelReturn", ["loss", "tb_dict", "disp_dict"])

    def model_fn(model, data, criterion, perfermance=False, vis=False, device="cuda", epoch=0, num_class=4, w=None):
        # imgs, gts, _ = data
        imgs, gts = data[:2]
        imgs = imgs.to(device)
        gts = torch.squeeze(gts, 1).to(device)
        gts = gts.to(device)
        loss = 0
        tb_dict = {}
        disp_dict = {}

        if not perfermance:

            net_out = model(imgs)[0]  # [d1]

            # add weight
            ce_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1,4,2,1]).float().to(device))(net_out, gts.long())
            # dc_loss = DiceLoss()(net_out,make_one_hot(gts.unsqueeze(1).long(),num_classes=4).to(device))

            loss = ce_loss

            tb_dict.update({"CE_loss": loss.item()})
            disp_dict.update({"CE_loss": loss.item()})

        if perfermance:
            gts_ = gts.unsqueeze(1)
            model.eval()
            net_out_eval = model(imgs)[0]
            net_out = F.softmax(net_out_eval, dim=1)
            _, preds = torch.max(net_out, 1)
            preds = preds.unsqueeze(1)
            cal_perfer(make_one_hot(preds, num_class), make_one_hot(gts_.long(), num_class), tb_dict)
            model.train()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_fn

def model_fn_decorator():
    ModelReturn = namedtuple("ModelReturn", ["loss", "tb_dict", "disp_dict"])

    def model_fn(model, data, criterion, perfermance=False, device="cuda", num_class=4):

        imgs, gts = data[:2]
        imgs = imgs.to(device)
        gts = torch.squeeze(gts, 1).to(device)
        gts = gts.to(device)
        loss = 0
        tb_dict = {}
        disp_dict = {}
        if not perfermance:

            net_out = model(imgs)[0]  # [d1]

            ce_loss = torch.nn.CrossEntropyLoss()(net_out[0], gts.long())
            # dc_loss = DiceLoss()(net_out,make_one_hot(gts.unsqueeze(1).long(),num_classes=4).to(device))

            loss = ce_loss

            tb_dict.update({"CE_loss": loss.item()})
            disp_dict.update({"CE_loss": loss.item()})

        if perfermance:
            gts_ = gts.long().unsqueeze(1)
            model.eval()
            net_out_eval = model(imgs)[0]
            net_out = F.softmax(net_out_eval, dim=1)
            _, preds = torch.max(net_out, 1)
            preds = preds.unsqueeze(1)
            cal_perfer(make_one_hot(preds, num_class), make_one_hot(gts_, num_class), tb_dict)
            model.train()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_fn
def model_backbone0dist_decorator():
    ModelReturn = namedtuple("ModelReturn", ["loss", "tb_dict", "disp_dict"])

    def model_fn(model, data, criterion, perfermance=False, vis=False, device="cuda", epoch=0, num_class=4, w=None):
        # imgs, gts, _ = data
        imgs, gts = data[:2]
        imgs = imgs.to(device)
        gts = torch.squeeze(gts, 1).to(device)
        gts = gts.to(device)
        loss = 0
        tb_dict = {}
        disp_dict = {}

        if not perfermance:

            net_out = model(imgs)  # [d1]
            mask = net_out[0]
            dist = net_out[1]
            # ==============================================================================================================
            # calculate distance map
            imgs_ = imgs.cpu()
            batchsize = imgs_.shape[0]
            shape = list(imgs_.numpy().shape)
            gts_onehot = make_one_hot(gts.unsqueeze(1).long(), num_classes=4)
            shape[1] = 4
            dist_onebatch = np.zeros(shape[1:])
            dis_all = np.zeros(shape)
            for i in range(batchsize):
                for c in range(num_class):
                    dst, labels = cv2.distanceTransformWithLabels(gts_onehot[i, c, ...].numpy().astype(np.uint8),
                                                                  cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
                                                                  labelType=cv2.DIST_LABEL_PIXEL)
                    # dst = list(dst)
                    dist_onebatch[c] = dst
                dis_all[i] = dist_onebatch
            dis_all_train = torch.from_numpy(dis_all).float().to(device)
            # print(dis_all_train.shape)
            # ==============================================================================================================

            ce_loss = torch.nn.CrossEntropyLoss()(mask, gts.long())
            dist_loss = torch.nn.MSELoss()(dist,dis_all_train)/dist.flatten().size()[0]
            # dc_loss = DiceLoss()(net_out,make_one_hot(gts.unsqueeze(1).long(),num_classes=4).to(device))

            loss = ce_loss + dist_loss

            tb_dict.update({"CE_loss": ce_loss.item()})
            tb_dict.update({"dist_loss": dist_loss.item()})
            tb_dict.update({"total_loss": loss.item()})

            disp_dict.update({"CE_loss": ce_loss.item()})
            disp_dict.update({"dist_loss": dist_loss.item()})
            disp_dict.update({"total_loss": loss.item()})


        if perfermance:
            gts_ = gts.unsqueeze(1)
            model.eval()
            net_out_eval = model(imgs)[0]
            net_out = F.softmax(net_out_eval, dim=1)
            _, preds = torch.max(net_out, 1)
            preds = preds.unsqueeze(1)
            cal_perfer(make_one_hot(preds, num_class), make_one_hot(gts_.long(), num_class), tb_dict)
            model.train()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_fn

def model_backbone0_smoothAnddist_decorator():
    ModelReturn = namedtuple("ModelReturn", ["loss", "tb_dict", "disp_dict"])

    def model_fn(model, data, criterion, perfermance=False, vis=False, device="cuda", epoch=0, num_class=4, w=None):
        # imgs, gts, _ = data
        imgs, gts = data[:2]
        imgs = imgs.to(device)
        gts = torch.squeeze(gts, 1).to(device)
        gts = gts.to(device)
        loss = 0
        tb_dict = {}
        disp_dict = {}

        if not perfermance:
            net_out = model(imgs)  # [d1]
            mask = net_out[0]
            # ==============================================================================================================
            # calculate distance map
            imgs_ = imgs.cpu()
            batchsize = imgs_.shape[0]
            shape = list(imgs_.numpy().shape)
            gts_onehot = make_one_hot(gts.unsqueeze(1).long(), num_classes=4)
            shape[1] = 4
            dist_onebatch = np.zeros(shape[1:])
            dis_all = np.zeros(shape)
            for i in range(batchsize):
                for c in range(num_class):
                    dst, labels = cv2.distanceTransformWithLabels(gts_onehot[i, c, ...].numpy().astype(np.uint8),
                                                                  cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
                                                                  labelType=cv2.DIST_LABEL_PIXEL)
                    # dst = list(dst)

                    dist_onebatch[c] = dst/np.max(dst) if np.max(dst) else dst
                dis_all[i] = dist_onebatch
            dis_all_train = torch.from_numpy(dis_all).float().to(device)
            # print(dis_all_train.shape)
            # ==============================================================================================================

            # ==============================================================================================================
            # calculate mask gradient
            mask_ = torch.softmax(mask,dim=1) # softmax
            mask_ = mask_.cpu()
            batchsize = mask_.detach().shape[0]
            shape = list(mask_.detach().numpy().shape)
            grad_onebatch = np.zeros(shape[1:])
            grad_all = np.zeros(shape)

            for i in range(batchsize):
                for c in range(num_class):
                    cv2.imwrite('mask_i_c.png',mask_[i,c,...].detach().numpy())
                    mask_i_c = cv2.imread('mask_i_c.png',0)
                    os.remove('mask_i_c.png')
                    laplacian = cv2.Laplacian(mask_i_c, cv2.CV_64F)

                    # dst = list(dst)
                    grad_onebatch[c] = laplacian
                grad_all[i] = grad_onebatch
            grad_all_train = torch.from_numpy(grad_all).float().to(device)
            # print(dis_all_train.shape)
            # ==============================================================================================================

            ce_loss = torch.nn.CrossEntropyLoss()(mask, gts.long())

            smooth_loss = smooth_dist_loss(mask_true=make_one_hot(gts.unsqueeze(1).long(),num_classes=4).to(device),
                                              mask_pred=mask, grad_pred=grad_all_train, dist_true=dis_all_train)

            loss = ce_loss + smooth_loss

            tb_dict.update({"CE_loss": ce_loss.item()})
            tb_dict.update({"smooth_loss": smooth_loss.item()})
            tb_dict.update({"total_loss": loss.item()})

            disp_dict.update({"CE_loss": ce_loss.item()})
            disp_dict.update({"smooth_loss": smooth_loss.item()})
            disp_dict.update({"total_loss": loss.item()})


        if perfermance:
            gts_ = gts.unsqueeze(1)
            model.eval()
            net_out_eval = model(imgs)[0]
            net_out = F.softmax(net_out_eval, dim=1)
            _, preds = torch.max(net_out, 1)
            preds = preds.unsqueeze(1)
            cal_perfer(make_one_hot(preds, num_class), make_one_hot(gts_.long(), num_class), tb_dict)
            model.train()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_fn

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

def cal_perfer(preds, masks, tb_dict):
    LV_dice = []  # 1
    MYO_dice = []  # 2
    RV_dice = []  # 3
    LV_hausdorff = []
    MYO_hausdorff = []
    RV_hausdorff = []

    for i in range(preds.shape[0]):
        LV_dice.append(dice(preds[i, 3, :, :], masks[i, 3, :, :]))
        RV_dice.append(dice(preds[i, 1, :, :], masks[i, 1, :, :]))
        MYO_dice.append(dice(preds[i, 2, :, :], masks[i, 2, :, :]))

        # LV_hausdorff.append(cal_hausdorff_distance(preds[i,1,:,:],masks[i,1,:,:]))
        # RV_hausdorff.append(cal_hausdorff_distance(preds[i,3,:,:],masks[i,3,:,:]))
        # MYO_hausdorff.append(cal_hausdorff_distance(preds[i,2,:,:],masks[i,2,:,:]))
    tb_dict.update({"LV_dice": np.mean(LV_dice)})
    tb_dict.update({"RV_dice": np.mean(RV_dice)})
    tb_dict.update({"MYO_dice": np.mean(MYO_dice)})
    # print('LV_dice:{} RV_dice:{} MYO:{}'.format(np.mean(LV_dice),np.mean(RV_dice),np.mean(MYO_dice)))
    # tb_dict.update({"LV_hausdorff": np.mean(LV_hausdorff)})
    # tb_dict.update({"RV_hausdorff": np.mean(RV_hausdorff)})
    # tb_dict.update({"MYO_hausdorff": np.mean(MYO_hausdorff)})

