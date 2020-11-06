import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import cv2
from collections import namedtuple
from libs.losses.Cal_Diceloss import make_one_hot, DiceLoss

def model_backbone0_decorator():
    ModelReturn = namedtuple("ModelReturn", ["loss", "tb_dict", "disp_dict"])

    def model_fn(model, data, criterion, perfermance=False, vis=False, device="cuda", epoch=0, num_class=4, w=None):
        # imgs, gts, _ = data
        imgs, gts = data[:2]
        imgs = imgs.to(device)
        gts = torch.squeeze(gts, 1).to(device)
        gts = gts.to(device)

        net_out = model(imgs)[0]  # [d1]

        ce_loss = torch.nn.CrossEntropyLoss()(net_out, gts.long())
        dc_loss = DiceLoss()(net_out,make_one_hot(gts.unsqueeze(1).long(),num_classes=4).to(device))

        loss = dc_loss

        tb_dict = {}
        disp_dict = {}
        tb_dict.update({"DC_loss": loss.item()})

        disp_dict.update({"DC_loss": loss.item()})

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

        net_out = model(imgs)[0]  # [d1]

        # ce_loss = torch.nn.CrossEntropyLoss()(net_out[0], gts.long())
        dc_loss = DiceLoss()(net_out,make_one_hot(gts.unsqueeze(1).long(),num_classes=4).to(device))

        loss = dc_loss

        tb_dict = {}
        disp_dict = {}
        tb_dict.update({"DC_loss": loss.item()})

        disp_dict.update({"DC_loss": loss.item()})

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

