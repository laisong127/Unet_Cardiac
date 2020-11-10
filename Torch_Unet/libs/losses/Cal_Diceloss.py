import torch
import numpy as np

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1e-6, p=1, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss =  - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                # print(predict[:, i].shape)
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


class BinaryTverskyLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1e-6, p=2, alpha=0.7, reduction='mean'):
        super(BinaryTverskyLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        TP = torch.sum(torch.mul(predict, target), dim=1)
        FN = torch.sum(torch.mul(1 - predict, target), dim=1)
        FP = torch.sum(torch.mul(predict, 1 - target), dim=1)

        loss = 1 - (TP + self.smooth) / (TP + self.alpha * FN + (1-self.alpha) * FP + self.smooth)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
class TverskyLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(TverskyLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        Binary_TverskyLoss = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                Tversky_loss = Binary_TverskyLoss(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    Tversky_loss *= self.weights[i]
                total_loss += Tversky_loss

        return total_loss/target.shape[1]

def tversky(y_pred, y_true):
    smooth = 1e-6
    true_pos = torch.sum(y_true * y_pred)
    false_neg = torch.sum(y_true * (1-y_pred))
    false_pos = torch.sum((1-y_true)*y_pred)
    alpha = 0.7

    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_pred, y_true):
    '''

        :param y_pred: N,C,H,W
        :param y_true: N,H,W
        :return:
        '''
    num_class = y_pred.shape[1]
    batch_size = y_pred.shape[0]
    for b in range(batch_size):
        batch_loss = 0
        onebatch_loss = 0
        for i in range(num_class):
            y_pred_i = y_pred[b,i,...]
            y_true_i = (y_true[b,...]==i).float()
            tversky_loss_b_i = 1- tversky(y_pred_i,y_true_i)
            print(b,i,tversky_loss_b_i)
            onebatch_loss+=tversky_loss_b_i
        onebatch_loss /= num_class
        print(b,onebatch_loss)
        batch_loss += onebatch_loss
    loss = batch_loss / batch_size
    return loss

def soft_dice(y_pred,y_true):
    '''

    :param y_pred: N,C,H,W
    :param y_true: N,H,W
    :return:
    '''
    Dice = []
    Dice_add = 0
    Class_num = y_pred.shape[1]
    for i in range(Class_num):
        y_true_i = (y_true==i).float()
        y_pred_i = y_pred[:,i,...]
        intersect = torch.sum(y_pred_i * y_true_i)
        denominator = torch.sum(y_pred_i) + torch.sum(y_true_i)
        dice_i = (2*intersect+(1e-6))/(denominator+(1e-6))
        Dice.append(dice_i)
        Dice_add+=dice_i
    Dice = np.array(Dice)
    Dice_mean = Dice_add/Class_num
    return -Dice_mean, Dice
def AMTA_loss(pred_dist,y_true_dist,pred_mask,gts_onehot):
    # fg_loss, _ = soft_dice(pred_fg,y_true_foreground)
    # mask_loss, _ = soft_dice(pred_mask,gts)
    # total_loss = fg_loss+mask_loss
    dist_criterion = torch.nn.MSELoss()
    mask_criterion = DiceLoss()
    dist_loss = dist_criterion(pred_dist,y_true_dist)
    mask_loss = mask_criterion(pred_mask,gts_onehot)
    total_loss = dist_loss + mask_loss

    return dist_loss, mask_loss, total_loss
def spatialloss(p_pred,p_true):
    '''

    :param p_pred: (N,-1)
    :param p_true: (N,-1)
    :return: loss
    '''

    loss = torch.sum((p_pred-p_true)**2)


    return loss
def smooth_dist_loss(mask_true, mask_pred, grad_pred, dist_true):

    loss = torch.abs(grad_pred)*torch.abs(mask_pred-mask_true)*(torch.exp(dist_true)-1)
    loss = loss.mean()

    return loss

def soft_spatial_loss(y_pred,y_true,p_pred,p_true,l = 0.1):

    soft_dice_loss,_ = soft_dice(y_pred,y_true)
    spatial_loss = spatialloss(p_pred,p_true)
    total_loss= soft_dice_loss + l*spatial_loss

    return soft_dice_loss,spatial_loss, total_loss

if __name__=='__main__':
    a = np.array([[1,2,3,4],[5,6,7,8]])
    print(len(a))
    # y_pred = torch.randn(1,4,2,2)
    # y_true = torch.tensor([[1,2],
    #                        [3,0]])
    # y_true = y_true.float()
    # print(soft_spatial_loss(y_pred,y_true,y_pred,y_true))
