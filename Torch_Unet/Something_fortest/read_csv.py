import csv
import numpy as np
# print((0.9215466853295324+0.855151651986132+0.8792753308983629)/3) # 88.5
with open('/home/laisong/Downloads/run-.-tag-val_LV_dice (3).csv', 'r') as f:
    LV_DICE = csv.reader(f)
    LV_DICE = list(LV_DICE)
    LV_DICE = LV_DICE[1:]
lv_dice = []
for i in LV_DICE:
    lv_dice.append(i[2])
lv_dice = np.array(lv_dice)
with open('/home/laisong/Downloads/run-.-tag-val_RV_dice (3).csv', 'r') as f:
    RV_DICE = csv.reader(f)
    RV_DICE = list(RV_DICE)
    RV_DICE = RV_DICE[1:]
rv_dice = []
for i in RV_DICE:
    rv_dice.append(i[2])
rv_dice = np.array(rv_dice)
with open('/home/laisong/Downloads/run-.-tag-val_MYO_dice (3).csv', 'r') as f:
    MYO_DICE =  csv.reader(f)
    MYO_DICE = list(MYO_DICE)
    MYO_DICE = MYO_DICE[1:]
myo_dice = []
for i in MYO_DICE:
    myo_dice.append(i[2])
myo_dice = np.array(myo_dice)

ALL_DICE = np.vstack([lv_dice,rv_dice,myo_dice])
ALL_DICE = np.array(ALL_DICE).astype(float)
ALL_DICE_MEAN = np.mean(ALL_DICE,axis=0)
# MAX_DICE_MEAN = ALL_DICE_MEAN.max()
# print(MAX_DICE_MEAN)
MAX_INDEX = np.argmax(ALL_DICE_MEAN,axis=0)
print(MAX_INDEX)
print(ALL_DICE_MEAN[MAX_INDEX])
print(ALL_DICE[:,MAX_INDEX])
#
# """
# [0.91339988 0.83291519 0.86281389] U-Net
# [0.91304803 0.8284381  0.86850137] U-NetDF
# [0.91439402 0.80746138 0.86231297] U-Net-steplr
# [0.9215466853295324 0.855151651986132 0.8792753308983629] U-Net-GroupNorm-batchsize=8
# [0.92948765 0.82661664 0.87752163] U-NetDF-GroupNorm-batchsize=8_a=0.5
# [0.92672282 0.84598827 0.88483167] U-NetDF-GroupNorm-batchsize=8_a=1.0
# """