import os
import torch
from libs.network.unet import U_Net
from libs.network.unet_df import U_NetDF
from libs.configs.config_acdc import cfg
from tools.train import create_dataloader
ckpt_path = '/home/laisong/github/DirectionalFeature/result/result_cleanUnet_GroupNorm/ckpt/model_best.pth'

ckpt = torch.load(ckpt_path)
model = U_Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_state_dict(ckpt['model_state'])
# model.eval()

# a = torch.randn(1, 1, 224, 224)
# out = model(a)
# print(out)
train_loader, test_loader = create_dataloader()
for cur_it, batch in enumerate(train_loader):
    print(batch[0])
# test_loader = iter(test_loader)
# batch = next(test_loader)
# print(batch[4])
# img,gt = batch[:2]
# batch_size = img.shape[0]
# img = img.to(device)
# out = model(img)[0]
# print(out.shape)
# for i in range(batch_size):
#     out_i = out[i]
#     print(out_i.shape)
