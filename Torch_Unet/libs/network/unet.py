import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional  as F
import numpy as np
from torchsummary import summary


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(ch_out),
            # nn.GroupNorm(num_groups=8, num_channels=ch_out),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(ch_out),
            # nn.GroupNorm(num_groups=8, num_channels=ch_out),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    # nn.BatchNorm2d(ch_out),
            # nn.GroupNorm(num_groups=8, num_channels=ch_out),
            nn.InstanceNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
#=======================================================================================================================
class Isensee_U_Net(nn.Module):
    def __init__(self, img_ch=1, num_class=4, base_channel = 48):
        super(Isensee_U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=base_channel)
        self.Conv2 = conv_block(ch_in=base_channel, ch_out=2*base_channel)
        self.Conv3 = conv_block(ch_in=2*base_channel, ch_out=4*base_channel)
        self.Conv4 = conv_block(ch_in=4*base_channel, ch_out=8*base_channel)
        self.Conv5 = conv_block(ch_in=8*base_channel, ch_out=16*base_channel)

        self.Up5 = up_conv(ch_in=16*base_channel, ch_out=8*base_channel)
        self.Up_conv5 = conv_block(ch_in=16*base_channel, ch_out=8*base_channel)

        self.Up4 = up_conv(ch_in=8*base_channel, ch_out=4*base_channel)
        self.Up_conv4 = conv_block(ch_in=8*base_channel, ch_out=4*base_channel)

        self.Up3 = up_conv(ch_in=4*base_channel, ch_out=2*base_channel)
        self.Up_conv3 = conv_block(ch_in=4*base_channel, ch_out=2*base_channel)

        self.Up2 = up_conv(ch_in=2*base_channel, ch_out=base_channel)
        self.Up_conv2 = conv_block(ch_in=2*base_channel, ch_out=base_channel)

        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Deepconv1 = nn.Conv2d(4*base_channel, num_class, kernel_size=1, stride=1, padding=0, bias=True)
        self.Deepconv2 = nn.Conv2d(2*base_channel, num_class, kernel_size=1, stride=1, padding=0, bias=True)

        self.Conv_1x1 = nn.Conv2d(base_channel, num_class, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(16*base_channel*16*16,100)
        self.fc2 = nn.Linear(100,8*8)
        self.activate = nn.ReLU()


    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.dropout(x3)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.dropout(x4)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.dropout(x5)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5) # 512
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.dropout(d5)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5) # 256
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.dropout(d4)

        d4 = self.Up_conv4(d4)
        deep1 = self.Deepconv1(d4)
        deep1_up = self.Upsample(deep1)

        d3 = self.Up3(d4) # 128
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.dropout(d3)

        d3 = self.Up_conv3(d3)
        deep2 = self.Deepconv2(d3)
        deep2_sum_deep1 = deep2+deep1_up
        deep2_sum_deep1_up = self.Upsample(deep2_sum_deep1)

        d2 = self.Up2(d3) # 64
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        deep_out = d1+deep2_sum_deep1_up

        return [deep_out,d1]
#=======================================================================================================================
    
#=======================================================================================================================
class U_Net(nn.Module):
    def __init__(self,img_ch=1,num_class=4, selfeat=False):
        super(U_Net,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,num_class,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return [d1]

# =======================================================================================================================

if __name__ == "__main__":

    # summary(net, (1, 256, 256),batch_size=1)

    def opCounter(model):
        type_size = 4  # float
        params = list(model.parameters())
        k = 0
        for i in params:
            l = 1
            print("该层的结构：" + str(list(i.size())))
            for j in i.size():
                l *= j
            print("该层参数和：" + str(l))
            k = k + l
        print("总参数数量和：" + str(k))
        print('Model {} : params: {:4f}M'.format(model._get_name(), k * type_size / 1000 / 1000))

    model = Isensee_U_Net()
    opCounter(model)
#=======================================================================================================================
#  Model Isensee_U_Net : params: 156.332736M M
#  Model U_Net : params: 138.049552 M
#  Model CleanU_Net : params: 26.921776M
#=======================================================================================================================
