# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.nn.functional import sigmoid
class Conv_block2(nn.Module):
    def __init__(self, in_ch, out_ch, size):
        super(Conv_block2, self).__init__()
        self.padding = [(size[0] - 1) // 2, (size[1] - 1) // 2]
        self.conv = nn.Conv2d(in_ch, out_ch, size, padding=self.padding, stride=2)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class Conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, size):
        super(Conv_block, self).__init__()
        self.padding = [(size[0] - 1) // 2, (size[1] - 1) // 2]
        self.conv = nn.Conv2d(in_ch, out_ch, size, padding=self.padding, stride=1)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double_conv, self).__init__()

        self.in_ch = in_ch
        self.mid_mid = out_ch // 7
        self.out_ch = out_ch
        self.conv1x1_mid = Conv_block(self.in_ch, self.out_ch, [1, 1])
        self.conv1x1_2 = nn.Conv2d(self.out_ch, self.out_ch, 1)
        self.conv3x3_3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_2_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_1_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x1_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv1x3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])

        self.conv3x3_3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_1_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_2_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x1_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv1x3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        # self.conv1x1_2 = Conv_block(self.mid_mid, self.mid_mid, [1, 1])
        self.conv1x1_1 = nn.Conv2d(self.out_ch, self.out_ch, 1)
        self.rel = nn.ReLU(inplace=True)
        # if self.in_ch > self.out_ch:
        #     self.short_connect = nn.Conv2d(in_ch, out_ch, 1, padding=0)

    def forward(self, x):

        x0 = x[:, 0:self.mid_mid, ...]
        x1 = x[:, self.mid_mid:self.mid_mid * 2, ...]
        x2 = x[:, self.mid_mid * 2:self.mid_mid * 3, ...]
        x3 = x[:, self.mid_mid * 3:self.mid_mid * 4, ...]
        x4 = x[:, self.mid_mid * 4:self.mid_mid * 5, ...]
        x5 = x[:, self.mid_mid * 5:self.mid_mid * 6, ...]
        x6 = x[:, self.mid_mid * 6:self.mid_mid * 7, ...]
        x1 = self.conv1x3_1(x1)
        x2 = self.conv3x1_1(x2 + x1)
        x3 = self.conv3x3_1(x3 + x2)
        x4 = self.conv3x3_1_1(x4 + x3)
        x5= self.conv3x3_2_1(x5+x4)
        x6 = self.conv3x3_2_1(x5 + x6)
        xxx = self.conv1x1_1(torch.cat((x0, x1, x2, x3, x4,x5,x6), dim=1))
        x0 = xxx[:, 0:self.mid_mid, ...]
        x1_2 = xxx[:, self.mid_mid:self.mid_mid * 2, ...]
        x2_2 = xxx[:, self.mid_mid * 2:self.mid_mid * 3, ...]
        x3_2 = xxx[:, self.mid_mid * 3:self.mid_mid * 4, ...]
        x4_2 = xxx[:, self.mid_mid * 4:self.mid_mid * 5, ...]
        x5_2 = xxx[:, self.mid_mid * 5:self.mid_mid * 6, ...]
        x6_2 = xxx[:, self.mid_mid * 6:self.mid_mid * 7, ...]
        x1 = self.conv1x3_2(x1_2 )
        x2 = self.conv3x1_2(x1 + x2_2)
        x3 = self.conv3x3_2(x2 + x3_2)
        x4 = self.conv3x3_1_2(x3 + x4_2)
        x5=self.conv3x3_2_2(x4+x5_2)
        x6 = self.conv3x3_2_2(x5 + x6_2)
        xxx = torch.cat((x0, x1, x2, x3, x4,x5,x6), dim=1)

        # if self.in_ch > self.out_ch:
        #     x = self.short_connect(x)
        return self.rel(xxx + x)


class Conv_down_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_down_2, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=2, bias=True)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class Conv_down(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''
    def __init__(self, in_ch, out_ch, flage):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_down, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.flage = flage
        if self.in_ch == 1:
            self.first = nn.Sequential(
                Conv_block(self.in_ch, self.out_ch, [3, 3]),
                Double_conv(self.out_ch, self.out_ch),
            )
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv_down = Conv_down_2(self.out_ch, self.out_ch)
        else:
            self.conv = Double_conv(self.in_ch, self.out_ch)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv_down = Conv_down_2(self.out_ch, self.out_ch)
            if self.flage==True:
                self.conv1x1=nn.Conv2d(self.out_ch*2,self.out_ch*2,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        if self.in_ch == 1:
            x = self.first(x)
            pool_x = torch.cat((self.pool(x), self.conv_down(x)), dim=1)
        else:
            x = self.conv(x)
            if self.flage == True:
                pool_x = torch.cat((self.pool(x), self.conv_down(x)), dim=1)
                pool_x=self.conv1x1(pool_x)
            else:
                pool_x = None
        return pool_x, x

def hdc(image, num=2):
    for i in range(num):
        for j in range(num):
            if i==0 and j==0:
                x1 = image[:, :, i::num, j::num]
            else:
                x1 = torch.cat((x1, image[:, :, i::num, j::num]), dim=1)
    return x1


class Conv_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_up, self).__init__()
        self.up = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, stride=1)
        self.conv = Double_conv(in_ch, out_ch)
        self.interp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2 = nn.Conv2d(in_ch+out_ch, out_ch, kernel_size=1, padding=0, stride=1)

    def forward(self, x1, x2):
        x1 = self.interp(x1)

        x1 = torch.cat((x1, x2), dim=1)
        x1=self.up2(x1)
        x1 = self.conv(x1)
        return x1

class Conv_up2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_up2, self).__init__()
        self.up = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, stride=1)
        self.conv = Double_conv(in_ch, out_ch)
        self.interp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x1, x2,x3):
        x1 = self.interp(x1)
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2,x3), dim=1)
        x1 = self.conv(x1)
        return x1

class CleanU_Net_block(nn.Module):
    def __init__(self, in_channels, out_channels,num_filters):
        super(CleanU_Net_block, self).__init__()
        self.filter=num_filters
        self.first = Conv_block2(in_channels, self.filter, [3, 3])
        self.Conv_down1 = Conv_down(self.filter, self.filter, True)
        self.Conv_down2 = Conv_down(self.filter*2, self.filter*2, True)
        self.Conv_down3 = Conv_down(self.filter*4, self.filter*4, True)
        self.Conv_down5 = Conv_down(self.filter*8, self.filter*8, False)
        self.Conv_up2 = Conv_up2(self.filter*8+self.filter*4, self.filter*4)
        self.Conv_up3 = Conv_up2(self.filter*4+self.filter*2, self.filter*2)
        self.Conv_up4 = Conv_up2(self.filter*2+self.filter, self.filter)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Conv_out = nn.Conv2d(self.filter, out_channels, 1, padding=0, stride=1)

    def forward(self, x,con1,con2,con3):
        # x = hdc(x)
        x = self.first(x)
        x, conv1 = self.Conv_down1(x)
        x, conv2 = self.Conv_down2(x)
        x, conv3 = self.Conv_down3(x)
        _, x = self.Conv_down5(x)
        x = self.Conv_up2(x, conv3,con3)
        x = self.Conv_up3(x, conv2,con2)
        x = self.Conv_up4(x, conv1,con1)
        x = self.up(x)
        x = self.Conv_out(x)

        return  x
class CleanU_Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=4,num_filters=70):
        super(CleanU_Net, self).__init__()
        self.filter=num_filters
        self.first = Conv_block(in_channels*4, self.filter, [3, 3])
        self.Conv_down1 = Conv_down(self.filter, self.filter, True)
        self.Conv_down2 = Conv_down(self.filter*2, self.filter*2, True)
        self.Conv_down3 = Conv_down(self.filter*4, self.filter*4, True)
        # self.Conv_down4 = Conv_down(512, 512,True)
        self.Conv_down5 = Conv_down(self.filter*8, self.filter*8, False)

        # self.Conv_up1 = Conv_up(1024,280)
        self.Conv_up2 = Conv_up(self.filter*8, self.filter*4)
        self.Conv_up1_3 = Conv_up(self.filter*4, self.filter*2)
        self.Conv_up1_4 = Conv_up(self.filter*2, self.filter)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Conv_out1 = nn.Conv2d(self.filter, out_channels, 1, padding=0, stride=1)

    def forward(self, x):
        x = hdc(x)
        x = self.first(x)
        x, conv1 = self.Conv_down1(x)
        x, conv2 = self.Conv_down2(x)
        x, conv3 = self.Conv_down3(x)
        _, x = self.Conv_down5(x)
        x = self.Conv_up2(x, conv3)
        x1 = self.Conv_up1_3(x, conv2)
        x1 = self.Conv_up1_4(x1, conv1)
        x1 = self.up1(x1)
        x1 = self.Conv_out1(x1)
        return [x1]

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


if __name__ == "__main__":
    def opCounter(model):
        type_size = 4  # float占据4个字节
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

    model = CleanU_Net()
    opCounter(model)
    # total = sum([param.nelement() for param in model.parameters()])
    # print('Number of paramter:{}'.format(total/1e6))
