import torch.nn as nn
import torch.nn.functional as F
import torch


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes=3, ndf=64, n_channel=1):
        super(FCDiscriminator, self).__init__()
        self.conv0 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_channel, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=1) # add to original
        self.classifier = nn.Linear(ndf*16, 2)
        self.avgpool = nn.AvgPool2d((8, 8))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # self.sigmoid = nn.Sigmoid()

    def forward(self, map, feature):
        map_feature = self.conv0(map)
        image_feature = self.conv1(feature)
        x = torch.add(map_feature, image_feature)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = self.leaky_relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = self.up_sample(x)
        # x = self.sigmoid(x)
        return x

if __name__=='__main__':
     map = torch.randn(1, 3, 256, 256)
     image = torch.randn(1,1,256,256)
     D = FCDiscriminator(num_classes=3)
     Doutput = D(map,image)
     print(Doutput)