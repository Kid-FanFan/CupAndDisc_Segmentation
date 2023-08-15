# -*- coding: utf-8 -*-
# @time : 2022/4/2 9:02
# @author : Precision
# @file : UModel.py
# @project : Glaucoma_Segmentation_Unet
import torch
from torch import nn
from torch.nn import functional as F
from MyResNet import  BasicBlock

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Conv_Block(nn.Module):
    def __init__(self,inChannel,outChannel):
        super(Conv_Block,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inChannel,outChannel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(outChannel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(outChannel, outChannel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(outChannel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self,x):
        return self.layer(x)

class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
        self.ChannelCon = ChannelAttention(channel)
        self.SpatialCon = SpatialAttention()
    def forward(self, x):

        return self.layer(x)

class conv1(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(conv1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            BasicBlock(out_ch,out_ch),
            #nn.BatchNorm2d(out_ch),
            #nn.ReLU(inplace=True),
        )
        self.ChannelCon = ChannelAttention(out_ch)
        self.SpatialCon = SpatialAttention()
    def forward(self, x):
        x = self.conv1(x)
        x = self.ChannelCon(x) * x
        x = self.ChannelCon(x) * x
        # print('double_conv', x.shape)
        return x

class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample,self).__init__()
        self.layer = nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feaatire_map):
        up = F.interpolate(x,scale_factor=2,mode='nearest')
        out = self.layer(up)
        return torch.cat((out,feaatire_map),dim=1)

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.conv1 = conv1(3,64)
        self.down1 = DownSample(64)
        self.conv2 = conv1(64,128)
        self.down2 = DownSample(128)
        self.conv3 = conv1(128,256)
        self.down3 = DownSample(256)
        self.conv4 = conv1(256,512)
        self.down4 = DownSample(512)
        self.conv5 = conv1(512,1024)
        self.up1 = UpSample(1024)
        self.conv6 = Conv_Block(1024,512)
        self.up2 = UpSample(512)
        self.conv7 = Conv_Block(512,256)
        self.up3 = UpSample(256)
        self.conv8 = Conv_Block(256,128)
        self.up4 = UpSample(128)
        self.conv9 = Conv_Block(128,64)
        self.out = nn.Conv2d(64,3,3,1,1)
        self.Th = nn.Sigmoid()
        self.res = Conv_Block
    def forward(self, x):
        R1 = self.conv1(x)
        R2 = self.conv2(self.down1(R1))
        R3 = self.conv3(self.down2(R2))
        R4 = self.conv4(self.down3(R3))
        R5 = self.conv5(self.down4(R4))
        O1 = self.conv6(self.up1(R5, R4))
        O2 = self.conv7(self.up2(O1, R3))
        O3 = self.conv8(self.up3(O2, R2))
        O4 = self.conv9(self.up4(O3, R1))

        return self.Th(self.out(O4))

if __name__ == '__main__' :
    x=torch.randn(2,3,640,640)
    net = Unet()
    print(net(x).shape)