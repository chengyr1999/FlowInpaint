from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from base_model import BaseNetwork
from network import *
from spectral_norm import use_spectral_norm
# from spectral_norm import use_spectral_norm
# from network import *
# from base_model import BaseNetwork



class FlowFuseNet(BaseNetwork):
    def __init__(self, inc1 = 3, inc2 = 4):
        super(FlowFuseNet, self).__init__()

        #DOWN
        self.dconv1 = Conv(inc1*3, 32)
        self.dconv2 = Conv(inc2, 32)
        self.dconv01_1 = DownConv(32, 64)
        self.agg01 = AggregateBlock(64)
        self.dconv01_2 = DownConv(32, 64)
        self.dconv02_1 = DownConv(64, 128)
        self.dconv02_2 = DownConv(64, 128)
        self.agg02 = AggregateBlock(128)
        self.dconv03_1 = DownConv(128, 256)
        self.dconv03_2 = DownConv(128, 256)
        self.agg03 = AggregateBlock(256)
        self.dconv04_1 = DownConv(256, 512)
        self.dconv04_2 = DownConv(256, 512)
        self.agg04 = AggregateBlock(512)
        self.dconv05_1 = DownConv(512, 512)
        self.dconv05_2 = DownConv(512, 512)

        #UP
        self.upconv1 = UpConv(512*2,512)
        self.atte1 = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10., n_down=4, fuse=True)

        self.upconv2 = UpConv(512*3,256)
        self.atte2 = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10., n_down=3, fuse=True)

        self.upconv3 = UpConv(256*3,128)
        self.atte3 = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10., n_down=2, fuse=True)

        self.upconv4 = UpConv(128*3,64)
        self.atte4 = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10., n_down=1, fuse=True)

        self.upconv5 = UpConv(64*3,32)
        self.out = Up(32*3,3)

        
    def forward(self, warp_img, mask_img, mask):

        down1 = self.dconv1(warp_img)
        down2 = self.dconv2(torch.cat([mask_img, mask],1))
        down1_1 = self.dconv01_1(down1)
        down1_1 =self.agg01(down1_1)
        down1_2 = self.dconv01_2(down2)
        
        down2_1 = self.dconv02_1(down1_1)
        down2_1 =self.agg02(down2_1)
        down2_2 = self.dconv02_2(down1_2)

        down3_1 = self.dconv03_1(down2_1)
        down3_1 =self.agg03(down3_1)
        down3_2 = self.dconv03_2(down2_2)

        down4_1 = self.dconv04_1(down3_1)
        down4_1 =self.agg04(down4_1)
        down4_2 = self.dconv04_2(down3_2)

        down5_1 = self.dconv05_1(down4_1)
        down5_2 = self.dconv05_2(down4_2)

        up5 = self.upconv1(torch.cat([down5_1,down5_2],1))
        up5 = F.interpolate(up5, scale_factor=2, mode='bilinear', align_corners=True)
        up5, flow5 = self.atte1(up5, up5, mask)

        up4 = self.upconv2(torch.cat([down4_1, down4_2, up5],1))
        up4 = F.interpolate(up4, scale_factor=2, mode='bilinear', align_corners=True)
        up4, flow4 = self.atte2(up4, up4, mask)

        up3 = self.upconv3(torch.cat([down3_1, down3_2, up4],1))
        up3 = F.interpolate(up3, scale_factor=2, mode='bilinear', align_corners=True)
        up3, flow3 = self.atte3(up3, up3, mask)

        up2 = self.upconv4(torch.cat([down2_1, down2_2, up3],1))
        up2 = F.interpolate(up2, scale_factor=2, mode='bilinear', align_corners=True)
        up2, flow2 = self.atte4(up2, up2, mask)

        up1 = self.upconv5(torch.cat([down1_1, down1_2, up2],1))
        up1 = F.interpolate(up1, scale_factor=2, mode='bilinear', align_corners=True)
        
        out = self.out(torch.cat([down1, up1, down2],1))
        
       
        return out



class Discriminator(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=False, use_sn=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        cnum = 64
        self.encoder = nn.Sequential(
        use_spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=cnum,
            kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
        nn.LeakyReLU(0.2, inplace=True),

        use_spectral_norm(nn.Conv2d(in_channels=cnum, out_channels=cnum*2,
            kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
        nn.LeakyReLU(0.2, inplace=True),
        
        use_spectral_norm(nn.Conv2d(in_channels=cnum*2, out_channels=cnum*4,
            kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
        nn.LeakyReLU(0.2, inplace=True),

        use_spectral_norm(nn.Conv2d(in_channels=cnum*4, out_channels=cnum*8,
            kernel_size=5, stride=1, padding=1, bias=False), use_sn=use_sn),
        nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Conv2d(in_channels=cnum*8, out_channels=1, kernel_size=5, stride=1, padding=1)
        if init_weights:
            self.init_weights()


    def forward(self, x):
        x = self.encoder(x)
        label_x = self.classifier(x)
        if self.use_sigmoid:
            label_x = torch.sigmoid(label_x)
        return label_x

if __name__ == '__main__':
    model = FlowFuseNet()
    # model = Inpainter()

    x = torch.ones([10,9,256,256])
    y = torch.ones([10,3,256,256])
    m = torch.ones([10,1,256,256])

    # print(model)
    model(x,y,m)
    # model(x,x,x,m)

