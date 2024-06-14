from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.moduleZero = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleOne = nn.Sequential( 
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
 
        self.moduleTwo = nn.Sequential( 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
 
        self.moduleThr = nn.Sequential( 
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
 
        self.moduleFour = nn.Sequential( 
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
 
        self.moduleFive = nn.Sequential( 
            nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
 
    def forward(self, tensorInput): 
        tensorZero = self.moduleZero(tensorInput)
        tensorOne = self.moduleOne(tensorZero)
        tensorTwo = self.moduleTwo(tensorOne)
        tensorThr = self.moduleThr(tensorTwo)
        tensorFou = self.moduleFour(tensorThr)
        tensorFiv = self.moduleFive(tensorFou)
     
        return [tensorZero, tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv]

class Decoder(nn.Module):
    def __init__(self, intLevel):
        super(Decoder, self).__init__()
 
        intPrevious = [ None, 81+32+2+2, 81+64+2+2, 81+96+2+2, 81+128+2+2, 81, 81][intLevel+1]
        intCurrent = [81+16+2+2, 81+32+2+2, 81+64+2+2, 81+96+2+2, 81+128+2+2, 81,None][intLevel+0]
        if intLevel < 6:
            self.dblBackwarp = [None, 10, 5.0, 2.5, 1.25, 0.625, 0.3125, None][intLevel + 1]

        if intLevel < 6:
            self.moduleUpflow = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        
        if intLevel < 6:
            self.moduleUpfeat = nn.ConvTranspose2d(in_channels=intPrevious + 128 + 96 + 64 + 32 + 16 + 1, out_channels=2, kernel_size=4, stride=2, padding=1)

        if intLevel < 6:
            self.moduleBackward = self.warp
 
        self.moduleCorreleaky = nn.LeakyReLU(inplace=False, negative_slope=0.1)
 
        self.moduleZero = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent+ 1 , out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
        self.moduleOne = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128+1 , out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
 
        self.moduleTwo = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128 + 96+1 , out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
 
        self.moduleThr = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128 + 96 + 64+1 , out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
 
        self.moduleFou = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128 + 96 + 64 + 32 +1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
 
        self.moduleFiv = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128 + 96 + 64 + 32 + 16+1, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

    def corr(self, refimg_fea, targetimg_fea):
        maxdisp=4
        b,c,h,w = refimg_fea.shape
        targetimg_fea = F.unfold(targetimg_fea, (2*maxdisp+1,2*maxdisp+1), padding=maxdisp).view(b,c,2*maxdisp+1, 2*maxdisp+1**2,h,w)
        cost = refimg_fea.view(b,c,h,w)[:,:,np.newaxis, np.newaxis]*targetimg_fea.view(b,c,2*maxdisp+1, 2*maxdisp+1**2,h,w)
        cost = cost.sum(1)
 
        b, ph, pw, h, w = cost.size()
        cost = cost.view(b, ph * pw, h, w)/refimg_fea.size(1)
        return cost
    
    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask
    
    def forward(self, tensorFirst, tensorSecond, objectPrevious, mask):
    
        tensorFirst = F.interpolate(tensorFirst, scale_factor=2, mode='bilinear', align_corners=True)
        tensorSecond = F.interpolate(tensorSecond, scale_factor=2, mode='bilinear', align_corners=True)
        tensorFlow = None
        tensorFeat = None
        
        if objectPrevious is None:
            tensorFlow = None
            tensorFeat = None
            tensorVolume = self.moduleCorreleaky(self.corr(tensorFirst, tensorSecond))
            tensorFeat = torch.cat([tensorVolume,mask], 1)
        elif objectPrevious is not None:
            tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow']) 
            tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat']) 
            tensorVolume = self.moduleCorreleaky(self.corr(tensorFirst, self.moduleBackward(tensorSecond, tensorFlow*self.dblBackwarp)))
            tensorFeat = torch.cat([tensorVolume, tensorFirst, tensorFlow, tensorFeat,mask], 1)
       
        tensorFeat = torch.cat([self.moduleZero(tensorFeat), tensorFeat], 1)
       
        tensorFeat = torch.cat([self.moduleOne(tensorFeat), tensorFeat], 1)
     
        tensorFeat = torch.cat([self.moduleTwo(tensorFeat), tensorFeat], 1) 
        tensorFeat = torch.cat([self.moduleThr(tensorFeat), tensorFeat], 1) 
        tensorFeat = torch.cat([self.moduleFou(tensorFeat), tensorFeat], 1)
        tensorFlow = self.moduleFiv(tensorFeat)
     
        return {
            'tensorFlow': tensorFlow,
            'tensorFeat': tensorFeat
        }
 
class Backwarp(nn.Module):
    def __init__(self):
        super(Backwarp, self).__init__()
 
    def forward(self, tensorInput, tensorFlow):
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(
            tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(
            tensorFlow.size(0), -1, -1, tensorFlow.size(3))
        tensorGrid = torch.cat([tensorHorizontal, tensorVertical], 1).cuda()
        tensorPartial = tensorFlow.new_ones(tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3)) 
        tensorInput = torch.cat([tensorInput, tensorPartial], 1) 
        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                                tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)
        tensorOutput = F.grid_sample(input=tensorInput, grid=(tensorGrid + tensorFlow).permute(0, 2, 3, 1),
                                    mode='bilinear', padding_mode='zeros')
       
        tensorMask = tensorOutput[:, -1:, :, :]
        tensorMask[tensorMask > 0.999] = 1.0
        tensorMask[tensorMask < 1.0] = 0.0
 
        return tensorOutput[:, :-1, :, :] * tensorMask
 
    def backwarp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
 
        if x.is_cuda:
            grid = grid.cuda()
        
        vgrid = Variable(grid) + flo
 
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
 
        vgrid = vgrid.permute(0, 2, 3, 1)
 
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
 
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
 
        return output * mask
    
class Refiner(nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()
 
        self.moduleMain = nn.Sequential(
            nn.Conv2d(in_channels=81+16+2+2+128+96+64+32+16+1, out_channels=128, kernel_size=3, stride=1, padding=1,  dilation=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2,  dilation=2),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4,  dilation=4),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8,  dilation=8),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16,  dilation=16),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1,  dilation=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1,  dilation=1)
        )
 
    def forward(self, tensorInput):
        return self.moduleMain(tensorInput)

class FlowInterpNet(nn.Module):
    def __init__(self, model_path=None):
        super(FlowInterpNet, self).__init__()
        self.model_path = model_path
        self.moduleExtractor = FeatureExtractor()
        
        self.down = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.moduleZero = Decoder(0)
        self.moduleOne = Decoder(1)
        self.moduleTwo = Decoder(2)
        self.moduleThr = Decoder(3)
        self.moduleFou = Decoder(4)
        self.moduleFiv = Decoder(5)
        self.moduleRefiner = Refiner()

        if self.model_path != None:
            print('self.model_path is',self.model_path)
            self.load_state_dict(torch.load(self.model_path))
 
    def forward(self, tensorPre, tensorInter, mask):
        #1. 两张图像帧构造两个特征金字塔 tensorImage=(batchsize,channel,height,width)
        #   - tensorFirst : [tensorOne...]
        #   - tensorSecond : [tensorOne...]
        tensorPre = self.moduleExtractor(tensorPre)
        tensorInter = self.moduleExtractor(tensorInter)
        
        # frame:Inter-->Pre
        d1_1 = self.down(1-mask)
        d1_2 = self.down(d1_1)
        d1_3 = self.down(d1_2)
        d1_4 = self.down(d1_3)
        d1_5 = self.down(d1_4)
        d1_6 = self.down(d1_5)
        objectEstimate = self.moduleFiv(tensorInter[-1], tensorPre[-1], None, d1_5)
        flow1_5 = objectEstimate['tensorFlow']
        objectEstimate = self.moduleFou(tensorInter[-2], tensorPre[-2], objectEstimate, d1_4)
        flow1_4 = objectEstimate['tensorFlow']
        objectEstimate = self.moduleThr(tensorInter[-3], tensorPre[-3], objectEstimate, d1_3)
        flow1_3 = objectEstimate['tensorFlow']
        objectEstimate = self.moduleTwo(tensorInter[-4], tensorPre[-4], objectEstimate, d1_2) 
        flow1_2 = objectEstimate['tensorFlow']
        objectEstimate = self.moduleOne(tensorInter[-5], tensorPre[-5], objectEstimate, d1_1) 
        flow1_1 = objectEstimate['tensorFlow']
        objectEstimate = self.moduleZero(tensorInter[-6], tensorPre[-6], objectEstimate, 1-mask) 
        flow1_0 = objectEstimate['tensorFlow']
        flow1_0= flow1_0 + self.moduleRefiner(objectEstimate['tensorFeat'])

        if self.training:
            return [flow1_0,flow1_1, flow1_2, flow1_3]
        else:
            return flow1_0

if __name__ == "__main__":
    model = PWC_Net().cuda()
    import numpy as np
    img = torch.tensor(np.zeros([5,3,256,256]).astype(np.float32)).cuda()
    mask = torch.tensor(np.zeros([5,1,256,256]).astype(np.float32)).cuda()

    flow1, flow2 = model(img,img,img,mask)