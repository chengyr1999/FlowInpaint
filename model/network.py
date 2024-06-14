import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AggregateBlock(nn.Module):
    def __init__(self, dim = None, rates = [1,2,4,8]):
        super(AggregateBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(self.rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)), 
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim//4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        return out

class Conv(nn.Module):
    def __init__(self, inc = None, outc = None,kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        self.norm_layer = nn.BatchNorm2d
        downconv = nn.Conv2d(inc, outc, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = self.norm_layer(outc)
        
        down_model = [downconv, downnorm, downrelu]
        self.model = nn.Sequential(*down_model)
    def forward(self, x):
        return self.model(x)

class DownConv(nn.Module):
    def __init__(self, inc = None, outc = None,kernel_size=3, stride=2, padding=1):
        super(DownConv, self).__init__()
        downconv = nn.Conv2d(inc, outc, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        down_model = [downconv, downrelu]
        self.model = nn.Sequential(*down_model)
    def forward(self, x):
        return self.model(x)

class UpConv(nn.Module):
    def __init__(self, inc = None, outc = None, kernel_size=3, stride=1, padding=1):
        super(UpConv, self).__init__()
        
        upconv = nn.Conv2d(inc, outc, kernel_size=kernel_size,
                            stride=stride, padding=1, bias=False)
        uprelu = nn.ReLU(inplace=True)
        up_model = [upconv, uprelu]
        self.model = nn.Sequential(*up_model)
    def forward(self, x):
        return self.model(x)

class Up(nn.Module):
    def __init__(self, inc = None, outc = None, kernel_size=3, stride=1, padding=1):
        super(Up, self).__init__()
        upconv1 = nn.Conv2d(inc, int(inc//2), kernel_size=3, stride=1, padding=1)
        uprelu = nn.ReLU()
        upconv2 = nn.Conv2d(int(inc//2), outc, kernel_size=3, stride=1, padding=1)
        upTahn = nn.Tanh()
       
        up_model = [upconv1, uprelu, upconv2, upTahn]
        self.model = nn.Sequential(*up_model)
    def forward(self, x):
        return self.model(x)

class ContextualAttention(nn.Module):
    """ Contextual attention layer implementation. 
        Contextual attention is first introduced in publication: 
         `Generative Image Inpainting with Contextual Attention, Yu et al`
    Args:
        ksize: Kernel size for contextual attention
        stride: Stride for extracting patches from b
        rate: Dilation for matching
        softmax_scale: Scaled softmax for attention
    """

    def __init__(self,
                 ksize=3,
                 stride=1,
                 rate=2,
                 fuse_k=3,
                 softmax_scale=10.,
                 n_down=2,
                 fuse=True): 
        super().__init__()

        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        # self.device_ids = device_ids
        self.n_down = n_down
        # self.return_flow = return_flow
        self.register_buffer('fuse_weight', torch.eye(
            fuse_k).view(1, 1, fuse_k, fuse_k))

    def forward(self, f, b, mask=None):
        """
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
        """
        device = f.device
        # get shapes
        raw_int_fs, raw_int_bs = list(f.size()), list(b.size())   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksize=kernel,
                                      stride=self.rate*self.stride,
                                      rate=1, padding='auto')  # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        L = raw_w.size(2)
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1./self.rate,
                          mode='nearest', recompute_scale_factor=False)
        b = F.interpolate(b, scale_factor=1./self.rate,
                          mode='nearest', recompute_scale_factor=False)
        int_fs, int_bs = list(f.size()), list(b.size())   # b*c*h*w
        # split tensors along the batch dimension
        f_groups = torch.split(f, 1, dim=0)
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksize=self.ksize,
                                  stride=self.stride,
                                  rate=1, padding='auto')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, L)
        w = w.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros(
                [int_bs[0], 1, int_bs[2], int_bs[3]], device=device)
        else:
            # mask = F.interpolate(
            #     mask, size=raw_int_bs, mode='nearest', recompute_scale_factor=False)
            mask = F.interpolate(
                mask, scale_factor=1./((2**self.n_down)*self.rate), mode='nearest', recompute_scale_factor=False)
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksize=self.ksize,
                                  stride=self.stride,
                                  rate=1, padding='auto')
        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, L)
        m = m.permute(0, 4, 1, 2, 3)    # m shape: [N, L, C, k, k]
        m = m[0]    # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]

        mm = (torch.mean(m, dim=[1, 2, 3], keepdim=True) == 0.).to(
            torch.float32)
        mm = mm.permute(1, 0, 2, 3)  # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(torch.sum(torch.square(wi), dim=[
                                1, 2, 3], keepdim=True)).clamp_min(1e-4)
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            yi = F.conv2d(xi, wi_normed, stride=1, padding=(self.ksize-1)//2) # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                # (B=1, I=1, H=32*32, W=32*32)
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                # (B=1, C=1, H=32*32, W=32*32)
                yi = F.conv2d(yi, self.fuse_weight, stride=1,
                              padding=(self.fuse_k-1)//2)
                # (B=1, 32, 32, 32, 32)
                yi = yi.contiguous().view(
                    1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])
                yi = yi.permute(0, 2, 1, 4, 3)

                yi = yi.contiguous().view(
                    1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = F.conv2d(yi, self.fuse_weight, stride=1,
                              padding=(self.fuse_k-1)//2)
                yi = yi.contiguous().view(
                    1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()

            # (B=1, C=32*32, H=32, W=32)
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # [1, L, H, W]
            self.return_flow = True
            if self.return_flow:
                offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

                if int_bs != int_fs:
                    # Normalize the offset value to match foreground dimension
                    times = (int_fs[2]*int_fs[3])/(int_bs[2]*int_bs[3])
                    offset = ((offset + 1).float() * times - 1).to(torch.int64)
                offset = torch.cat([torch.div(offset, int_fs[3], rounding_mode='trunc'),
                                    offset % int_fs[3]], dim=1)  # 1*2*H*W
                offsets.append(offset)

            # deconv for patch pasting
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(
                yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y = y.contiguous().view(raw_int_fs)

        # if not self.return_flow:
        #     return y, None

        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(int_fs[0], 2, *int_fs[2:])

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(int_fs[2], device=device).view(
            [1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3])
        w_add = torch.arange(int_fs[3], device=device).view(
            [1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1)
        offsets = offsets - torch.cat([h_add, w_add], dim=1)
        # to flow image
        flow = torch.from_numpy(flow_to_image(
            offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)
        # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.long()).cpu().data.numpy()))

        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate,
                                 mode='bilinear', align_corners=True)

        return y, flow

# ----------------------------------------------------------------------------

def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))

# ----------------------------------------------------------------------------

def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img

# ----------------------------------------------------------------------------

def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC,
               2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM,
               0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel

# ----------------------------------------------------------------------------


def extract_image_patches(images, ksize, stride, rate, padding='auto'):
    """
    Extracts sliding local blocks \\
    see also: https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
    """

    padding = rate*(ksize-1)//2 if padding == 'auto' else padding

    unfold = torch.nn.Unfold(kernel_size=ksize,
                             dilation=rate,
                             padding=padding,
                             stride=stride)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks

# ----------------------------------------------------------------------------