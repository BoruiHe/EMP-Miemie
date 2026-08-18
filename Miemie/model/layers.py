import torch as tr
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(y)
        return y

class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker, stride=1, groups=8):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=stride, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

class motion_norm_v1(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(tr.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(tr.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):   
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y
    
class motion_norm_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(tr.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(tr.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y
    
class TemporalBlock(nn.Module):
    def __init__(self, motion_dim, seq_len, out_seq_len=None, norm='ours'):
        super().__init__()
        if out_seq_len:
            self.fc = nn.Linear(seq_len, out_seq_len)
        else:
            self.fc = nn.Linear(seq_len, seq_len)
        if norm == 'ours':
            self.norm = motion_norm_v1(motion_dim)
        elif norm == 'bn':
            self.norm = nn.BatchNorm1d(motion_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm([motion_dim, seq_len])
        self._init_parameters()
        self.skipconnection = (self.fc.in_features == self.fc.out_features)

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x_ = self.fc(x)
        x_ = self.norm(x_)
        if self.skipconnection: 
            x_ = x + x_

        return x_
    
class SpatialBlock(nn.Module):
    def __init__(self, motion_dim, seq_len, norm='ours'):
        super().__init__()
        self.fc = nn.Linear(motion_dim, motion_dim)
        if norm == 'ours':
            self.norm = motion_norm_v2(motion_dim)
        elif norm == 'bn':
            self.norm = nn.BatchNorm1d(seq_len)
        elif norm == 'ln':
            self.norm = nn.LayerNorm([seq_len, motion_dim])
        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x_ = self.fc(x)
        x_ = self.norm(x_)
        x_ = x + x_

        return x_
    
class FastFourierTransform(nn.Module):
    def __init__(self, threshold, return_all=False):
        super(FastFourierTransform, self).__init__()
        self.threshold = threshold
        self.return_all = return_all
        self.ht_activation = tr.nn.Hardtanh(0,255)
    
    def forward(self, batched_img_seq):
        
        bis = batched_img_seq.float()
        bs, seq_len, H, W, C = bis.shape
        device = batched_img_seq.device
        assert self.threshold <= min(H//2, W//2), f'{self.threshold} is too large for image size H:{H}, W:{W}.'

        f = tr.fft.fft2(bis, dim=(2,3))
        fshift = tr.fft.fftshift(f)

        rows, cols = int(H/2), int(W/2)
        mask = tr.zeros((H, W, C), dtype=tr.uint8).to(device)
        mask[(rows - self.threshold):(rows + self.threshold), (cols - self.threshold):(cols + self.threshold)] = 1
        fshift = fshift * mask               
        ishift = tr.fft.ifftshift(fshift)
        i_img = tr.fft.ifft2(ishift, dim=(2,3))
        LF_feat = tr.abs(i_img)

        HF_feat = (bis - LF_feat)
        HF_feat = HF_feat*(255/HF_feat.max())*3

        hf_ = self.ht_activation(HF_feat)
        lf_ = self.ht_activation(LF_feat)
        
        if self.return_all:
            return hf_, lf_
        else:
            return lf_
        
def FFT_for_Period(x, k):
    # [B, T, C]
    xf = tr.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0

    _, top_list = tr.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list

    return period, abs(xf).mean(-1)[:, top_list]

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = tr.stack(res_list, dim=-1).mean(-1)
        return res
    
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.motion_dim, configs.inc_dim, num_kernels=2),
            nn.GELU(),
            Inception_Block_V1(configs.inc_dim, configs.motion_dim, num_kernels=2)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = tr.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = tr.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = tr.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = tr.sum(res * period_weight, -1)
        res = res + x
        return res


if __name__ == '__main__':
    bs = 16
    N = 60
    C = 25

    class arguments():
        def __init__(self):
            self.seq_len = 0
            self.pred_len = 12
            self.top_k = 3
            self.inc_dim = 32
            self.motion_dim = 25
    
    args = arguments()
    a = tr.randn(bs, args.motion_dim, args.motion_dim)
    # tb = TimesBlock(args)
    # print(tb(a).shape)

    # a = tr.randn(bs, C, N)
    norm_1 = motion_norm_v1(C)
    # norm_2 = motion_norm_v2(N)
    # norm_3 = nn.BatchNorm1d(C)
    out_1 = norm_1(a)
    # out_2 = norm_2(a)
    # out_3 = norm_3(a)
    # tb = TemporalBlock(25, 30, 20)

    # H, W = 96, 96
    # L = 30

    # motion = tr.randn(bs, N, C)
    # exo_traj = tr.randn((bs, L, C))
    # img = tr.randint(0, 255, (bs, H, W, 3)).float()
    # img = tr.randint(0, 255, (1, 2, 2, 3)).float()/255

    # import pickle as pk
    # with open('/home/bhe100/Desktop/data/RP/2025_05_08_stdev0.25/bufs_00_4.pkl', 'rb') as f:
    #     (buffers, _, _, _) = pk.load(f, encoding='latin1')
    #     img = np.stack([buffers['images'][0][:,:,::-1],buffers['images'][-1][:,:,::-1]])

    # img = tr.from_numpy(img).repeat(2,1,1,1,1)

    # fft_layer = FastFourierTransform(30, return_all=True)
    # hf, lf = fft_layer(img)
    # hf, lf = hf.type(tr.uint8), lf.type(tr.uint8)
    # import matplotlib.pyplot as pt
    # fig, axes = pt.subplots(4,3)
    # for i in range(2):
    #     for j in range(2):
    #         axes[i*2+j, 0].imshow(hf[i,j])
    #         axes[i*2+j, 1].imshow(lf[i,j])
    #         axes[i*2+j, 2].imshow(img[i,j])
    # pt.show()