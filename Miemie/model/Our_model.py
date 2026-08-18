import os
import sys
sys.path.insert(0, os.getcwd())
import math
import torch as tr
from torch import nn
# import torch.nn.functional as F
from model.layers import TemporalBlock, SpatialBlock, FastFourierTransform, TimesBlock, get_dct_matrix, Inception


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.modalities = configs.modalities
        if configs.modalities != 'V':
            self.motion_head = MotionModule(configs)

        if ('V' in configs.modalities):
            self.egovision_head = EgovisionModule(configs)
        
        if (configs.modalities == 'PP') or (configs.modalities == 'PVP'):
            self.plan_head = TrajModule(configs)

        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(len(configs.modalities)*configs.d_model, configs.num_class),
        )

        self._init_parameters()

    def forward(self, motion, imgs, traj_plan):
        output = []
        if self.modalities != 'V':
            mot_ = self.motion_head(motion)
            output.append(mot_)

        if (self.modalities == 'PP') or (self.modalities == 'PVP'):
            pl_ = self.plan_head(motion, traj_plan)
            output.append(pl_)

        if ('V' in self.modalities):
            imgs_ = self.egovision_head(imgs)
            output.append(imgs_)

        output = tr.cat(output, dim=-1)
        output = self.classifier(output)

        return output
    
    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.classifier[1].weight, a=0)
        nn.init.constant_(self.classifier[1].bias, 0)


class MotionModule(nn.Module):
    def __init__(self, configs):
        super(MotionModule, self).__init__()
        # DCT and IDCT matrices for seq_len
        dct_m, idct_m = get_dct_matrix(configs.seq_len)
        dct_m, idct_m = tr.from_numpy(dct_m).float().unsqueeze(0), tr.from_numpy(idct_m).float().unsqueeze(0)
        self.register_buffer('dct_seq', dct_m)
        self.register_buffer('idct_seq', idct_m)

        self.spa_mlp = nn.ModuleList([SpatialBlock(configs.motion_dim, configs.seq_len, configs.normalization) for _ in range(2)])
        self.tem_mlp = nn.Sequential(*[TemporalBlock(configs.motion_dim, configs.seq_len, norm=configs.normalization) for _ in range(configs.num_Templayers)])
        self.act = nn.GELU()

        self.projection = nn.Linear(configs.motion_dim*configs.seq_len, configs.d_model)

        self.apply(self._init_parameters)

    def forward(self, motion):
        # motion: (bs, seq_len, motion_dim)

        mot_ = tr.matmul(self.dct_seq, motion)
        mot_ = self.spa_mlp[0](mot_)
        mot_ = mot_.permute(0,2,1)
        mot_ = self.tem_mlp(mot_)
        mot_ = mot_.permute(0,2,1)
        mot_ = self.spa_mlp[1](mot_)
        mot_ = tr.matmul(self.idct_seq, mot_)
        
        mot_ = self.act(mot_)
        mot_ = mot_.flatten(start_dim=1)
        mot_ = self.projection(mot_)
    
        return mot_
    
    def _init_parameters(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=1e-8)
            nn.init.constant_(layer.bias, 0)
    

class EgovisionModule(nn.Module):
    def __init__(self, configs):
        super(EgovisionModule, self).__init__()
        self.H, self.W = 128, 128
        self.seq_len = configs.seq_len

        self.fft_block = FastFourierTransform(threshold=configs.fft_threshold, return_all=False)

        channel_list = [3] + [5*(i+1) for i in range(configs.num_convlayers)]
        conv_block_list = []
        for i in range(configs.num_convlayers):
            conv_block_list.extend([nn.Conv3d(channel_list[i], channel_list[i+1], kernel_size=(3), dilation=(1), stride=(1,2,2), padding=(1)), 
                                    nn.ReLU(inplace=True), 
                                    ]) 
        self.conv_block = nn.Sequential(*conv_block_list)

        self.dsize = self._getDynamicSize()
        self.projection = nn.Linear(self.dsize, configs.d_model)

        self.apply(self._init_weights)

    def forward(self, imgs):
        # imgs: (bs, C, seq_len, H, W)

        imgs_ = imgs.permute(0,2,3,4,1)
        imgs_ = self.fft_block(imgs_)
        imgs_ = imgs_.permute(0,4,1,2,3)

        imgs_ = self.conv_block(imgs_)

        imgs_ = imgs_.flatten(start_dim=1)
        imgs_ = self.projection(imgs_)

        return imgs_

    def _getDynamicSize(self):
        a = tr.zeros(1, 3, self.seq_len, self.H, self.W)
        return self.conv_block(a).numel()
    
    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, a=0)
            nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Conv3d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(layer.bias, -bound, bound)
        elif isinstance(layer, nn.BatchNorm1d):
            nn.init.constant_(layer.bias, 0)


class TrajModule(nn.Module):
    def __init__(self, configs):
        super(TrajModule, self).__init__()
        self.body = nn.ModuleList([TimesBlock(configs) for _ in range(configs.num_Timeslayers)])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.05)
        self.projection = nn.Linear(configs.motion_dim * (configs.seq_len + configs.pred_len), configs.d_model)
        # self.projection = nn.Linear(configs.motion_dim * (configs.seq_len), configs.d_model)

        self._init_parameters()
        
    def forward(self, motion, plannedtraj):
        # plannedtraj: (bs, seq_len, motion_dim)

        pl_ = plannedtraj # - motion[:, -1:, :]
        for blocks in self.body:
            pl_ = blocks(pl_)
        pl_ = self.act(pl_)
        pl_ = self.dropout(pl_)
        pl_ = pl_.reshape(pl_.shape[0], -1)
        pl_ = self.projection(pl_)

        return pl_
    
    def _init_parameters(self):
        nn.init.xavier_uniform_(self.projection.weight, gain=1e-8)
        nn.init.constant_(self.projection.bias, 0)


if __name__ == "__main__":
    from utils.miscellaneous import set_random_seeds
    from torchinfo import summary


    set_random_seeds(608)
    class conf():
        def __init__(self):
            self.num_class = 2
            self.seq_len = 60
            self.pred_len = 30
            self.fft_threshold = 30
            self.d_model = 1024
            self.motion_dim = 25
            self.num_Templayers = 4
            self.num_convlayers = 4
            self.num_Timeslayers = 2
            self.inc_dim = 64
            self.top_k = 3
            self.modalities = 'PVP'
    args = conf()

    model = Model(args)
    summary(model, input_size=[(2,60,25), (2,3,60,128,128), (2,90,25)])
    print(model(tr.randn(2,60,25), tr.randn(2,3,60,128,128), tr.randn(2,90,25)).shape)

    model = EgovisionModule(args)
    summary(model, input_size=[(3,3,60,128,128)])
    print(model(tr.randn(2,3,60,128,128)).shape)

    # model = TrajModule(args)
    # print(model(tr.randn(2,60,25), tr.randn(2,30,25)).shape)
    pass