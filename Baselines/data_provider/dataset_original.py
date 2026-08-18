# import sys
# import os   
# sys.path.insert(0, os.getcwd())
from torch.utils.data import Dataset
from utils.tools import load_scaler
import torch.nn.functional as F
import pickle as pk
import torch as tr
import numpy as np
import os


class Miedataset(Dataset):
    def __init__(self, args, mode):
        super().__init__()
        self.task = args.task
        self.exp_dir = os.path.join(args.checkpoints, args.model_id, str(args.seed))
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.mode = mode
        self.exogenous = args.exogenous
        self.data = args.data
        self.ps = args.pred_len
        
        if mode == 'pred':
            with open(os.path.join(self.exp_dir, 'intermediate_files', f'idx_splits_RP_{self.seq_len}_{self.ps}.pkl'), 'rb') as f:
                idx_splits = pk.load(f)
            self.idx_splits = []
            for mode in idx_splits:
                self.idx_splits.extend(idx_splits[mode])
        else:
            with open(os.path.join(self.exp_dir, 'intermediate_files', f'idx_splits_{self.data}_{self.seq_len}_{self.ps}.pkl'), 'rb') as f:
                self.idx_splits = pk.load(f)[mode]
            if self.data=='RP':
                with open(os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP', 'epi_indices.pkl'), 'rb') as f:
                    self.epi_idx = pk.load(f)
                self.joint_index = list(range(25))
            elif self.data=='VP':
                self.fps = args.fps
                self.joint_index = list(range(17)) + list(range(18,26)) # remove dummy joint head_cam
        
        if self.exogenous:
            self.scaler_act, self.scaler_fut, self.scaler_tar = load_scaler(self.exp_dir, dataset=self.data, seq_len=self.seq_len, ps=self.ps, exogenous=self.exogenous)
        else:
            self.scaler_act, self.scaler_fut, self.scaler_tar = load_scaler(self.exp_dir, dataset=self.data, seq_len=self.seq_len, ps=self.ps, exogenous=True)
    
    def __len__(self):
        return len(self.idx_splits)
    
    def __getitem__(self, index):
        if self.data=='RP':
            bufs_file, idx = self.idx_splits[index]
            s_end = idx
            s_begin = s_end - self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.ps

            seq_x_idx = np.asarray(self.epi_idx[bufs_file])[s_begin:s_end]
            seq_y_idx = np.asarray(self.epi_idx[bufs_file])[r_begin:r_end]
            seq_x_mark_idx = np.asarray(self.epi_idx[bufs_file])[s_begin:s_end]
            seq_y_mark_idx = np.asarray(self.epi_idx[bufs_file])[r_begin:r_end]

            with open(os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP', bufs_file), 'rb') as f:
                (buffers, elapsed, _, _) = pk.load(f, encoding='latin1')
            seq_x = np.asarray(buffers['position'][seq_x_idx], dtype='float32')*(np.pi/180)

            if self.task == 'classification':
                num_steps = int(bufs_file[-5])
                timepoints = np.asarray(elapsed, dtype='float32')
                if num_steps == 6:
                    label = np.array(0, dtype='float32')
                else:
                    stop_sign = 2.25*num_steps
                    label = np.array((timepoints[seq_y_idx]>stop_sign)[-1], dtype='float32')
                seq_y = tr.from_numpy(label).long()
                seq_y = F.one_hot(seq_y, 2)
            elif self.task == 'regression':
                seq_y = np.asarray(buffers['position'][seq_y_idx], dtype='float32')*(np.pi/180)
                self.scaler_fut.transform(seq_y)

            if self.exogenous:
                # for Ours: 
                exo_var = np.asarray(buffers['target'][seq_y_idx[:,None], self.joint_index], dtype='float32')*(np.pi/180)
                self.scaler_tar.transform(exo_var)
                seq_x_mark = seq_y_mark = exo_var
            else:
                seq_x_mark = np.asarray(seq_x_mark_idx, dtype=np.short)
                seq_y_mark = np.asarray(seq_y_mark_idx, dtype=np.short)
                seq_x_mark = seq_x_mark[:,np.newaxis]
                seq_y_mark = seq_y_mark[:,np.newaxis]

            # targets = np.asarray(buffers['target'][fur_indices], dtype='float32')*(np.pi/180)
            # num_steps = int(bufs_file[-5])
            # if num_steps == 0:
            #     label = np.array(1, dtype='float32')
            # else:
            #     stop_sign = 2.25*num_steps
            #     label = np.array(elapsed[lab_index]>stop_sign, dtype='float32')
            
        elif self.data=='VP':
            bufs_file, idx = self.idx_splits[index]
            s_end = idx
            s_begin = s_end - self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.ps

            seq_x_idx = np.asarray(range(s_begin,s_end), dtype='int16')
            seq_y_idx = np.asarray(range(r_begin,r_end), dtype='int16')
            seq_x_mark_idx = np.asarray(range(s_begin,s_end), dtype='int16')
            seq_y_mark_idx = np.asarray(range(r_begin,r_end), dtype='int16')

            data = np.load(os.path.join(os.path.dirname(os.getcwd()), 'data', 'VP', f'fps{self.fps}', bufs_file))
            seq_x = np.asarray(data['ajp'][seq_x_idx[:,None], self.joint_index], dtype='float32')

            if self.task == 'classification':
                seq_y = np.asarray(data['labels'][seq_y_idx][-1], dtype='float32')
                seq_y = tr.from_numpy(seq_y).long()
                seq_y = F.one_hot(seq_y, 2)
            elif self.task == 'regression':
                seq_y = np.asarray(data['ajp'][seq_y_idx[:,None], self.joint_index], dtype='float32')
                self.scaler_fut.transform(seq_y)
            
            if self.exogenous:
                # for Ours: 
                exo_var = np.asarray(data['traj'][seq_y_idx[:,None], self.joint_index], dtype='float32')
                self.scaler_tar.transform(exo_var)
                seq_x_mark = seq_y_mark = exo_var
            else:
                seq_x_mark = np.asarray(seq_x_mark_idx, dtype=np.short)
                seq_y_mark = np.asarray(seq_y_mark_idx, dtype=np.short)
                seq_x_mark = seq_x_mark[:,np.newaxis]
                seq_y_mark = seq_y_mark[:,np.newaxis]
                
            # targets = np.asarray(data['traj'][fur_indices,joint_index], dtype='float32')
            # label = np.asarray(data['labels'][lab_index], dtype='float32')

        self.scaler_act.transform(seq_x)
        if self.mode == 'testing':
            return seq_x, seq_y, seq_x_mark, seq_y_mark, bufs_file+str(idx)
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark


if __name__ == '__main__':
    from torch.utils.data import DataLoader


    class hyperparameters():
        def __init__(self):
        # datasets parameters
            self.bs = 64
            self.motion_dim = 25
            self.seq_len = 60 # the length of input sequence
            self.label_len = 30
            self.exogenous = False
            self.task = 'classification'
            self.checkpoints = 'checkpoints'
            self.model_id = 'Autoformer/Autoformer_VP_lr0.00000005_dmodel512_dff2048'
            self.seed = '608'
            self.seq_len = 60
            self.label_len = 30
            self.pred_len = 30

            self.data = None
            self.fps = None

    hp = hyperparameters()
    hp.data = 'VP'
    hp.fps = 30

    print('---------')
    print('Testing block for dataloader:')
    print(f'Data: {hp.data}')

    for m in ['testing']:
        dataset = Miedataset(hp, m)
        print(len(dataset))
        dataloader = DataLoader(dataset, batch_size=hp.bs, shuffle=True)
        print(len(dataloader))

        for i, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(dataloader):
            print(f'Batch: {i}')
            print(seq_x.shape, seq_x_mark.shape, seq_y.shape, seq_y_mark.shape)
            print(seq_x.dtype, seq_x_mark.dtype, seq_y.dtype, seq_y_mark.dtype)

            print(seq_y)

        # for i, (index, file) in enumerate(dataloader):
        #     print(f'Batch: {i}')
        #     print(f'{index}: {file}')