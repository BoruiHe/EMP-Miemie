from utils.miscellaneous import reorder_joints
from torch.utils.data import Dataset
import torchvision.transforms as trf
import torch.nn.functional as F
import pickle as pk
import torch as tr
import numpy as np
import os


class Miedataset_Egofalls(Dataset):
    def __init__(self, args, mode):
        super().__init__()
        self.exp_dir = os.path.join(args.checkpoints, str(args.seed))
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.fps = args.fps
        self.mode = mode
        self.data = args.data
        self.cdv = args.do_cdv

        if self.data == 'RP':
            with open(os.path.join(self.exp_dir, 'intermediate_files', f'idx_splits_RP_{self.seq_len}_{self.pred_len}.pkl'), 'rb') as f:
                self.idx_splits = pk.load(f)[mode]
            with open(os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP', 'epi_indices.pkl'), 'rb') as f:
                self.epi_idx = pk.load(f)
            self.joint_index = list(range(25))
        elif self.data == 'VP':
            if args.do_cdv and (mode == 'testing'):
                with open(os.path.join(self.exp_dir, 'intermediate_files', f'idx_splits_RP_{self.seq_len}_{self.pred_len}.pkl'), 'rb') as f:
                    idx_splits = pk.load(f)
                with open(os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP', 'epi_indices.pkl'), 'rb') as f:
                    self.epi_idx = pk.load(f)
                self.idx_splits = []
                self.idx_splits.extend(idx_splits[mode])
                self.data = 'RP'
                self.joint_index = list(range(25))
            else:
                with open(os.path.join(self.exp_dir, 'intermediate_files', f'idx_splits_VP_{self.seq_len}_{self.pred_len}.pkl'), 'rb') as f:
                    self.idx_splits = pk.load(f)[mode]
                self.joint_index = list(range(17)) + list(range(18,26)) # remove dummy joint head_cam

    def __len__(self):
        return len(self.idx_splits)
    
    def __getitem__(self, index):
        if self.data=='RP':
            bufs_file, idx = self.idx_splits[index]
            
            his_indices = np.asarray(self.epi_idx[bufs_file])[np.linspace(0, idx, num=10, endpoint=False).astype(np.uint8)]
            lab_index = np.asarray(self.epi_idx[bufs_file])[idx+self.pred_len-1]
            with open(os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP', bufs_file), 'rb') as f:
                (buffers, elapsed, _, _) = pk.load(f, encoding='latin1')
            frames = np.asarray(buffers['images'][his_indices].tolist(), dtype='float32').transpose(0, 3, 1, 2)/255.

            num_steps = int(bufs_file[-5])
            if num_steps == 0:
                label = np.array(1, dtype='float32')
            elif num_steps == 6:
                label = np.array(0, dtype='float32')
            else:
                stop_sign = 2.25*num_steps
                label = np.array(elapsed[lab_index]>stop_sign, dtype='float32')
        elif self.data=='VP':
            file, idx = self.idx_splits[index]

            his_indices = np.asarray(np.linspace(0, idx, num=10, endpoint=False).astype(np.uint8), dtype='int16')
            lab_index = idx+self.pred_len-1
            data = np.load(os.path.join(os.path.dirname(os.getcwd()), 'data', 'VP', f'fps{self.fps}', file))
            frames = np.asarray(data['imgs'][his_indices], dtype='float32').transpose(0, 3, 1, 2)/255.
            label = np.asarray(data['labels'][lab_index], dtype='float32')

        label = tr.from_numpy(label).long()

        frames = tr.from_numpy(frames).float()
        resize = trf.Resize((224, 224))
        frames = resize(frames).permute(1,0,2,3)
        return frames, frames, frames, frames, label
        

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from utils.preprocessing import RP_preprocessing, VP_preprocessing, RP_scaler, VP_scaler


    class arguments():
        def __init__(self):
            self.checkpoints= os.path.join('checkpoints', 'debug')
            self.seed = 608
            self.seq_len = 24
            self.pred_len = 12
            self.relax_factor = 0
            self.data = 'RP'
            self.do_cdv = False

    args = arguments()
    # VP_preprocessing(args.seq_len, args.pred_len, [0.8,0.1,0.1], os.path.join(args.checkpoints, str(args.seed)), args.fps, relax_factor=args.relax_factor)
    # VP_scaler(os.path.join(args.checkpoints, str(args.seed)), args.seq_len, args.pred_len, args.fps)
    for m in ['training', 'validation', 'testing']:
        dataset = Miedataset(args, m)
        print(f'dataset length: {len(dataset)}')
        dataloader = DataLoader(dataset, batch_size=4)
        print(f'dataloader length: {len(dataloader)}')

        # f,n = 0,0
        for (frames, curr_joint_angles, future_joint_angles, targets, labels) in dataloader:
            # print(frames, curr_joint_angles, future_joint_angles, targets, labels)
            print(frames.shape, curr_joint_angles.shape, future_joint_angles.shape, targets.shape, labels.shape)
            # print(frames.dtype, curr_joint_angles.dtype, future_joint_angles.dtype, targets.dtype, labels.dtype)
        #     f += (labels==1).sum()
        #     n += (labels==0).sum()
        # for (f, i) in dataloader:
        #     print(f'{i}, {f}')
        # print(f'Failure video clips: {f}, success video clips": {n}\n')