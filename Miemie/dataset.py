from utils.miscellaneous import reorder_joints
from torch.utils.data import Dataset
import torchvision.transforms as trf
import torch.nn.functional as F
import pickle as pk
import torch as tr
import numpy as np
import os


class Miedataset(Dataset):
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
            # self.joint_index = [0, 3, 4, 5, 6, 8, 12, 13, 16, 17, 18, 19, 22, 23, 24]
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

        self.scaler_act, self.scaler_fut, self.scaler_tar = self._load_scaler()

    def _load_scaler(self):
        with open(os.path.join(self.exp_dir, 'intermediate_files', f'scaler_act_{self.data}_{self.seq_len}_{self.pred_len}.pkl'), 'rb') as f:
            scaler_1 = pk.load(f)
        with open(os.path.join(self.exp_dir, 'intermediate_files', f'scaler_fut_{self.data}_{self.seq_len}_{self.pred_len}.pkl'), 'rb') as f:
            scaler_2 = pk.load(f)
        with open(os.path.join(self.exp_dir, 'intermediate_files', f'scaler_tar_{self.data}_{self.seq_len}_{self.pred_len}.pkl'), 'rb') as f:
            scaler_3 = pk.load(f)
        return scaler_1, scaler_2, scaler_3

    def __len__(self):
        return len(self.idx_splits)
    
    def __getitem__(self, index):
        if self.data=='RP':
            bufs_file, idx = self.idx_splits[index]

            his_indices = np.asarray(self.epi_idx[bufs_file])[(idx-self.seq_len):idx]
            fur_indices = np.asarray(self.epi_idx[bufs_file])[(idx-self.seq_len):(idx+self.pred_len)]
            lab_index = np.asarray(self.epi_idx[bufs_file])[idx+self.pred_len-1]
            with open(os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP', bufs_file), 'rb') as f:
                (buffers, elapsed, _, _) = pk.load(f, encoding='latin1')
            frames = np.asarray(buffers['images'][his_indices].tolist(), dtype='float32').transpose(0, 3, 1, 2)/255.
            actuals = np.asarray(buffers['position'][his_indices[:,None], self.joint_index], dtype='float32')*(np.pi/180)
            future = np.asarray(buffers['position'][fur_indices[:,None], self.joint_index], dtype='float32')*(np.pi/180)
            targets = np.asarray(buffers['target'][fur_indices[:,None], self.joint_index], dtype='float32')*(np.pi/180)

            if self.cdv and (self.mode == 'testing'):
                actuals = reorder_joints(actuals)
                future = reorder_joints(future)
                targets = reorder_joints(targets)

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

            his_indices = np.asarray(range(idx-self.seq_len, idx), dtype='int16')
            fur_indices = np.asarray(range((idx-self.seq_len), (idx+self.pred_len)), dtype='int16')
            lab_index = idx+self.pred_len-1
            data = np.load(os.path.join(os.path.dirname(os.getcwd()), 'data', 'VP', f'fps{self.fps}', file))
            frames = np.asarray(data['imgs'][his_indices], dtype='float32').transpose(0, 3, 1, 2)/255.
            actuals = data['ajp'][his_indices[:,None], self.joint_index]
            future = data['ajp'][fur_indices[:,None], self.joint_index]
            targets = data['traj'][fur_indices[:,None], self.joint_index]
            label = np.asarray(data['labels'][lab_index], dtype='float32')

        self.scaler_act.transform(actuals)
        self.scaler_fut.transform(future)
        self.scaler_tar.transform(targets)
        label = tr.from_numpy(label).long()
        label = F.one_hot(label, 2)
        frames = tr.from_numpy(frames).float()
        resize = trf.Resize((128,128))
        frames = resize(frames).permute(1,0,2,3)
        return actuals, future, frames, targets, label
        

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from utils.preprocessing import RP_preprocessing, VP_preprocessing, RP_scaler, VP_scaler


    class arguments():
        def __init__(self):
            self.checkpoints= os.path.join('checkpoints', 'VP_pl30_P_LR0.1_finetune')
            self.seed = 608
            self.seq_len = 60
            self.pred_len = 30
            self.relax_factor = 0
            self.fps = 30
            self.data = 'VP'
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
            print(frames, curr_joint_angles, future_joint_angles, targets, labels)
            # print(frames.shape, curr_joint_angles.shape, future_joint_angles.shape, targets.shape, labels.shape)
            # print(frames.dtype, curr_joint_angles.dtype, future_joint_angles.dtype, targets.dtype, labels.dtype)
        #     f += (labels==1).sum()
        #     n += (labels==0).sum()
        # for (f, i) in dataloader:
        #     print(f'{i}, {f}')
        # print(f'Failure video clips: {f}, success video clips": {n}\n')