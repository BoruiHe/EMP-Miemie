import os
import yaml
import glob
import torch
import random
import numpy as np
import pickle as pk


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def save_hp_to_yaml(args):
    with open(os.path.join(args.checkpoints, 'hp.yml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

def get_params_dict(args, model):
    params_dict_ls = []

    if args.modalities.count('P') >= 1:
        params_dict = {}
        params_dict['params'] = model.motion_head.parameters()
        params_dict['lr'] = args.learning_rate[0]
        params_dict['weight_decay'] = args.weight_decay[0]
        params_dict_ls.append(params_dict)

    if args.modalities.count('V') == 1:
        params_dict = {}
        params_dict['params'] = model.egovision_head.parameters()
        params_dict['lr'] = args.learning_rate[1]
        params_dict['weight_decay'] = args.weight_decay[1]
        params_dict_ls.append(params_dict)

    if args.modalities.count('P') == 2:
        params_dict = {}
        params_dict['params'] = model.plan_head.parameters()
        params_dict['lr'] = args.learning_rate[2]
        params_dict['weight_decay'] = args.weight_decay[2]
        params_dict_ls.append(params_dict)

    params_dict = {}
    params_dict['params'] = model.classifier.parameters()
    params_dict['lr'] = args.learning_rate[3]
    params_dict['weight_decay'] = args.weight_decay[3]
    params_dict_ls.append(params_dict)

    return params_dict_ls

def adjust_learning_rate(optimizer, epoch, args):
    lr_adjust = {epoch: [lr * (0.5 ** ((epoch - 1) // 1)) for lr in args.learning_rate]}
    if epoch in lr_adjust.keys():
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_adjust[epoch][i]
        print(f'Updating learning rate to {lr_adjust[epoch]}.')

class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/checkpoint.pth')
        self.val_loss_min = val_loss

def rgba2rgb(rgba, background=(255,255,255)):
    n, row, col, _ = rgba.shape
    rgb = np.zeros((n, row, col, 3), dtype='float32')
    r, g, b, a = rgba[:,:,:,0], rgba[:,:,:,1], rgba[:,:,:,2], rgba[:,:,:,3]
    a = np.asarray(a, dtype='float32') / 255

    rgb[:,:,:,0] = r * a + (1.0 - a) * background[0]
    rgb[:,:,:,1] = g * a + (1.0 - a) * background[1]
    rgb[:,:,:,2] = b * a + (1.0 - a) * background[2]

    return np.asarray(rgb, dtype='uint8')

def reorder_joints(motion_tensor_for_RP):
    # motion_tensor_for_RP: (bs, T, motion_dim)

    RP_joints = ['abs_y', 'abs_x', 'abs_z', 'bust_y', 'bust_x', 'head_z', 'head_y', 'l_shoulder_y', 'l_shoulder_x', 'l_arm_z', 
                 'l_elbow_y', 'r_shoulder_y', 'r_shoulder_x', 'r_arm_z', 'r_elbow_y', 'l_hip_x', 'l_hip_z', 'l_hip_y', 'l_knee_y', 'l_ankle_y', 
                 'r_hip_x', 'r_hip_z', 'r_hip_y', 'r_knee_y', 'r_ankle_y']

    VP_joints = {'r_hip_x': 0, 'r_hip_z': 1, 'r_hip_y': 2, 'r_knee_y': 3, 'r_ankle_y': 4, 'l_hip_x': 5, 'l_hip_z': 6, 'l_hip_y': 7, 'l_knee_y': 8, 
                'l_ankle_y': 9, 'abs_y': 10, 'abs_x': 11, 'abs_z': 12, 'bust_y': 13, 'bust_x': 14, 'head_z': 15, 'head_y': 16, 'head_cam': 17, 
                'l_shoulder_y': 18, 'l_shoulder_x': 19, 'l_arm_z': 20, 'l_elbow_y': 21, 'r_shoulder_y': 22, 'r_shoulder_x': 23, 'r_arm_z': 24, 
                'r_elbow_y': 25}
    
    VP_joints.pop('head_cam')
    new_order = []
    for joint_name in VP_joints.keys():
        new_order.append(RP_joints.index(joint_name))

    return motion_tensor_for_RP[:,new_order]

def variance():

    # SimP
    joint_angles_collection = []
    index = list(range(26))
    index.remove(17)
    index = np.array(index)

    dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'VP', 'fps30')
    files = os.listdir(dataset_path)
    for file_name in files:
        file = np.load(os.path.join(dataset_path, file_name))
        joint_angles = file['ajp'][:, index]
        joint_angles_collection.append(joint_angles)
    print('num of datapoints:', len(joint_angles_collection))
    var = np.var(joint_angles_collection, axis=0)
    ave_var = var.mean()
    print('SimP Variance:', ave_var)

    # RP
    joint_angles_collection = []
    dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP')
    all_entries = os.listdir(dataset_path)
    all_directories = [entry for entry in all_entries if os.path.isdir(os.path.join(dataset_path, entry))]

    for dir in all_directories:
        files = glob.glob(os.path.join(dataset_path, dir, 'bufs_*.pkl'))
        for file_name in files:
            with open(file_name, 'rb') as f:
                (buffers, _, _, _) = pk.load(f, encoding='latin1')
            joint_angles = buffers['position'][:972, :] * (np.pi/180)
            joint_angles_collection.append(joint_angles)

    print('num of datapoints:', len(joint_angles_collection))
    var = np.var(joint_angles_collection, axis=0)
    ave_var = var.mean()
    print('RP Variance:', ave_var)


# def gen_velocity(m):
#     dm = m[:, 1:] - m[:, :-1]
#     return dm

# def MSE_siMLPe(pred, gt):
#     mse = torch.nn.MSELoss(reduction='none')(pred, gt)
#     return mse.mean(dim=2, keepdim=True).sum(dim=0, keepdim=True).squeeze(), mse.std(dim=2, keepdim=True).sum(dim=0, keepdim=True).squeeze()

# def MSE(pred, gt, mode):
#     loss_fn = torch.nn.MSELoss(reduction='sum')
#     b,c,n,h,w = gt.shape
#     if mode == 'mean':
#         return loss_fn(pred, gt)/(b*n)
#     elif mode == 'sum':
#         return loss_fn(pred, gt)/n

# def MSE_traj(pred, gt, mode):
#     loss_fn = torch.nn.MSELoss(reduction='sum')
#     b,_ = gt.shape
#     if mode == 'mean':
#         return loss_fn(pred, gt)/b
#     elif mode == 'sum':
#         return loss_fn(pred, gt)


if __name__ == '__main__':
    variance()
    # import torch as tr
    
    # a = tr.arange(2*25)
    # a = a.reshape(1,2,25)
    # print(a)
    # print(reorder_joints(a))
    # gt = torch.ones((2,60,25))
    # pred = torch.zeros((2,60,25))
    # ave = np.zeros([60])
    pass