import pickle as pk
import numpy as np
import random
import torch
import yaml
import os


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
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
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def load_scaler(exp_path, dataset, seq_len, ps, exogenous=False):
    with open(os.path.join(exp_path, 'intermediate_files', f'scaler_act_{dataset}_{seq_len}_{ps}.pkl'), 'rb') as f:
        scaler_1 = pk.load(f)
    with open(os.path.join(exp_path, 'intermediate_files', f'scaler_fut_{dataset}_{seq_len}_{ps}.pkl'), 'rb') as f:
        scaler_2 = pk.load(f)
    if exogenous:
        with open(os.path.join(exp_path, 'intermediate_files', f'scaler_tar_{dataset}_{seq_len}_{ps}.pkl'), 'rb') as f:
            scaler_3 = pk.load(f)
        return scaler_1, scaler_2, scaler_3
    else:
        return scaler_1, scaler_2
    
def save_hp_to_yaml(args):
    with open(os.path.join(args.checkpoints, args.model_id, 'hp.yml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

