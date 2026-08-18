from data_provider.dataset import Miedataset
from torch.utils.data import DataLoader
import os


def data_provider(args, flag):

    if flag == 'testing':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        Data = Miedataset
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data = Miedataset_Pred
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        Data = Miedataset
    
    data_set = Data(args, mode= flag)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    
    return data_set, data_loader
