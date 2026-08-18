import os
import argparse
import numpy as np
from EXP.encoder import train_encoder, test_encoder, Egofalls_cpt
from utils.miscellaneous import set_random_seeds, save_hp_to_yaml
from utils.preprocessing import RP_preprocessing, VP_preprocessing, RP_scaler, VP_scaler, ER_scaler


parser = argparse.ArgumentParser(description='Process user defined hyperparameters')

# Exp
parser.add_argument('--exp_tag', type=str, required=True, help='Something we need to know about this experiment.')
parser.add_argument('--num_class', type=int, default=2, help='The number of classes for classification tasks.')
parser.add_argument('--modalities', type=str, default='PVP', help='Change it for ablation study, options: [PVP, PV, PP, P].')
parser.add_argument('--do_cdv', action='store_true', help='Whether to do cross data validation, True for yes, default is False.')


# Ours
parser.add_argument('--pred_len', type=int, required=True, help='Prediction horizon.')
parser.add_argument('--d_model', type=int, default=1024, help='The dimension of hidden features for the classifier.')
parser.add_argument('--num_Templayers', type=int, default=6, help='The num of TemporalBlocks in the Motion module.')
parser.add_argument('--normalization', default='ours', choices=['bn', 'ln', 'ours'], help='The normalization layers used in MotionModule.')
parser.add_argument('--num_Timeslayers', type=int, default=2, help='The num of TimesBlocks in the Traj module.')
parser.add_argument('--inc_dim', type=int, default=256, help='The num of channels in Inception Blocks.')
parser.add_argument('--top_k', type=int, default=3, help='The frequencies of top K amplitude will be taken.')
parser.add_argument('--num_convlayers', type=int, default=4, help='The num of conv3D layers in EgovisionModule.')
parser.add_argument('--num_SPAconvlayers', type=int, default=6, help='The num of spatial conv layers in EgovisionModule.')
parser.add_argument('--num_TEMconvlayers', type=int, default=4, help='The num of temporal conv layers in EgovisionModule.')
parser.add_argument('--hid_SPA', type=int, default=64, help='The dimension of hidden spatial features.')
parser.add_argument('--hid_TEM', type=int, default=64, help='The dimension of hidden temporal features.')
parser.add_argument('--fft_threshold', type=int, default=30, help='FFT threshold.')
parser.add_argument('--motion_dim', type=int, default=25, help='The DoF of Poppy.')


# Egofalls
parser.add_argument('--Egofalls', action='store_true', default=False, help='True for model Egofalls.')
parser.add_argument('--C', type=float, help='SVC regularization parameter.')
parser.add_argument('--kfold', default=4, help='k-fold validation.')


# Dataset
parser.add_argument('--data', type=str, required=True, help='The name of dataset.')
parser.add_argument('--fps', type=str, default=30, help='Freq for the VP, options: [16, 30].')
parser.add_argument('--relax_factor', type=int, default=0, help='Whether to follow i.i.d sampling for calssification. 0 for yes, 1 to 5 for more examples.')
parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
parser.add_argument('--num_workers', type=int, default=10, help='Data loader num workers.')


# optimization
parser.add_argument('--train_epochs', type=int, default=10, help='Training epochs.')
parser.add_argument('--patience', type=int, default=3, help='Early stopping patience.')
parser.add_argument('--Vlr', type=float)
parser.add_argument('--Vwd', type=float)
parser.add_argument('--learning_rate', type=float, default=[0.0005, 0.00001, 0.0001, 0.0001], nargs='*', help='Optimizer learning rate.')
parser.add_argument('--weight_decay', type=float, default=[0.0001, 0.0005, 0.0001, 0.0001], nargs='*', help='Optimizer weight decay.')

args = parser.parse_args()
# args = parser.parse_args('--data ExchangeRate --exp_tag ExchangeRate_pl12 --modalities P --pred_len 12 --motion_dim 8 --num_workers 0 --learning_rate 0.01 0.00001 0.0001 0.0001'.split())
if (args.data == 'RP') or args.do_cdv or (args.data == 'ExchangeRate'):
    args.seq_len = 24
else:
    args.seq_len = 60
# if args.do_cdv:
#     args.fps = 16
args.checkpoints = os.path.join('checkpoints', args.exp_tag)
# args.learning_rate[3] = args.Vlr
# args.weight_decay[3] = args.Vwd
print(args)
seeds = [608, 1247, 3224, 2025, 5, 57, 234, 235, 369, 403, 451, 620, 762, 842, 1099, 1260, 1381, 1456, 1496, 1530, 1700, 1772, 1951, 2236, 2401, 2512, 2561, 2619, 2968, 3065]
acc_list = []
eps = 0
for seed in seeds:
    # set random seed
    set_random_seeds(seed)
    args.seed = seed
    print(f'\n-------{seed}-------')
    
    path_if = os.path.join(args.checkpoints, str(seed), 'intermediate_files')
    if not os.path.exists(path_if):
        os.makedirs(path_if)

    # save model hyperparameters
    save_hp_to_yaml(args)
    
    # data preprocessing
    path_idx_splits = os.path.join(path_if, f'idx_splits_{args.data}_{args.seq_len}_{args.pred_len}.pkl')
    if not os.path.exists(path_idx_splits):
        if args.data == 'RP':
            RP_preprocessing(args.seq_len, args.pred_len, [0.8,0.2], os.path.join(args.checkpoints, str(seed)), relax_factor=args.relax_factor)
            RP_scaler(os.path.join(args.checkpoints, str(seed)), args.seq_len, args.pred_len,)
        elif args.data == 'ExchangeRate':
            ER_scaler(os.path.join(args.checkpoints, str(seed)), args.seq_len, args.pred_len,)
        else:
            if args.do_cdv:
                VP_preprocessing(args.seq_len, args.pred_len, [0.8,0.2], os.path.join(args.checkpoints, str(seed)), args.fps, CDV=args.do_cdv, relax_factor=args.relax_factor)
                VP_scaler(os.path.join(args.checkpoints, str(seed)), args.seq_len, args.pred_len, args.fps)
                RP_preprocessing(args.seq_len, args.pred_len, None, os.path.join(args.checkpoints, str(seed)), CDV=args.do_cdv, relax_factor=args.relax_factor)
                RP_scaler(os.path.join(args.checkpoints, str(seed)), args.seq_len, args.pred_len, CDV=args.do_cdv)
            else:
                VP_preprocessing(args.seq_len, args.pred_len, [0.8,0.1,0.1], os.path.join(args.checkpoints, str(seed)), args.fps, relax_factor=args.relax_factor)
                VP_scaler(os.path.join(args.checkpoints, str(seed)), args.seq_len, args.pred_len, args.fps)
    
    if args.Egofalls:
        acc_list.append(Egofalls_cpt(args))
    else:
        eps += train_encoder(args)
        acc_list.append(test_encoder(args))

f = open(f'result_{args.exp_tag}.txt', 'a')
f.write(f'Average ACC: {np.array(acc_list).mean()}, Std: {np.array(acc_list).std()}, eps: {eps/len(seeds)}')
f.write('\n')
f.write('\n')
f.close()

np.save(args.checkpoints + '/acc_list.npy', np.array(acc_list))
