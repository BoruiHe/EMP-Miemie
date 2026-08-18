import argparse
import os
import torch
import numpy as np
from EXP.exp_regression import Exp_Main as Exp_Main_r
from EXP.exp_classification_skl import Exp_Main as Exp_Main_c
from utils.preprocessing import RP_preprocessing, VP_preprocessing, RP_scaler, VP_scaler
from utils.tools import save_hp_to_yaml, set_random_seeds


def main():
    parser = argparse.ArgumentParser(description='Good luck Miemie')

    # basic config
    parser.add_argument('--is_training', type=int, default=True, help='status')
    parser.add_argument('--model_id', type=str, required=True, help='model description')
    parser.add_argument('--model', type=str, required=True, help='model name, options: [Autoformer, DLinear, TiDE, TimeMixer, TimeXer, Flowformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, help='dataset type')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M]; M:multivariate predict multivariate.')
    parser.add_argument('--fps', type=str, default=30, help='freq for the VP, options: [16, 30].')
    parser.add_argument('--freq', type=str, default='ts', help='freq for time features encoding. No other choices unless more datasets are included')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--label_len', type=int, default=30, help='start token length')
    parser.add_argument('--pred_len', type=int, default=30, help='prediction sequence length')
    parser.add_argument('--exogenous', action='store_true', help='whether the model takes exogenous variables, False for no, default is False.')

    # forecasting task
    parser.add_argument('--relax_factor', type=int, default=0, help='Whether to follow i.i.d sampling for calssification. 0 for yes, 1 to 5 for more examples.')
    parser.add_argument('--num_class', type=int, default=2, help='The number of classes for classification tasks.')
    parser.add_argument('--task', type=str, required=True, help='task type, option: [classification, regression].')
    parser.add_argument('--do_cdv', action='store_true', help='whether to do cross data validation, True for yes, default is False.')

    # TimesNet
    parser.add_argument('--top_k', type=int, default=3, help='Frequencies of top k amplitude.')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    # Flowformer
    parser.add_argument('--max_len', type=int, default=60, help='maximum of seq_len')
    # TimeMixer
    parser.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--down_sampling_layers', type=int, default=1, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg', help='down sampling method, only support avg, max, conv')
    # TimeXer
    parser.add_argument('--patch_len', type=int, default=24, help='patch length for short term forecasting')
    # DLinear
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate/channel (individually).')
    # Autoformer 
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    # Common parameters
    parser.add_argument('--enc_in', type=int, default=25, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=25, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=25, help='output size')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if (args.data == 'RP'):
        args.seq_len = 24
    else:
        args.seq_len = 60
    print(args)

    if args.task == 'classification':
        Exp = Exp_Main_c
    elif args.task == 'regression':
        Exp = Exp_Main_r
    seeds = [608, 1247, 3224, 2025, 5]
    if args.task == 'classification':
        avg = []
    else:
        avg_mse, avg_mae, avg_mape = 0, 0, 0
    for seed in seeds:
        # fix random seed
        set_random_seeds(seed)
        args.seed = seed
        print(f'\n-------{seed}-------')
        
        path_if = os.path.join(args.checkpoints, args.model_id, str(seed), 'intermediate_files')
        if not os.path.exists(path_if):
            os.makedirs(path_if)
        
        # save model hyperparameters
        save_hp_to_yaml(args)

        # data preprocessing
        path_idx_splits = os.path.join(path_if, f'idx_splits_{args.data}_{args.seq_len}_{args.pred_len}.pkl')
        if not os.path.exists(path_idx_splits):
            if args.data == 'RP':
                RP_preprocessing(args.seq_len, args.pred_len, [0.8,0.1,0.1], os.path.join(args.checkpoints, args.model_id, str(seed)), relax_factor=args.relax_factor)
                RP_scaler(os.path.join(args.checkpoints, args.model_id, str(seed)), args.seq_len, args.pred_len,)
            else:
                VP_preprocessing(args.seq_len, args.pred_len, [0.8,0.1,0.1], os.path.join(args.checkpoints, args.model_id, str(seed)), args.fps, relax_factor=args.relax_factor)
                VP_scaler(os.path.join(args.checkpoints, args.model_id, str(seed)), args.seq_len, args.pred_len, args.fps)

        exp = Exp(args)  # set experiments

        if args.is_training:
            print('<<<<<<< training >>>>>>>')
            exp.train(seed)

            print('>>>>>>> testing <<<<<<<')
            if args.task == 'classification':
                avg.append(exp.test(seed))
            else:
                avg = exp.test(seed)
                avg_mse += avg[0]
                avg_mae += avg[1]
                avg_mape += avg[2]

            # if args.do_cdv:
            #     # cross data validation
            #     RP_preprocessing(args.seq_len, args.pred_len, [0.8,0.1,0.1], os.path.join(args.checkpoints, args.model_id, str(seed)), CDV=True)
            #     RP_scaler(os.path.join(args.checkpoints, args.model_id, str(seed)), args.seq_len, args.label_len, args.pred_len,)

            #     print('>>>>>>> predicting <<<<<<<')
            #     exp.predict(seed, True)

            torch.cuda.empty_cache()
        else:
            print('>>>>>>> testing <<<<<<<')
            exp.test(seed, test=1)

            torch.cuda.empty_cache()
    
    if args.task == 'classification':
        np.save(os.path.join(args.checkpoints, args.model_id) + '/acc_list.npy', np.array(avg))
        f = open(f'{args.model_id}.txt', 'a')
        f.write(f'Average ACC: {np.array(avg).mean()}, STD: {np.array(avg).std()}')
        f.write('\n')
        f.write('\n')
        f.close()
    else:
        f = open(f'{args.model_id}.txt', 'a')
        f.write('Average MSE, MAE, MAPE: {}, {}, {}'.format(avg_mse/len(seeds), avg_mae/len(seeds), avg_mape/len(seeds)))
        f.write('\n')
        f.write('\n')
        f.close()
    

if __name__ == "__main__":
    main()
