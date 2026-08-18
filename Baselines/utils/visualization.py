import sys
import os   
sys.path.insert(0, os.getcwd())
from utils.tools import load_scaler
from utils.metrics import comprehensive_metric
import matplotlib.pyplot as pt
import warnings
import numpy as np
import os


warnings.filterwarnings("ignore")

def visual(true, preds, name):
    """
    Results (prediciton) visualization
    """
    # if i % 20 == 0:
    # input = batch_x.detach().cpu().numpy()
    # gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
    # pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
    fig, axes = pt.subplots(5,5, figsize=(19.2, 10.8))
    for i in range(25):
        axes[i//5,i%5].plot(true, label='GroundTruth', linewidth=2)
        axes[i//5,i%5].plot(preds, label='Prediction', linewidth=2)
        
    pt.legend()
    pt.savefig(name, bbox_inches='tight', dpi=300)

def Autoformer_finetune_plot():
    '''
    #A80E29 #BF3147 #E93A32 #D55D69 #F2A9BB 
    '''
    seeds = [608, 1247, 3224]
    fig, axes = pt.subplots(1,2,figsize=(19.2, 10.8))

    MAPE, MSE = [], []
    lrs = [0.01, 0.005, 0.001, '0.0001', '0.00001']
    for lr in lrs:
        suffix = f'lr{lr}'
        mape, mse = 0, 0
        for seed in seeds:
            p = f'checkpoints/Autoformer/Autoformer_VP_{suffix}'
            scaler_act, scaler_fut = load_scaler(p, 'VP', 60, 30)
                
            # path = os.path.join(p, str(seed), 'results', 'metrics.npy')
            # metrics = np.load(path)
            
            path_pred = os.path.join(p, str(seed), 'results', 'pred.npy')
            path_true = os.path.join(p, str(seed), 'results', 'true.npy')
            pred = np.load(path_pred)
            true = np.load(path_true)
            metrics = comprehensive_metric(pred, true, scaler_act, scaler_fut)

            mape += metrics[0].item()
            mse += metrics[1].item()
            
        mape /= len(seeds)
        mse /= len(seeds)
        MAPE.append(mape)
        MSE.append(mse)
    
    axes[0].plot(range(len(lrs)), MAPE, 's-', color='#A80E29', label='MAPE(learning rate)')
    axes[1].plot(range(len(lrs)), MSE, 's-', color='#A80E29', label='MSE(learning rate)')
    location = [(-0.1, MAPE[0]+0.015, MSE[0]+0.035), (0.7, MAPE[1]+0.015, MSE[1]+0.05), (1.9, MAPE[2]+0.05, MSE[2]+0.1), (2.8, MAPE[3]+0.015, MSE[3]+0.05), (3.9, MAPE[4]+0.015, MSE[4]+0.05)]
    for i, lr in zip(location, lrs):
        lr = '{:.0e}'.format(float(lr))
        axes[0].text(i[0], i[1], lr)
        axes[1].text(i[0], i[2], lr)
    print(min(MAPE), 'lr',lrs[MAPE.index(min(MAPE))])
   
    MAPE, MSE = [], []
    # d_models = [16, 32, 64, 128, 256, 512, 1024]
    d_models = [16, 64, 256, 512, 1024]
    for d_model in d_models:
        if d_model == 512:
            suffix = 'lr0.001'
        else:
            suffix = f'dmodel{d_model}'
        p = f'checkpoints/Autoformer/Autoformer_VP_{suffix}'
        mape, mse = 0, 0
        for seed in seeds:
            scaler_act, scaler_fut = load_scaler(p, 'VP', 60, 30)
                
            # path = os.path.join(p, str(seed), 'results', 'metrics.npy')
            # metrics = np.load(path)

            path_pred = os.path.join(p, str(seed), 'results', 'pred.npy')
            path_true = os.path.join(p, str(seed), 'results', 'true.npy')
            pred = np.load(path_pred)
            true = np.load(path_true)
            metrics = comprehensive_metric(pred, true, scaler_act, scaler_fut)

            mape += metrics[0].item()
            mse += metrics[1].item()

        mape /= len(seeds)
        mse /= len(seeds)
        MAPE.append(mape)
        MSE.append(mse)
    
    axes[0].plot(range(len(d_models)), MAPE, 'D-', color='#E93A32', label='MAPE(d_model)')
    axes[1].plot(range(len(d_models)), MSE, 'D-', color='#E93A32', label='MSE(d_model)')
    location = [(-0.1, MAPE[0]+0.015, MSE[0]+0.035), (0.9, MAPE[1]+0.015, MSE[1]+0.05), (2.1, MAPE[2]+0.02, MSE[2]), (2.8, MAPE[3]+0.015, MSE[3]+0.05), (3.9, MAPE[4]+0.015, MSE[4]+0.05)]
    for i, d_model in zip(location, d_models):
        axes[0].text(i[0], i[1], f'{d_model}')
        axes[1].text(i[0], i[2], f'{d_model}')
    print(min(MAPE), 'd_model',d_models[MAPE.index(min(MAPE))])
    
    MAPE, MSE = [], []
    d_ffs = [512, 1024, 2048, 4096, 8192]
    for d_ff in d_ffs:
        if d_ff == 2048:
            suffix = 'dmodel16'
        else:
            suffix = f'dff{d_ff}'
        p = f'checkpoints/Autoformer/Autoformer_VP_{suffix}'
        mape, mse = 0, 0
        for seed in seeds:
            scaler_act, scaler_fut = load_scaler(p, 'VP', 60, 30)
                
            # path = os.path.join(p, str(seed), 'results', 'metrics.npy')
            # metrics = np.load(path)

            path_pred = os.path.join(p, str(seed), 'results', 'pred.npy')
            path_true = os.path.join(p, str(seed), 'results', 'true.npy')
            pred = np.load(path_pred)
            true = np.load(path_true)
            metrics = comprehensive_metric(pred, true, scaler_act, scaler_fut)
            
            mape += metrics[0].item()
            mse += metrics[1].item()

        mape /= len(seeds)
        mse /= len(seeds)
        MAPE.append(mape)
        MSE.append(mse)
    
    axes[0].plot(range(len(d_ffs)), MAPE, 'o-', color='#F2A9BB', label='MAPE(d_ff)')
    axes[1].plot(range(len(d_ffs)), MSE, 'o-', color='#F2A9BB', label='MSE(d_ff)')
    for i, d_ff in zip(range(len(d_ffs)), d_ffs):
        axes[0].text(i-0.1, MAPE[i]-0.035, f'{d_ff}')
        axes[1].text(i-0.1, MSE[i]-0.15, f'{d_ff}')
    print(min(MAPE), 'd_ff',d_ffs[MAPE.index(min(MAPE))])
    
    axes[0].legend()
    axes[1].legend()
    axes[0].xaxis.set_visible(False)
    axes[1].xaxis.set_visible(False)
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0,3))
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0,2))
    pt.tight_layout()
    # pt.savefig('plots/Autoformer_finetune_ch.png', dpi=300)
    pt.show()

def DLinear_finetune_plot():
    '''
    #A80E29 #BF3147 #E93A32 #D55D69 #F2A9BB 
    '''
    seeds = [608, 1247, 3224]
    fig, axes = pt.subplots(1,2,figsize=(19.2, 10.8))

    MAPE, MSE = [], []
    lrs = [0.01, 0.005, 0.001, 0.0005, '0.0001']
    for lr in lrs:
        suffix = f'ind_lr{lr}'
        mape, mse = 0, 0
        for seed in seeds:
            p = f'checkpoints/DLinear_VP_{suffix}/{seed}'
            scaler_act, scaler_fut = load_scaler(p, 'VP', 60, 30)
                
            # path = os.path.join(p, str(seed), 'results', 'metrics.npy')
            # metrics = np.load(path)

            path_pred = os.path.join(p, 'pred.npy')
            path_true = os.path.join(p, 'true.npy')
            pred = np.load(path_pred)
            true = np.load(path_true)
            metrics = comprehensive_metric(pred, true, scaler_act, scaler_fut)

            mape += metrics[0].item()
            mse += metrics[1].item()
            
        mape /= len(seeds)
        mse /= len(seeds)
        MAPE.append(mape)
        MSE.append(mse)
    
    axes[0].plot(range(len(lrs)), MAPE, 's-', color='#A80E29', label='MAPE(Ind+learning rate)')
    axes[1].plot(range(len(lrs)), MSE, 's-', color='#A80E29', label='MSE(Ind+learning rate)')
    location = [(-0.1, MAPE[0]-0.0002, MSE[0]-0.00002), (0.8, MAPE[1]+0.00004, MSE[1]+0.000005), (2, MAPE[2]-0.00015, MSE[2]-0.000015), (2.8, MAPE[3]+0.0001, MSE[3]+0.00001), (3.9, MAPE[4]+0.0001, MSE[4]+0.00001)]
    for i, lr in zip(location, lrs):
        lr = '{:.0e}'.format(float(lr))
        axes[0].text(i[0], i[1], lr)
        axes[1].text(i[0], i[2], lr)
    print(min(MAPE), 'lr',lrs[MAPE.index(min(MAPE))])

    MAPE, MSE = [], []
    lrs = [0.01, 0.005, 0.001, 0.0005, '0.0001']
    for lr in lrs:
        suffix = f'd_lr{lr}'
        p = f'checkpoints/DLinear_VP_{suffix}'
        mape, mse = 0, 0
        for seed in seeds:
            scaler_act, scaler_fut = load_scaler(p, 'VP', 60, 30)
                
            # path = os.path.join(p, str(seed), 'results', 'metrics.npy')
            # metrics = np.load(path)

            path_pred = os.path.join(p, 'pred.npy')
            path_true = os.path.join(p, 'true.npy')
            pred = np.load(path_pred)
            true = np.load(path_true)
            metrics = comprehensive_metric(pred, true, scaler_act, scaler_fut)

            mape += metrics[0].item()
            mse += metrics[1].item()
            
        mape /= len(seeds)
        mse /= len(seeds)
        MAPE.append(mape)
        MSE.append(mse)
    
    axes[0].plot(range(len(lrs)), MAPE, 'o-', color='#E93A32', label='MAPE(D+learning rate)')
    axes[1].plot(range(len(lrs)), MSE, 'o-', color='#E93A32', label='MSE(D+learning rate)')
    location = [(-0.1, MAPE[0]-0.00015, MSE[0]-0.000015), (0.8, MAPE[1]+0.0001, MSE[1]+0.00001), (1.9, MAPE[2]-0.00015, MSE[2]-0.000015), (2.9, MAPE[3]-0.00015, MSE[3]-0.000015), (3.9, MAPE[4]+0.0001, MSE[4]+0.00001)]

    for i, lr in zip(location, lrs):
        lr = '{:.0e}'.format(float(lr))
        axes[0].text(i[0], i[1], lr)
        axes[1].text(i[0], i[2], lr)
    print(min(MAPE), 'lr',lrs[MAPE.index(min(MAPE))])
    
    axes[0].legend()
    axes[1].legend()
    axes[0].xaxis.set_visible(False)
    axes[1].xaxis.set_visible(False)
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0,2))
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0,2))
    pt.tight_layout()
    # pt.savefig('plots/DLinear_finetune_ch.png', dpi=300)
    # pt.show()

def TiDE_finetune_plot():
    '''
    #A80E29 #BF3147 #E93A32 #D55D69 #F2A9BB 
    '''
    seeds = [608, 1247, 3224]
    fig, axes = pt.subplots(1,2,figsize=(19.2, 10.8))

    MAPE, MSE = [], []
    lrs = [0.5, 0.1, 0.05, 0.01, 0.001]
    for lr in lrs:
        suffix = f'lr{lr}'
        mape, mse = 0, 0
        for seed in seeds:
            p = f'checkpoints/TiDE_VP_{suffix}/{seed}'
            scaler_act, scaler_fut = load_scaler(p, 'VP', 60, 30)
                
            # path = os.path.join(p, str(seed), 'results', 'metrics.npy')
            # metrics = np.load(path)

            path_pred = os.path.join(p, 'pred.npy')
            path_true = os.path.join(p, 'true.npy')
            pred = np.load(path_pred)
            true = np.load(path_true)
            metrics = comprehensive_metric(pred, true, scaler_act, scaler_fut)

            mape += metrics[0].item()
            mse += metrics[1].item()
            
        mape /= len(seeds)
        mse /= len(seeds)
        MAPE.append(mape)
        MSE.append(mse)
    
    axes[0].plot(range(len(lrs)), MAPE, 's-', color='#A80E29', label='MAPE(learning rate)')
    axes[1].plot(range(len(lrs)), MSE, 's-', color='#A80E29', label='MSE(learning rate)')
    location = [(-0.1, MAPE[0]+0.00005, MSE[0]-0.000015), (1, MAPE[1]-0.00008, MSE[1]-0.00001), (1.8, MAPE[2]+0.00005, MSE[2]+0.000005), (2.8, MAPE[3]+0.00005, MSE[3]-0.00001), (3.9, MAPE[4]+0.00005, MSE[4]+0.000005)]
    for i, lr in zip(location, lrs):
        lr = '{:.0e}'.format(float(lr))
        axes[0].text(i[0], i[1], lr)
        axes[1].text(i[0], i[2], lr)
    print(min(MAPE), 'lr',lrs[MAPE.index(min(MAPE))])

    MAPE, MSE = [], []
    d_models = [64, 128, 256, 512, 1024]
    for d_model in d_models:
        if d_model == 512:
            suffix = 'lr0.1'
        else:
            suffix = f'dmodel{d_model}'
        p = f'checkpoints/TiDE_VP_{suffix}'
        mape, mse = 0, 0
        for seed in seeds:
            scaler_act, scaler_fut = load_scaler(p, 'VP', 60, 30)
                
            # path = os.path.join(p, str(seed), 'results', 'metrics.npy')
            # metrics = np.load(path)

            path_pred = os.path.join(p, 'pred.npy')
            path_true = os.path.join(p, 'true.npy')
            pred = np.load(path_pred)
            true = np.load(path_true)
            metrics = comprehensive_metric(pred, true, scaler_act, scaler_fut)

            mape += metrics[0].item()
            mse += metrics[1].item()
            
        mape /= len(seeds)
        mse /= len(seeds)
        MAPE.append(mape)
        MSE.append(mse)
    
    axes[0].plot(range(len(d_models)), MAPE, 'D-', color='#E93A32', label='MAPE(d_model)')
    axes[1].plot(range(len(d_models)), MSE, 'D-', color='#E93A32', label='MSE(d_model)')
    location = [(0, MAPE[0]+0.000025, MSE[0]+0.000005), (1, MAPE[1]-0.00005, MSE[1]-0.000008), (2, MAPE[2]-0.00005, MSE[2]-0.000008), (2.8, MAPE[3]+0.000025, MSE[3]+0.000005), (3.9, MAPE[4]+0.000025, MSE[4]+0.000005)]
    for i, d_model in zip(location, d_models):
        axes[0].text(i[0], i[1], d_model)
        axes[1].text(i[0], i[2], d_model)
    print(min(MAPE), 'd_model',d_models[MAPE.index(min(MAPE))])

    MAPE, MSE = [], []
    d_ffs = [64, 128, 512, 1024, 2048]
    for d_ff in d_ffs:
        if d_ff == 2048:
            suffix = 'dmodel128'
        else:
            suffix = f'dff{d_ff}'
        mape, mse = 0, 0
        for seed in seeds:
            p = f'checkpoints/TiDE_VP_{suffix}'
                
            # path = os.path.join(p, str(seed), 'results', 'metrics.npy')
            # metrics = np.load(path)

            path_pred = os.path.join(p, 'pred.npy')
            path_true = os.path.join(p, 'true.npy')
            pred = np.load(path_pred)
            true = np.load(path_true)
            metrics = comprehensive_metric(pred, true, scaler_act, scaler_fut)

            mape += metrics[0].item()
            mse += metrics[1].item()
            
        mape /= len(seeds)
        mse /= len(seeds)
        MAPE.append(mape)
        MSE.append(mse)
    
    axes[0].plot(range(len(d_ffs)), MAPE, 'o-', color='#F2A9BB', label='MAPE(d_ff)')
    axes[1].plot(range(len(d_ffs)), MSE, 'o-', color='#F2A9BB', label='MSE(d_ff)')
    location = [(0, MAPE[0]+0.000025, MSE[0]+0.000005), (1, MAPE[1]-0.00005, MSE[1]-0.000008), (1.9, MAPE[2]-0.00008, MSE[2]-0.000012), (2.95, MAPE[3]+0.000025, MSE[3]+0.00001), (3.9, MAPE[4]+0.00004, MSE[4]+0.000013)]
    for i, d_ff in zip(location, d_ffs):
        axes[0].text(i[0], i[1], d_ff)
        axes[1].text(i[0], i[2], d_ff)
    print(min(MAPE), 'd_ff',d_ffs[MAPE.index(min(MAPE))])
    
    axes[0].legend()
    axes[1].legend()
    axes[0].xaxis.set_visible(False)
    axes[1].xaxis.set_visible(False)
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0,2))
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0,2))
    pt.tight_layout()
    # pt.savefig('plots/TiDE_finetune_ch.png', dpi=300)
    # pt.show()

def TimeXer_finetune_plot():
    '''
    #A80E29 #BF3147 #E93A32 #D55D69 #F2A9BB 
    '''
    seeds = [608, 1247, 3224]
    fig, axes = pt.subplots(1,2,figsize=(19.2, 10.8))

    MAPE, MSE = [], []
    lrs = [0.001, '0.0005', '0.0001', '0.00005', '0.00001']
    for lr in lrs:
        suffix = f'lr{lr}_dmodel512_dff512'
        mape, mse = 0, 0
        for seed in seeds:
            p = f'checkpoints/TimeXer/TimeXer_VP_{suffix}/{seed}'
            scaler_act, scaler_fut = load_scaler(p, 'VP', 60, 30)
            
            # path = os.path.join(p, str(seed), 'results', 'metrics.npy')
            # metrics = np.load(path)

            path_pred = os.path.join(p, 'pred.npy')
            path_true = os.path.join(p, 'true.npy')
            pred = np.load(path_pred)
            true = np.load(path_true)
            metrics = comprehensive_metric(pred, true, scaler_act, scaler_fut)

            mape += metrics[0].item()
            mse += metrics[1].item()
            
        mape /= len(seeds)
        mse /= len(seeds)
        MAPE.append(mape)
        MSE.append(mse)
    
    axes[0].plot(range(len(lrs)), MAPE, 's-', color='#A80E29', label='MAPE(learning rate)')
    axes[1].plot(range(len(lrs)), MSE, 's-', color='#A80E29', label='MSE(learning rate)')
    location = [(-0.1, MAPE[0]+0.0001, MSE[0]+0.00001), (1, MAPE[1]+0.0002, MSE[1]+0.00002), (1.8, MAPE[2]+0.00005, MSE[2]+0.000005), (2.8, MAPE[3]+0.00005, MSE[3]+0.000005), (3.9, MAPE[4]+0.00005, MSE[4]+0.000005)]
    for i, lr in zip(location, lrs):
        lr = '{:.0e}'.format(float(lr))
        axes[0].text(i[0], i[1], lr)
        axes[1].text(i[0], i[2], lr)
    print(min(MAPE), 'lr',lrs[MAPE.index(min(MAPE))])

    MAPE, MSE = [], []
    d_models = [128, 256, 512, 1024, 2048]
    for d_model in d_models:
        suffix = f'lr0.0001_dmodel{d_model}_dff512'
        mape, mse = 0, 0
        for seed in seeds:             
            p = f'checkpoints/TimeXer/TimeXer_VP_{suffix}/{seed}'
            scaler_act, scaler_fut = load_scaler(p, 'VP', 60, 30)

            # path = os.path.join(p, str(seed), 'results', 'metrics.npy')
            # metrics = np.load(path)

            path_pred = os.path.join(p, 'pred.npy')
            path_true = os.path.join(p, 'true.npy')
            pred = np.load(path_pred)
            true = np.load(path_true)
            metrics = comprehensive_metric(pred, true, scaler_act, scaler_fut)

            mape += metrics[0].item()
            mse += metrics[1].item()
            
        mape /= len(seeds)
        mse /= len(seeds)
        MAPE.append(mape)
        MSE.append(mse)
    
    axes[0].plot(range(len(d_models)), MAPE, 'D-', color='#E93A32', label='MAPE(d_model)')
    axes[1].plot(range(len(d_models)), MSE, 'D-', color='#E93A32', label='MSE(d_model)')
    location = [(-0.1, MAPE[0]-0.0002, MSE[0]-0.00003), (1, MAPE[1]-0.0002, MSE[1]-0.00002), (2, MAPE[2]-0.00015, MSE[2]-0.00002), (2.8, MAPE[3]+0.000025, MSE[3]+0.000005), (3.9, MAPE[4]+0.000025, MSE[4]+0.000005)]
    for i, d_model in zip(location, d_models):
        axes[0].text(i[0], i[1], d_model)
        axes[1].text(i[0], i[2], d_model)
    print(min(MAPE), 'd_model',d_models[MAPE.index(min(MAPE))])

    MAPE, MSE = [], []
    d_ffs = [128, 256, 512, 1024, 2048]
    for d_ff in d_ffs:
        suffix = f'lr0.0001_dmodel256_dff{d_ff}'
        mape, mse = 0, 0
        for seed in seeds:
            p = f'checkpoints/TimeXer/TimeXer_VP_{suffix}/{seed}'
            scaler_act, scaler_fut = load_scaler(p, 'VP', 60, 30)

            # path = os.path.join(p, str(seed), 'results', 'metrics.npy')
            # metrics = np.load(path)

            path_pred = os.path.join(p, 'pred.npy')
            path_true = os.path.join(p, 'true.npy')
            pred = np.load(path_pred)
            true = np.load(path_true)
            metrics = comprehensive_metric(pred, true, scaler_act, scaler_fut)

            mape += metrics[0].item()
            mse += metrics[1].item()
            
        mape /= len(seeds)
        mse /= len(seeds)
        MAPE.append(mape)
        MSE.append(mse)
    
    axes[0].plot(range(len(d_ffs)), MAPE, 'o-', color='#F2A9BB', label='MAPE(d_ff)')
    axes[1].plot(range(len(d_ffs)), MSE, 'o-', color='#F2A9BB', label='MSE(d_ff)')
    location = [(0, MAPE[0]-1, MSE[0]+0.000015), (1, MAPE[1]-0.0002, MSE[1]-0.00004), (1.9, MAPE[2]-0.0002, MSE[2]-0.00003), (2.8, MAPE[3]+0.000025, MSE[3]+0.000005), (3.9, MAPE[4]+0.00015, MSE[4]+0.00002)]
    for i, d_ff in zip(location, d_ffs):
        axes[0].text(i[0], i[1], d_ff)
        axes[1].text(i[0], i[2], d_ff)
    print(min(MAPE), 'd_ff',d_ffs[MAPE.index(min(MAPE))])
    
    axes[0].legend()
    axes[1].legend()
    axes[0].xaxis.set_visible(False)
    axes[1].xaxis.set_visible(False)
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0,2))
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0,2))
    pt.tight_layout()
    # pt.savefig('plots/TimeXer_finetune_ch.png', dpi=300)
    pt.show()

def TimeMixer_finetune_plot():
    '''
    #A80E29 #BF3147 #E93A32 #D55D69 #F2A9BB 
    '''
    seeds = [608, 1247, 3224]
    fig, axes = pt.subplots(1,2,figsize=(19.2, 10.8))

    MAPE, MSE = [], []
    lrs = ['0.00001', '0.000005', '0.000001', '0.0000005', '0.0000001']
    for lr in lrs:
        suffix = f'lr{lr}_dmodel512_dff512'
        mape, mse = 0, 0
        for seed in seeds:
            p = f'checkpoints/TimeMixer/TimeMixer_VP_{suffix}/{seed}'
            scaler_act, scaler_fut = load_scaler(p, 'VP', 60, 30)
                
            # path = os.path.join(p, str(seed), 'results', 'metrics.npy')
            # metrics = np.load(path)

            path_pred = os.path.join(p, 'pred.npy')
            path_true = os.path.join(p, 'true.npy')
            pred = np.load(path_pred)
            true = np.load(path_true)
            metrics = comprehensive_metric(pred, true, scaler_act, scaler_fut)

            mape += metrics[0].item()
            mse += metrics[1].item()
            
        mape /= len(seeds)
        mse /= len(seeds)
        MAPE.append(mape)
        MSE.append(mse)
    
    axes[0].plot(range(len(lrs)), MAPE, 's-', color='#A80E29', label='MAPE(learning rate)')
    axes[1].plot(range(len(lrs)), MSE, 's-', color='#A80E29', label='MSE(learning rate)')
    location = [(-0.1, MAPE[0]+0.0001, MSE[0]+0.00001), (1, MAPE[1]+0.0002, MSE[1]+0.00002), (1.8, MAPE[2]+0.00005, MSE[2]+0.000005), (2.8, MAPE[3]+0.00005, MSE[3]+0.000005), (3.9, MAPE[4]+0.00005, MSE[4]+0.000005)]
    for i, lr in zip(location, lrs):
        lr = '{:.0e}'.format(float(lr))
        axes[0].text(i[0], i[1], lr)
        axes[1].text(i[0], i[2], lr)
    print(min(MAPE), 'lr',lrs[MAPE.index(min(MAPE))])

    MAPE, MSE = [], []
    d_models = [128, 256, 512, 1024, 2048]
    for d_model in d_models:
        suffix = f'lr0.0000005_dmodel{d_model}_dff512'
        mape, mse = 0, 0
        for seed in seeds:
            p = f'checkpoints/TimeMixer/TimeMixer_VP_{suffix}/{seed}'
                
            # path = os.path.join(p, str(seed), 'results', 'metrics.npy')
            # metrics = np.load(path)

            path_pred = os.path.join(p, 'pred.npy')
            path_true = os.path.join(p, 'true.npy')
            pred = np.load(path_pred)
            true = np.load(path_true)
            metrics = comprehensive_metric(pred, true, scaler_act, scaler_fut)

            mape += metrics[0].item()
            mse += metrics[1].item()
            
        mape /= len(seeds)
        mse /= len(seeds)
        MAPE.append(mape)
        MSE.append(mse)
    
    axes[0].plot(range(len(d_models)), MAPE, 'D-', color='#E93A32', label='MAPE(d_model)')
    axes[1].plot(range(len(d_models)), MSE, 'D-', color='#E93A32', label='MSE(d_model)')
    location = [(-0.1, MAPE[0]-0.0002, MSE[0]-0.00003), (1, MAPE[1]-0.0002, MSE[1]-0.00002), (2, MAPE[2]-0.00015, MSE[2]-0.00002), (2.8, MAPE[3]+0.000025, MSE[3]+0.000005), (3.9, MAPE[4]+0.000025, MSE[4]+0.000005)]
    for i, d_model in zip(location, d_models):
        axes[0].text(i[0], i[1], d_model)
        axes[1].text(i[0], i[2], d_model)
    print(min(MAPE), 'd_model',d_models[MAPE.index(min(MAPE))])

    # MAPE, MSE = [], []
    # d_ffs = [8, 16, 32, 64, 128]
    # for d_ff in d_ffs:
    #     if d_ff == 32:
    #         suffix = 'dmodel16'
    #     else:
    #         suffix = f'dff{d_ff}'
    #     mape, mse = 0, 0
    #     for seed in seeds:
    #         p = f'checkpoints/TimeMixer/TimeMixer_VP_{suffix}'
                
    #         # path = os.path.join(p, str(seed), 'results', 'metrics.npy')
    #         # metrics = np.load(path)

    #         path_pred = os.path.join(p, 'pred.npy')
    #         path_true = os.path.join(p, 'true.npy')
    #         pred = np.load(path_pred)
    #         true = np.load(path_true)
    #         metrics = comprehensive_metric(pred, true, scaler_act, scaler_fut)

    #         mape += metrics[0].item()
    #         mse += metrics[1].item()
            
    #     mape /= len(seeds)
    #     mse /= len(seeds)
    #     MAPE.append(mape)
    #     MSE.append(mse)
    
    # axes[0].plot(range(len(d_ffs)), MAPE, 'o-', color='#F2A9BB', label='MAPE(d_ff)')
    # axes[1].plot(range(len(d_ffs)), MSE, 'o-', color='#F2A9BB', label='MSE(d_ff)')
    # location = [(0, MAPE[0]+0.00015, MSE[0]+0.000015), (1, MAPE[1]-0.0002, MSE[1]-0.00004), (1.9, MAPE[2]-0.0002, MSE[2]-0.00003), (2.8, MAPE[3]+0.000025, MSE[3]+0.000005), (3.9, MAPE[4]+0.00015, MSE[4]+0.00002)]
    # for i, d_ff in zip(location, d_ffs):
    #     axes[0].text(i[0], i[1], d_ff)
    #     axes[1].text(i[0], i[2], d_ff)
    # print(min(MAPE), 'd_ff',d_ffs[MAPE.index(min(MAPE))])
    
    axes[0].legend()
    axes[1].legend()
    axes[0].xaxis.set_visible(False)
    axes[1].xaxis.set_visible(False)
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0,2))
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0,2))
    pt.tight_layout()
    # pt.savefig('plots/TimeMixer_finetune_ch_.png', dpi=300)
    pt.show(block=True)

def MovingAvg_fineturn_plot():
    '''
    #A80E29 #BF3147 #E93A32 #D55D69 #F2A9BB 
    '''
    seeds = [608, 1247, 3224]
    fig, axes = pt.subplots(1,2,figsize=(19.2, 10.8))

    p = 'checkpoints/moving_avg_VP'
    mape, mse = 0, 0
    for seed in seeds:
        scaler_act, scaler_fut = load_scaler(p, 'VP', 60, 30)
        
        # path = os.path.join(p, str(seed), 'results', 'metrics.npy')
        # metrics = np.load(path)

        path_pred = os.path.join(p, 'pred.npy')
        path_true = os.path.join(p, 'true.npy')
        pred = np.load(path_pred)
        true = np.load(path_true)
        metrics = comprehensive_metric(pred, true, scaler_act, scaler_fut)

        mape += metrics[0].item()
        mse += metrics[1].item()
        
    mape /= len(seeds)
    mse /= len(seeds)
    
    axes[0].hlines(y=mape, xmin=0, xmax=5, linestyle='--', color='#A80E29', label='MAPE(learning rate)')
    axes[1].hlines(y=mse, xmin=0, xmax=5, linestyle='--', color='#A80E29', label='MSE(learning rate)')
    print(mape)

    # axes[0].set_ylim(0, 0.2)
    # axes[1].set_ylim(0, 0.2)
    axes[0].legend()
    axes[1].legend()
    axes[0].xaxis.set_visible(False)
    axes[1].xaxis.set_visible(False)
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0,2))
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0,2))
    pt.tight_layout()
    # pt.savefig('plots/MovingAvg_finetune.png', dpi=300)
    # pt.show()

def steps_periodic_plot():
    color_r = ['#A80E29', '#E93A32', '#F2A9BB']
    color_b = ['#32417A', '#7C94B7', '#C7DEE9']
    p = '/home/bhe100/Desktop/data/VP/fps30'
    files = os.listdir(p)
    for file in files:
        # if file[0] == 'f':
        fig, axes = pt.subplots(5,5, figsize=(19.2, 10.8))
        ajp = np.load(p+f'/{file}')['ajp']
        joint_index = list(range(17)) + list(range(18,26))
        ajp = ajp[:,joint_index]
        for i in range(25):
            for c in range(6):
                color = color_r[c//2] if c % 2 == 0 else color_b[c//2]
                axes[i//5,i%5].plot(range(0, 61), ajp[range(61*c, 61*(c+1)),i]*(180/np.pi), c=color, label=f'{c+1}')
        pt.tight_layout()
        pt.legend()
        pt.savefig(f'plots/steps_periodic_plot/{file[:6]}.png', dpi=300)
        # pt.show()
        pt.close()


if __name__ == "__main__":
    # steps_periodic_plot()
    # print('--------MovingAvg---------')
    # MovingAvg_fineturn_plot()
    # print('--------Autoformer---------')
    # Autoformer_finetune_plot()
    # print('--------Dlinear---------')
    # DLinear_finetune_plot()
    # print('--------TiDE---------')
    # TiDE_finetune_plot()
    # print('--------TimeXer---------')
    # TimeXer_finetune_plot()
    print('--------TimeMixer---------')
    TimeMixer_finetune_plot()
    # a = np.array([  6.1529056e-05, 2.2112727e-04, 4.7176384e-04, 7.5660797e-04, 1.0225066e-03,
    #                 1.2534405e-03, 1.4309181e-03, 1.5711450e-03, 1.7211763e-03, 1.8657354e-03,
    #                 1.9761694e-03, 2.0847304e-03, 2.1749348e-03, 2.2598889e-03, 2.3462533e-03,
    #                 2.4442873e-03, 2.5046847e-03, 2.5354123e-03, 2.5590041e-03, 2.5845824e-03,
    #                 2.5828716e-03, 2.5991532e-03, 2.6024785e-03, 2.6078841e-03, 2.6174104e-03,
    #                 2.6228137e-03, 2.6376073e-03, 2.6767962e-03, 2.7060371e-03, 2.7162200e-03])
    # print(np.mean(a))