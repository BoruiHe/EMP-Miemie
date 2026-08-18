import numpy as np
import torch as tr
import pickle as pk
import matplotlib.pyplot as pt
import torch.nn.functional as F


def majority_voting_ensemble(retrain=False):
    seeds = [608, 1247, 3224, 2025, 5]
    if retrain:
        model_name = retrain
    else:
        model_name = 'EEP'
    
    # VP (SimP)
    for ds in ['RP', 'VP']:
        if ds == 'VP':
            pred_spans = [15, 30, 45, 60]
        else:
            pred_spans = [6, 12, 18, 24]
        for ps in pred_spans:
            Ego = f'ensemble/{ds}/{ds}_Egofalls_pl{ps}'

            if ds == 'VP':
                if retrain == 'Flowformer':
                    Flo = f'ensemble/VP/Flowformer_VP_PT_pl{ps}'
                else:
                    Flo = f'ensemble/VP/Flowformer_VP_lr0.0001_dmodel512_dff512_pl{ps}'
                if retrain == 'TimesNet':
                    Tim = f'ensemble/VP/TimesNet_VP_PT_pl{ps}'
                else:
                    Tim = f'ensemble/VP/TimesNet_VP_topk2_numk4_lr0.01_dmodel256_dff64_pl{ps}'
            else:
                if retrain == 'Flowformer':
                    Flo = f'ensemble/RP/Flowformer_RP_PT_pl{ps}'
                else:
                    Flo = f'ensemble/RP/Flowformer_RP_lr0.0001_dmodel1024_dff256_pl{ps}'
                if retrain == 'TimesNet':
                    Tim = f'ensemble/RP/TimesNet_RP_PT_pl{ps}'
                else:
                    Tim = f'ensemble/RP/TimesNet_RP_topk1_numk2_lr0.0001_dmodel256_dff1024_pl{ps}'

            acc = 0
            for seed in seeds:
                A = np.load(Ego+f'/{seed}/true.npy')
                a = F.one_hot(tr.from_numpy(A), 2).numpy()
                B = np.load(Flo+f'/{seed}/true.npy')
                C = np.load(Tim+f'/{seed}/true.npy')
                assert (a==B).all(), 'Ground truths are not the same.'
                assert (C==B).all(), 'Ground truths are not the same.'
                true = A

                predEgo = np.load(Ego+f'/{seed}/pred.npy')
                predFlo = np.load(Flo+f'/{seed}/pred.npy')
                predTim = np.load(Tim+f'/{seed}/pred.npy')

                pred = np.array([predEgo, np.argmax(predFlo, axis=1), np.argmax(predTim, axis=1)])

                # majority vote
                pred = pred.sum(axis=0)
                pred = np.where(pred>=2, 1, 0)
                acc_ = (pred == true).sum() / pred.size
                acc += acc_
                
            print(f'{ds}, {model_name}, ensemble Acc at prediction span {ps}: {100*acc/len(seeds)}')

def representative_frames():
    with open('video_example/bufs_01_5.pkl', 'rb') as f:
        (buffers, _, _, _) = pk.load(f, encoding='latin1')
    RP_frames = buffers['images'][-1]
    RP_frames = RP_frames[:,:,::-1]

    VP_frames = np.load('video_example/f_0428_turnleft.npz')['imgs']
    VP_frames = VP_frames[1]

    pt.rc('text', usetex=True)
    pt.rc('font', family='serif')
    fig, axes = pt.subplots(1,2, figsize=(15*128/300, 960/300), dpi=300)
    axes[0].imshow(RP_frames)
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[0].set_title('A representative frame from RP', fontsize=15)

    axes[1].imshow(VP_frames)
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    axes[1].set_title('A representative frame from SimP', fontsize=15)

    pt.tight_layout()
    # pt.show(block=True)
    pt.savefig('plots/representative_frames.pdf', dpi=300)

# majority_voting_ensemble('Flowformer')
representative_frames()
pass