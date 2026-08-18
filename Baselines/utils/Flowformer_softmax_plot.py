import numpy as np
import matplotlib.pyplot as pt
import torch as tr
import os


fig, axes = pt.subplots(5,3, figsize=(19.2, 10.8))
bar_colors = ['blue', 'blue', 'blue', 'blue', 'blue', 'red']
for idx, lr in enumerate([128,256,512,1024,2048]):
    path = f'checkpoints/Flowformer_VP_lr0.001_dmodel512_dff{lr}'

    for i, seed in enumerate([608, 1247, 3224]):
        pred = np.load(os.path.join(path, str(seed), 'pred.npy'))
        true = np.load(os.path.join(path, str(seed), 'true.npy'))   

        stats = {
            '90~100': 0,
            '80~90': 0,
            '70~80': 0,
            '60~70': 0,
            '50~60': 0,
            '<50': 0
        }

        data = tr.softmax(tr.from_numpy(pred), dim=1)[:,0]
        temp = (data >= 0.9)
        stats['90~100'] += temp.sum()
        data = data[~temp]

        temp = (data >= 0.8)
        stats['80~90'] += temp.sum()
        data = data[~temp]

        temp = (data >= 0.7)
        stats['70~80'] += temp.sum()
        data = data[~temp]

        temp = (data >= 0.6)
        stats['60~70'] += temp.sum()
        data = data[~temp]

        temp = (data >= 0.5)
        stats['50~60'] += temp.sum()
        stats['<50'] += (~temp).sum()

        axes[idx, i].bar(list(stats.keys()), list(stats.values()), label=list(stats.keys()), color=bar_colors)
        axes[idx, i].set_title(path)

pt.tight_layout()
pt.savefig('plots/Flowformer_softmax.png', dpi=300)
# pt.show()
pass