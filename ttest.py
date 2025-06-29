from scipy import stats
import numpy as np


for ds in ['VP']:
    if ds == 'RP':
        a = np.load(f'{ds}_pl12_PP_qttest.npy')
        b = np.load(f'TimesNet_{ds}_pl12_topk1_numk2_lr0.0001_dmodel64_dff256_pl6_qttest.npy')
        c = np.load(f'Flowformer_{ds}_pl12_qttest.npy')

    else:    
        a = np.load(f'{ds}_pl30_PVP_qttest.npy')
        b = np.load(f'TimesNet_VP_topk2_numk4_lr0.01_dmodel256_dff64_pl30_qttest.npy')
        c = np.load(f'Flowformer_{ds}_pl30_qttest.npy')

    print(f'vs. TimesNet:')
    if stats.levene(a,b)[1] <= 0.05:
        print(stats.ttest_ind(a,b, equal_var=False))
    else:
        print(stats.ttest_ind(a,b))

    print(f'vs. Flowformer:')
    if stats.levene(a,c)[1] <= 0.05:
        print(stats.ttest_ind(a,c, equal_var=False))
    else:
        print(stats.ttest_ind(a,c))