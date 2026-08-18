from scipy import stats
import numpy as np


# for ds in ['VP']:
#     if ds == 'RP':
#         a = np.load(f'{ds}_pl12_PP_qttest.npy')
#         b = np.load(f'TimesNet_{ds}_pl12_topk1_numk2_lr0.0001_dmodel64_dff256_pl6_qttest.npy')
#         c = np.load(f'Flowformer_{ds}_pl12_qttest.npy')

#     else:    
#         a = np.load(f'{ds}_pl30_PVP_qttest.npy')
#         b = np.load(f'TimesNet_VP_topk2_numk4_lr0.01_dmodel256_dff64_pl30_qttest.npy')
#         c = np.load(f'Flowformer_{ds}_pl30_qttest.npy')

#     print(f'vs. TimesNet:')
#     if stats.levene(a,b)[1] <= 0.05:
#         print(stats.ttest_ind(a,b, equal_var=False))
#     else:
#         print(stats.ttest_ind(a,b))

#     print(f'vs. Flowformer:')
#     if stats.levene(a,c)[1] <= 0.05:
#         print(stats.ttest_ind(a,c, equal_var=False))
#     else:
#         print(stats.ttest_ind(a,c))

alg_PVP = {}
alg_PVP['11'] = np.array([0.965, 0.98, 0.97, 0.98, 0.97])
alg_PVP['22'] = np.array([0.955, 0.98, 0.97, 0.96, 0.96])
alg_PVP['33'] = np.array([0.965, 0.985, 0.94, 0.985, 0.96])

alg_Ego = {}
alg_Ego['11'] = np.load('Egofalls_pl11_acc_list.npy')
alg_Ego['22'] = np.load('Egofalls_pl22_acc_list.npy')
alg_Ego['33'] = np.load('Egofalls_pl33_acc_list.npy')

alg_TimesNet = {}
alg_TimesNet['11'] = np.array([0.755, 0.955, 0.885, 0.88, 0.805])
alg_TimesNet['22'] = np.array([0.82, 0.85, 0.94, 0.695, 0.885])
alg_TimesNet['33'] = np.array([0.7, 0.82, 0.94, 0.91, 0.915])

alg_Flowformer = {}
alg_Flowformer['11'] = np.array([0.945, 0.955, 0.935, 0.95, 0.93])
alg_Flowformer['22'] = np.array([0.945, 0.97, 0.94, 0.955, 0.935])
alg_Flowformer['33'] = np.array([0.965, 0.97, 0.975, 0.96, 0.93])

for target, txt in zip([alg_Ego, alg_TimesNet, alg_Flowformer], ['alg_Ego', 'alg_TimesNet', 'alg_Flowformer']):
    print(f'vs. {txt}:')

    for pl in ['11', '22', '33']:
        print(f'PS: {pl}')
        if stats.levene(alg_PVP[pl], target[pl])[1] <= 0.05:
            print(stats.ttest_ind(alg_PVP[pl], target[pl], equal_var=False).pvalue)
        else:
            print(stats.ttest_ind(alg_PVP[pl], target[pl]).pvalue)