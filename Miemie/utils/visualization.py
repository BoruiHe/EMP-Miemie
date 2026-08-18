import os
import numpy as np
import torch as tr
import pandas as pd
import pickle as pk
import matplotlib
import matplotlib.pyplot as pt
import statsmodels.tsa.api as smt
from matplotlib.ticker import FormatStrFormatter


def episodes_distribution():
    sigma_ls = [0.040, 0.035, 0.010, 0.005, 0.0]
    with open('data_generation/saved_labels.pkl', 'rb') as f:
        labels = pk.load(f)

    # drop 3 original trajectories. 23 --> 20
    labels = labels[:11] + labels[13:18] + labels[19:]

    fig, axes = pt.subplots(len(labels), len(sigma_ls), sharex=True, sharey=True, figsize=(12.80*2, 9.60*4))
    for n, label in enumerate(labels):
        # collect data
        path = f'videos/{label}'
        statistic = {}
        for sigma in sigma_ls:
            statistic[sigma] = []
        all_files = os.listdir(path)
        for file in all_files:
            head_z = np.load(os.path.join(path, file))['hc_z']
            file_std = file.split('_')[0]
            if float(file_std) in sigma_ls:
                statistic[float(file_std)].append(head_z)
            
        # visualization
        for sigma, color in zip(sigma_ls, ['slateblue', 'lightcoral', 'darkorange', 'limegreen', 'black']):
            ax = axes[n, sigma_ls.index(sigma)]
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.set_title(f'{label}, {sigma}')
            for head_z in statistic[sigma]:
                ax.plot(range(len(head_z)), head_z, color=color)
    pt.tight_layout()
    pt.savefig('plots/perturbation.png')

def episodes_z_coor():
    pt.rc('text', usetex=True)
    pt.rc('font', family='serif')
    fig, axes = pt.subplots(1, 2, sharex=True, sharey=True, figsize=(12.8, 9.6))
    files = os.listdir(os.path.join('data', 'episodes'))
    for n, file_name in enumerate(files):
            
        # visualization
        ax = axes[n]
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_title(f'{file_name[:-4]}', fontsize=20)
        file = np.load(os.path.join('data', 'episodes', file_name))
        head_z = file['pos'][:, 17, -1]
        stop_sign = (~file['labels']).sum()
        ax.plot(range(len(head_z)), head_z)
        ax.axvline(x=stop_sign-15, color='#BF3147', linestyle='--', label='The latest frame for prediction span 15', linewidth=2)
        ax.axvline(x=stop_sign-30, color='#E93A32', linestyle='--', label='The latest frame for prediction span 30', linewidth=2)
        ax.axvline(x=stop_sign-45, color='#D55D69', linestyle='--', label='The latest frame for prediction span 45', linewidth=2)
        ax.axvline(x=stop_sign-60, color='#F2A9BB', linestyle='--', label='The latest frame for prediction span 60', linewidth=2)
        ax.legend(fontsize=15)
        ax.set_xlabel('Timesteps', fontsize=20, loc='center')
        ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400], [0, 50, 100, 150, 200, 250, 300, 350, 400], fontsize=20)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)    

    axes[0].set_ylabel(r'Head $z$ coordinate', fontsize=20, loc='center')

    pt.tight_layout()
    # pt.show(block=True)
    pt.savefig('plots/epi_z_coor.pdf', dpi=300)

def destinations_distribution():
    if os.name =='nt':
        pt.rc('text', usetex=True)
        pt.rc('font', family='serif')
        with open('data/destination.pkl', 'rb') as f:
            destination = pk.load(f)
        f_dest, n_dest = destination['f_dest'], destination['n_dest']
        fig, ax = pt.subplots(1, 1, figsize=(12.80, 9.60)) 
        for c, coorF, coorN in zip(range(len(f_dest)), f_dest, n_dest):
            ax.scatter(coorF[0], coorF[1], c='crimson', marker='X', s=60, label='Falling' if c == 0 else '')
            ax.scatter(coorN[0], coorN[1], c='mediumspringgreen', marker='o', s=60, label='Non-fall' if c == 0 else '')
        ax.spines.left.set_position('zero')
        ax.spines.bottom.set_position('zero')
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_xticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
        ax.set_yticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8], [-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8], fontsize=20)
        # ax.set_title('Destination distribution', color='r', fontsize=30)
        ax.legend(fontsize=20)
        ax.grid(True, which='both')
        # pt.show(block=True)
        pt.savefig('plots/dest_distribution.pdf', dpi=300)

        print(f'f: {len(f_dest)}, n: {len(n_dest)}.')
    else:
        path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'VP', 'fps30')
        f_dest, n_dest = [], []
        for file_name in os.listdir(path):
            if file_name[0] == 'n':
                n_dest.append(np.load(os.path.join(path, file_name))['pos'][-1,17,:-1])
            elif file_name[0] == 'f':
                f_dest.append(np.load(os.path.join(path, file_name))['pos'][-1,17,:-1])
        destination = {
            'n_dest': n_dest,
            'f_dest': f_dest
        }
        with open('destination.pkl', 'wb') as f:
            pk.dump(destination, f)

def autocorrelation():
    pt.rcParams['text.usetex'] = True
    name_list = ['ETTh1.csv', 'ETTm1.csv', 'PEMS03.npz', 'Weather.csv', 'ILI.csv', 'Traffic.csv', 'Electricity.csv', 'Exchange_rate.csv', 
                 '2025_03_12_stdev0.0/bufs_02_6.pkl', '2025_03_09_stdev0.5/bufs_02_0.pkl', 'fps30/f_0000_walk1.npz', 'fps30/n_0000_walk1.npz']
    fig, axes = pt.subplots(4,3, sharey=True, figsize=(19.2, 10.8))
    axes[0, 0].set_ylabel('Autocorrelation', fontsize=20, loc='center')
    axes[1, 0].set_ylabel('Autocorrelation', fontsize=20, loc='center')
    axes[2, 0].set_ylabel('Autocorrelation', fontsize=20, loc='center')
    axes[3, 0].set_ylabel('Autocorrelation', fontsize=20, loc='center')

    # Other
    for name in name_list[:8]:
        data, data_log = [], []
        path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'other_data', name)
        if '.csv' in name:
            df_raw = pd.read_csv(path)
            df_raw = df_raw.drop('date', axis=1)
            data = df_raw.values.astype(np.float32)
        elif '.npz' in name:
            data = np.load(path)['data']
            data = data.squeeze()
        _, col = data.shape

        for c in range(col):
            autocorrelation = smt.stattools.acf(data[:,c], nlags=720)
            axes[name_list.index(name) // 3, name_list.index(name) % 3].plot(autocorrelation)
        axes[name_list.index(name) // 3, name_list.index(name) % 3].set_title(name[:-4], fontsize=20)
        if 'ETTh' in name:
            axes[name_list.index(name) // 3, name_list.index(name) % 3].set_xlabel('Time (hours)', fontsize=20, color='red', loc='center')
        elif 'ETTm' in name:
            axes[name_list.index(name) // 3, name_list.index(name) % 3].set_xlabel('Time (15 minutes)', fontsize=20, color='red', loc='center')
        elif 'Weather' in name:
            axes[name_list.index(name) // 3, name_list.index(name) % 3].set_xlabel('Time (10 minutes)', fontsize=20, color='red', loc='center')
        elif 'ILI' in name:
            axes[name_list.index(name) // 3, name_list.index(name) % 3].set_xlabel('Time (weeks)', fontsize=20, color='red', loc='center')
        elif 'Exchange_rate' in name:
            axes[name_list.index(name) // 3, name_list.index(name) % 3].set_xlabel('Time (days)', fontsize=20, color='red', loc='center')
        elif 'Traffic' in name:
            axes[name_list.index(name) // 3, name_list.index(name) % 3].set_xlabel('Time (hours)', fontsize=20, color='red', loc='center')
        elif 'Electricity' in name:
            axes[name_list.index(name) // 3, name_list.index(name) % 3].set_xlabel('Time (hours)', fontsize=20, color='red', loc='center')
        elif 'PEMS' in name:
            axes[name_list.index(name) // 3, name_list.index(name) % 3].set_xlabel('Time (5 minutes)', fontsize=20, color='red', loc='center')
        axes[name_list.index(name) // 3, name_list.index(name) % 3].tick_params(axis='both', which='major', labelsize=20)
    
    # RP
    for name in name_list[8:10]:
        file_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP', name)
        with open(file_path, 'rb') as f:
            (buffers, _, _, _) = pk.load(f, encoding='latin1')
        with open(os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP', 'epi_indices.pkl'), 'rb') as f:
            epi_indices = pk.load(f)[name]
        pos = buffers['position']
        _, col = pos.shape

        for c in range(col):
            autocorrelation = smt.stattools.acf(pos[:,c][epi_indices], nlags=120)
            axes[name_list.index(name) // 3, name_list.index(name) % 3].plot(autocorrelation)
        if name[-5] == '6':
            title = 'RP Walk Epi'
        else:
            title = 'RP Fall Epi'
        axes[name_list.index(name) // 3, name_list.index(name) % 3].set_title(title, fontsize=20)
        axes[name_list.index(name) // 3, name_list.index(name) % 3].set_xlabel('Time (time steps)', fontsize=20, color='red', loc='center')
        axes[name_list.index(name) // 3, name_list.index(name) % 3].tick_params(axis='both', which='major', labelsize=20)

    # VP
    joint_index = list(range(17)) + list(range(18,26))
    for name in name_list[10:]:
        file_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'VP', name)
        pos = np.load(file_path)['ajp'][:,joint_index]
        _, col = pos.shape

        for c in range(col):
            autocorrelation = smt.stattools.acf(pos[:,c], nlags=368)
            axes[name_list.index(name) // 3, name_list.index(name) % 3].plot(autocorrelation)
        if name[6] == 'f':
            title = 'SimP Fall Epi'
        else:
            title = 'SimP Walk Epi'
        axes[name_list.index(name) // 3, name_list.index(name) % 3].set_title(title, fontsize=20)
        axes[name_list.index(name) // 3, name_list.index(name) % 3].set_xlabel('Time (time steps)', fontsize=20, color='red', loc='center')
        axes[name_list.index(name) // 3, name_list.index(name) % 3].tick_params(axis='both', which='major', labelsize=20)

    pt.tight_layout()
    # pt.show()
    pt.savefig('plots/autocorrelation.pdf', dpi=300)

def autocorrelation_short():
    matplotlib.rcParams['text.usetex'] = True
    name_list = ['ETTh1.csv', 'PEMS03.npz', 'f_0000_walk1.npz']
    fig, axes = pt.subplots(1,3, sharey=True, figsize=(19.2, 2.7))
    axes[0].set_ylabel('Autocorrelation', fontsize=20, loc='center')

    # Other
    for name in name_list[:2]:
        data, data_log = [], []
        path = os.path.join(os.path.dirname(os.getcwd()), 'other_data', name)
        if '.csv' in name:
            df_raw = pd.read_csv(path)
            df_raw = df_raw.drop('date', axis=1)
            data = df_raw.values.astype(np.float32)
        elif '.npz' in name:
            data = np.load(path)['data']
            data = data.squeeze()
        _, col = data.shape

        for c in range(col):
            autocorrelation = smt.stattools.acf(data[:,c], nlags=720)
            axes[name_list.index(name) % 3].plot(autocorrelation)
        axes[name_list.index(name) % 3].set_title(name[:-4], fontsize=20)
        axes[name_list.index(name) % 3].tick_params(axis='both', which='major', labelsize=20)
        if 'ETTh' in name:
            axes[name_list.index(name) % 3].set_xlabel('Time (hours)', fontsize=20, color='red', loc='center')
        elif 'ETTm' in name:
            axes[name_list.index(name) % 3].set_xlabel('Time (15 minutes)', fontsize=20, color='red', loc='center')
        elif 'Weather' in name:
            axes[name_list.index(name) % 3].set_xlabel('Time (10 minutes)', fontsize=20, color='red', loc='center')
        elif 'ILI' in name:
            axes[name_list.index(name) % 3].set_xlabel('Time (weeks)', fontsize=20, color='red', loc='center')
        elif 'Exchange_rate' in name:
            axes[name_list.index(name) % 3].set_xlabel('Time (days)', fontsize=20, color='red', loc='center')
        elif 'Traffic' in name:
            axes[name_list.index(name) % 3].set_xlabel('Time (hours)', fontsize=20, color='red', loc='center')
        elif 'Electricity' in name:
            axes[name_list.index(name) % 3].set_xlabel('Time (hours)', fontsize=20, color='red', loc='center')
        elif 'PEMS' in name:
            axes[name_list.index(name) % 3].set_xlabel('Time (5 minutes)', fontsize=20, color='red', loc='center')

    # VP
    joint_index = list(range(17)) + list(range(18,26))
    for name in name_list[2:]:
        file_path = os.path.join('data', name)
        pos = np.load(file_path)['ajp'][:,joint_index]
        _, col = pos.shape

        for c in range(col):
            autocorrelation = smt.stattools.acf(pos[:,c], nlags=368)
            axes[name_list.index(name) % 3].plot(autocorrelation)
        if name[0] == 'f':
            title = 'SimP Fall Epi'
        else:
            title = 'SimP Walk Epi'
        axes[name_list.index(name) % 3].set_title(title, fontsize=20)
        axes[name_list.index(name) % 3].tick_params(axis='both', which='major', labelsize=20)
        axes[name_list.index(name) % 3].set_xlabel('Time (time steps)', fontsize=25, color='red', loc='center')

    pt.tight_layout()
    # pt.show()
    pt.savefig('plots/autocorrelation_short.pdf', dpi=300)

def fft_visualization(threshold):
    with open('/home/bhe100/Desktop/data/RP/2025_05_08_stdev0.25/bufs_00_4.pkl', 'rb') as f:
        (buffers, _, _, _) = pk.load(f, encoding='latin1')
        img = np.stack([buffers['images'][0][:,:,::-1], buffers['images'][-1][:,:,::-1]])
        img = tr.from_numpy(img)

    bs, H, W, C = img.shape
    img_ = img.double()

    # f1 = np.fft.fft2(img_.numpy(), axes=(0,1))
    f = tr.fft.fft2(img_, dim=(1,2))
    fshift = tr.fft.fftshift(f)

    rows, cols = int(H/2), int(W/2)
    mask = tr.zeros((H, W, C), dtype=tr.uint8)
    mask[rows-threshold:rows+threshold, cols-threshold:cols+threshold] = 1
    fshift = fshift * mask               
    ishift = tr.fft.ifftshift(fshift)
    i_img = tr.fft.ifft2(ishift, dim=(1,2))
    LF_feat = tr.abs(i_img)

    HF_feat = (img - LF_feat)
    HF_feat = HF_feat*(255/HF_feat.max())*3
    
    ht_activation = tr.nn.Hardtanh(0,255)
    hf = ht_activation(HF_feat)
    lf = ht_activation(LF_feat)

    HF_feat = (hf/255).float()
    LF_feat = (lf/255).float()

    fig, axes = pt.subplots(bs,3)
    for t in range(bs):
        axes[t, 0].imshow(HF_feat[t])
        axes[t, 1].imshow(LF_feat[t])
        axes[t, 2].imshow(img[t])
    pt.show()

def main_results_histogram():
    pt.rcParams['text.usetex'] = True
    performance, error = [], []
    fig, ax = pt.subplots(1,1,figsize=(12.80*1.5, 9.60))
    a = np.load('RP_Egofalls.npy')
    b = np.load('VP_Egofalls.npy')
    performance.append(a.mean()*100)
    performance.append(b.mean()*100)
    error.append(a.std()*100)
    error.append(b.std()*100)

    a = np.load('Flowformer_RP_lr0.0001_dmodel1024_dff256_pl12_qttest.npy')
    b = np.load('Flowformer_VP_lr0.0001_dmodel512_dff512_pl30_qttest.npy')
    performance.append(a.mean()*100)
    performance.append(b.mean()*100)
    error.append(a.std()*100)
    error.append(b.std()*100)

    a = np.load('TimesNet_RP_topk1_numk2_lr0.0001_dmodel256_dff1024_pl12_qttest.npy')
    b = np.load('TimesNet_VP_topk2_numk4_lr0.01_dmodel256_dff64_pl30_qttest.npy')
    performance.append(a.mean()*100)
    performance.append(b.mean()*100)
    error.append(a.std()*100)
    error.append(b.std()*100)

    a = np.load('RP_pl12_PVP_qttest.npy')
    b = np.load('VP_pl30_PVP_qttest.npy')
    performance.append(a.mean()*100)
    performance.append(b.mean()*100)
    error.append(a.std()*100)
    error.append(b.std()*100)

    models = ('Egoalls_RP', 'Egoalls_SimP', 'Flowformer_RP', 'Flowformer_SimP', 'TimesNet_RP', 'TimesNet_SimP', 'Ours_RP', 'Ours_SimP')
    y_pos = np.arange(len(models))
    cols = ['#C7DEE9', '#C7DEE9', '#A4D9F4', '#A4D9F4', '#7C94B7', '#7C94B7', '#32417A', '#32417A']
    hbars = ax.barh(y_pos, performance, xerr=error, color=cols, align='center')
    ax.set_yticks(y_pos, labels=models, fontsize=40)
    ax.set_xticks([0.0, 20, 40, 60, 80, 100, 120], labels=['0', '20', '40', '60', '80', '100', ''], fontsize=40)
    ax.bar_label(hbars, fmt='%.2f', fontsize=40)
    ax.set_xlabel('Accuracy', fontsize=40)

    pt.tight_layout()
    pt.show()
    # pt.savefig('main_results_histogram_long.pdf', dpi=300)


if __name__ == "__main__":
    # episodes_z_coor()
    # destinations_distribution()
    # autocorrelation_short()
    # fft_visualization(30)
    main_results_histogram()