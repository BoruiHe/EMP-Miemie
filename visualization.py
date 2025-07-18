import os
import numpy as np
import torch as tr
import pandas as pd
import pickle as pk
import matplotlib.pyplot as pt
import statsmodels.tsa.api as smt
from torchvision.transforms.functional import rgb_to_grayscale
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

def destinations_distribution():
    if os.name =='nt':
        pt.rc('text', usetex=True)
        with open('data_generation/destination.pkl', 'rb') as f:
            destination = pk.load(f)
        f_dest, n_dest = destination['f_dest'], destination['n_dest']
        fig, ax = pt.subplots(1, 1, figsize=(12.80, 9.60)) 
        for coorF, coorN in zip(f_dest, n_dest):
            ax.scatter(coorF[0], coorF[1], c='crimson', marker='X', s=60)
            ax.scatter(coorN[0], coorN[1], c='mediumspringgreen', marker='o', s=60)
        ax.spines.left.set_position('zero')
        ax.spines.bottom.set_position('zero')
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_xticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6], [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6], fontsize=20)
        ax.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6], [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6], fontsize=20)
        pt.savefig('plots/dest_distribution.pdf')

        print(f'f: {len(f_dest)}, n: {len(n_dest)}.')
    else:
        path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'VP')
        f_dest, n_dest = [], []
        for num in range(int(len(os.listdir(path))/2)):
            n_dest.append(np.load(os.path.join(path, f'n_{num}.npz'))['pos'][-1,-1,:-1])
            f_dest.append(np.load(os.path.join(path, f'f_{num}.npz'))['pos'][-1,-1,:-1])
        destination = {
            'n_dest': n_dest,
            'f_dest': f_dest
        }
        with open('destination.pkl', 'wb') as f:
            pk.dump(destination, f)

def autocorrelation():
    pt.rcParams['text.usetex'] = True
    name_list = ['ETTh1.csv', 'ETTm1.csv', 'PEMS03.npz', 'Weather.csv', 'ILI.csv', 'Traffic.csv', 'Electricity.csv', 'Exchange_rate.csv', 
                 '2025_03_12_stdev0.0/bufs_02_6.pkl', '2025_03_09_stdev0.5/bufs_02_0.pkl', 'fps30/n_0000_walk1.npz', 'fps30/f_0000_walk1.npz']
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

def traj_and_real_pos():
    pt.rcParams['text.usetex'] = True
    name_list = ['2025_03_12_stdev0.0/bufs_02_6.pkl', '2025_03_09_stdev0.5/bufs_02_0.pkl', 'fps30/f_0000_walk1.npz', 'fps30/n_0000_walk1.npz']
    fig, axes = pt.subplots(1,1)
    axes.set_ylabel('Joint angles', fontsize=20, loc='center')
    # axes[1, 0].set_ylabel('Joint angles', fontsize=20, loc='center')

    # RP
    for name in name_list[1:2]:
        file_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP', name)
        with open(file_path, 'rb') as f:
            (buffers, _, _, _) = pk.load(f, encoding='latin1')
        with open(os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP', 'epi_indices.pkl'), 'rb') as f:
            epi_indices = pk.load(f)[name]
        pos = buffers['position']
        traj = buffers['target']

        axes.plot(pos[:,3][epi_indices], label='Real')
        axes.plot(traj[:,3][epi_indices], label='Traj.')
        # if name[-5] == '6':
        #     title = 'RP Walk Epi'
        # else:
        #     title = 'RP Fall Epi'
        # axes.set_title(title, fontsize=20)
        axes.set_xlabel('Time (time steps)', fontsize=20, color='red', loc='center')
        axes.tick_params(axis='both', which='major', labelsize=20)
        axes.legend(fontsize=20)

    # VP
    # joint_index = list(range(17)) + list(range(18,26))
    # for name in name_list[2:]:
    #     file_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'VP', name)
    #     pos = np.load(file_path)['ajp'][:,joint_index]
    #     traj = np.load(file_path)['traj'][:,joint_index]

    #     axes[name_list.index(name) // 2, name_list.index(name) % 2].plot(pos[:,20], label='Real')
    #     axes[name_list.index(name) // 2, name_list.index(name) % 2].plot(traj[:,20], label='Traj.')
    #     if name[6] == 'f':
    #         title = 'SimP Fall Epi'
    #     else:
    #         title = 'SimP Walk Epi'
    #     axes[name_list.index(name) // 2, name_list.index(name) % 2].set_title(title, fontsize=20)
    #     axes[name_list.index(name) // 2, name_list.index(name) % 2].set_xlabel('Time (time steps)', fontsize=20, color='red', loc='center')
    #     axes[name_list.index(name) // 2, name_list.index(name) % 2].tick_params(axis='both', which='major', labelsize=20)
    #     axes[name_list.index(name) // 2, name_list.index(name) % 2].legend(fontsize=20)

    pt.tight_layout()
    # pt.show()
    pt.savefig('plots/traj_real_pos.pdf', dpi=300)

def real_pos():
    pt.rcParams['text.usetex'] = True
    name = '2025_03_09_stdev0.5/bufs_02_0.pkl'
    fig, axes = pt.subplots(1,1)
    axes.set_ylabel('Joint angles', fontsize=20, loc='center')
    # axes[1, 0].set_ylabel('Joint angles', fontsize=20, loc='center')

    # RP
    file_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP', name)
    with open(file_path, 'rb') as f:
        (buffers, _, _, _) = pk.load(f, encoding='latin1')
    with open(os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP', 'epi_indices.pkl'), 'rb') as f:
        epi_indices = pk.load(f)[name]
    pos = buffers['position']
    _, col = pos.shape

    for p in [2,5,11]:
        axes.plot(pos[:,p][epi_indices], label=f'{p}')

    # if name[-5] == '6':
    #     title = 'RP Walk Epi'
    # else:
    #     title = 'RP Fall Epi'
    # axes.set_title(title, fontsize=20)
    axes.set_xlabel('Time (time steps)', fontsize=20, color='red', loc='center')
    axes.tick_params(axis='both', which='major', labelsize=20)
    axes.legend(fontsize=20)

    pt.tight_layout()
    # pt.show()
    pt.savefig('plots/real_pos.pdf', dpi=300)

def decoder_image(img, mean, std):
    inputs_decoder = []
    for ss, m, s in zip(img, mean, std):
        ss = np.array(ss * s)
        ss = np.array(ss + m)
        ss = ss * 255
        inputs_decoder.append(ss)
    return np.stack(inputs_decoder)

def fft_visualization(threshold):
    # with open('/home/bhe100/Desktop/data/RP/2025_05_08_stdev0.25/bufs_00_4.pkl', 'rb') as f:
    #     (buffers, _, _, _) = pk.load(f, encoding='latin1')
    #     imgs = np.stack([buffers['images'][0][:,:,::-1], buffers['images'][-1][:,:,::-1]])
    # bs,H,W,C = img.shape
    
    data = np.load('/home/bhe100/Desktop/data/VP/fps30/f_0000_walk1.npz')
    imgs = data['imgs']
    # imgs: (bs, H, W, C)
    bs,H,W,C = imgs.shape
    for img in imgs[:1]: 
        img = tr.from_numpy(img)
        
        f = tr.fft.fft2(img, dim=(0,1))
        fshift = tr.fft.fftshift(f)

        rows, cols = int(H/2), int(W/2)
        mask = tr.zeros((H, W, C), dtype=tr.uint8)
        mask[rows-threshold:rows+threshold, cols-threshold:cols+threshold] = 1
        fshift = fshift * mask               
        ishift = tr.fft.ifftshift(fshift)
        i_img = tr.fft.ifft2(ishift, dim=(0,1))
        LF_feat = tr.abs(i_img)

        HF_feat = (img - LF_feat)
        HF_feat = HF_feat*(255/HF_feat.max())*3
        
        ht_activation = tr.nn.Hardtanh(0,255)
        hf = ht_activation(HF_feat).type(tr.uint8)
        lf = ht_activation(LF_feat).type(tr.uint8)

        fig, axes = pt.subplots(1,3)
        axes[0].imshow(rgb_to_grayscale(hf.permute(2,0,1)).squeeze(), cmap='gray')
        axes[1].imshow(lf)
        axes[2].imshow(img, cmap='gray')
        pt.show()

def main_results_histogram():
    pt.rcParams['text.usetex'] = True
    performance, error = [], []
    fig, ax = pt.subplots(1,1)
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

    models = ('Egoalls_RP', 'Egoalls_VP', 'Flowformer_RP', 'Flowformer_VP', 'TimesNet_RP', 'TimesNet_VP', 'Ours_RP', 'Ours_VP')
    y_pos = np.arange(len(models))
    cols = ['#C7DEE9', '#C7DEE9', '#A4D9F4', '#A4D9F4', '#7C94B7', '#7C94B7', '#32417A', '#32417A']
    hbars = ax.barh(y_pos, performance, xerr=error, color=cols, align='center')
    ax.set_yticks(y_pos, labels=models, fontsize=15)
    ax.set_xticks([0.0, 20, 40, 60, 80, 100, 120], labels=['0', '20', '40', '60', '80', '100', ''], fontsize=20)
    ax.bar_label(hbars, fmt='%.2f', fontsize=15)
    ax.set_xlabel('Accuracy', fontsize=15)

    pt.tight_layout()
    # pt.show()
    pt.savefig('main_results_histogram.pdf', dpi=300)

def vis_MH():
    class conf():
        def __init__(self, dataset):
            self.modalities = 'PVP'
            self.d_model = 1024
            self.num_class = 2
            if dataset == 'RP':
                self.seq_len = 24
                self.pred_len = 12
            else:
                self.seq_len = 60
                self.pred_len = 30
            self.num_Templayers = 6
            self.num_Timeslayers = 2
            self.num_convlayers = 4
            self.num_SPAconvlayers = 6
            self.num_TEMconvlayers = 6
            self.hid_SPA = 64
            self.hid_TEM = 64
            self.motion_dim = 25
            self.fft_threshold = 30
            self.top_k = 3
            self.inc_dim = 256

    cm = 'inferno'
    plt.rcParams['font.size'] = 14
    plt.rcParams['text.usetex'] = True

    for p in ['RP_pl12_PVP', 'VP_pl30_PVP']:
        model = Model(conf(p[:2]))
        
        seed = 5
        model_p = os.path.join('checkpoints', p, str(seed), 'checkpoint.pth')
        model.load_state_dict(torch.load(model_p, weights_only=True, map_location=torch.device('cpu')))

        weights = model.motion_head.projection.weight.permute(1,0).detach().numpy()
        fig, ax = plt.subplot_mosaic([['top', 'top'],
                                    ['lower left', 'lower right']],
                                    layout='constrained')
        im1=ax['top'].imshow(weights[:100,], cmap=cm)
        im2=ax['lower left'].imshow(weights, cmap=cm)
        if 'RP' in p:
            im3=ax['lower right'].imshow(weights[-25:,:25], cmap=cm)
        else:
            im3=ax['lower right'].imshow(weights[250:275,:25], cmap=cm)
        fig.colorbar(im2, pad=0.03, ax=ax.values())
        # plt.show(block=True)
        plt.savefig(os.path.join(f'{p[:2]}_MotionHead_vis.pdf'), dpi=300)
        plt.close()


if __name__ == "__main__":
    # destinations_distribution()
    # autocorrelation()
    traj_and_real_pos()
    # real_pos()
    # fft_visualization(30)
