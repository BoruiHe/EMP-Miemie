import os
import csv
import numpy as np
import pickle as pk
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
    name_list = ['ETTh1.csv', 'ETTh2.csv', 'ETTm1.csv', 'ETTm2.csv', 'traffic.txt', 'electricity.txt', 'exchange_rate.txt', '2025_03_09_stdev0.5/bufs_02_0.pkl']
    fig, axes = pt.subplots(3,3)
    axes[0, 0].set_ylabel('Autocorrelation', fontsize=12, color='red', loc='center')
    axes[1, 0].set_ylabel('Autocorrelation', fontsize=12, color='red', loc='center')
    axes[2, 0].set_ylabel('Autocorrelation', fontsize=12, color='red', loc='center')

    for name in name_list[:4]:
        data, data_log = [], []
        path = f'other_data/{name}'
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter=',', quotechar='|')
            for row in reader:
                data_log.append(row)
        for row in data_log[1:]:
            data.append(np.asarray([float(i) for i in row[1:]], dtype='float32'))
        data = np.asarray(data, dtype='float32')
        _, col = data.shape

        for c in range(col):
            autocorrelation = smt.stattools.acf(data[:,c], nlags=720)
            axes[name_list.index(name) // 3, name_list.index(name) % 3].plot(autocorrelation)
        axes[name_list.index(name) // 3, name_list.index(name) % 3].set_title(name)
        if 'h' in name:
            axes[name_list.index(name) // 3, name_list.index(name) % 3].set_xlabel('Time (hours)', fontsize=12, color='red', loc='center')
        else:
            axes[name_list.index(name) // 3, name_list.index(name) % 3].set_xlabel('Time (15 minutes)', fontsize=12, color='red', loc='center')
    
    for name in name[4:6]:
        data, data_log = [], []
        path = f'other_data/{name}'
        with open(path, 'r') as f:
            text = f.readlines()
        for row in text:
            data_log.extend(row.split())
        for row in data_log:
            data.append(np.asarray([float(i) for i in row.split(',')], dtype='float32'))
        data = np.asarray(data, dtype='float32')
        _, col = data.shape

        for c in range(col):
            autocorrelation = smt.stattools.acf(data[:,c], nlags=720)
            axes[name_list.index(name) // 3, name_list.index(name) % 3].plot(autocorrelation)
        axes[name_list.index(name) // 3, name_list.index(name) % 3].set_title(name)
        if 'traffic' in name:
            axes[name_list.index(name) // 3, name_list.index(name) % 3].set_xlabel('Time (hours)', fontsize=12, color='red', loc='center')
        else:
            axes[name_list.index(name) // 3, name_list.index(name) % 3].set_xlabel('Time (15 minutes)', fontsize=12, color='red', loc='center')
    
    name = 'exchange_rate.txt'
    path = f'other_data/{name}'
    data = []
    with open(path, 'r') as f:
        text = f.readlines()
    for record in text:
        data.append(np.asarray([float(i) for i in record[:-1].split(',')]))
    data = np.asarray(data, dtype='float32')
    _, col = data.shape

    for c in range(col):
        autocorrelation = smt.stattools.acf(data[:,c], nlags=data.shape[0])
        axes[name_list.index(name) // 3, name_list.index(name) % 3].plot(autocorrelation)
    axes[name_list.index(name) // 3, name_list.index(name) % 3].set_title(name)
    axes[name_list.index(name) // 3, name_list.index(name) % 3].set_xlabel('Time (days)', fontsize=12, color='red', loc='center')
    
    for name in name_list[7:]:
        file_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP', name)
        with open(file_path, 'rb') as f:
            (buffers, _, _, _) = pk.load(f, encoding='latin1')
        with open(os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP', 'epi_indices.pkl'), 'rb') as f:
            epi_indices = pk.load(f)[name]
        pos = buffers['position']
        _, col = pos.shape

        for c in range(col):
            autocorrelation = smt.stattools.acf(pos[:,c][epi_indices], nlags=12)
            axes[name_list.index(name) // 3, name_list.index(name) % 3].plot(autocorrelation)
        axes[name_list.index(name) // 3, name_list.index(name) % 3].set_title('RP')
        axes[name_list.index(name) // 3, name_list.index(name) % 3].set_xlabel('Time (time steps)', fontsize=12, color='red', loc='center')

    pt.show()
    # pt.savefig('plots/autocorrelation.pdf')


if __name__ == "__main__":
    # destinations_distribution()
    autocorrelation()