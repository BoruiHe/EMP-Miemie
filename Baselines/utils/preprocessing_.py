from sklearn.preprocessing import StandardScaler
import pickle as pk
import numpy as np
import random
import os


def RP_preprocessing(seq_len, ps, ratio, exp_path, CDV=False, relax_factor=0):
    '''
    RP_preprocessing: STRICTLY seperate datasets with no infomation leakage.
    That is to say, video clips in the validation and testing sets should never be seen in training.
    CDK --> True for cross data validation, default is False.
    '''
    root = os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP')
    indices_file_exist = os.path.isfile(os.path.join(root, 'epi_indices.pkl'))

    files = [[], [], []]
    folders = [['2025_03_09_stdev0.5', '2025_03_12_stdev0.0', '2025_03_12_stdev0.25', '2025_03_14_stdev0.125'],
               ['2025_05_07_stdev0.0', '2025_05_07_stdev0.25', '2025_05_08_stdev0.0', '2025_05_08_stdev0.25'],
               ['2025_05_09_stdev0.0']]
    # Seperate videos into 3 sets based on the location.
    for folder, loc, idx in zip(folders, ['office', 'lab', 'hall'], [0, 1, 2]):
        for subfolder in folder:
            files[idx].extend([subfolder+f'/{file}' for file in os.listdir(os.path.join(root, subfolder)) if ('bufs' in file)])
        print(f'{len(files[idx])} episodes at {loc}.')
    print('-------------------------')
    
    # if cross data validation, all data in RP is taken as inputs
    if CDV:
        training_epi, validation_epi = [], []
        testing_epi = []
        for fi in files:
            for i in fi:
                testing_epi.append(i)
    else:
        # office + lab for training and hall for testing
        failure_epi, successes_epi = [], []
        for idx, fi in enumerate(files[:2]):
            failure_epi.append([f for f in fi if f[-5]!='6'])
            successes_epi.append(list(set(fi) - set(failure_epi[idx])))

        training_epi, validation_epi, testing_epi = [], [], []
        for sepi, fepi in zip(successes_epi, failure_epi):
            sepi, fepi = set(sepi), set(fepi)
            # Traininig set
            training_epi.extend(random.sample(list(sepi), int(len(sepi)*ratio[0])))
            sepi = sepi - set(training_epi)
            training_epi.extend(random.sample(list(fepi), int(len(fepi)*ratio[0])))
            fepi = fepi - set(training_epi)
            training_epi.sort()
            # Validation set
            validation_epi.extend(list(sepi))
            sepi = sepi - set(validation_epi)
            validation_epi.extend(list(fepi))
            fepi = fepi - set(validation_epi)
            validation_epi.sort()

        # Testing set
        testing_epi = files[-1]

    # Each video contribute several video clips to the corresponding set.
    if indices_file_exist:
        with open(os.path.join(root, 'epi_indices.pkl'), 'rb') as f:
            epi_indices = pk.load(f)
    else:
        epi_indices = {}

    shortest_len, num_suc, num_fai = 10**4, 0, 0
    idx_splits = {
        'training': [],
        'validation': [],
        'testing': []
    }
    for epi_set, key in zip([training_epi, validation_epi, testing_epi], idx_splits.keys()):
        failure, successes = [], []
        for file in epi_set:
            num_steps = int(file[-5])
            with open(os.path.join(root, file), 'rb') as f:
                (buffers, elapsed, _, _) = pk.load(f, encoding='latin1')
            # Data at high frequency should align with the data at the lowest frequency, which are images.
            frames = buffers['images']
            if indices_file_exist:
                indices = epi_indices[file]
            else:
                epi_indices[file] = indices = [i for i in range(len(frames)) if frames[i] is not None]
            timepoints = np.asarray(elapsed, dtype='float32')[indices]
            if num_steps == 0:
                stop_sign = 2.25 # in seconds
                stop_sign = (timepoints < stop_sign).sum() # in # of frames
                if stop_sign < shortest_len: shortest_len = stop_sign
                qualified_f_indices = range(seq_len, seq_len+1+relax_factor)
                failure.extend(zip([file]*len(qualified_f_indices), qualified_f_indices))

            elif 6 > num_steps > 0:
                stop_sign = 2.25*num_steps # in seconds
                stop_sign = (timepoints < stop_sign).sum() # in # of frames
                qualified_f_indices = list(range(stop_sign-ps+1, stop_sign-ps+1+(relax_factor+1)))
                failure.extend(zip([file]*len(qualified_f_indices), qualified_f_indices))

            elif num_steps == 6:
                qualified_s_indices = random.sample(list(range(seq_len, timepoints.size-ps+1)), (relax_factor+1))
                successes.extend(zip([file]*len(qualified_s_indices), qualified_s_indices))

        # make the VC dataset perfectly balanced
        if len(successes) != len(failure):
            s_folder1 = [i for i in successes if i[0][:10] in [f[:10] for f in folders[0]]]
            s_folder2 = [i for i in successes if i[0][:10] in [f[:10] for f in folders[1]]]

            f_folder1 = [i for i in failure if i[0][:10] in [f[:10] for f in folders[0]]]
            f_folder2 = [i for i in failure if i[0][:10] in [f[:10] for f in folders[1]]]

            if len(s_folder1) > len(f_folder1):
                s_folder1 = random.sample(s_folder1, len(f_folder1))
            elif len(s_folder1) < len(f_folder1):
                f_folder1 = random.sample(f_folder1, len(s_folder1))
            else:
                pass
            successes = s_folder1 + s_folder2
            failure = f_folder1 + f_folder2

        num_suc += len(successes)
        num_fai += len(failure)
        print(f'{key}: {len(successes)} successes; {len(failure)} failure.')
        idx_splits[key] = successes + failure

    # Save a file for aligning available frames once and for all.
    if not indices_file_exist:
        with open(root+'/epi_indices.pkl', 'wb') as f:
            pk.dump(epi_indices, f)

    # Save an intermediate file for data loading.
    with open(os.path.join(exp_path, 'intermediate_files', f'idx_splits_RP_{seq_len}_{ps}.pkl'), 'wb') as outfile:
        pk.dump(idx_splits, outfile)

    # Output statistics for checking the scale of the VP dataset and debugging.
    print('---------------------------------------------')
    print(f'RP dataset has {num_suc} available non-fall data points and {num_fai} fall data points.')
    print(f'The exact number of data points one can get from RP depends on historical sequences length and will not exceed {len(failure)*2}.')
    print(f'Warning: Minimum length: {shortest_len}; seq_len: {seq_len}; ps: {ps}.')
    print('---------------------------------------------')
    print(f'You get {len(idx_splits['training']+idx_splits['validation']+idx_splits['testing'])} data points.')
    print('---------------------------------------------')

def VP_preprocessing(seq_len, ps, ratio, exp_path, fps, relax_factor=0):
    '''
    VP_preprocessing: STRICTLY seperate datasets with no infomation leakage.
    That is to say, video clips in the validation and testing sets should never be seen in training.
    '''
    root = os.path.join(os.path.dirname(os.getcwd()), 'data', 'VP', f'fps{fps}')
    files = os.listdir(root)

    # Seperate videos into 3 sets.
    failure_epi = [f for f in files if f[0]=='f']
    successes_epi = list(set(files) - set(failure_epi))
    splits = [int(len(failure_epi)*i) for i in ratio]
    failure_epi.sort()
    successes_epi.sort()

    successes_epi, failure_epi = set(successes_epi), set(failure_epi)
    # Traininig set
    training_epi = random.sample(list(successes_epi), splits[0])
    successes_epi = successes_epi - set(training_epi)
    training_epi.extend(random.sample(list(failure_epi), splits[0]))
    failure_epi = failure_epi - set(training_epi)
    training_epi.sort()
    # Validation set
    validation_epi = random.sample(list(successes_epi), splits[1])
    successes_epi = successes_epi - set(validation_epi)
    validation_epi.extend(random.sample(list(failure_epi), splits[1]))
    failure_epi = failure_epi - set(validation_epi)
    validation_epi.sort()
    # Testing set
    testing_epi = random.sample(list(successes_epi), splits[2])
    successes_epi = successes_epi - set(testing_epi)
    testing_epi.extend(random.sample(list(failure_epi), splits[2]))
    failure_epi = failure_epi - set(testing_epi)
    testing_epi.sort()

    # Each video contribute several video clips to the corresponding set.
    idx_splits = {
        'training': [],
        'validation': [],
        'testing': []
    }
    early_falls, num_suc, num_fai, vd_len = 0, 0, 0, 0
    issue_f, issue_n = 0, 0
    files_f, files_n = [], []
    for epi_set, key in zip([training_epi, validation_epi, testing_epi], idx_splits.keys()):
        failure, successes = [], []
        for file in epi_set:
            data = np.load(os.path.join(root, file))
            vd_len += len(data['labels'])
            if file[0] == 'n':
                qualified_s_indices = random.sample(list(range(seq_len, data['labels'].size-ps+1)), (relax_factor+1))
                successes.extend(zip([file]*len(qualified_s_indices), qualified_s_indices))
                if (data['labels'] != 0).any():
                    issue_n += 1
                    files_n.append(file)
            else:
                stop_sign = (~data['labels']).sum()
                if stop_sign < seq_len + ps:
                    early_falls += 1
                    files_f.append(file)
                # qualified_s_indices = list(range(seq_len, stop_sign-ps+1))
                # successes.extend(zip([file]*len(qualified_s_indices), qualified_s_indices))
                qualified_f_indices = list(range(stop_sign-ps+1, stop_sign-ps+1+(relax_factor+1)))
                failure.extend(zip([file]*len(qualified_f_indices), qualified_f_indices))
                if (data['labels'][stop_sign] != 1):
                    issue_f += 1

        # make the VC dataset perfectly balanced
        successes = random.sample(successes, len(failure))
        num_suc += len(successes)
        num_fai += len(failure)
        print(f'{key}: {len(successes)} successes; {len(failure)} failure.')
        idx_splits[key] = successes + failure

    # Save an intermediate file for data loading. 
    with open(os.path.join(exp_path, 'intermediate_files', f'idx_splits_VP_{seq_len}_{ps}.pkl'), 'wb') as outfile:
        pk.dump(idx_splits, outfile)

    # Output statistics for checking the scale of the VP dataset and debugging.
    print('---------------------------------------------')
    print(f'VP dataset ({fps} fps) has {num_suc} available non-fall data points and {num_fai} fall data points.')
    print(f'The exact number of data points one can get from RP depends on historical sequences length and will not exceed {num_fai*2}.')
    print(f'Info: Early Falls: {early_falls}; Average Video Length: {vd_len/len(files)}; Sequence length: {seq_len}; Prediction Span: {ps}; Issues: {issue_f} {issue_n}.')
    print('Issues n:', files_n)
    print('Issees f:', files_f)
    print('---------------------------------------------')
    print(f'You get {len(idx_splits['training']+idx_splits['validation']+idx_splits['testing'])} data points.')
    print('---------------------------------------------')

def RP_scaler(exp_path, seq_len, ps):
    with open(os.path.join(exp_path, 'intermediate_files', f'idx_splits_RP_{seq_len}_{ps}.pkl'), 'rb') as f:
        idx_splits = pk.load(f)['training']
    with open(os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP', 'epi_indices.pkl'), 'rb') as f:
        epi_idx = pk.load(f)
    # joint_index = [0, 3, 4, 5, 6, 8, 12, 13, 16, 17, 18, 19, 22, 23, 24] # remove dummy joint head_cam
    joint_index = list(range(25))

    A, B, C = [], [], []
    for i in range(len(idx_splits)):
        bufs_file, idx = idx_splits[i]
        his_indices = np.asarray(epi_idx[bufs_file])[(idx-seq_len):idx]
        fur_indices = np.asarray(epi_idx[bufs_file])[idx:(idx+ps)]
        with open(os.path.join(os.path.dirname(os.getcwd()), 'data', 'RP', bufs_file), 'rb') as f:
            (buffers, _, _, _) = pk.load(f, encoding='latin1')
        actuals = np.asarray(buffers['position'][his_indices[:,None], joint_index], dtype='float32')*(np.pi/180)
        future = np.asarray(buffers['position'][fur_indices[:,None], joint_index], dtype='float32')*(np.pi/180)
        targets = np.asarray(buffers['target'][fur_indices[:,None], joint_index], dtype='float32')*(np.pi/180)
        A.append(actuals)
        B.append(future)
        C.append(targets)
    
    A = np.asarray(A).reshape(-1,len(joint_index))
    B = np.asarray(B).reshape(-1,len(joint_index))
    C = np.asarray(C).reshape(-1,len(joint_index))

    scaler_act, scaler_fut, scaler_tar = StandardScaler(), StandardScaler(), StandardScaler()
    scaler_act.fit(A)
    scaler_fut.fit(B)
    scaler_tar.fit(C)
    with open(os.path.join(exp_path, 'intermediate_files', f'scaler_act_RP_{seq_len}_{ps}.pkl'), 'wb') as f:
        pk.dump(scaler_act, f)
    with open(os.path.join(exp_path, 'intermediate_files', f'scaler_fut_RP_{seq_len}_{ps}.pkl'), 'wb') as f:
        pk.dump(scaler_fut, f)
    with open(os.path.join(exp_path, 'intermediate_files', f'scaler_tar_RP_{seq_len}_{ps}.pkl'), 'wb') as f:
        pk.dump(scaler_tar, f)

def VP_scaler(exp_path, seq_len, ps, fps):
    with open(os.path.join(exp_path, 'intermediate_files', f'idx_splits_VP_{seq_len}_{ps}.pkl'), 'rb') as f:
        idx_splits = pk.load(f)['training']
    joint_index = list(range(17)) + list(range(18,26)) # remove dummy joint head_cam

    A, B, C = [], [], []
    for i in range(len(idx_splits)):
        file, idx = idx_splits[i]
        his_indices = np.asarray(range(idx-seq_len, idx), dtype='int16')
        fur_indices = np.asarray(range(idx, idx+ps), dtype='int16')
        data = np.load(os.path.join(os.path.dirname(os.getcwd()), 'data', 'VP', f'fps{fps}', file))
        actuals = data['ajp'][his_indices[:,None], joint_index]
        future = data['ajp'][fur_indices[:,None], joint_index]
        targets = data['traj'][fur_indices[:,None], joint_index]
        A.append(actuals)
        B.append(future)
        C.append(targets)
    
    A = np.asarray(A).reshape(-1,len(joint_index))
    B = np.asarray(B).reshape(-1,len(joint_index))
    C = np.asarray(C).reshape(-1,len(joint_index))

    scaler_act, scaler_fut, scaler_tar = StandardScaler(), StandardScaler(), StandardScaler()
    scaler_act.fit(A)
    scaler_fut.fit(B)
    scaler_tar.fit(C)
    with open(os.path.join(exp_path, 'intermediate_files', f'scaler_act_VP_{seq_len}_{ps}.pkl'), 'wb') as f:
        pk.dump(scaler_act, f)
    with open(os.path.join(exp_path, 'intermediate_files', f'scaler_fut_VP_{seq_len}_{ps}.pkl'), 'wb') as f:
        pk.dump(scaler_fut, f)
    with open(os.path.join(exp_path, 'intermediate_files', f'scaler_tar_VP_{seq_len}_{ps}.pkl'), 'wb') as f:
        pk.dump(scaler_tar, f)


if __name__ == '__main__':
    exp_path = os.path.join('checkpoints', 'debug', '608')
    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
    RP_preprocessing(24, 12, [0.8, 0.2], exp_path)
    # VP_preprocessing(60, 30, hyperparameters_real['splits'], exp_path, fps=30)
    # VP_scaler(exp_path, 60, 30)
