from utils import run_traj, array_from_design, hook_for_data_collection, rgba2rgb
from humanoid import PoppyHumanoidEnv
import pybullet as pb
import pickle as pk
import numpy as np
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', help='sampling rate', type=int, required=True)
    parser.add_argument('--number', help='the number of videos of the type', type=int, required=True)
    parser.add_argument('--base', help='the number of existing videos', type=int, required=True)
    args = parser.parse_args('--fps 16 --number 30 --base 0'.split())
    # args = parser.parse_args('--number 2000 --base 1000 '.split())

    if not (240 % args.fps == 0):
        raise Exception('Improper FPS.')
    else:
        print(f'You are cellecting data at {args.fps} fps, which means recording data every {240 // args.fps} time steps.')
    timestep = .2
    num_cycles = 3 # total number of cycles per episode; each cycle consists of two footsteps (one for each foot)

    npz = np.load(f'rrt_400.npz')
    all_samples = npz['samples']
    all_falls = npz['falls']
    all_objectives = npz['objectives']

    # filter out the falls. ---> The only filter I need.
    all_samples = all_samples[~all_falls]
    all_objectives = all_objectives[~all_falls]
    all_falls = all_falls[~all_falls]

    # scale up jerk
    all_objectives[:,-1] *= .1 / all_objectives[:,-1].max()

    # sort by objectives lexicographically
    lex = np.lexsort(all_objectives.T[::-1])

    saved_idx = np.load('saved_examples.npz')['idx']
    with open('saved_labels.pkl', 'rb') as f:
        saved_labels = pk.load(f)

    # drop 3 original trajectories. 23 --> 20
    saved_labels = saved_labels[:11] + saved_labels[13:18] + saved_labels[19:]
    saved_idx = saved_idx[list(range(11)) + list(range(13,18)) + list(range(19,23))]

    saved_labels = [saved_labels[6], saved_labels[7], saved_labels[17]]
    saved_idx = saved_idx[[6, 7, 17]]

    sigma_ls = [0.040, 0.035, 0.010, 0.005]
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'VP', f'fps{args.fps}')
    if not os.path.isdir(dataset_path): os.makedirs(dataset_path)
    failure_counter, nonfailure_counter = int(args.base/2), int(args.base/2)
    num_per_traj = int(args.number/len(saved_idx))
    num_per_traj_per_cls = int(num_per_traj/2)
    print('----------------------------------------')
    print(f'VP: Generating {args.number} episodes based on {len(saved_idx)} trajectories, {num_per_traj}/traj.') 
    print(f'VP: {num_per_traj_per_cls} falls + {num_per_traj_per_cls} non-falls = {num_per_traj}/traj.')
    if (num_per_traj_per_cls <= 0) or (num_per_traj%2 != 0) or (args.number%len(saved_idx) != 0):
        print('Improper size.')
    # if args.base < len(os.listdir(dataset_path)): 
    #     raise Exception('Warning: the args.base < the actual number of files. Continue collecting data may overwrite existing files.\nComment out this Exception for skipping safty check.')
    print('----------------------------------------')

    # collect data for the VP dataset
    for c, (n, l) in enumerate(zip(saved_idx, saved_labels), start=1):
        f_counter, n_counter, t = 0, 0, 0
        while (f_counter < num_per_traj_per_cls) or (n_counter < num_per_traj_per_cls):
            flag = False
            env = PoppyHumanoidEnv(args.fps, pb.POSITION_CONTROL, show=False, step_hook=hook_for_data_collection)
            step_array = array_from_design(env, all_samples[lex[n]])
            _ = run_traj(env, step_array, num_cycles, timestep, data_collection=True, sigma=sigma_ls[t%len(sigma_ls)])
            if (n_counter < num_per_traj_per_cls) and (np.asarray(env.obs.labels, dtype='bool') == 0).all(): # non-failure episode
                print((np.asarray(env.obs.labels, dtype='bool') == 0).all())
                print(np.asarray(env.obs.labels, dtype='bool'))
                filename = 'n_' + str(nonfailure_counter+n_counter).zfill(4) + f'_{l}.npz'
                n_counter += 1
                flag = True
                print(filename)
                print('-------------------')
            elif (f_counter < num_per_traj_per_cls) and (np.asarray(env.obs.labels, dtype='bool')[:90] == 0).all(): # failure episode
                print((np.asarray(env.obs.labels, dtype='bool')[:90] == 0).all())
                print(np.asarray(env.obs.labels, dtype='bool'))
                filename = 'f_' + str(nonfailure_counter+f_counter).zfill(4) + f'_{l}.npz'
                f_counter += 1
                flag = True
                print(filename)
                print('-------------------')
            if flag:
                np.savez_compressed(os.path.join(dataset_path, filename),
                                    # imgs=rgba2rgb(np.asarray(env.obs.imgs, dtype='uint8')),
                                    # hc_z=np.array(env.obs.hc_z, dtype='float32'),
                                    # ajp=np.asarray(env.obs.actual_joint_pos, dtype='float32'),
                                    # traj=np.asarray(env.obs.trajectory, dtype='float32'),
                                    pos=np.asarray(env.obs.pos, dtype='float32'),
                                    # ori=np.asarray(env.obs.ori, dtype='float32'),
                                    labels=np.asarray(env.obs.labels, dtype='bool')
            )
            env.close()
            t += 1
        print(f'{c}/{len(saved_labels)}: design {n}, {l} is done.')
        failure_counter += f_counter
        nonfailure_counter += n_counter
