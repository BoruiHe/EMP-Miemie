from utils import run_traj, array_from_design, hook_for_data_collection, rgba2rgb
from humanoid import PoppyHumanoidEnv
import pybullet as pb
import pickle as pk
import numpy as np
import os


fps = 16
timestep = .2
num_cycles = 2 # total number of cycles per episode; each cycle consists of two footsteps (one for each foot)

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

# collect data for the VP dataset
sigma_ls = [0.040, 0.035, 0.010, 0.005]
for n, l in zip(saved_idx, saved_labels):
    folder_path = os.path.join(f'fps{fps}', f'{l}')
    if not os.path.isdir(folder_path): os.makedirs(folder_path)
    print(f'design {n}: {all_objectives[lex[n]].round(4)}')
    for i in range(len(sigma_ls)*30):
        env = PoppyHumanoidEnv(fps, pb.POSITION_CONTROL, show=False, step_hook=hook_for_data_collection)
        step_array = array_from_design(env, all_samples[lex[n]])
        result = run_traj(env, step_array, num_cycles, timestep, data_collection=True, sigma=sigma_ls[i%len(sigma_ls)])
        positions = result[0]
        np.savez_compressed(os.path.join(folder_path, f'{sigma_ls[i%len(sigma_ls)]}_{i}.npz'), 
                            # imgs=rgba2rgb(np.asarray(env.obs.imgs, dtype='uint8')),
                            hc_z=np.array(env.obs.hc_z, dtype='float32'),
                            # ajp=np.asarray(env.obs.actual_joint_pos, dtype='float32'),
                            # traj=np.asarray(env.obs.trajectory, dtype='float32'),
                            # linkpos=np.asarray(env.obs.linkpos, dtype='float32'),
                            # ori=np.asarray(env.obs.ori, dtype='float32'),
                            # labels=np.asarray(env.obs.labels, dtype='bool')
        )
        env.close()