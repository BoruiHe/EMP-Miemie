from utils import run_traj, array_from_design, hook_for_data_collection, rgba2rgb, create_video_from_images
from humanoid import PoppyHumanoidEnv
import matplotlib.pyplot as pt
import pybullet as pb
import pickle as pk
import numpy as np
import os


skip_statistics = True
select = False # False for not selecting samples, if you already got lists of indices and labels, True for selecting samples and entering labels.
visualization = True
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

if not skip_statistics:
    obj_names = ("progress", "lateral", "spin", "velocity", "jerk")
    fig, axs = pt.subplots(1, len(obj_names), figsize=(len(obj_names)*3, 3))
    for i, (name, ax) in enumerate(zip(obj_names, axs)):
        ax.hist(all_objectives[:,i])
        ax.set_xlabel(name)
    axs[0].set_ylabel("Frequency")
    pt.tight_layout()
    pt.show()

    pt.figure()
    pt.plot(all_objectives[:,0], all_objectives[:,4], 'k.')
    pt.xlabel("progress")
    pt.ylabel("jerk")
    pt.show()

print(all_objectives.shape)

# select interesting examples
if select:
    env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=True, step_hook=None)
    cam = (1.200002670288086,
        15.999960899353027,
        -31.799997329711914,
        (-0.010284600779414177, -0.012256712652742863, 0.14000000059604645))
    pb.resetDebugVisualizerCamera(*cam)

    print([env.joint_name[i] for i in range(len(env.joint_name))])
    
    saved_idx = []
    saved_labels = []
    saved_idx = np.load('saved_examples.npz')['idx']
    for n in saved_idx: # range(all_samples.shape[0]):
        print(f"design {n}: {all_objectives[lex[n]].round(4)}")
        step_array = array_from_design(env, all_samples[lex[n]])
        result = run_traj(env, step_array, num_cycles, timestep)
        positions = result[0]

        label = input('enter your label:')
        if label != '':
            saved_idx.append(n)
            saved_labels.append(label)
    env.close()
    # saved_idx = np.savez_compressed('saved_examples.npz', idx=np.array(saved_idx))
    # with open('saved_labels.pkl', 'wb') as f:
    #     pk.dump(saved_labels, f)
else:
    saved_idx = np.load('saved_examples.npz')['idx']
    with open('saved_labels.pkl', 'rb') as f:
        saved_labels = pk.load(f)

# collect data for the VP dataset
if visualization:
    for n, l in zip(saved_idx, saved_labels):
        env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=False, step_hook=hook_for_data_collection)
        print(f"design {n}: {all_objectives[lex[n]].round(4)}")
        step_array = array_from_design(env, all_samples[lex[n]])
        result = run_traj(env, step_array, num_cycles, timestep, data_collection=True)
        positions = result[0]
        np.savez_compressed(os.path.join('videos', l+'.npz'), imgs=rgba2rgb(np.array(env.obs.imgs, dtype='uint8')))
        create_video_from_images(np.array(env.obs.imgs), os.path.join('videos', l+'.avi'), fps=30)
        env.close()