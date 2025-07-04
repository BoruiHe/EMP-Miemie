import pybullet as pb
import numpy as np


class observer():
    def __init__(self) -> None:
        self.imgs = []
        self.trajectory = []
        self.actual_joint_pos = []
        self.pos = []
        self.ori = []
        self.hc_z = [] # z coordinates of the head camera
        self.labels = [] # frame-wise labels

def array_from_design(env, design):
    # design[0]: init pos
    # design[1:T-1]: finite difference acceleration
    init_pos = design[0]
    last_pos = env.mirror_position(init_pos)

    acc = design[1:]
    vel = acc.cumsum(axis=0) # initial velocity zero
    pos = init_pos + vel.cumsum(axis=0)

    return np.concatenate((pos, last_pos[np.newaxis]), axis=0)

def make_cycle_traj(env, step_traj, num_cycles):
    traj = []
    for cycle in range(num_cycles):
        traj.extend(step_traj)
        traj.extend([
            # (dur, env.angle_dict(env.mirror_position(env.angle_array(pos))))
            (dur, env.mirror_dict(pos))
            for (dur, pos) in step_traj])
    return traj

def traj_from_array(env, arr, num_cycles, timestep):
    # assumes arr angles are in degrees
    step_traj = [(timestep, env.angle_dict(pos, convert=False)) for pos in arr]
    cycle_traj = make_cycle_traj(env, step_traj, num_cycles)
    init_dict = env.mirror_dict(step_traj[-1][1])
    return init_dict, cycle_traj

def run_traj(env, step_array, num_cycles, timestep, data_collection=False, sigma=0):
    # step_array[t]: angle array at timestep t (1-based) for a single footstep.
    # it is mirrored for the second foot of the cycle
    # the cycle is repeated the requested number of times
    # step_array[-1] is mirrored for the initial pose before the trajectory

    init_dict, cycle_traj = traj_from_array(env, step_array, num_cycles, timestep)

    # start fresh
    env.reset()

    # check ankle positions in initial position
    init_pos = env.angle_array(init_dict)
    env.set_position(init_pos)
    init_base, init_quat = pb.getBasePositionAndOrientation(env.robot_id)
    ankle_z = []
    for lr in "lr":
        state = pb.getLinkState(env.robot_id, env.joint_index[lr + "_ankle_y"])
        ankle_z.append(state[0][2])
    ankle_z = min(ankle_z)

    # for stable init poses with feet flat on the ground, ankle_z should be ~.01265
    # so adjust base accordingly and stable poses will settle
    dz = .01265 - ankle_z
    init_base = init_base[:2] + (init_base[2] + dz,)
    
    # drop from slightly higher to avoid ground collision
    drop_base = init_base[:2] + (init_base[2] + .003,)
    pb.resetBasePositionAndOrientation(env.robot_id, drop_base, init_quat)

    # Do not record any data during the settle phase.
    env.track_trajectory([(2, init_dict)], datacollection=data_collection)

    # try performing trajectory with another settle at end
    positions = env.track_trajectory(cycle_traj + [(2, init_dict)], datacollection=data_collection, sigma=sigma)

    # get state after settle
    post_base, post_quat = pb.getBasePositionAndOrientation(env.robot_id)
    post_linv, post_angv = pb.getBaseVelocity(env.robot_id)

    # return results
    return positions, init_base, init_quat, post_base, post_quat, post_linv, post_angv

def add_perturbation(target, sigma, steps, sampling_rate):
    '''
    add perturbation on each waypoint (= elements in the cycle_traj, see run_traj())
    '''
    num_joints = len(target)
    if sigma > 0.01:
        factor = 1.5/(1+2.71828**(-(steps // (240 // sampling_rate))+90))
    else:
        factor = 1
    perturbation = np.random.normal(0, sigma, num_joints)
    return target + factor*perturbation
    
def create_video_from_images(images, output_video_path, fps):
    '''
    This function convert a sequence of RGBA images into a video for visualization.
    Input: (N, H, W, C)
    '''
    import cv2
    N, height, width, _ = images.shape
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

    for i in range(N):
        image = cv2.cvtColor(images[i], cv2.COLOR_RGBA2RGB)
        image = image[:,:,::-1]
        video.write(image)

    cv2.destroyAllWindows()
    video.release()
    print(f'Video saved to {output_video_path}')

def rgba2rgb(rgba, background=(255,255,255)):
    '''
    This function convert a sequence of RGBA images into that of RGB images.
    Input: (N, H, W, C)
    '''
    n, H, W, _ = rgba.shape
    rgb = np.zeros((n, H, W, 3), dtype='float32')
    r, g, b, a = rgba[:,:,:,0], rgba[:,:,:,1], rgba[:,:,:,2], rgba[:,:,:,3]
    a = np.asarray(a, dtype='float32') / 255

    rgb[:,:,:,0] = r * a + (1.0 - a) * background[0]
    rgb[:,:,:,1] = g * a + (1.0 - a) * background[1]
    rgb[:,:,:,2] = b * a + (1.0 - a) * background[2]

    return np.asarray(rgb, dtype='uint8')

def hook_for_data_collection(env, action, failure):
    # # image capture
    # rgba = env.get_camera_image()
    # env.obs.imgs.append(rgba)

    # actual joints position capture
    env.obs.actual_joint_pos.append(env.get_position())

    # # trajectory capture
    # env.obs.trajectory.append(action)

    # # z-coordinates for plotting z vs. timesteps
    # env.obs.hc_z.append(pb.getLinkState(env.robot_id, env.joint_index['head_cam'], computeLinkVelocity=1)[0][2]) # getLinkState()[0] is (x,y,z), head_cam index = 17

    # link positions and orientation (including the base at the end)
    # pb.getBasePositionAndOrientation(env.robot_id) --> (pos, ori).sss
    # pb.getLinkStates(env.robot_id, range(len(env.joint_index)) --> (pos, ori, ...).
    # The ori is quaternion (x,y,z,w).
    states = pb.getLinkStates(env.robot_id, range(len(env.joint_index)), computeLinkVelocity=0)
    states = states + (pb.getBasePositionAndOrientation(env.robot_id), )
    # pos
    env.obs.pos.append(np.array([state[0] for state in states], dtype='float32'))
    # ori
    env.obs.ori.append(np.asarray([state[1] for state in states], dtype='float32'))# pb.getBasePositionAndOrientation(env.robot_id) (pos, ori). The ori is quaternion.

    # frame-wise labels
    if failure:
        env.obs.labels.append(np.True_) # True --> fall; False --> non-fall
    else:
        env.obs.labels.append(env.obs.pos[-1][17][2] < 0.7) # True --> fall; False --> non-fall