import numpy as np
import pybullet as pb
import os, time, random
from pybullet_data import getDataPath
from utils import observer, add_perturbation


class PoppyHumanoidEnv():
    def __init__(self,
        sampling_rate,
        control_mode=pb.POSITION_CONTROL,
        timestep=1/240,
        control_period=1,
        show=True,
        step_hook=None,
    ):
        self.sampling_rate = sampling_rate
        self.step_hook = step_hook
        self.control_mode = control_mode
        self.timestep = timestep
        self.control_period = control_period
        self.show = show
        if self.step_hook:
            self.obs = observer()
            self.num_step = 0

        self.client_id = pb.connect(pb.GUI if show else pb.DIRECT)
        if show: pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)
        pb.setTimeStep(timestep)
        pb.setGravity(0, 0, -9.81)
        pb.setAdditionalSearchPath(getDataPath())
        plane = pb.loadURDF('plane.urdf')
        # pb.changeVisualShape(plane, -1, textureUniqueId=pb.loadTexture(os.path.join('humanoid/furniture/floor_textures/carpet.jpg')))

        '''
        load 5 cubes as background. box size is (1,1,1). z_pos(height) = globalScaling/2
        back, front, right, left, top(sky)
        '''
        gs = 10
        pos_ls = [[0,gs,gs/2], [0,-gs,gs/2], [gs,0,gs/2], [-gs,0,gs/2], [0,0,3*gs/2]]
        texture_path = os.path.join('humanoid', 'wall_textures')
        texture_files = os.listdir(texture_path)
        texture_files.remove('sky.png')
        texture_files = [f for f in texture_files if os.path.isfile(os.path.join(texture_path, f))]
        texture_files = random.sample(texture_files, 4) + ['sky.png']
        for i in range(5):
            cubeid = pb.loadURDF('cube.urdf', useFixedBase=1, globalScaling=gs)
            pb.resetBasePositionAndOrientation(cubeid, pos_ls[i], pb.getQuaternionFromEuler([np.pi / 2, 0, 0]))
            pb.changeVisualShape(cubeid, -1, textureUniqueId=pb.loadTexture(os.path.join(texture_path, texture_files[i])))
        record = []
        obj = pb.loadURDF('humanoid/furniture/barbecue_1/barbecue.urdf', useFixedBase=True, globalScaling=0.5)
        denominator, x, y = self.whereami(record)
        pb.resetBasePositionAndOrientation(obj, [x,y,0], pb.getQuaternionFromEuler([0, 0, np.pi/denominator]))

        obj = pb.loadURDF('humanoid/furniture/bed_1/bed.urdf', useFixedBase=True, globalScaling=1)
        denominator, x, y = self.whereami(record)
        pb.resetBasePositionAndOrientation(obj, [x,y,0], pb.getQuaternionFromEuler([0, 0, np.pi/denominator]))

        obj = pb.loadURDF('humanoid/furniture/bookshelf_1/bookshelf.urdf', useFixedBase=True, globalScaling=1)
        denominator, x, y = self.whereami(record)
        pb.resetBasePositionAndOrientation(obj, [x,y,0], pb.getQuaternionFromEuler([0, 0, np.pi/denominator]))

        obj = pb.loadURDF('humanoid/furniture/barrel_1/barrel.urdf', useFixedBase=True, globalScaling=0.5)
        denominator, x, y = self.whereami(record)
        pb.resetBasePositionAndOrientation(obj, [x,y,0], pb.getQuaternionFromEuler([0, 0, np.pi/denominator]))

        obj = pb.loadURDF('humanoid/furniture/piano_1/piano.urdf', useFixedBase=True, globalScaling=1)
        denominator, x, y = self.whereami(record)
        pb.resetBasePositionAndOrientation(obj, [x,y,0], pb.getQuaternionFromEuler([0, 0, np.pi/denominator]))

        obj = pb.loadURDF('humanoid/furniture/table_1/table.urdf', useFixedBase=True, globalScaling=0.5)
        denominator, x, y = self.whereami(record)
        pb.resetBasePositionAndOrientation(obj, [x,y,0], pb.getQuaternionFromEuler([0, 0, np.pi/denominator]))
        
        fpath = os.path.join(os.getcwd(), 'humanoid')
        pb.setAdditionalSearchPath(fpath)
        self.robot_id = pb.loadURDF(
            'poppy_humanoid.pybullet.urdf',
            basePosition = (0, 0, .43),
            baseOrientation = pb.getQuaternionFromEuler((0,0,0))
        )

        self.num_joints = pb.getNumJoints(self.robot_id)
        self.joint_name, self.joint_index, self.joint_fixed = {}, {}, {}
        for i in range(self.num_joints):
            info = pb.getJointInfo(self.robot_id, i)
            name = info[1].decode('UTF-8')
            self.joint_name[i] = name
            self.joint_index[name] = i
            self.joint_fixed[i] = (info[2] == pb.JOINT_FIXED)
        
        self.initial_state_id = pb.saveState(self.client_id)

    # Get mirrored version of position across left/right halves of body
    def mirror_position(self, position):
        mirrored = np.empty(len(position))
        for i, name in self.joint_name.items():
            sign = 1 if name[-2:] == "_y" else -1 # don't negate y-axis rotations
            mirror_name = name # swap right and left
            if name[:2] == "l_": mirror_name = "r_" + name[2:]
            if name[:2] == "r_": mirror_name = "l_" + name[2:]
            mirrored[self.joint_index[mirror_name]] = position[i] * sign        
        return mirrored

    # version of mirror_position for angle dict input and output
    def mirror_dict(self, position_dict):
        mirrored = {}
        for (name, angle) in position_dict.items():
            sign = 1 if name[-2:] == "_y" else -1 # don't negate y-axis rotations
            mirror_name = name # swap right and left
            if name[:2] == "l_": mirror_name = "r_" + name[2:]
            if name[:2] == "r_": mirror_name = "l_" + name[2:]
            mirrored[mirror_name] = angle * sign
        return mirrored
    
    def step_simple(self, action):
        pb.setJointMotorControlArray(
            self.robot_id,
            jointIndices = range(len(self.joint_index)),
            controlMode = self.control_mode,
            targetPositions = action,
            targetVelocities = [0]*len(action),
            positionGains = [.25]*len(action), # important for constant position accuracy
        )
        for _ in range(self.control_period):
            pb.stepSimulation()
    
    # override step and goto for humanoid walking
    def step(self, action, counter=None, failure=False, datacollect=False):
        pb.setJointMotorControlArray(
            self.robot_id,
            jointIndices = range(len(self.joint_index)),
            controlMode = self.control_mode,
            targetPositions = action,
        )
        pb.stepSimulation()
        if datacollect:
            if self.num_step % (240 // self.sampling_rate) == 0:
                if isinstance(counter, int):
                    counter += 1
                self.step_hook(self, action, failure)
                # if fall, all labels in the duration are True. True --> fall; False --> non-fall
                if (self.obs.pos[-1][17][2] < 0.7) and (not failure):
                    self.obs.labels[-counter:] = [np.True_ for _ in self.obs.labels[-counter:]]
                    counter = None
                    failure = True
            self.num_step += 1
    
    # Get image from head camera
    def get_camera_image(self):

        # Get current pose of head camera
        # link index should be same as parent joint index?
        state = pb.getLinkState(self.robot_id, self.joint_index["head_cam"])
        pos, quat = state[:2]
        M = np.array(pb.getMatrixFromQuaternion(quat)).reshape((3,3)) # local z-axis is third column

        # Calculate camera target and up vector
        camera_position = tuple(p + d for (p,d) in zip(pos, .1*M[:,2]))
        target_position = tuple(p + d for (p,d) in zip(pos, .4*M[:,2]))
        up_vector = tuple(M[:,1])
        
        # Capture image
        width, height = 128, 96
        view = pb.computeViewMatrix(
            cameraEyePosition = camera_position,
            cameraTargetPosition = target_position, # focal point
            cameraUpVector = up_vector,
        )
        proj = pb.computeProjectionMatrixFOV(
            fov = 90,
            aspect = height/width,
            nearVal = 0.01,
            farVal = 18.0,
        )
        # rgba shape is (height, width, 4)
        _, _, rgba, _, _ = pb.getCameraImage(
            width, height, view, proj,
            flags = pb.ER_NO_SEGMENTATION_MASK) # not much speed difference
        return rgba
    
    def reset(self):
        pb.restoreState(stateId = self.initial_state_id)
    
    def close(self):
        pb.disconnect()

    # get/set joint angles as np.array
    def get_position(self):
        states = pb.getJointStates(self.robot_id, range(len(self.joint_index)))
        return np.array([state[0] for state in states])
    
    def set_position(self, position):
        for p, angle in enumerate(position):
            pb.resetJointState(self.robot_id, p, angle)

    # convert a pypot style dictionary {... name:angle ...} to joint angle array
    # if convert == True, convert from degrees to radians
    def angle_array(self, angle_dict, convert=True):
        angle_array = np.zeros(self.num_joints)
        for name, angle in angle_dict.items():
            angle_array[self.joint_index[name]] = angle
        if convert: angle_array *= np.pi / 180
        return angle_array
    
    # convert back from dict to array
    def angle_dict(self, angle_array, convert=True):
        return {
            name: angle_array[j] * (180/np.pi if convert else 1)
            for j, name in enumerate(self.joint_index)}
    
    # poppy_wrapper-style trajectory tracker
    def track_trajectory(self, trajectory, binsize=None, overshoot=None, ms_rpms = 0.165, hang=False, datacollection=False, sigma=0):
        # trajectory = [..., (duration (sec), waypoint) ...]
        # waypoint[name] = angle (deg)
        # returns buffers[t,a] = angle a at timestep t in radians
        
        # pybul doesn't have fast array-version maxvel and pos ctrl is unrealistically fast
        # linearly interpolate waypoints to throttle speed

        buffers = []
        failure = False     
        for (duration, waypoint) in trajectory:
            current = self.get_position()
            target = self.angle_array(waypoint)
            if datacollection and (duration == 0.2):
                target = add_perturbation(target, sigma, self.num_step, self.sampling_rate)

            num_steps = int(duration / (self.timestep * self.control_period) + 1)
            weights = np.linspace(0, 1, num_steps).reshape(-1,1)
            interp = weights * target + (1 - weights) * current

            positions = np.empty((num_steps, self.num_joints))
            if (duration == 0.2) and (not failure):
                counter = 0
            for a, action in enumerate(interp):
                if datacollection:
                    if duration == 0.2:
                        self.step(action, counter, failure, datacollect=True)
                    else:
                        self.step(action, datacollect=False)
                else:
                    self.step_simple(action)
                positions[a] = self.get_position()
                
                if hang: input('..')
            if (duration == 0.2) and counter is not None:
                counter -= len(interp)
            buffers.append(positions)

        return np.concatenate(buffers, axis=0)
    
    # Run IK, accounting for fixed joints
    def inverse_kinematics(self, link_indices, target_positions, num_iters=1000):

        angles = pb.calculateInverseKinematics2(
            self.robot_id,
            link_indices,
            target_positions,
            # residualThreshold=1e-4, # default 1e-4 not enough for ergo jr
            maxNumIterations=num_iters, # default 20 usually not enough
        )

        a = 0
        result = self.get_position()
        for r in range(self.num_joints):
            if not self.joint_fixed[r]:
                result[r] = angles[a]
                a += 1
        
        return result

    def whereami(self, list_of_pos):
        '''
        This method samples proper coordinates (X, Y) for loaded furniture.
        '''
        ori_denominator = np.random.uniform(0, np.pi)
        if len(list_of_pos) == 0:
            pos_x = random.uniform(-4.5, 4.5)
            pos_y = random.uniform(-4.5, -2)
            list_of_pos.append((pos_x,pos_y))
        else:
            pos_x = random.uniform(-4.5, 4.5)
            pos_y = random.uniform(-4.5, -2)
            while not np.asarray([((pos_x-pos[0])**2 + (pos_y-pos[1])**2)>=1 for pos in list_of_pos]).all():
                pos_x = random.uniform(-4.5, 4.5)
                pos_y = random.uniform(-4.5, -2)
            list_of_pos.append((pos_x,pos_y))
        return ori_denominator, pos_x, pos_y