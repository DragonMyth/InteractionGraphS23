
import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import numpy as np
import math
import time
import copy
from enum import Enum
from collections import deque

from basecode.bullet import bullet_client
from basecode.bullet import bullet_utils as bu
from basecode.motion import kinematics_simple as kinematics
from basecode.math import mmMath
from basecode.utils import basics
from basecode.rl import action as ac

import env_humanoid_tracking
import sim_obstacle

import sim_agent

import pickle
import gzip

import interaction
import render_module as rm

import pose_embedding
import torch

time_checker_auto_play = basics.TimeChecker()

def load_motions(motion_files, skel, char_info, verbose):
    assert motion_files is not None
    motion_file_names = []
    for names in motion_files:
        head, tail = os.path.split(names)
        motion_file_names.append(tail)
    if isinstance(motion_files[0], str):
        motion_dict = {}
        motion_all = []
        for i, file in enumerate(motion_files):
            ''' If the same file is already loaded, do not load again for efficiency'''
            if file in motion_dict:
                motion = motion_dict[file]
            else:
                if file.endswith('bvh'):
                    motion = kinematics.Motion(skel=skel,
                                               file=file, 
                                               scale=1.0, 
                                               load_skel=False,
                                               v_up_skel=char_info.v_up, 
                                               v_face_skel=char_info.v_face, 
                                               v_up_env=char_info.v_up_env)
                elif file.endswith('bin'):
                    motion = pickle.load(open(file, "rb"))
                elif file.endswith('gzip') or file.endswith('gz'):
                    with gzip.open(file, "rb") as f:
                        motion = pickle.load(f)
                else:
                    raise Exception('Unknown Motion File Type')
                if verbose: 
                    print('Loaded: %s'%file)
            motion_all.append(motion)
    elif isinstance(motion_files[0], kinematics.Motion):
        motion_all = motion_files
    else:
        raise Exception('Unknown Type for Reference Motion')

    return motion_all, motion_file_names

class Action(ac.ActionBase):
    def initialize(self, args):
        normalizer_pose = args[0]
        self.val_max = np.hstack([normalizer_pose.norm_val_max])
        self.val_min = np.hstack([normalizer_pose.norm_val_min])
        self.normalizer_pose = normalizer_pose

        self.s_idx_pose = 0
        self.e_idx_pose = self.s_idx_pose + self.normalizer_pose.dim

    def get_pose(self, val):
        return val[self.s_idx_pose:self.e_idx_pose]

    def norm_to_real_pose(self, val):
        pose_norm = self.get_pose(val)
        pose_real = self.normalizer_pose.norm_to_real(pose_norm)
        return pose_real

    def norm_to_real(self, val):
        pose = self.norm_to_real_pose(val)
        return np.hstack([pose])

    def real_to_norm_pose(self, val):
        pose_real = self.get_pose(val)
        pose_norm = self.normalizer_pose.real_to_norm(pose_real)
        return pose_norm

    def real_to_norm(self, val):
        pose = self.real_to_norm_pose(val)
        return np.hstack([pose])

class Env(object):
    class Task(Enum):
        Imitation=0
        Heading=1
        Carry=2
        Dribble=3
        Fight=4
        Chase=5
        @classmethod
        def from_string(cls, string):
            if string=="imitation": return cls.Imitation
            if string=="heading": return cls.Heading
            if string=="carry": return cls.Carry
            if string=="dribble": return cls.Dribble
            if string=="fight": return cls.Fight
            if string=="chase": return cls.Chase
            raise NotImplementedError
    class ActionMode(Enum):
        Absolute=0 # Use an absolute posture as an action
        Relative=1 # Use a relative posture from a reference posture as an action
        @classmethod
        def from_string(cls, string):
            if string=="absolute": return cls.Absolute
            if string=="relative": return cls.Relative
            raise NotImplementedError
    class StateChoice(Enum):
        Body=0
        Task=1
        @classmethod
        def from_string(cls, string):
            if string=="body": return cls.Body
            if string=="task": return cls.Task
            raise NotImplementedError
    class EarlyTermChoice(Enum):
        SimDiv=0
        SimWindow=1
        TaskComplete=2
        Falldown=3
        RootFail=4
        LowReward=5
        @classmethod
        def from_string(cls, string):
            if string=="sim_div": return cls.SimDiv
            if string=="sim_window": return cls.SimWindow
            if string=="task_complete": return cls.TaskComplete
            if string=="falldown": return cls.Falldown
            if string=="root_fail": return cls.RootFail
            if string=="low_reward": return cls.LowReward
            raise NotImplementedError
    def __init__(self, config):
        ''' Configure misc information '''
        task = config['task']
        if isinstance(task, str):
            self._tasks = [Env.Task.from_string(task)]
        elif isinstance(task, list):
            assert len(task) > 0
            self._tasks = [Env.Task.from_string(s) for s in task]
        else:
            raise Exception

        project_dir      = config['project_dir']
        char_info_module = config['character'].get('char_info_module')
        sim_char_file    = config['character'].get('sim_char_file')
        base_motion_file = config['character'].get('base_motion_file')
        ref_motion_scale = config['character'].get('ref_motion_scale')
        environment_file = config['character'].get('environment_file')
        ref_motion_file  = config['character'].get('ref_motion_file')
        self_collision   = config['character'].get('self_collision')
        actuation        = config['character'].get('actuation')
        
        ''' Append project_dir to the given file path '''
        
        if project_dir is not None:
            for i in range(len(char_info_module)):
                char_info_module[i] = os.path.join(project_dir, char_info_module[i])
                sim_char_file[i]    = os.path.join(project_dir, sim_char_file[i])
                base_motion_file[i] = os.path.join(project_dir, base_motion_file[i])
            if environment_file is not None:
                for i in range(len(environment_file)):
                    environment_file[i] = os.path.join(project_dir, environment_file[i])
            if Env.Task.Imitation in self._tasks:
                for i in range(len(ref_motion_file)):
                    for j in range(len(ref_motion_file[i])):
                        ref_motion_file[i][j] = os.path.join(project_dir, ref_motion_file[i][j])

        ''' Create a base tracking environment '''

        self._base_env = env_humanoid_tracking.Env(
            fps_sim=config['fps_sim'],
            fps_con=config['fps_con'],
            verbose=config['verbose'],
            char_info_module=char_info_module,
            sim_char_file=sim_char_file,
            ref_motion_scale=ref_motion_scale,
            self_collision=self_collision,
            contactable_body=config['early_term']['falldown_contactable_body'],
            actuation=actuation,
            )

        self._pb_client = self._base_env._pb_client
        self._dt_con = 1.0/config['fps_con']

        self._num_agent = self._base_env._num_agent
        assert self._num_agent == len(base_motion_file)

        ''' Create Simulated Agents '''
        self._sim_agent = [self._base_env._agent[i] for i in range(self._num_agent)]

        self._v_up = self._base_env._v_up

        ''' State '''
        self._state_choices = [Env.StateChoice.from_string(s) for s in config['state']['choices']]

        ''' Early Terminations '''
        self._early_term_choices = [Env.EarlyTermChoice.from_string(s) for s in config['early_term']['choices']]
        self._early_term_choices += [Env.EarlyTermChoice.SimDiv, Env.EarlyTermChoice.SimWindow]

        self._reward_fn_def = config['reward']['fn_def']
        self._reward_fn_map = config['reward']['fn_map']
        self._reward_names = [self.get_reward_names(
            self._reward_fn_def[self._reward_fn_map[i]]) for i in range(self._num_agent)]

        '''
        Check the existence of reward definitions, which are defined in our reward map
        '''
        assert len(self._reward_fn_map) == self._num_agent
        for key in self._reward_fn_map:
            assert key in self._reward_fn_def.keys()

        self._verbose = config['verbose']

        if Env.EarlyTermChoice.LowReward in self._early_term_choices:
            self._et_low_reward_thres = config['early_term']['low_reward_thres']
            self._rew_queue = self._num_agent * [None]
            for i in range(self._num_agent):
                self._rew_queue[i] = deque(maxlen=int(1.0/self._dt_con))
        
        ''' The environment automatically terminates after 'sim_window' seconds '''
        self._sim_window_time = config['sim_window']
        ''' 
        The environment continues for "eoe_margin" seconds after end-of-episode is set by TRUE.
        This is useful for making the controller work for boundaries of reference motions
        '''
        self._eoe_margin = config['eoe_margin']

        self._action_type = Env.ActionMode.from_string(config['action']['type'])
        self._pose_embedding = None
        
        if config['action'].get('pose_embedding'):
            import pose_embedding
            self._pose_embedding = pose_embedding.Autoencoder(dim_feature, dim_embedding)

        self._imit_window = [0.05, 0.15]

        if Env.Task.Fight in self._tasks or Env.Task.Chase in self._tasks:
            self._ground_height = 1.0
        else:
            self._ground_height = 0.0

        ''' Load Skeleton for Reference Motion '''
        self._ref_motion_skel = []
        self._ref_poses, self._ref_vels = [], []
        self._ref_xform_root = []
        for i in range(self._num_agent):
            m = kinematics.Motion(file=base_motion_file[i],
                                  scale=1.0, 
                                  load_motion=True,
                                  v_up_skel=self._sim_agent[i]._char_info.v_up, 
                                  v_face_skel=self._sim_agent[i]._char_info.v_face, 
                                  v_up_env=self._sim_agent[i]._char_info.v_up_env, 
                                  )
            self._ref_motion_skel.append(m.skel)
            self._ref_poses.append(m.get_pose_by_frame(0))
            self._ref_vels.append(m.get_velocity_by_frame(0))
            
            T_face = self._ref_poses[-1].get_facing_transform(self._ground_height)
            T_root = self._ref_poses[-1].get_root_transform()
            self._ref_xform_root.append(np.dot(mmMath.invertSE3(T_face), T_root))

        ''' Create Kinematic Agents '''
        self._kin_agent = []
        for i in range(self._num_agent):
            self._kin_agent.append(
                sim_agent.SimAgent(pybullet_client=self._base_env._pb_client, 
                                   model_file=sim_char_file[i],
                                   char_info=self._sim_agent[i]._char_info,
                                   ref_scale=ref_motion_scale[i],
                                   self_collision=config['self_collision'],
                                   kinematic_only=True,
                                   verbose=config['verbose']))

        ''' Configure action space '''
        self._action_info = []
        for i in range(self._num_agent):
            dim = self._sim_agent[i].get_num_dofs()
            normalizer_pose = basics.Normalizer(real_val_max=config['action']['range_max']*np.ones(dim),
                                                real_val_min=config['action']['range_min']*np.ones(dim),
                                                norm_val_max=config['action']['range_max_pol']*np.ones(dim),
                                                norm_val_min=config['action']['range_min_pol']*np.ones(dim))
            self._action_info.append(Action(dim=normalizer_pose.dim, init_args=[normalizer_pose]))

        if Env.Task.Imitation in self._tasks:
            ''' Load Reference Motion '''
            self._ref_motion_all = []
            self._ref_motion_file_names = []
            for i in range(self._num_agent):
                ref_motion_all, ref_motion_file_names = \
                    load_motions(ref_motion_file[i], 
                                 self._ref_motion_skel[i],
                                 self._sim_agent[i]._char_info,
                                 self._verbose)
                self._ref_motion_all.append(ref_motion_all)
                self._ref_motion_file_names.append(ref_motion_file_names)

            self._ref_motion = self.sample_ref_motion()
            self._ref_motion_scale = ref_motion_scale

        if Env.Task.Heading in self._tasks:
            self._target_vel_dir = self._num_agent * [None]
            self._target_vel_angle = self._num_agent * [None]
            self._target_vel_len = self._num_agent * [None]
        
        if Env.Task.Fight in self._tasks:
            tf_ring = mmMath.R2Q(mmMath.I_SO3())
            self._ring_id = self._pb_client.loadURDF(
                environment_file[0], [0, 0, 0.5], tf_ring, useFixedBase=True)
            self._ring_size = [3.0]
            # ring boundaries
            r = self._ring_size[0]
            h = self._ground_height
            self._ring_boundaries = [np.array([r*math.cos(theta), r*math.sin(theta), h]) \
                for theta in np.linspace(0.0, 2*math.pi, num=8, endpoint=False)]
        
        if Env.Task.Chase in self._tasks:
            tf_ring = mmMath.R2Q(mmMath.I_SO3())
            self._ring_id = self._pb_client.loadURDF(
                environment_file[0], [0, 0, 0.5], tf_ring, useFixedBase=True)
            self._ring_size = [8.0, 8.0]
            # 8 corners of the ring
            w_x, w_y, h = self._ring_size[0], self._ring_size[1] ,self._ground_height
            self._ring_boundaries = [np.array([0.5*w_x, 0.5*w_y, h]),
                                     np.array([0.5*w_x, -0.5*w_y, h]),
                                     np.array([-0.5*w_x, 0.5*w_y, h]),
                                     np.array([-0.5*w_x, -0.5*w_y, h]),
                                     np.array([0.5*w_x, 0.0, h]),
                                     np.array([-0.5*w_x, 0.0, h]),
                                     np.array([0.0, 0.5*w_y, h]),
                                     np.array([0.0, -0.5*w_y, h])]

        self._com_vel = self._num_agent * [None]
        for i in range(self._num_agent):
            self._com_vel[i] = deque(maxlen=int(1.0/self._dt_con))

        self._dim_action = np.sum([info.dim for info in self._action_info])

        ''' Start time of the environment '''
        self._start_time = 0.0
        ''' Elapsed time after the environment starts '''
        self._elapsed_time = 0.0
        ''' For tracking the length of current episode '''
        self._episode_len = 0.0

        ''' 
        Any necessary information needed for training this environment.
        This can be set by calling "set_learning_info". 
        '''
        self._learning_info = {}

        self.add_noise = config['add_noise']

        self._pb_state_id = self._pb_client.saveState()

        self.reset(add_noise=False)

        if self._verbose:
            print('----- Environment Created -----')
            for i in range(self._num_agent):
                print('[Agent%d]: state(%d) and action(%d)'%(i, len(self.state(i)), self._action_info[i].dim))
            print('-------------------------------')

    def action_range(self, idx):
        return self._action_info[idx].val_min, self._action_info[idx].val_max

    def dim_action(self, idx):
        return self._action_info[idx].dim

    def dim_state(self, idx):
        return len(self.state(idx))

    def dim_state_body(self, idx):
        return len(self.state_body(idx))

    def dim_state_task(self, idx):
        return len(self.state_task(idx))

    def set_learning_info(self, info):
        self._learning_info = info

    def update_learning_info(self, info):
        self._learning_info.update(info)

    def interaction_mesh_vertices(self, agent="sim"):
        points = []
        for i in range(self._num_agent):
            if agent=="sim":
                points += self._sim_agent[i].interaction_mesh_samples()
            elif agent=="kin":
                points += self._kin_agent[i].interaction_mesh_samples()
            else:
                raise NotImplementedError
        return points

    def agent_avg_position(self, agents=None):
        if agents is None: agents=self._sim_agent
        return np.mean([(agent.get_root_state())[0] for agent in agents], axis=0)

    def agent_ave_facing_position(self, agents=None):
        if agents is None: agents=self._sim_agent
        return np.mean([agent.get_facing_position(self._ground_height) for agent in agents], axis=0)

    def throw_obstacle(self):
        size = np.random.uniform(0.1, 0.3, 3)
        p = self.agent_avg_position()
        self._base_env.throw_obstacle(size, p)

    def get_target_velocity(self, idx):
        R = mmMath.exp(self._sim_agent[idx]._char_info.v_up_env, self._target_vel_angle[idx])
        return self._target_vel_len[idx] * np.dot(R, self._target_vel_dir[idx])

    def split_action(self, action):
        assert len(action)%self._num_agent == 0
        dim_action = len(action)//self._num_agent
        actions = []
        idx = 0
        for i in range(self._num_agent):
            actions.append(action[idx:idx+dim_action])
            idx += dim_action
        return actions

    def compute_target_pose(self, i, action, action_info, clamp=True, normalized=True):
        agent = self._sim_agent[i]
        char_info = agent._char_info
        
        ''' the current posture should be deepcopied because action will modify it '''
        if self._action_type == Env.ActionMode.Relative:
            ref_pose = copy.deepcopy(self.get_current_pose_from_motion(i))
        else:
            ref_pose = copy.deepcopy(self._ref_poses[i])

        a_norm = action if normalized else action_info.real_to_norm(action)
        a_norm = action_info.clamp(a_norm) if clamp else a_norm
        a_real = action_info.norm_to_real(a_norm)

        dof_cnt = 0
        for j in agent._joint_indices:
            joint_type = agent.get_joint_type(j)
            ''' Fixed joint will not be affected '''
            if joint_type == self._pb_client.JOINT_FIXED:
                continue
            ''' If the joint do not have correspondance, use the reference posture itself'''
            if char_info.bvh_map[j] == None:
                continue
            if self._action_type == Env.ActionMode.Relative:
                T = ref_pose.get_transform(char_info.bvh_map[j], local=True)
            elif self._action_type == Env.ActionMode.Absolute:
                T = ref_pose.skel.get_joint(char_info.bvh_map[j]).xform_from_parent_joint
            else:
                raise NotImplementedError
            R, p = mmMath.T2Rp(T)
            if joint_type == self._pb_client.JOINT_SPHERICAL:
                dR = mmMath.exp(a_real[dof_cnt:dof_cnt+3])
                dof_cnt += 3
            elif joint_type == self._pb_client.JOINT_REVOLUTE:
                axis = agent.get_joint_axis(j)
                angle = a_real[dof_cnt:dof_cnt+1]
                dR = mmMath.exp(axis, angle)
                dof_cnt += 1
            else:
                raise NotImplementedError
            T_new = mmMath.Rp2T(np.dot(R, dR), p)
            ref_pose.set_transform(char_info.bvh_map[j], T_new, do_ortho_norm=False, local=True)

        return ref_pose
    
    def reset(self, start_time=None, add_noise=None):

        if add_noise is None:
            add_noise = self.add_noise 

        self._start_time = 0.0
        self._elapsed_time = 0.0
        self._target_pose = [None for i in range(self._num_agent)]
        init_poses, init_vels = [], []

        if Env.Task.Imitation in self._tasks:
            self._ref_motion = self.sample_ref_motion()
            
            if start_time is not None: 
                self._elapsed_time = self._start_time = start_time
            else:
                ''' Select a start time randomly '''
                t = np.random.uniform(0.0, self.end_time_of_ref_motion(0))
                self._elapsed_time = self._start_time = t

            for i in range(self._num_agent):
                ''' Set the state of simulated agent by using the state of reference motion '''
                cur_pose, cur_vel = self.get_current_pose_from_motion(i), self.get_current_vel_from_motion(i)
                ''' Add noise to the state if necessary '''
                if add_noise:
                    cur_pose, cur_vel = self._base_env.add_noise_to_pose_vel(self._sim_agent[i], cur_pose, cur_vel)
                init_poses.append(cur_pose)
                init_vels.append(cur_vel)
        else:
            for i in range(self._num_agent):
                cur_pose, cur_vel = self._ref_poses[i], self._ref_vels[i]
                if add_noise:
                    cur_pose, cur_vel = self._base_env.add_noise_to_pose_vel(self._sim_agent[i], cur_pose, cur_vel)
                init_poses.append(cur_pose)
                init_vels.append(cur_vel)

        self._base_env.reset(time=self._start_time, 
                             poses=init_poses, 
                             vels=init_vels, 
                             pb_state_id=self._pb_state_id)

        if Env.Task.Imitation in self._tasks:
            for i in range(self._num_agent):
                self._kin_agent[i].set_pose(cur_pose, cur_vel)
        
        self.end_of_episode = False
        self.end_of_episode_reason = []

        self.end_of_episode_intermediate = False
        self.end_of_episode_reason_intermediate = []
        self.time_elapsed_after_end_of_episode = 0.0

        for i in range(self._num_agent):
            self._com_vel[i].clear()
            self._com_vel[i].append(self._sim_agent[i].get_com_and_com_vel()[1])

        if Env.EarlyTermChoice.LowReward in self._early_term_choices:
            for i in range(self._num_agent):
                self._rew_queue[i].clear()
                for j in range(self._rew_queue[i].maxlen):
                    self._rew_queue[i].append(self.reward_max())

        self._episode_len = 0.0
            
        if Env.Task.Heading in self._tasks:
            for i in range(self._num_agent):
                self._target_vel_dir[i] = self._sim_agent[0].get_facing_direction()
                self._target_vel_angle[i] = 0.0
                self._target_vel_len[i] = 1.0

        if Env.Task.Dribble in self._tasks:
            # Generate the ball and goal randomly
            ball_radius = 0.15
            ball_pos = self._sim_agent[0].project_to_ground(np.random.uniform(-7.5, 7.5, size=3)) \
                + ball_radius * self._sim_agent[0]._char_info.v_up_env
            self._ball = sim_obstacle.Obstacle("ball", 
                                               2*self._sim_window_time,
                                               sim_obstacle.Shape.SPHERE, 
                                               mass=3.0,
                                               size=ball_radius*np.ones(3),
                                               p=ball_pos,
                                               Q=mmMath.R2Q(mmMath.I_SO3()),
                                               v=np.zeros(3),
                                               w=np.zeros(3))
            self._base_env._obs_manager.launch(self._ball)
            self._goal_pos = self._sim_agent[0].project_to_ground(np.random.uniform(-10, 10, size=3))

        if Env.Task.Fight in self._tasks or Env.Task.Chase in self._tasks:
            def place_agent_at(agent, theta, pos):
                R, p = mmMath.T2Rp(agent.get_root_transform())
                R, p[:2] = np.dot(mmMath.rotZ(theta), R), pos
                agent.set_root_transform(mmMath.Rp2T(R,p))
            if Env.Task.Fight in self._tasks:
                def pick_a_random_xform_on_the_ring(ring_size, boundary_margin=0.5):
                    phi = np.random.uniform(0, 2*math.pi)
                    r = np.random.uniform(0, ring_size[0]-boundary_margin)
                    pos = r * np.array([math.cos(phi), math.sin(phi)])
                    theta = np.random.uniform(0, 2*math.pi)
                    return theta, pos
                min_dist_btw_agents = 1.0
            else:
                def pick_a_random_xform_on_the_ring(ring_size, boundary_margin=0.5):
                    w, h = 0.5*ring_size[0] - boundary_margin, 0.5*ring_size[1] - boundary_margin
                    pos = np.array([np.random.uniform(-w, w), np.random.uniform(-h, h)])
                    theta = np.random.uniform(0, 2*math.pi)
                    return theta, pos
                min_dist_btw_agents = 1.0
            # Place characters at random positions
            while True:
                # Choose 2 positions on the ring which is far more than 1.0m
                theta1, pos1 = pick_a_random_xform_on_the_ring(self._ring_size)
                theta2, pos2 = pick_a_random_xform_on_the_ring(self._ring_size)
                if np.linalg.norm(pos2-pos1) > min_dist_btw_agents: break
            place_agent_at(self._sim_agent[0], theta1, pos1)
            place_agent_at(self._sim_agent[1], theta2, pos2)

    def sample_ref_motion(self):
        ref_indices = []
        ref_motions = []
        for i in range(self._num_agent):
            idx = np.random.randint(len(self._ref_motion_all[i]))
            ref_indices.append(idx)
            ref_motions.append(self._ref_motion_all[i][idx])
        if self._verbose:
            print('Ref. motions selected: ', ref_indices)
        return ref_motions

    def get_current_pose_from_motion(self, idx):
        return self._ref_motion[idx].get_pose_by_time(self._elapsed_time)

    def get_current_vel_from_motion(self, idx):
        return self._ref_motion[idx].get_velocity_by_time(self._elapsed_time)

    def callback_step_prev(self):
        return

    def callback_step_after(self):
        return
    
    def step(self, action):

        profile = False
        
        if profile:
            print('-----------------------------------------')
            time_checker = basics.TimeChecker()

        self.callback_step_prev()

        ''' Collect data for reward computation before the current step'''
        rew_data_prev = [self.reward_data(i) for i in range(self._num_agent)]

        ''' Increase elapsed time '''
        self._elapsed_time += self._dt_con
        self._episode_len += self._dt_con

        assert len(action) == self._num_agent

        if profile:
            print('> compute_target_pose')
            time_checker.print_time()
        
        
        for i in range(self._num_agent):
            if isinstance(action[i], kinematics.Posture):
                self._target_pose[i] = action[i]
            elif isinstance(action[i], np.ndarray):
                self._target_pose[i] = self.compute_target_pose(i, action[i], self._action_info[i])
            else:
                print(type(action[i]))
                raise NotImplementedError
        
        for i in range(self._num_agent):
            self._com_vel[i].append(self._sim_agent[i].get_com_and_com_vel()[1])
        
        ''' Update simulation '''
        self._base_env.step(self._target_pose)

        if profile:
            print('> simulation')
            time_checker.print_time()

        ''' Things after the current step '''

        if Env.Task.Imitation in self._tasks:
            ''' Set kinematic character '''
            ''' This is necessary to compute the reward correctly '''
            for i in range(self._num_agent):
                self._kin_agent[i].set_pose(self.get_current_pose_from_motion(i),
                                            self.get_current_vel_from_motion(i))
        if Env.Task.Heading in self._tasks:
            for i in range(self._num_agent):
                self._target_vel_angle[i] += np.random.uniform(-0.15, 0.15)

        self.callback_step_after()

        ''' Collect data for reward computation after the current step'''
        rew_data_next = [self.reward_data(i) for i in range(self._num_agent)]

        ''' 
        Check conditions for end-of-episode. 
        If 'eoe_margin' is larger than zero, the environment will continue for some time.
        '''
        
        if not self.end_of_episode_intermediate:
            eoe_reason = []
            for i in range(self._num_agent):
                eoe_reason += self.inspect_end_of_episode_per_agent(i)
            if Env.EarlyTermChoice.TaskComplete in self._early_term_choices:
                eoe_reason += self.inspect_end_of_episode_task()

            self.end_of_episode_intermediate = len(eoe_reason) > 0
            self.end_of_episode_reason_intermediate = eoe_reason

        if self.end_of_episode_intermediate:
            self.time_elapsed_after_end_of_episode += self._dt_con
            if self.time_elapsed_after_end_of_episode >= self._eoe_margin:
                self.end_of_episode = True
                self.end_of_episode_reason = self.end_of_episode_reason_intermediate

        if self._verbose and self.end_of_episode:
            print('=================EOE=================')
            print('Reason:', self.end_of_episode_reason)
            print('TIME: (start:%02f ) (cur:%02f) (elapsed:%02f) (time_after_eoe: %02f)'\
                %(self._start_time, 
                  self._elapsed_time, 
                  self._elapsed_time-self._start_time, 
                  self.time_elapsed_after_end_of_episode))
            print('=====================================')

        if profile:
            print('> inspect eoe and reward_data')
            time_checker.print_time()
            print('-----------------------------------------')

        ''' Compute rewards '''
        
        rews, infos = [], []
        for i in range(self._num_agent):
            r, rd = self.reward(i, rew_data_prev, rew_data_prev, action)
            rews.append(r)
            info = {
                'eoe_reason': self.end_of_episode_reason,
                'rew_detail': rd,
                'learning_info': self._learning_info
            }
            infos.append(info)
            if Env.EarlyTermChoice.LowReward in self._early_term_choices:
                self._rew_queue[i].append(r)
        
        return rews, infos

    def state(self, idx):
        state = []
        
        if Env.StateChoice.Body in self._state_choices:
            state.append(self.state_body(idx, return_stacked=True))
        if Env.StateChoice.Task in self._state_choices:
            state.append(self.state_task(idx))

        return np.hstack(state)

    def state_body(self, 
                   idx,
                   T_ref=None, 
                   include_com=True,
                   include_p=True, 
                   include_Q=True, 
                   include_v=True, 
                   include_w=True, 
                   return_stacked=True):
        return self._state_body(self._sim_agent[idx],
                                T_ref, 
                                include_com, 
                                include_p, 
                                include_Q, 
                                include_v, 
                                include_w, 
                                return_stacked)

    def _state_body(self, 
                    agent, 
                    T_ref=None, 
                    include_com=True, 
                    include_p=True, 
                    include_Q=True, 
                    include_v=True, 
                    include_w=True, 
                    return_stacked=True):
        if T_ref is None: T_ref = agent.get_facing_transform(self._ground_height)

        R_ref, p_ref = mmMath.T2Rp(T_ref)
        R_ref_inv = R_ref.transpose()

        link_states = []
        link_states.append(agent.get_root_state())
        ps, Qs, vs, ws = agent.get_link_states()
        for j in agent._joint_indices:
            link_states.append((ps[j], Qs[j], vs[j], ws[j]))

        state = []
        for i, s in enumerate(link_states):
            p, Q, v, w = s[0], s[1], s[2], s[3]
            if include_p:
                p_rel = np.dot(R_ref_inv, p - p_ref)
                state.append(p_rel) # relative position w.r.t. the reference frame
            if include_Q:
                Q_rel = mmMath.R2Q(np.dot(R_ref_inv, mmMath.Q2R(Q)))
                Q_rel = mmMath.post_process_Q(Q_rel, normalize=True, half_space=True)
                state.append(Q_rel) # relative rotation w.r.t. the reference frame
            if include_v:
                v_rel = np.dot(R_ref_inv, v)
                state.append(v_rel) # relative linear vel w.r.t. the reference frame
            if include_w:
                w_rel = np.dot(R_ref_inv, w)
                state.append(w_rel) # relative angular vel w.r.t. the reference frame
            if include_com:
                if i==0:
                    p_com = agent._link_masses[i] * p
                    v_com = agent._link_masses[i] * v
                else:
                    p_com += agent._link_masses[i] * p
                    v_com += agent._link_masses[i] * v

        if include_com:
            p_com /= agent._link_total_mass
            v_com /= agent._link_total_mass
            state.append(np.dot(R_ref_inv, p_com - p_ref))
            state.append(np.dot(R_ref_inv, v_com))
        
        if return_stacked:
            return np.hstack(state)
        else:
            return state

    def state_imitation(self, sim_agent, kin_agent, poses, vels, include_abs, include_rel):

        assert len(poses) == len(vels)

        R_sim, p_sim = mmMath.T2Rp(sim_agent.get_facing_transform(self._ground_height))
        R_sim_inv = R_sim.transpose()
        state_sim = self._state_body(sim_agent, None, return_stacked=False)
        
        state = []
        state_kin_orig = kin_agent.save_states()
        for pose, vel in zip(poses, vels):
            kin_agent.set_pose(pose, vel)
            state_kin = self._state_body(kin_agent, None, return_stacked=False)
            # Add pos/vel values
            if include_abs:
                state.append(np.hstack(state_kin))
            # Add difference of pos/vel values
            if include_rel:
                for j in range(len(state_sim)):
                    if len(state_sim[j])==3: 
                        state.append(state_sim[j]-state_kin[j])
                    elif len(state_sim[j])==4:
                        state.append(self._pb_client.getDifferenceQuaternion(state_sim[j], state_kin[j]))
                    else:
                        raise NotImplementedError
            ''' Add facing frame differences '''
            R_kin, p_kin = mmMath.T2Rp(kin_agent.get_facing_transform(self._ground_height))
            state.append(np.dot(R_sim_inv, p_kin - p_sim))
            state.append(np.dot(R_sim_inv, kin_agent.get_facing_direction()))
        kin_agent.restore_states(state_kin_orig)

        return np.hstack(state)

    def state_interaction(self):
        state = []

        vertices = self.interaction_mesh_vertices()
        im = self._interaction_meshes[self._ref_motion[0].time_to_frame(self._elapsed_time)]
        coordinates_new = im.get_laplacian_coordinates(vertices)
        coordinates_old = im.laplacian_coordinates
        for i in range(len(vertices)):
            state.append(coordinates_old[i]-coordinates_new[i])

        return np.hstack(state)

    def state_task(self, idx):        
        state = []

        if Env.Task.Imitation in self._tasks:
            poses, vels = [], []
            ref_motion = self._ref_motion[idx]
            for dt in self._imit_window:
                t = basics.clamp(self._elapsed_time + dt, ref_motion.times[0], ref_motion.times[-1])
                poses.append(ref_motion.get_pose_by_time(t))
                vels.append(ref_motion.get_velocity_by_time(t))
            state.append(self.state_imitation(self._sim_agent[idx],
                                              self._kin_agent[idx],
                                              poses,
                                              vels,
                                              include_abs=True,
                                              include_rel=True))

        if Env.Task.Heading in self._tasks:
            agent = self._sim_agent[idx]
            
            vel = self.get_target_velocity(idx)

            R, p = mmMath.T2Rp(agent.get_facing_transform(self._ground_height))
            R_inv = R.transpose()
            
            com, com_vel = agent.get_com_and_com_vel()
            
            state.append(np.dot(R_inv, vel))
            state.append(np.dot(R_inv, com_vel - vel))
        
        if Env.Task.Dribble in self._tasks:
            agent = self._sim_agent[idx]

            R, p = mmMath.T2Rp(agent.get_facing_transform(self._ground_height))
            R_inv = R.transpose()
            
            com, _ = agent.get_com_and_com_vel()
            com = agent.project_to_ground(com)

            p_ball = agent.project_to_ground(self._ball.p)
            R_ball = mmMath.Q2R(self._ball.Q)

            p_goal = agent.project_to_ground(self._goal_pos)
            
            state.append(np.dot(R_inv, p_ball-p))
            state.append(mmMath.logSO3(np.dot(R_inv, R_ball)))
            state.append(np.dot(R_inv, self._ball.v))
            state.append(np.dot(R_inv, self._ball.w))
            
            state.append(np.dot(R_inv, p_goal-p))
            state.append(np.dot(R_inv, p_goal-p_ball))
        
        if Env.Task.Fight in self._tasks or Env.Task.Chase in self._tasks:
            assert idx in [0, 1]
            
            agent = self._sim_agent[idx]
            opponent_idx = 1 if idx == 0 else 0
            
            T = agent.get_facing_transform(self._ground_height)
            R, p = mmMath.T2Rp(T)
            R_inv = R.transpose()
            for p_boundary in self._ring_boundaries:
                state.append(np.dot(R_inv, p_boundary-p))
            # Opponent 
            # R_opponent, p_opponent = mmMath.T2Rp(opponent.get_facing_transform(self._ground_height))
            # state.append(mmMath.logSO3(np.dot(R_inv, R_opponent)))
            # state.append(np.dot(R_inv, p_opponent - p))
            state.append(self.state_body(opponent_idx, 
                                         T_ref=T,
                                         include_com=True,
                                         include_p=True, 
                                         include_Q=False, 
                                         include_v=False, 
                                         include_w=False, 
                                         return_stacked=True))

        return np.hstack(state)

    def reward_data(self, idx):
        data = {}

        data['sim_root_pQvw'] = self._sim_agent[idx].get_root_state()
        data['sim_link_pQvw'] = self._sim_agent[idx].get_link_states()
        data['sim_joint_pv'] = self._sim_agent[idx].get_joint_states()
        data['sim_facing_frame'] = self._sim_agent[idx].get_facing_transform(self._ground_height)
        data['sim_com'], data['sim_com_vel'] = self._sim_agent[idx].get_com_and_com_vel()

        if Env.Task.Imitation in self._tasks:
            data['kin_root_pQvw'] = self._kin_agent[idx].get_root_state()
            data['kin_link_pQvw'] = self._kin_agent[idx].get_link_states()
            data['kin_joint_pv'] = self._kin_agent[idx].get_joint_states()
            data['kin_facing_frame'] = self._kin_agent[idx].get_facing_transform(self._ground_height)
            data['kin_com'], data['kin_com_vel'] = self._kin_agent[idx].get_com_and_com_vel()

        if Env.Task.Heading in self._tasks:
            data['sim_com_vel_avg'] = np.mean(self._com_vel[idx], axis=0)
            data['sim_pose'] = self._sim_agent[idx].get_pose(self._ref_motion_skel[idx])
            data['target_vel'] = self.get_target_velocity(i)
        
        if Env.Task.Dribble in self._tasks:
            data['sim_com_vel_avg'] = np.mean(self._com_vel[idx], axis=0)
            data['sim_pose'] = self._sim_agent[idx].get_pose(self._ref_motion_skel[idx])
            data['ball_pQvw'] = copy.deepcopy((self._ball.p, self._ball.Q, self._ball.v, self._ball.w))
            data['goal_pos'] = copy.deepcopy(self._goal_pos)
        
        if Env.Task.Fight in self._tasks:
            data['sim_com_vel_avg'] = np.mean(self._com_vel[idx], axis=0)
            data['out_of_ring'] = self.check_out_of_ring(self._sim_agent[idx])
            data['falldown'] = self._base_env.check_falldown(self._sim_agent[idx], self._ring_id)
            data['root_height'] = self._sim_agent[idx].get_root_height_from_ground(self._ground_height)
            data['is_standing'] = data['root_height'] >= 0.5
        
        if Env.Task.Chase in self._tasks:
            data['sim_com_vel_avg'] = np.mean(self._com_vel[idx], axis=0)
            data['out_of_ring'] = self.check_out_of_ring(self._sim_agent[idx])
            data['falldown'] = self._base_env.check_falldown(self._sim_agent[idx], self._ring_id)
            data['root_height'] = self._sim_agent[idx].get_root_height_from_ground(self._ground_height)
            data['is_standing'] = data['root_height'] >= 0.5
            data['touched'] = self.check_touched()

        data.update(self.reward_data_auxiliary(idx))

        return data

    def reward_data_auxiliary(self, idx):
        return {}

    def reward_max(self):
        return 1.0

    def reward_min(self):
        return 0.0

    def return_max(self, gamma):
        assert gamma < 1.0
        return 1.0 / (1.0 - gamma)

    def return_min(self):
        return 0.0

    def reward_task_heading(self, idx, data_prev, data_next, action, error):
        diff = data_next[idx]['target_vel'] - \
            self._sim_agent[idx].project_to_ground(data_next[idx]['sim_com_vel_avg'])
        error['com_vel'] = np.dot(diff, diff)

    def reward_task_dribble(self, idx, data_prev, data_next, action, error):
        v_com_avg = data_next[idx]['sim_com_vel_avg']
        p_com = data_next[idx]['sim_com']
        p_ball, Q_ball, v_ball, w_ball = data_next[idx]['ball_pQvw']
        p_goal = data_next[idx]['goal_pos']

        dir_to_ball = mmMath.normalize2(self._sim_agent[idx].project_to_ground(p_ball-p_com))
        v_com_avg_proj = self._sim_agent[idx].project_to_ground(v_com_avg)
        diff = min(0, np.dot(v_com_avg_proj, dir_to_ball) - 1.0)
        error['vel_to_ball'] = np.dot(diff, diff)

        diff = self._sim_agent[idx].project_to_ground(p_ball-p_com)
        error['stay_at_ball'] = np.dot(diff, diff)

        dir_to_goal = mmMath.normalize2(self._sim_agent[idx].project_to_ground(p_goal-p_ball))
        v_ball_proj = self._sim_agent[idx].project_to_ground(v_ball)
        diff = min(0, np.dot(v_ball_proj, dir_to_goal) - 1.0)
        error['vel_to_goal'] = np.dot(diff, diff)

        diff = self._sim_agent[idx].project_to_ground(p_goal-p_ball)
        error['stay_at_goal'] = np.dot(diff, diff)

    def reward_task_fight(self, idx, data_prev, data_next, action, error):
        idx_opponent = 1 if idx == 0 else 0
            
        root_height = data_next[idx]['root_height']
        v_com_avg = data_next[idx]['sim_com_vel_avg']
        p_com = data_next[idx]['sim_com']
        p_com_opponent = data_next[idx_opponent]['sim_com']
        p_center = np.zeros(3)

        is_standing = data_next[idx]['is_standing']
        is_standing_opponent = data_next[idx_opponent]['is_standing']

        if 'high_root' in self._reward_names[idx]:
            diff = min(0, root_height - 0.5)
            error['high_root'] = np.dot(diff, diff)
        
        if 'vel_to_center' in self._reward_names[idx]:
            dir_to_center = mmMath.normalize2(self._sim_agent[idx].project_to_ground(p_center-p_com))
            v_com_avg_proj = self._sim_agent[idx].project_to_ground(v_com_avg)
            diff = min(0, np.dot(v_com_avg_proj, dir_to_center) - 1.0)
            error['vel_to_center'] = np.dot(diff, diff)
        
        if 'stay_at_center' in self._reward_names[idx]:
            diff = self._sim_agent[idx].project_to_ground(p_center-p_com)
            error['stay_at_center'] = np.dot(diff, diff)

        if 'push_opponent' in self._reward_names[idx]:
            diff = min(0, np.linalg.norm(p_com_opponent[:2]-p_center[:2]) - self._ring_size[0])
            error['push_opponent'] = np.dot(diff, diff)

        if 'out_of_ring' in self._reward_names[idx]:
            error['out_of_ring'] = -1.0 if data_next[idx]['out_of_ring'] else 0.0

        if 'falldown' in self._reward_names[idx]:
            error['falldown'] = -1.0 if data_next[idx]['falldown'] else 0.0

        if 'win' in self._reward_names[idx]:
            out_of_ring = data_next[idx]['out_of_ring']
            out_of_ring_opponent = data_next[idx_opponent]['out_of_ring']
            if is_standing and not out_of_ring and out_of_ring_opponent:
                error['win'] = 1.0
            elif out_of_ring and not out_of_ring_opponent and is_standing_opponent:
                error['win'] = -1.0
            else:
                error['win'] = 0.0

    def reward_task_chase(self, idx, data_prev, data_next, action, error):
        assert idx in [0, 1]
        idx_opponent = 1 if idx == 0 else 0

        root_height = data_next[idx]['root_height']
        v_com_avg = data_next[idx]['sim_com_vel_avg']
        v_com_avg_opponent = data_next[idx_opponent]['sim_com_vel_avg']
        p_com = data_next[idx]['sim_com']
        p_com_opponent = data_next[idx_opponent]['sim_com']
        is_standing = data_next[idx]['is_standing']
        is_standing_opponent = data_next[idx_opponent]['is_standing']
        
        if idx == 0:
            if 'high_root' in self._reward_names[idx]:
                diff = min(0, root_height - 0.5)
                error['high_root'] = np.dot(diff, diff)
            
            if 'vel_to_prey' in self._reward_names[idx]:
                dir_to_prey = mmMath.normalize2(self._sim_agent[idx].project_to_ground(p_com_opponent-p_com))
                v_com_avg_proj = self._sim_agent[idx].project_to_ground(v_com_avg)
                diff = min(0, np.dot(v_com_avg_proj, dir_to_prey) - 1.0)
                error['vel_to_prey'] = np.dot(diff, diff)
            
            if 'stay_at_prey' in self._reward_names[idx]:
                diff = self._sim_agent[idx].project_to_ground(p_com_opponent-p_com)
                error['stay_at_prey'] = np.dot(diff, diff)
            
            if 'away_from_boundary' in self._reward_names[idx]:
                dist_from_boundary = min(max(0.0, 0.5*self._ring_size[0]-abs(p_com[0])), max(0.0, 0.5*self._ring_size[1]-abs(p_com[1])))
                diff = min(0, dist_from_boundary - 1.0)
                error['away_from_boundary'] = np.dot(diff, diff)
            
            if 'vel_to_prey_rel' in self._reward_names[idx]:
                dir_to_prey = mmMath.normalize2(self._sim_agent[idx].project_to_ground(p_com_opponent-p_com))
                v_com_avg_rel_proj = self._sim_agent[idx].project_to_ground(v_com_avg-v_com_avg_opponent)
                diff = min(0, np.dot(v_com_avg_rel_proj, dir_to_prey) - 1.0)
                error['vel_to_prey_rel'] = np.dot(diff, diff)
            
            if 'out_of_ring' in self._reward_names[idx]:
                error['out_of_ring'] = -1.0 if data_next[idx]['out_of_ring'] else 0.0
            
            if 'falldown' in self._reward_names[idx]:
                error['falldown'] = -1.0 if data_next[idx]['falldown'] else 0.0
            
            if 'win' in self._reward_names[idx]:
                error['win'] = 1.0 if data_next[idx]['touched'] else 0.0
        else:
            if 'high_root' in self._reward_names[idx]:
                diff = min(0, root_height - 0.5)
                error['high_root'] = np.dot(diff, diff)

            if 'vel_from_hunter' in self._reward_names[idx]:
                dir_from_hunter = mmMath.normalize2(self._sim_agent[idx].project_to_ground(p_com-p_com_opponent))
                v_com_avg_proj = self._sim_agent[idx].project_to_ground(v_com_avg)
                diff = min(0, np.dot(v_com_avg_proj, dir_from_hunter) - 1.0)
                error['vel_from_hunter'] = np.dot(diff, diff)

            if 'away_from_hunter' in self._reward_names[idx]:
                dist_from_hunter = np.linalg.norm(p_com[:2] - p_com_opponent[:2])
                diff = min(0, dist_from_hunter - 2.0)
                error['away_from_hunter'] = np.dot(diff, diff)

            if 'away_from_boundary' in self._reward_names[idx]:
                dist_from_boundary = min(max(0.0, 0.5*self._ring_size[0]-abs(p_com[0])), max(0.0, 0.5*self._ring_size[1]-abs(p_com[1])))
                diff = min(0, dist_from_boundary - 1.0)
                error['away_from_boundary'] = np.dot(diff, diff)

            if 'vel_from_hunter_rel' in self._reward_names[idx]:
                dir_from_hunter = mmMath.normalize2(self._sim_agent[idx].project_to_ground(p_com-p_com_opponent))
                v_com_avg_rel_proj = self._sim_agent[idx].project_to_ground(v_com_avg-v_com_avg_opponent)
                diff = min(0, np.dot(v_com_avg_rel_proj, dir_from_hunter) - 1.0)
                error['vel_from_hunter_rel'] = np.dot(diff, diff)

            if 'out_of_ring' in self._reward_names[idx]:
                error['out_of_ring'] = -1.0 if data_next[idx]['out_of_ring'] else 0.0

            if 'falldown' in self._reward_names[idx]:
                error['falldown'] = -1.0 if data_next[idx]['falldown'] else 0.0

            if 'win' in self._reward_names[idx]:
                error['win'] = -1.0 if data_next[idx]['touched'] else 0.0

    def reward_auxiliary(self, idx, data_prev, data_next, action, error):
        return

    def reward(self, idx, data_prev, data_next, action):        
        error = {}

        if Env.Task.Imitation in self._tasks:
            self.reward_task_imitation(idx, data_prev, data_next, action, error)
        
        if Env.Task.Heading in self._tasks:
            self.reward_task_heading(idx, data_prev, data_next, action, error)
        
        if Env.Task.Dribble in self._tasks:
            self.reward_task_dribble(idx, data_prev, data_next, action, error)
        
        if Env.Task.Fight in self._tasks:
            self.reward_task_fight(idx, data_prev, data_next, action, error)
        
        if Env.Task.Chase in self._tasks:
            self.reward_task_chase(idx, data_prev, data_next, action, error)

        self.reward_auxiliary(idx, data_prev, data_next, action, error)

        rew_fn_def = self._reward_fn_def[self._reward_fn_map[idx]]
        rew, rew_detail = self.compute_reward(error, rew_fn_def)

        return rew, rew_detail

    # def get_w_superposition(self, idx):
    #     w_superposition = {}
        
    #     if Env.Task.Fight in self._tasks:
    #         if 'reward_phase' in self._learning_info.keys():
    #             reward_phase = self._learning_info['reward_phase']
    #             assert 0.0 <= reward_phase <= 1.0
    #             w_help = 1.0 - reward_phase
    #             w_goal = reward_phase
    #             w_superposition['high_root']      = w_help
    #             w_superposition['vel_to_center']  = w_help
    #             w_superposition['stay_at_center'] = w_help
    #             w_superposition['push_opponent']  = w_help
    #             w_superposition['win']            = w_goal

    #     if Env.Task.Chase in self._tasks:
    #         if 'reward_phase' in self._learning_info.keys():
    #             reward_phase = self._learning_info['reward_phase']
    #             assert 0.0 <= reward_phase <= 1.0
    #             w_help = 1.0 - reward_phase
    #             w_goal = reward_phase
    #             if idx == 0:
    #                 w_superposition['high_root']          = w_help
    #                 w_superposition['vel_to_prey']        = w_help
    #                 w_superposition['stay_at_prey']       = w_help
    #                 w_superposition['away_from_boundary'] = w_help
    #                 w_superposition['win']                = w_goal
    #             else:
    #                 w_superposition['high_root']          = w_help
    #                 w_superposition['vel_from_hunter']    = w_help
    #                 w_superposition['away_from_hunter']   = w_help
    #                 w_superposition['away_from_boundary'] = w_help
    #                 w_superposition['win']                = w_goal
        
    #     return w_superposition

    def get_reward_names(self, fn_def):
        rew_names = set()
        op = fn_def['op']

        if op in ['add', 'mul']:
            for child in fn_def['child_nodes']:
                rew_names = rew_names.union(self.get_reward_names(child))
        elif op == 'leaf':
            rew_names.add(fn_def['name'])
        else:
            raise NotImplementedError

        return rew_names

    def pretty_print_rew_detail(self, rew_detail, prefix=str()):
        print("%s > name:   %s"%(prefix, rew_detail['name']))
        print("%s   value:  %s"%(prefix, rew_detail['value']))
        print("%s   weight: %s"%(prefix, rew_detail['weight']))
        print("%s   op: %s"%(prefix, rew_detail['op']))
        for child in rew_detail["child_nodes"]:
            self.pretty_print_rew_detail(child, prefix+"\t")

    def compute_reward(self, error, fn_def):
        ''' compute reward accroding to the definition tree '''
        op = fn_def['op']
        n = fn_def['name'] if 'name' in fn_def.keys() else 'noname'
        w = fn_def['weight'] if 'weight' in fn_def.keys() else 1.0

        rew_detail = {'name': n, 'value': 0.0, 'op': op, 'weight': w, 'child_nodes': []}

        if op == 'add':
            rew = 0.0
            for child in fn_def['child_nodes']:
                r, rd = self.compute_reward(error, child)
                rew += r
                rew_detail['child_nodes'].append(rd)
        elif op == 'mul':
            rew = 1.0
            for child in fn_def['child_nodes']:
                r, rd = self.compute_reward(error, child)
                rew *= r
                rew_detail['child_nodes'].append(rd)
        elif op == 'leaf':
            if 'kernel' in fn_def.keys():
                kernel = fn_def['kernel']
            else:
                kernel = None

            if 'weight_schedule' in fn_def.keys():
                timesteps_total = self._learning_info['timesteps_total']
                w *= basics.lerp_from_paired_list(
                    timesteps_total, fn_def['weight_schedule'])
            
            if kernel is None or kernel['type'] == "none":
                e = error[n]
            elif kernel['type'] == "gaussian":
                e = math.exp(-kernel['scale']*error[n])
            else:
                raise NotImplementedError
            
            rew = w*e
        else:
            raise NotImplementedError

        rew_detail['value'] = rew

        return rew, rew_detail

    def reward_task_imitation(self, idx, data_prev, data_next, action, error):

        sim_agent = self._sim_agent[idx]
        char_info = sim_agent._char_info

        data = data_next[idx]

        sim_root_p, sim_root_Q, sim_root_v, sim_root_w = data['sim_root_pQvw']
        sim_link_p, sim_link_Q, sim_link_v, sim_link_w = data['sim_link_pQvw']
        sim_joint_p, sim_joint_v = data['sim_joint_pv']
        sim_facing_frame = data['sim_facing_frame']
        R_sim_f, p_sim_f = mmMath.T2Rp(sim_facing_frame)
        R_sim_f_inv = R_sim_f.transpose()
        sim_com, sim_com_vel = data['sim_com'], data['sim_com_vel']
        
        kin_root_p, kin_root_Q, kin_root_v, kin_root_w = data['kin_root_pQvw']
        kin_link_p, kin_link_Q, kin_link_v, kin_link_w = data['kin_link_pQvw']
        kin_joint_p, kin_joint_v = data['kin_joint_pv']
        kin_facing_frame = data['kin_facing_frame']
        R_kin_f, p_kin_f = mmMath.T2Rp(kin_facing_frame)
        R_kin_f_inv = R_kin_f.transpose()
        kin_com, kin_com_vel = data['kin_com'], data['kin_com_vel']

        indices = range(len(sim_joint_p))

        if 'pose_pos' in self._reward_names[idx]:
            error['pose_pos'] = 0.0
            for j in indices:
                joint_type = sim_agent.get_joint_type(j)
                if joint_type == self._pb_client.JOINT_FIXED:
                    continue
                elif joint_type == self._pb_client.JOINT_SPHERICAL:
                    dQ = self._pb_client.getDifferenceQuaternion(sim_joint_p[j], kin_joint_p[j])
                    _, diff_pose_pos = self._pb_client.getAxisAngleFromQuaternion(dQ)
                else:
                    diff_pose_pos = sim_joint_p[j] - kin_joint_p[j]
                error['pose_pos'] += char_info.joint_weight[j] * np.dot(diff_pose_pos, diff_pose_pos)
            if len(indices) > 0:
                error['pose_pos'] /= len(indices)

        if 'pose_vel' in self._reward_names[idx]:
            error['pose_vel'] = 0.0
            for j in indices:
                joint_type = sim_agent.get_joint_type(j)
                if joint_type == self._pb_client.JOINT_FIXED:
                    continue
                else:
                    diff_pose_vel = sim_joint_v[j] - kin_joint_v[j]
                error['pose_vel'] += char_info.joint_weight[j] * np.dot(diff_pose_vel, diff_pose_vel)
            if len(indices) > 0:
                error['pose_vel'] /= len(indices)

        if 'ee' in self._reward_names[idx]:
            error['ee'] = 0.0
            
            for j in char_info.end_effector_indices:
                sim_ee_local = np.dot(R_sim_f_inv, sim_link_p[j]-p_sim_f)
                kin_ee_local = np.dot(R_kin_f_inv, kin_link_p[j]-p_kin_f)
                diff_pos =  sim_ee_local - kin_ee_local
                error['ee'] += np.dot(diff_pos, diff_pos)

            if len(char_info.end_effector_indices) > 0:
                error['ee'] /= len(char_info.end_effector_indices)

        if 'root' in self._reward_names[idx]:
            diff_root_p = sim_root_p - kin_root_p
            _, diff_root_Q = self._pb_client.getAxisAngleFromQuaternion(
                self._pb_client.getDifferenceQuaternion(sim_root_Q, kin_root_Q))
            diff_root_v = sim_root_v - kin_root_v
            diff_root_w = sim_root_w - kin_root_w
            error['root'] = 1.0 * np.dot(diff_root_p, diff_root_p) + \
                            0.1 * np.dot(diff_root_Q, diff_root_Q) + \
                            0.01 * np.dot(diff_root_v, diff_root_v) + \
                            0.001 * np.dot(diff_root_w, diff_root_w)

        if 'com' in self._reward_names[idx]:
            diff_com = np.dot(R_sim_f_inv, sim_com-p_sim_f) - np.dot(R_kin_f_inv, kin_com-p_kin_f)
            diff_com_vel = sim_com_vel - kin_com_vel
            error['com'] = 1.0 * np.dot(diff_com, diff_com) + \
                           0.1 * np.dot(diff_com_vel, diff_com_vel)

    def end_time_of_ref_motion(self, idx):
        return self._ref_motion[idx].times[-1]

    def check_root_fail(self, sim_agent, kin_agent):
        p1, Q1, _, _ = sim_agent.get_root_state()
        p2, Q2, _, _ = kin_agent.get_root_state()
        _, angle = self._pb_client.getAxisAngleFromQuaternion(
            self._pb_client.getDifferenceQuaternion(Q1, Q2))
        dist = np.linalg.norm(p2-p1)
        return angle > 0.3 * math.pi or dist > 1.0

    def check_out_of_ring(self, agent):
        result = False
        if Env.Task.Fight in self._tasks:
            p_com, _ = agent.get_com_and_com_vel()
            result = np.linalg.norm(p_com[:2]) > self._ring_size[0]
        if Env.Task.Chase in self._tasks:
            p_com, _ = agent.get_com_and_com_vel()
            result = (abs(p_com[0]) > 0.5*self._ring_size[0]) or (abs(p_com[1]) > 0.5*self._ring_size[1])
        return result

    # def check_out_of_ring(self, agent):
    #     return self._base_env.check_collision(agent._body_id, self._base_env._plane_id)

    def check_touched(self):
        return self._base_env.check_collision(
            self._sim_agent[0]._body_id, self._sim_agent[1]._body_id)

    def inspect_end_of_episode_task(self):
        eoe_reason = []

        ''' TODO: imitation should be a task '''
        if Env.Task.Imitation in self._tasks:
            for i in range(self._num_agent):
                check = self._elapsed_time >= self.end_time_of_ref_motion(i)
                if check: eoe_reason.append('[%s] end_of_motion'%self._sim_agent[i].get_name())

        if Env.Task.Fight in self._tasks:
            ''' When one of the agents get out of the ring '''
            for agent in self._sim_agent:
                check = self.check_out_of_ring(agent)
                if check: eoe_reason.append('[%s] out_of_ring'%agent.get_name())
            ''' When one of agents falldown '''
            check = self._base_env.check_falldown(self._sim_agent[0], self._ring_id) \
                or self._base_env.check_falldown(self._sim_agent[1], self._ring_id)
            if check: eoe_reason.append('falldown_on_ring')
        
        if Env.Task.Chase in self._tasks:
            ''' When one of the agents get out of the ring '''
            for agent in self._sim_agent:
                check = self.check_out_of_ring(agent)
                if check: eoe_reason.append('[%s] out_of_ring'%agent.get_name())
            ''' When one of agents falldown '''
            check = self._base_env.check_falldown(self._sim_agent[0], self._ring_id) \
                or self._base_env.check_falldown(self._sim_agent[1], self._ring_id)
            if check: eoe_reason.append('falldown_on_ring')
            ''' When the hunter catches the prey '''
            check = self.check_touched()
            if check: eoe_reason.append('touched')

        return eoe_reason

    def inspect_end_of_episode_per_agent(self, idx):
        eoe_reason = []
        name = self._sim_agent[idx].get_name()

        if Env.EarlyTermChoice.Falldown in self._early_term_choices:
            check = self._base_env.check_falldown(self._sim_agent[idx])
            if check: eoe_reason.append('[%s] falldown'%name)
        if Env.EarlyTermChoice.SimDiv in self._early_term_choices:
            check = self._base_env.is_sim_div(self._sim_agent[idx])
            if check: eoe_reason.append('[%s] sim_div'%name)
        if Env.EarlyTermChoice.SimWindow in self._early_term_choices:
            check = self._elapsed_time - self._start_time > self._sim_window_time
            if check: eoe_reason.append('[%s] sim_window'%name)
        if Env.EarlyTermChoice.LowReward in self._early_term_choices:
            check = np.mean(list(self._rew_queue[idx])) < self._et_low_reward_thres * self.reward_max()
            if check: eoe_reason.append('[%s] low_rewards'%name)

        return eoe_reason

    def render(self, flag, colors):
        rm.gl.glEnable(rm.gl.GL_LIGHTING)
        rm.gl.glEnable(rm.gl.GL_BLEND)
        rm.gl.glBlendFunc(rm.gl.GL_SRC_ALPHA, rm.gl.GL_ONE_MINUS_SRC_ALPHA)

        self._base_env.render(agent=rm.flag['sim_model'],
                              shadow=rm.flag['shadow'],
                              joint=rm.flag['joint'],
                              collision=rm.flag['collision'],
                              com_vel=rm.flag['com_vel'],
                              facing_frame=rm.flag['facing_frame'],
                              height=self._ground_height,
                              colors=colors)

        if rm.flag['target_pose']:
            for i in range(self._num_agent):
                if self._target_pose[i] is None: continue
                agent = self._kin_agent[i]
                agent_state = agent.save_states()
                agent.set_pose(self._target_pose[i])
                rm.gl.glPushAttrib(rm.gl.GL_LIGHTING|rm.gl.GL_DEPTH_TEST|rm.gl.GL_BLEND)
                rm.bullet_render.render_model(self._pb_client, 
                                              agent._body_id,
                                              draw_link=True,
                                              draw_link_info=False,
                                              draw_joint=rm.flag['joint'],
                                              draw_joint_geom=False, 
                                              ee_indices=agent._char_info.end_effector_indices,
                                              color=[colors[i][0], colors[i][1], colors[i][2], 0.5])
                rm.gl.glPopAttrib()
                agent.restore_states(agent_state)

        if rm.flag['kin_model']:
            for i in range(self._num_agent):
                agent = self._kin_agent[i]                
                rm.gl.glPushAttrib(rm.gl.GL_LIGHTING|rm.gl.GL_DEPTH_TEST|rm.gl.GL_BLEND)
                # frame = self._ref_motion[0].time_to_frame(self._elapsed_time)
                # im = self._interaction_meshes[frame]
                # for k in range(im.num_simplex):
                #     p1, p2, p3, p4 = im.get_simplex(k)
                #     rm.gl_render.render_tet_line(p1, p2, p3, p4, color=[0, 0.5, 0, 1.0])
                # for p in agent.interaction_mesh_samples():
                #     rm.gl_render.render_point(p, radius=0.02, color=[0, 1, 0, 1])
                rm.bullet_render.render_model(self._pb_client, 
                                              agent._body_id,
                                              draw_link=True,
                                              draw_link_info=False,
                                              draw_joint=rm.flag['joint'],
                                              draw_joint_geom=False, 
                                              ee_indices=agent._char_info.end_effector_indices,
                                              color=[colors[i][0], colors[i][1], colors[i][2], 0.5])
                if rm.flag['com_vel']:
                    p, Q, v, w = agent.get_root_state()
                    p, v = agent.get_com_and_com_vel()
                    rm.gl_render.render_arrow(p, p+v, D=0.01, color=[0.5, 0.5, 0.5, 1])
                rm.gl.glPopAttrib()

        for i in range(self._num_agent):
            # if self._ref_motion[i].num_frame() > 0:
            #     for j in range(len(self._imit_window)):
            #         t = self._elapsed_time + self._imit_window[j] - self._imit_window[0]
            #         pose = self._ref_motion[i].get_pose_by_time(t)
            #         vel = self._ref_motion[i].get_velocity_by_time(t)
            #         agent = self._kin_agent[i]
            #         agent.set_pose(pose, vel)
            #         if rm.flag['kin_model']:
            #             rm.gl.glPushAttrib(rm.gl.GL_LIGHTING|rm.gl.GL_DEPTH_TEST|rm.gl.GL_BLEND)
            #             rm.bullet_render.render_model(self._pb_client, 
            #                                           agent._body_id,
            #                                           draw_link=True,
            #                                           draw_link_info=False,
            #                                           draw_joint=rm.flag['joint'],
            #                                           draw_joint_geom=False, 
            #                                           ee_indices=agent._char_info.end_effector_indices,
            #                                           color=[colors[i][0], colors[i][1], colors[i][2], 0.75*(1.0-float(j)/len(self._imit_window))])
            #             rm.gl.glPopAttrib()
            if Env.Task.Heading in self._tasks:
                p1 = self._sim_agent[i].get_facing_position(self._ground_height)
                p2 = p1 + self.get_target_velocity(i)
                p3 = p1 + np.mean(self._com_vel[i], axis=0)
                rm.gl_render.render_arrow(p1, p2, D=0.05, color=[1.0,0.5,0.0], closed=True)
                rm.gl_render.render_arrow(p1, p3, D=0.05, color=[0.0,0.5,1.0], closed=True)
            
            if Env.Task.Dribble  in self._tasks:
                v_com_avg = np.mean(self._com_vel[i], axis=0)
                p_com, _ = self._sim_agent[i].get_com_and_com_vel()
                p_ball = self._ball.p
                p_goal = self._goal_pos

                dir_to_ball = self._sim_agent[i].project_to_ground(p_ball-p_com)
                v_com_avg_proj = mmMath.projectionOnVector(v_com_avg, dir_to_ball)

                p1 = self._sim_agent[i].get_facing_position(self._ground_height)
                p2 = p1 + np.mean(self._com_vel[i], axis=0)
                p3 = p1 + v_com_avg_proj
                rm.gl_render.render_arrow(p1, p2, D=0.05, color=[0.0,0.5,1.0], closed=True)
                rm.gl_render.render_arrow(p1, p3, D=0.05, color=[1.0,0.5,0.0], closed=True)
                rm.gl_render.render_cylinder(mmMath.p2T(self._goal_pos), 0.5, 0.1, color=[1, 0, 0, 1])
            
            if Env.Task.Fight in self._tasks or Env.Task.Chase in self._tasks:
                rm.bullet_render.render_model(self._pb_client, 
                                              self._ring_id,
                                              draw_link=True,
                                              draw_link_info=True,
                                              color=[0.5, 0.5, 0.5, 1.0])

class EnvWithPoseEmbedding1(Env):
    def __init__(self, config):
        super().__init__(config)

        project_dir = config.get("project_dir")
        file_model = config["action"].get("pose_embedding_model")
        file_dataset = config["action"].get("pose_embedding_dataset")
        if project_dir is not None:
            file_model = os.path.join(project_dir, file_model)
            file_dataset = os.path.join(project_dir, file_dataset)

        self.dim_feature = config["action"].get("pose_embedding_dim_feature")
        self.dim_embedding = config["action"].get("pose_embedding_dim_embedding")
        
        model = pose_embedding.Autoencoder(self.dim_feature, self.dim_embedding)
        model.load_state_dict(torch.load(file_model))

        self.pose_embedding_model = model
        self.pose_embedding_dataset = \
            pose_embedding.load_dataset(file_dataset, self._verbose)
    def dim_action(self, idx):
        return self.dim_embedding
    def action_range(self, idx):
        dim = self.dim_action(idx)
        return -np.ones(dim), np.ones(dim)
    def step(self, action):
        target_poses = []
        for i, a in enumerate(action):
            target_pose = copy.deepcopy(self._ref_poses[i])
            x = self.pose_embedding_model.decode(torch.Tensor(a)).detach().numpy()
            x = self.pose_embedding_dataset.postprocess_x(x)
            idx = 0
            for j in target_pose.skel.joints:
                if j==target_pose.skel.root_joint: continue
                target_pose.set_transform(j, mmMath.R2T(mmMath.exp(x[idx:idx+3])), local=True)
                idx += 3
            target_poses.append(target_pose)
        return super().step(target_poses)

class EnvWithPoseEmbedding2(Env):
    def __init__(self, config):
        super().__init__(config)

        project_dir = config.get("project_dir")
        file_model = config["action"].get("pose_embedding_model")
        file_dataset = config["action"].get("pose_embedding_dataset")
        if project_dir is not None:
            file_model = os.path.join(project_dir, file_model)
            file_dataset = os.path.join(project_dir, file_dataset)

        self.dim_feature = config["action"].get("pose_embedding_dim_feature")
        self.dim_embedding = config["action"].get("pose_embedding_dim_embedding")
        
        model = pose_embedding.Autoencoder(self.dim_feature, self.dim_embedding)
        model.load_state_dict(torch.load(file_model))

        self.pose_embedding_model = model
        self.pose_embedding_dataset = \
            pose_embedding.load_dataset(file_dataset, self._verbose)
    def dim_action(self, idx):
        ''' Pose from embedding + Delta Pose '''
        return self.dim_embedding + self.dim_feature
    def action_range(self, idx):
        dim1 = self.dim_embedding
        dim2 = self.dim_feature
        return np.hstack([-np.ones(dim1), -0.5*np.ones(dim2)]), np.hstack([np.ones(dim1), 0.5*np.ones(dim2)])
    def step(self, action):
        target_poses = []
        for i, a in enumerate(action):
            pose_latent, delta_pose = a[:self.dim_embedding], a[self.dim_embedding:]
            x = self.pose_embedding_model.decode(torch.Tensor(pose_latent)).detach().numpy()
            x = self.pose_embedding_dataset.postprocess_x(x)
            assert len(x) == len(delta_pose)
            target_pose = copy.deepcopy(self._ref_poses[i])
            idx = 0
            for j in target_pose.skel.joints:
                if j==target_pose.skel.root_joint: continue
                R = mmMath.exp(x[idx:idx+3])
                dR = mmMath.exp(delta_pose[idx:idx+3])
                target_pose.set_transform(j, mmMath.R2T(np.dot(R,dR)), local=True)
                idx += 3
            target_poses.append(target_pose)
        return super().step(target_poses)

class EnvWithPoseEmbedding3(Env):
    def __init__(self, config):
        super().__init__(config)

        project_dir = config.get("project_dir")
        file_model = config["action"].get("pose_embedding_model")
        file_dataset = config["action"].get("pose_embedding_dataset")
        if project_dir is not None:
            file_model = os.path.join(project_dir, file_model)
            file_dataset = os.path.join(project_dir, file_dataset)

        self.dim_feature = config["action"].get("pose_embedding_dim_feature")
        self.dim_embedding = config["action"].get("pose_embedding_dim_embedding")
        
        model = pose_embedding.Autoencoder(self.dim_feature, self.dim_embedding)
        model.load_state_dict(torch.load(file_model))

        self.pose_embedding_model = model
        self.pose_embedding_dataset = \
            pose_embedding.load_dataset(file_dataset, self._verbose)
    def dim_action(self, idx):
        ''' Pose from embedding + Delta Pose '''
        return self.dim_embedding + self.dim_feature
    def action_range(self, idx):
        dim1 = self.dim_embedding
        dim2 = self.dim_feature
        return np.hstack([-np.ones(dim1), -1.0*np.ones(dim2)]), np.hstack([np.ones(dim1), 1.0*np.ones(dim2)])
    def step(self, action):
        target_poses = []
        for i, a in enumerate(action):
            ''' action is composed of pose_embedding and pose_offset '''
            pose_latent, pose_offset = a[:self.dim_embedding], a[self.dim_embedding:]
            
            ''' Convert pose_embedding x to actual pose by running the decoder '''
            x = self.pose_embedding_model.decode(torch.Tensor(pose_latent)).detach().numpy()
            x = self.pose_embedding_dataset.postprocess_x(x)
            
            ''' Prepare postures to be applied '''
            assert len(x) == len(pose_offset)
            ref_pose = copy.deepcopy(self._ref_poses[i])
            target_pose = copy.deepcopy(self._ref_poses[i])
            
            idx = 0
            for j in ref_pose.skel.joints:
                if j==ref_pose.skel.root_joint: continue
                R = mmMath.exp(x[idx:idx+3])
                dR = mmMath.exp(pose_offset[idx:idx+3])
                ref_pose.set_transform(j, mmMath.R2T(R), local=True)
                target_pose.set_transform(j, mmMath.R2T(np.dot(R,dR)), local=True)
                idx += 3
            
            self._kin_agent[i].set_pose(ref_pose)
            target_poses.append(target_pose)
        return super().step(target_poses)
    def reward_data_auxiliary(self, idx):
        data = {}
        data['kin_root_pQvw'] = self._kin_agent[idx].get_root_state()
        data['kin_link_pQvw'] = self._kin_agent[idx].get_link_states()
        data['kin_joint_pv'] = self._kin_agent[idx].get_joint_states()
        return data
    def reward_auxiliary(self, idx, data_prev, data_next, action, error):
        sim_agent = self._sim_agent[idx]
        char_info = sim_agent._char_info

        data = data_next[idx]

        sim_root_p, sim_root_Q, sim_root_v, sim_root_w = data['sim_root_pQvw']
        sim_link_p, sim_link_Q, sim_link_v, sim_link_w = data['sim_link_pQvw']
        sim_joint_p, sim_joint_v = data['sim_joint_pv']
        R_sim_root_inv = mmMath.Q2R(sim_root_Q).transpose()
        
        kin_root_p, kin_root_Q, kin_root_v, kin_root_w = data['kin_root_pQvw']
        kin_link_p, kin_link_Q, kin_link_v, kin_link_w = data['kin_link_pQvw']
        kin_joint_p, kin_joint_v = data['kin_joint_pv']
        R_kin_root_inv = mmMath.Q2R(kin_root_Q).transpose()

        indices = range(len(sim_joint_p))

        if 'pose_pos' in self._reward_names[idx]:
            error['pose_pos'] = 0.0
            for j in indices:
                joint_type = sim_agent.get_joint_type(j)
                if joint_type == self._pb_client.JOINT_FIXED:
                    continue
                elif joint_type == self._pb_client.JOINT_SPHERICAL:
                    dQ = self._pb_client.getDifferenceQuaternion(sim_joint_p[j], kin_joint_p[j])
                    _, diff_pose_pos = self._pb_client.getAxisAngleFromQuaternion(dQ)
                else:
                    diff_pose_pos = sim_joint_p[j] - kin_joint_p[j]
                error['pose_pos'] += char_info.joint_weight[j] * np.dot(diff_pose_pos, diff_pose_pos)
            if len(indices) > 0:
                error['pose_pos'] /= len(indices)

        if 'ee' in self._reward_names[idx]:
            error['ee'] = 0.0
            
            for j in char_info.end_effector_indices:
                sim_ee_local = np.dot(R_sim_root_inv, sim_link_p[j]-sim_root_p)
                kin_ee_local = np.dot(R_kin_root_inv, kin_link_p[j]-kin_root_p)
                diff_pos =  sim_ee_local - kin_ee_local
                error['ee'] += np.dot(diff_pos, diff_pos)

            if len(char_info.end_effector_indices) > 0:
                error['ee'] /= len(char_info.end_effector_indices)

def keyboard_callback(key):
    global env
    global time_checker_auto_play
    if env is None: return

    if key in rm.toggle:
        rm.flag[rm.toggle[key]] = not rm.flag[rm.toggle[key]]
        print('Toggle:', rm.toggle[key], rm.flag[rm.toggle[key]])
    elif key == b'r':
        env.reset()
        time_checker_auto_play.begin()
    elif key == b'>':
        act = np.zeros(env._action_info.dim)
        # act = np.random.normal(size=env._action_info.dim)
        env.step(action=act)
        print('Elapsed Time:', env._elapsed_time, 'EOE:', env.end_of_episode_reason)
    elif key == b'<':
        print(env.state(env._sim_agent, env._ref_motion))
    elif key == b' ':
        env.step_forward()
        print(env._elapsed_time)
    elif key == b'a':
        print('Key[a]: auto_play:', not rm.flag['auto_play'])
        rm.flag['auto_play'] = not rm.flag['auto_play']
    elif key == b'p':
        env.pfnn.update()
        env._kin_agent.set_pose_by_xform(env.pfnn.character.joint_xform_by_ik)
    elif key == b'o':
        env.throw_obstacle()
    elif key == b'S':
        env._pfnn.command.save_history('data/temp/temp.pfnncommand.gzip')
    else:
        return False
    return True

def idle_callback(allow_auto_play=True):
    global env
    global time_checker_auto_play
    if env is None: return

    if allow_auto_play and rm.flag['auto_play'] and time_checker_auto_play.get_time(restart=False) >= env._dt_con:
        action = np.zeros(np.sum([env._action_info[i].dim for i in range(env._num_agent)]))
        env.step(action)
        time_checker_auto_play.begin()

    if rm.flag['follow_cam']:
        p = env.agent_avg_position()

        if np.allclose(env._v_up, np.array([0.0, 1.0, 0.0])):
            rm.viewer.update_target_pos(p, ignore_y=True)
        elif np.allclose(env._v_up, np.array([0.0, 0.0, 1.0])):
            rm.viewer.update_target_pos(p, ignore_z=True)
        else:
            raise NotImplementedError

def render_callback():
    global env
    if env is None: return

    if rm.flag['ground']:
        if rm.tex_id_ground is None:
            rm.tex_id_ground = rm.gl_render.load_texture(rm.file_tex_ground)
        rm.gl_render.render_ground_texture(rm.tex_id_ground,
                                        size=[40.0, 40.0], 
                                        dsize=[2.0, 2.0], 
                                        axis=kinematics.axis_to_str(env._v_up),
                                        origin=rm.flag['origin'],
                                        use_arrow=True,
                                        circle_cut=True)
    if rm.flag['fog']:
        density = 0.05;
        fogColor = [1.0, 1.0, 1.0, 1.0]
        rm.gl.glEnable(rm.gl.GL_FOG)
        rm.gl.glFogi(rm.gl.GL_FOG_MODE, rm.gl.GL_EXP2)
        rm.gl.glFogfv(rm.gl.GL_FOG_COLOR, fogColor)
        rm.gl.glFogf(rm.gl.GL_FOG_DENSITY, density)
        rm.gl.glHint(rm.gl.GL_FOG_HINT, rm.gl.GL_NICEST)
    else:
        rm.gl.glDisable(rm.gl.GL_FOG)

    env.render(rm.flag, rm.COLORS_FOR_AGENTS)

def overlay_callback():
    return

env = None
pi = None
if __name__ == '__main__':
    import argparse

    rm.initialize()
    
    def arg_parser():
        num_agent = 1
        parser = argparse.ArgumentParser()
        parser.add_argument('--env_mode', 
            choices=['imitation'], type=str, default='imitation')
        parser.add_argument('--action_type', 
            choices=['absolute', 'relative'], type=str, default='relative')
        parser.add_argument('--ref_motion_file', 
            action='append', type=str, default=num_agent*['data/motion/amass/amass_hierarchy1.bvh'])
        parser.add_argument('--char_info_module', 
            action='append', type=str, default=num_agent*['amass_char_info.py'])
        parser.add_argument('--ref_motion_scale', 
            action='append', type=float, default=num_agent*[1.0])
        parser.add_argument('--sim_char_file', 
            action='append', type=str, default=num_agent*['data/character/amass.urdf'])
        parser.add_argument('--base_motion_file', 
            action='append', type=str, default=num_agent*['data/motion/amass/amass_hierarchy.bvh'])
        return parser
    
    print('=====Multi-agent Controller=====')
    
    args = basics.parse_args_by_file(arg_parser, sys.argv)
    print(args.ref_motion_file)

    env = Env(config)

    cam_origin = env.agent_avg_position()

    if np.allclose(env._v_up, np.array([0.0, 1.0, 0.0])):
        cam_pos = cam_origin + np.array([0.0, 2.0, 3.0])
        cam_vup = np.array([0.0, 1.0, 0.0])
    elif np.allclose(env._v_up, np.array([0.0, 0.0, 1.0])):
        cam_pos = cam_origin + np.array([3.0, 0.0, 2.0])
        cam_vup = np.array([0.0, 0.0, 1.0])
    else:
        raise NotImplementedError

    cam = rm.camera.Camera(pos=cam_pos,
                           origin=cam_origin, 
                           vup=cam_vup, 
                           fov=45)

    rm.viewer.run(title='env',
                  cam=cam,
                  size=(1280, 720),
                  keyboard_callback=keyboard_callback,
                  render_callback=render_callback,
                  overlay_callback=overlay_callback,
                  idle_callback=idle_callback,
                  )
