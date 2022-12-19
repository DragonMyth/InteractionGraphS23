'''
python3 env_humanoid_kinematics.py

press the space key for proceeding one-step
press 'a' for proceeding continuously
press 'r' for reset the environment
'''


import os

''' 
This forces the environment to use only 1 cpu when running.
This is helpful to launch multiple environment simulatenously.
'''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import copy

import pybullet as pb
import pybullet_data

from bullet import bullet_client
from bullet import bullet_utils as bu

from fairmotion.ops import conversions
from fairmotion.ops import math
from fairmotion.utils import constants
from fairmotion.core.motion import Pose
from fairmotion.core.velocity import MotionWithVelocity
from fairmotion.data import bvh

import sim_agent
import importlib.util

import motion_utils

from abc import ABCMeta, abstractmethod

class Env(metaclass=ABCMeta):
    '''
    This environment defines a base environment where the simulated 
    characters exist and they are controlled by tracking controllers
    '''
    def __init__(self, 
                 fps,
                 past_window_size,
                 skip_frames,
                 char_info_module,
                 sim_char_file,
                 ref_motion_scale,
                 base_motion_file,
                 ref_motion_file,
                 verbose=False,
                 ):
        self._num_agent = len(sim_char_file)
        assert self._num_agent > 0
        assert self._num_agent == len(char_info_module)
        assert self._num_agent == len(ref_motion_scale)

        self._char_info = []
        for i in range(self._num_agent):
            ''' Load Character Info Moudle '''
            spec = importlib.util.spec_from_file_location(
                "char_info%d"%(i), char_info_module[i])
            char_info = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(char_info)
            self._char_info.append(char_info)

        self._v_up = self._char_info[0].v_up_env

        ''' Define PyBullet Client '''
        self._pb_client = bullet_client.BulletClient(
            connection_mode=pb.DIRECT, options=' --opengl2')
        self._pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        ''' timestep for control '''
        self._dt = 1.0/fps

        self._past_window_size = past_window_size
        self._skip_frames = skip_frames

        self._verbose = verbose

        self._agent = []
        for i in range(self._num_agent):
            self._agent.append(
                sim_agent.SimAgent(pybullet_client=self._pb_client, 
                                   model_file=sim_char_file[i],
                                   char_info=self._char_info[i],
                                   ref_scale=ref_motion_scale[i],
                                   self_collision=False,
                                   kinematic_only=True,
                                   verbose=verbose))

        self._base_pose = []
        for i in range(self._num_agent):
            m = bvh.load(file=base_motion_file[i],
                         motion=MotionWithVelocity(),
                         scale=1.0, 
                         load_skel=True,
                         load_motion=True,
                         v_up_skel=self._char_info[i].v_up, 
                         v_face_skel=self._char_info[i].v_face, 
                         v_up_env=self._char_info[i].v_up_env)
            self._base_pose.append(m.get_pose_by_frame(0))

        ''' Load Reference Motion '''

        self._ref_motion_all = []
        self._ref_motion_file_names = []
        for i in range(self._num_agent):
            ref_motion_all, ref_motion_file_names = \
                motion_utils.load_motions(
                    ref_motion_file[i], 
                    None,
                    self._agent[i]._char_info,
                    self._verbose)
            self._ref_motion_all.append(ref_motion_all)
            self._ref_motion_file_names.append(ref_motion_file_names)
        
        self._prev_poses = [[] for i in range(self._num_agent)]

        self._cur_ref_motion = [None for i in range(self._num_agent)]
        self._cur_time = np.zeros(self._num_agent)

        ''' Elapsed time after the environment starts '''
        self._elapsed_time = 0.0
        ''' For tracking the length of current episode '''
        self._episode_len = 0.0

    @abstractmethod
    def state(self, idx):
        raise NotImplementedError

    def dim_state(self, idx):
        return len(self.state(idx))

    def dim_action(self, idx):
        ''' characters DOF except for the root joint '''
        return 51

    def action_range(self, idx):
        dim_action = self.dim_action(idx)
        return -3.0 * np.ones(dim_action), 3.0 * np.ones(dim_action)

    @abstractmethod
    def reward(self, idx):
        raise NotImplementedError

    @abstractmethod
    def collect_step_info(self):
        raise NotImplementedError

    @abstractmethod
    def inspect_end_of_episode(self):
        raise NotImplementedError        

    def compute_pose_from_action(self, idx, action):
        agent = self._agent[idx]
        char_info = agent._char_info

        ref_pose = copy.deepcopy(self._base_pose[idx])

        dof_cnt = 0
        for j in agent._joint_indices:
            joint_type = agent.get_joint_type(j)
            ''' Fixed joint will not be affected '''
            if joint_type == self._pb_client.JOINT_FIXED:
                continue
            ''' If the joint do not have correspondance, use the reference posture itself'''
            if char_info.bvh_map[j] == None:
                continue

            T = ref_pose.skel.get_joint(char_info.bvh_map[j]).xform_from_parent_joint
            
            R, p = conversions.T2Rp(T)
            if joint_type == self._pb_client.JOINT_SPHERICAL:
                dR = conversions.A2R(action[dof_cnt:dof_cnt+3])
                dof_cnt += 3
            elif joint_type == self._pb_client.JOINT_REVOLUTE:
                axis = agent.get_joint_axis(j)
                angle = action[dof_cnt:dof_cnt+1]
                dR = conversions.A2R(axis*angle)
                dof_cnt += 1
            else:
                raise NotImplementedError
            T_new = conversions.Rp2T(np.dot(R, dR), p)
            ref_pose.set_transform(char_info.bvh_map[j], T_new, do_ortho_norm=False, local=True)

        return ref_pose

    def step(self, actions):

        ''' Increase elapsed time '''
        self._cur_time += self._dt
        self._elapsed_time += self._dt
        self._episode_len += self._dt

        ''' Update simulation '''
        for i, a in enumerate(actions):
            pose = self.compute_pose_from_action(i, a)
            self._agent[i].set_pose(pose=pose)
            self._prev_poses[i].append(pose)

        rews = [self.reward(i) for i in range(self._num_agent)]
        info = self.collect_step_info()

        return rews, info

    def reset(self):

        self._elapsed_time = 0.0
        self._episode_len = 0.0

        '''
        Sample a reference motion to start and set prev poses/vels by using it
        '''

        for i in range(self._num_agent):
            self._prev_poses[i].clear()

        for i in range(self._num_agent):
            idx = np.random.randint(len(self._ref_motion_all[i]))
            ref_motion = self._ref_motion_all[i][idx]
            margin = self._dt * self._past_window_size * self._skip_frames
            start_time = np.random.uniform(
                margin, 
                ref_motion.length() - margin
            )
            for j in range(-self._past_window_size*self._skip_frames-1, 0):
                time = max(0.0, start_time + j * self._dt)
                self._prev_poses[i].append(ref_motion.get_pose_by_time(time))
            self._agent[i].set_pose(self._prev_poses[i][-1])
            self._cur_ref_motion[i] = ref_motion
            self._cur_time[i] = start_time

    def render(self, rm, ground_height=0.0):
        colors = rm.COLORS_FOR_AGENTS

        rm.gl.glEnable(rm.gl.GL_LIGHTING)
        rm.gl.glEnable(rm.gl.GL_BLEND)
        rm.gl.glBlendFunc(rm.gl.GL_SRC_ALPHA, rm.gl.GL_ONE_MINUS_SRC_ALPHA)

        for i in range(self._num_agent):
            agent = self._agent[i]
            char_info = self._char_info[i]
            if rm.flag['sim_model']:
                rm.gl.glEnable(rm.gl.GL_DEPTH_TEST)
                if rm.flag['shadow']:
                    rm.gl.glPushMatrix()
                    d = np.array([1, 1, 1])
                    d = d - math.projectionOnVector(d, char_info.v_up_env)
                    offset = (0.001 + ground_height) * char_info.v_up_env
                    rm.gl.glTranslatef(offset[0], offset[1], offset[2])
                    rm.gl.glScalef(d[0], d[1], d[2])
                    rm.bullet_render.render_model(self._pb_client, 
                                               agent._body_id, 
                                               draw_link=True, 
                                               draw_link_info=False, 
                                               draw_joint=False, 
                                               draw_joint_geom=False, 
                                               ee_indices=None, 
                                               color=[0.5,0.5,0.5,1.0],
                                               lighting=False)
                    rm.gl.glPopMatrix()
                rm.bullet_render.render_model(self._pb_client, 
                                              agent._body_id,
                                              draw_link=True, 
                                              draw_link_info=True, 
                                              draw_joint=rm.flag['joint'], 
                                              draw_joint_geom=True, 
                                              ee_indices=char_info.end_effector_indices, 
                                              color=colors[i])
                if rm.flag['com_vel']:
                    p, Q, v, w = agent.get_root_state()
                    p, v = agent.get_com_and_com_vel()
                    rm.gl_render.render_arrow(p, p+v, D=0.01, color=[0, 0, 0, 1])
                if rm.flag['facing_frame']:
                    rm.gl.glPushAttrib(rm.gl.GL_LIGHTING|rm.gl.GL_DEPTH_TEST|rm.gl.GL_BLEND)
                    rm.gl.glEnable(rm.gl.GL_BLEND)
                    rm.gl_render.render_transform(
                        agent.get_facing_transform(ground_height), 
                        scale=0.5, 
                        use_arrow=True)
                    rm.gl.glPopAttrib()


