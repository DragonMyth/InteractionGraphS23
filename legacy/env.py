
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import numpy as np
import math
import time
import copy
from enum import Enum
from collections import deque

import pybullet as pb
import pybullet_data
from pybullet_utils import pd_controller_stable

from basecode.bullet import bullet_client
from basecode.bullet import bullet_utils as bu
from basecode.motion import kinematics_simple as kinematics
from basecode.math import mmMath
from basecode.utils import basics
from basecode.rl import action as ac

# import mimicpfnn_self.char_info as self.char_info
# import cmuV2_self.char_info as self.char_info
import sim_agent

import pickle
import gzip

import sim_obstacle

import importlib.util

DT_SIM = 1.0 / 480.0
DT_CON = 1.0 / 30.0
DT_MOT = 1.0 / 60.0

# For viewers
flag = {}
flag['follow_cam'] = True
flag['ground'] = True
flag['origin'] = False
flag['shadow'] = True
flag['sim_model'] = True
flag['kin_model'] = True
flag['joint'] = False
flag['com_vel'] = False
flag['collision'] = True
flag['overlay'] = False
flag['auto_play'] = False
flag['fog'] = False
flag['facing_frame'] = False

toggle = {}
toggle[b'0'] = 'follow_cam'
toggle[b'1'] = 'ground'
toggle[b'2'] = 'origin'
toggle[b'3'] = 'shadow'
toggle[b'4'] = 'sim_model'
toggle[b'5'] = 'kin_model'
toggle[b'6'] = 'joint'
toggle[b'7'] = 'com_vel'
toggle[b'8'] = 'collision'
toggle[b'9'] = 'overlay'
toggle[b'a'] = 'auto_play'
toggle[b'f'] = 'fog'
toggle[b'F'] = 'facing_frame'

time_checker_auto_play = basics.TimeChecker()

viewer = None
gl_render = None
bullet_render = None
gl = None
glu = None
glut = None
pfnn = None
camera = None

tex_id_ground = None
file_tex_ground = "data/image/grid2.png"

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
    class Mode(Enum):
        DeepMimic=0
        MimicPFNN=1
        MotionGraph=2
        @classmethod
        def from_string(cls, string):
            if string=="deepmimic": return cls.DeepMimic
            if string=="mimicpfnn": return cls.MimicPFNN
            if string=="motiongraph": return cls.MotionGraph
            raise NotImplementedError
    class ActionType(Enum):
        NONE=0  # No control
        SPD=1   # Stable PD Control
        PD=2    # PD Control
        CPD=3   # PD Control as Constraints of Simulation
        CP=4    # Position Control as Constraints of Simulation
        V=5     # Velocity Control as Constraints of Simulation
        @classmethod
        def from_string(cls, string):
            if string=="none": return cls.NONE
            if string=="spd": return cls.SPD
            if string=="pd": return cls.PD
            if string=="cpd": return cls.CPD
            if string=="cp": return cls.CP
            if string=="v": return cls.V
            raise NotImplementedError
    class ActionMode(Enum):
        Absolute=0 # Use an absolute posture as an action
        Relative=1 # Use a relative posture from a reference posture as an action
        @classmethod
        def from_string(cls, string):
            if string=="absolute": return cls.Absolute
            if string=="relative": return cls.Relative
            raise NotImplementedError
    class RewardMode(Enum):
        Sum=0 # Reward is a summation of given terms
        Mul=1 # Reward is a multiplication of given terms
        @classmethod
        def from_string(cls, string):
            if string=="sum": return cls.Sum
            if string=="mul": return cls.Mul
            raise NotImplementedError
    def __init__(self, 
                 dt_sim, 
                 dt_con, 
                 # Mode of this environment
                 mode="deepmimic",
                 verbose=False,
                 sim_window=math.inf, 
                 end_margin=1.5,
                 pfnn_command_type="autonomous",
                 pfnn_command_record=False,
                 pfnn_command_file=None,
                 state_choices=None,
                 state_imit_window=None,
                 action_type="spd",
                 action_mode="relative",
                 action_range_min=-1.5,
                 action_range_max=1.5,
                 action_range_min_pol=-15,
                 action_range_max_pol=15,
                 reward_choices=[],
                 reward_mode="sum",
                 ref_motion_files=None,
                 add_noise=False,
                 early_term_choices=[],
                 et_low_reward_thres=0.1,
                 et_falldown_contactable_body=[],
                 char_info_module=None,
                 sim_char_file=None,
                 base_motion_file=None,
                 motion_graph_file=None,
                 ref_motion_scale=1.0,
                 self_collision=True,
                 ref_motion_sample='random',
                 visualization=False,
                 reward_weight_scale=0.5,
                 ):
        ''' Load Modules for Visualzation if needed '''
        self.visualization = visualization
        if self.visualization:
            global viewer, gl_render, camera, bullet_render, gl, glu, glut
            from basecode.render import glut_viewer as viewer
            from basecode.render import gl_render
            from basecode.render import camera
            from basecode.bullet import bullet_render
            import OpenGL.GL as gl
            import OpenGL.GLU as glu
            import OpenGL.GLUT as glut

        ''' Load Character Info Moudle '''
        spec = importlib.util.spec_from_file_location("char_info", char_info_module)
        char_info = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(char_info)
        self.char_info = char_info
        self.sim_char_file = sim_char_file

        ''' Modfiy Contactable Body Parts '''
        contact_allow_all = True if 'all' in et_falldown_contactable_body else False
        for key in list(self.char_info.contact_allow_map.keys()):
            if contact_allow_all or key in et_falldown_contactable_body:
                self.char_info.contact_allow_map[key] = True

        ''' Define PyBullet Client '''
        self._pb_client = bullet_client.BulletClient(connection_mode=pb.DIRECT, options=' --opengl2')
        self._pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        ''' timestep for physics simulation '''
        self._dt_sim = dt_sim
        ''' timestep for control of dynamic controller '''
        self._dt_con = dt_con
        ''' timestep for control of kinematic motion controller '''
        self._dt_mot = DT_MOT

        assert self._dt_con > self._dt_sim
        assert self._dt_con > self._dt_mot
        
        self._num_iter_per_step_sim = int(self._dt_con / self._dt_sim)
        self._num_iter_per_step_mot = int(self._dt_con / self._dt_mot)

        ''' State Choices'''
        if state_choices is None or len(state_choices)==0: 
            state_choices = ['body', 'imit_abs', 'imit_rel']
        self._state_body        = True if 'body'       in state_choices else False
        self._state_imit_abs    = True if 'imit_abs'   in state_choices else False
        self._state_imit_rel    = True if 'imit_rel'   in state_choices else False

        ''' Early Terminations Choices'''
        self._et_sim_div = True
        self._et_sim_window = True
        self._et_task_complete = True if 'task_complete' in early_term_choices else False
        self._et_falldown      = True if 'falldown'      in early_term_choices else False
        self._et_root_fail     = True if 'root_fail'     in early_term_choices else False
        self._et_low_reward    = True if 'low_reward'    in early_term_choices else False

        ''' Reward Choices '''
        if reward_choices is None or len(reward_choices)==0:
            reward_choices = ['pose', 'vel', 'ee', 'root', 'com']
        self._rew_pose = True if 'pose' in reward_choices else False
        self._rew_vel  = True if 'vel'  in reward_choices else False
        self._rew_ee   = True if 'ee'   in reward_choices else False
        self._rew_root = True if 'root' in reward_choices else False
        self._rew_com  = True if 'com'  in reward_choices else False

        if self._et_low_reward:
            self._et_low_reward_thres = et_low_reward_thres
            self.rew_queue = deque(maxlen=int(1.0/self._dt_con))
        
        ''' The environment automatically terminates after 'sim_window' seconds '''
        self._sim_window_time = sim_window
        ''' 
        The environment continues for "end_margin" seconds after end-of-episode is set by TRUE.
        This is useful for making the controller work for boundaries of reference motions
        '''
        self._end_margin_time = end_margin
        self._verbose = verbose

        ''' Load Skeleton for Reference Motion '''
        self._ref_motion_skel = kinematics.Motion(file=base_motion_file, 
                                                  scale=1.0, 
                                                  load_motion=False,
                                                  v_up_skel=self.char_info.v_up, 
                                                  v_face_skel=self.char_info.v_face, 
                                                  v_up_env=self.char_info.v_up_env, 
                                                  ).skel
        # for j in self._ref_motion_skel.joints:
        #     print(mmMath.T2p(j.xform_from_parent_joint))

        self._mode = Env.Mode.from_string(mode)
        if self._mode == Env.Mode.DeepMimic:
            assert ref_motion_files is not None
            '''
            DeepMimic can also be trained by multiple motions.
            If we have multiple reference motions, randomly select one every episode.
            '''
            self._ref_motion_file_names = []
            for names in ref_motion_files:
                head, tail = os.path.split(names)
                self._ref_motion_file_names.append(tail)
            if isinstance(ref_motion_files[0], str):
                ref_motion_dict = {}
                self._ref_motion_all = []
                for file in ref_motion_files:
                    ''' If the same file is already loaded, do not load again for efficiency'''
                    if file in ref_motion_dict:
                        ref_motion = ref_motion_dict[file]
                    else:
                        if file.endswith('bvh'):
                            ref_motion = kinematics.Motion(skel=self._ref_motion_skel, 
                                                           file=file, 
                                                           scale=1.0, 
                                                           load_skel=False,
                                                           v_up_skel=self.char_info.v_up, 
                                                           v_face_skel=self.char_info.v_face, 
                                                           v_up_env=self.char_info.v_up_env)
                        elif file.endswith('bin'):
                            ref_motion = pickle.load(open(file, "rb"))
                        elif file.endswith('gzip') or file.endswith('gz'):
                            with gzip.open(file, "rb") as f:
                                ref_motion = pickle.load(f)
                        else:
                            raise Exception('Unknown Motion File Type')
                        ''' 
                        Move the first posture to the origin to prevent that
                        the character is simulated out of the ground plane
                        '''
                        p = ref_motion.get_pose_by_frame(0).get_facing_position()
                        ref_motion.translate(-p)
                        ref_motion_dict[file] = ref_motion
                        if self._verbose: 
                            print('Loaded: %s'%file)
                    self._ref_motion_all.append(ref_motion)
            elif isinstance(ref_motion_files[0], kinematics.Motion):
                self._ref_motion_all = ref_motion_files
            else:
                raise Exception('Unknown Type for Reference Motion')
            assert len(self._ref_motion_all) > 0
            ''' Some information to check which reference motion is hard to learn '''
            self._ref_motion_avg_ep_len = np.zeros(len(self._ref_motion_all))
            self._ref_motion_sample_prob = np.ones(len(self._ref_motion_all))

            self._ref_motion_idx = 0
            self._ref_motion_sample = ref_motion_sample
            self._ref_motion = self.sample_ref_motion()
            self._ref_motion_scale = ref_motion_scale
        elif self._mode == Env.Mode.MimicPFNN:
            global pfnn
            import pfnn
            ''' 
            Reference motion that the simulated character follows.
            This will be extended on the fly by PFNN controller.
            '''
            self._ref_motion = kinematics.Motion(skel=self._ref_motion_skel, 
                                                 file=base_motion_file, 
                                                 scale=1.0,
                                                 load_skel=False,
                                                 v_up_skel=self.char_info.v_up, 
                                                 v_face_skel=self.char_info.v_face, 
                                                 v_up_env=self.char_info.v_up_env)
            self._ref_motion.clear()
            '''
            Load a pre-computed PFNN controller
            '''
            self._pfnn_command_type = pfnn.Runner.UserInput.from_string(pfnn_command_type)
            self._pfnn = pfnn.Runner(user_input=pfnn_command_type, 
                                     record=pfnn_command_record,
                                     command_file=pfnn_command_file)
            self._ref_motion_scale = ref_motion_scale
        elif self._mode == Env.Mode.MotionGraph:
            ''' 
            Reference motion that the simulated character follows.
            This will be extended on the fly by MotionGraph controller.
            '''
            self._ref_motion = kinematics.Motion(skel=self._ref_motion_skel, 
                                                 file=base_motion_file, 
                                                 scale=1.0,
                                                 load_skel=False,
                                                 v_up_skel=self.char_info.v_up, 
                                                 v_face_skel=self.char_info.v_face, 
                                                 v_up_env=self.char_info.v_up_env)
            self._ref_motion.clear()
            '''
            Load a pre-computed MotionGraph controller
            '''
            with gzip.open(motion_graph_file, "rb") as f:
                self._mg = pickle.load(f)
            self._mg_node_idx = np.random.randint(self._mg.graph.number_of_nodes())
            self._ref_motion_scale = ref_motion_scale
        else:
            raise NotImplementedError

        self.setup_physics_scene(self_collision)

        self._rew_mode = Env.RewardMode.from_string(reward_mode)

        if state_imit_window is None or len(state_imit_window)==0: 
            state_imit_window = [0.3, 0.6, 0.9]
            
        self._imit_window = state_imit_window

        self._action_mode = Env.ActionMode.from_string(action_mode)
        self._action_type = action_type
        
        real_val_max = action_range_max
        real_val_min = action_range_min
        norm_val_max = action_range_max_pol
        norm_val_min = action_range_min_pol
        dim_pose = self._sim_agent.get_num_dofs()
        normalizer_pose = basics.Normalizer(real_val_max=real_val_max*np.ones(dim_pose),
                                            real_val_min=real_val_min*np.ones(dim_pose),
                                            norm_val_max=norm_val_max*np.ones(dim_pose),
                                            norm_val_min=norm_val_min*np.ones(dim_pose))
        self._action_info = Action(dim=normalizer_pose.dim, init_args=[normalizer_pose])

        ''' Start time of the environment '''
        self._start_time = 0.0
        ''' Elapsed time after the environment starts '''
        self._elapsed_time = 0.0
        ''' For tracking the length of current episode '''
        self._episode_len = 0.0
        
        ''' Create a Manager for Handling Obstacles '''
        self._obs_manager = sim_obstacle.ObstacleManager(
            self._pb_client, self._dt_con, self.char_info.v_up_env, self.visualization)

        self.add_noise = add_noise
        self.reward_weight_scale = reward_weight_scale

        self.reset()

        if self._verbose:
            print('----- Environment Created -----')
            print('Mode:', mode)
            print('Dimensions: state(%d) and action(%d)'%(len(self.state()), self._action_info.dim))
            print('-------------------------------')

    def throw_obstacle(self):
        size = np.random.uniform(0.1, 0.3, 3)
        p, Q, v, w = self._sim_agent.get_root_state()
        self._obs_manager.throw(p, size=size)

    def setup_physics_scene(self, self_collision):
        self._pb_client.resetSimulation()

        ''' Create Plane '''
        tf_plane = mmMath.R2Q(mmMath.getSO3FromVectors(np.array([0.0, 0.0, 1.0]), self.char_info.v_up_env))
        self._plane_id = self._pb_client.loadURDF("plane_implicit.urdf", [0, 0, 0],
                                        tf_plane,
                                        useMaximalCoordinates=True)
        self._pb_client.changeDynamics(self._plane_id, linkIndex=-1, lateralFriction=0.9)

        ''' Dynamics parameters '''
        assert np.allclose(np.linalg.norm(self.char_info.v_up_env), 1.0)
        gravity = -9.8 * self.char_info.v_up_env
        self._pb_client.setGravity(gravity[0], gravity[1], gravity[2])
        self._pb_client.setTimeStep(self._dt_sim)
        self._pb_client.setPhysicsEngineParameter(numSubSteps=2)
        self._pb_client.setPhysicsEngineParameter(numSolverIterations=10)
        # self._pb_client.setPhysicsEngineParameter(solverResidualThreshold=1e-10)

        ''' sim_agent is used for simulation whereas kin_agent is used only for kinematics '''
        self._sim_agent = sim_agent.SimAgent(pybullet_client=self._pb_client, 
                                             model_file=self.sim_char_file, 
                                             char_info=self.char_info, 
                                             ref_scale=self._ref_motion_scale,
                                             self_collision=self_collision,
                                             kinematic_only=False,
                                             verbose=self._verbose)
        self._kin_agent = sim_agent.SimAgent(pybullet_client=self._pb_client, 
                                             model_file=self.sim_char_file, 
                                             char_info=self.char_info,
                                             ref_scale=self._ref_motion_scale,
                                             self_collision=self_collision,
                                             kinematic_only=True,
                                             verbose=self._verbose)

    def add_noise_to_pose(self, agent, pose):
        ref_pose = copy.deepcopy(pose)
        dof_cnt = 0
        for j in agent._joint_indices:
            joint_type = agent.get_joint_type(j)
            ''' Fixed joint will not be affected '''
            if joint_type == self._pb_client.JOINT_FIXED:
                continue
            ''' If the joint do not have correspondance, use the reference posture itself'''
            if self.char_info.bvh_map[j] == None:
                continue
            T = ref_pose.get_transform(self.char_info.bvh_map[j], local=True)
            R, p = mmMath.T2Rp(T)
            if joint_type == self._pb_client.JOINT_SPHERICAL:
                dR = basics.random_disp_rot(mu_theta=self.char_info.noise_pose_mu[j],
                                            sigma_theta=self.char_info.noise_pose_sigma[j],
                                            lower_theta=self.char_info.noise_pose_lower[j],
                                            upper_theta=self.char_info.noise_pose_upper[j])
                dof_cnt += 3
            elif joint_type == self._pb_client.JOINT_REVOLUTE:
                theta = basics.truncnorm(mu=self.char_info.noise_pose_mu[j],
                                         sigma=self.char_info.noise_pose_sigma[j],
                                         lower=self.char_info.noise_pose_lower[j],
                                         upper=self.char_info.noise_pose_upper[j])
                joint_axis = self._sim_agent.get_joint_axis(j)
                dR = mmMath.exp(joint_axis, theta=theta)
                dof_cnt += 1
            else:
                raise NotImplementedError
            T_new = mmMath.Rp2T(np.dot(R, dR), p)
            ref_pose.set_transform(self.char_info.bvh_map[j], T_new, do_ortho_norm=False, local=True)
        return ref_pose

    def set_action(self, agent, action, clamp=True, normalized=True):
        ''' 
        Now, we assume that action is a deviation of the given posture.
        The current posture given by the reference motion will be modified by the given action.
        '''
        
        ''' the current posture should be deepcopied because action will modify it '''
        ref_pose = copy.deepcopy(self.get_current_pose_from_motion())

        a_norm = action if normalized else self._action_info.real_to_norm(action)
        a_norm = self._action_info.clamp(a_norm) if clamp else a_norm
        a_real = self._action_info.norm_to_real(a_norm)

        dof_cnt = 0
        for j in agent._joint_indices:
            joint_type = agent.get_joint_type(j)
            ''' Fixed joint will not be affected '''
            if joint_type == self._pb_client.JOINT_FIXED:
                continue
            ''' If the joint do not have correspondance, use the reference posture itself'''
            if self.char_info.bvh_map[j] == None:
                continue
            if self._action_mode == Env.ActionMode.Relative:
                T = ref_pose.get_transform(self.char_info.bvh_map[j], local=True)
            elif self._action_mode == Env.ActionMode.Absolute:
                T = ref_pose.skel.get_joint(self.char_info.bvh_map[j]).xform_from_parent_joint
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
            ref_pose.set_transform(self.char_info.bvh_map[j], T_new, do_ortho_norm=False, local=True)

        self._target_pose = ref_pose

    def tracking_control(self, agent, pose=None, vel=None):
        agent.tracking_control(pose, vel, self._action_type)
    
    def reset(self, start_time=None, change_ref_motion=True, ref_motion_idx=None):

        ''' remove obstacles in the scene '''
        self._obs_manager.clear()

        self._start_time = 0.0
        self._elapsed_time = 0.0
        self._target_pose = None

        if self._mode == Env.Mode.DeepMimic:
            if self._ref_motion_sample == 'adaptive':
                val = self._ref_motion_avg_ep_len[self._ref_motion_idx]
                self._ref_motion_avg_ep_len[self._ref_motion_idx] = 0.9*val + 0.1*self._episode_len
                ep_len_avg_total = np.mean(self._ref_motion_avg_ep_len)
                ep_len_avg_total += 1.0e-5
                self._ref_motion_sample_prob = np.clip(
                    np.exp(-5.0*(self._ref_motion_avg_ep_len-ep_len_avg_total)/ep_len_avg_total), 0.05, 0.95)
            
            if ref_motion_idx is not None:
                assert ref_motion_idx < len(self._ref_motion_all)
                self._ref_motion = self._ref_motion_all[ref_motion_idx]
            elif change_ref_motion:
                ''' Select a reference motion randomly if there exist multiple reference motions '''
                self._ref_motion = self.sample_ref_motion()
            
            if start_time is not None: 
                self._elapsed_time = self._start_time = start_time
            else:
                ''' Select a start time randomly '''
                t = np.random.uniform(0.0, self.end_time_of_ref_motion())
                self._elapsed_time = self._start_time = t
        
        elif self._mode == Env.Mode.MimicPFNN:
            ''' Reset PFNN and generate a few motions for the initialization '''
            self.generate_reference_motion_by_pfnn(self._imit_window[-1], reset=True)
        
        elif self._mode == Env.Mode.MotionGraph:
            met_dead_end = True
            while met_dead_end:
                met_dead_end = self.generate_reference_motion_by_mg(self._imit_window[-1], reset=True)
        
        else:
            raise NotImplementedError

        ''' Set the state of simulated agent by using the state of reference motion '''
        cur_pose, cur_vel = self.get_current_pose_from_motion(), self.get_current_vel_from_motion()
        ''' Add noise to the state if necessary '''
        if self.add_noise:
            cur_pose = self.add_noise_to_pose(self._sim_agent, cur_pose)
            
        self._sim_agent.set_pose(cur_pose, cur_vel)
        self._kin_agent.set_pose(cur_pose, cur_vel)
        
        self.end_of_episode = False
        self.end_of_episode_reason = None

        if self._et_low_reward:
            self.rew_queue.clear()
            for i in range(self.rew_queue.maxlen):
                self.rew_queue.append(self.reward_max())

        self._episode_len = 0.0

        self.draw_data = {}
        self.draw_data['state_imit'] = []

    def sample_ref_motion(self):
        '''
        Given a set of reference motions, 
        "random" picks a motion randomly with uniform probability
        "adaptive" picks a motion randomly where probabilities are inversely propotional to the average episode length
        '''
        if self._ref_motion_sample=='random':
            self._ref_motion_idx = np.random.randint(len(self._ref_motion_all))
        elif self._ref_motion_sample=='adaptive':
            for i in range(100):
                idx = np.random.randint(len(self._ref_motion_all))
                prob = self._ref_motion_sample_prob[idx]
                if np.random.uniform(0.0, 1.0) <= prob:
                    break
            self._ref_motion_idx = idx
        else:
            raise NotImplementedError
        return self._ref_motion_all[self._ref_motion_idx]

    def generate_reference_motion_by_pfnn(self, end_time, reset=True):
        if reset:
            self._pfnn.reset()
            self._ref_motion.clear()
            self._ref_motion.add_one_frame(0.0, copy.deepcopy(self._pfnn.character.joint_xform_by_ik))
        t = 0.0 if self._ref_motion.num_frame()==0 else self._ref_motion.times[-1]
        while t <= end_time:
            for _ in range(self._num_iter_per_step_mot):
                self._pfnn.update()
            t += self._dt_con
            self._ref_motion.add_one_frame(t, copy.deepcopy(self._pfnn.character.joint_xform_by_ik))

    def generate_reference_motion_by_mg(self, end_time, reset=True):
        if reset:
            self._mg_node_idx = np.random.randint(self._mg.graph.number_of_nodes())
            self._ref_motion.clear()
        graph = self._mg.graph
        node_idx = self._mg_node_idx
        t = 0.0 if self._ref_motion.num_frame()==0 else self._ref_motion.times[-1]
        while t < end_time:
            num_out_edges = graph.out_degree(node_idx)
            if num_out_edges == 0: break
            ''' Append the selected motion to the current motion '''
            self._ref_motion.append(graph.nodes[node_idx]['motion'], blend_window=self._mg.blend_window)
            ''' Jump to adjacent node (motion) randomly '''
            node_idx = list(graph.successors(node_idx))[np.random.randint(num_out_edges)]
            t = self._ref_motion.times[-1]
        self._mg_node_idx = node_idx
        ''' If it returns True then it means that we met dead-end '''
        return t < end_time

    def get_current_pose_from_motion(self):
        return self._ref_motion.get_pose_by_time(self._elapsed_time)

    def get_current_vel_from_motion(self):
        return self._ref_motion.get_velocity_by_time(self._elapsed_time)
    
    def step_forward(self):
        self._pb_client.stepSimulation()
        self._elapsed_time += self._dt_sim
    
    def step(self, action):

        profile = False
        
        if profile:
            print('-----------------------------------------')
            time_checker = basics.TimeChecker()

        rew_data_prev = self.reward_data()

        ''' Increase elapsed time '''
        self._elapsed_time += self._dt_con
        self._episode_len += self._dt_con

        if profile:
            print('> reward_data')
            time_checker.print_time()

        if self._mode == Env.Mode.MimicPFNN:
            ''' Update motion (PFNN) controller '''
            for _ in range(self._num_iter_per_step_mot):
                self._pfnn.update()
                # if isinstance(self._pfnn.command, pfnn.JoystickCommand):
                #     self._pfnn.update(self._sim_agent)
                # else:
                #     self._pfnn.update()

            self._ref_motion.add_one_frame(
                self._ref_motion.times[-1]+self._dt_con, copy.deepcopy(self._pfnn.character.joint_xform_by_ik))
        elif self._mode == Env.Mode.MotionGraph:
            ''' Extend motion when remaining time is shorter than the imitation window '''
            time_remaining = self._ref_motion.times[-1] - self._elapsed_time
            if time_remaining <= self._imit_window[-1]:
                stride = self._mg.motions[0].times[-1]
                met_dead_end = self.generate_reference_motion_by_mg(
                    self._elapsed_time+self._imit_window[-1]+stride, reset=False)
                if met_dead_end:
                    print('DeadEnd!!! MotionGraph')

        if profile:        
            print('> motion:add_one_frame')
            time_checker.print_time()

        ''' Set target posture as the result of motion controller '''
        self.set_action(self._sim_agent, action)

        # p, v = self._kin_agent.get_com_and_com_vel()
        # print(np.linalg.norm(v)*3.6)

        if profile:
            print('> set_action')
            time_checker.print_time()

        ''' Update simulation '''
        for _ in range(self._num_iter_per_step_sim):
            self._sim_agent.tracking_control(pose=self._target_pose, 
                                             vel=None, 
                                             method=self._action_type)
            self._pb_client.stepSimulation()
        # print('++++++++++++++++++++++++++++++++++++')
        # print(self._sim_agent.get_joint_torques())
        # print('++++++++++++++++++++++++++++++++++++')

        self._obs_manager.update()

        if profile:
            print('> simulation')
            time_checker.print_time()

        ''' Set kinematic character '''
        ''' This is necessary to compute the reward correctly '''
        self._kin_agent.set_pose(self.get_current_pose_from_motion(),
                                 self.get_current_vel_from_motion())

        if profile:
            print('> set_pose')
            time_checker.print_time()

        ''' Check end-of-episode and compute reward accordingly '''
        self.inspect_end_of_episode(self._sim_agent, self._kin_agent)
        rew_data_next = self.reward_data()

        if profile:
            print('> inspect eoe and reward_data')
            time_checker.print_time()
            print('-----------------------------------------')

        rew, rew_detail = self.reward(rew_data_prev, rew_data_next, action)
        if self._et_low_reward: 
            self.rew_queue.append(rew)
            # print(np.mean(list(self.rew_queue)))

        return rew, rew_detail

    def state(self, sim_agent=None, kin_agent=None, ref_motion=None):
        if sim_agent is None: sim_agent = self._sim_agent
        if kin_agent is None: kin_agent = self._kin_agent
        if ref_motion is None: ref_motion = self._ref_motion

        state = []
        
        if self._state_body:
            state.append(self.state_body(sim_agent, include_com=True, return_stacked=True))
        state.append(self.state_imit_motion(sim_agent,
                                            kin_agent, 
                                            ref_motion, 
                                            include_abs=self._state_imit_abs, 
                                            include_rel=self._state_imit_rel))

        return np.hstack(state)

    def state_task(self):
        return []

    def state_param(self):
        return []

    def state_body(self, agent=None, T_ref=None, include_com=False, return_stacked=True):
        if agent is None: agent = self._sim_agent
        if T_ref is None: T_ref = agent.get_facing_transform()

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
            p_rel = np.dot(R_ref_inv, p - p_ref)
            Q_rel = mmMath.R2Q(np.dot(R_ref_inv, mmMath.Q2R(Q)))
            Q_rel = mmMath.post_process_Q(Q_rel, normalize=True, half_space=True)
            v_rel = np.dot(R_ref_inv, v)
            w_rel = np.dot(R_ref_inv, w)
            state.append(p_rel) # relative position w.r.t. the reference frame
            state.append(Q_rel) # relative rotation w.r.t. the reference frame
            state.append(v_rel) # relative linear vel w.r.t. the reference frame
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

    def state_imit_pfnn(self):
        return self._pfnn.controller.x_p.copy()

    # def state_imit_pfnn(self):
    #     x = self._pfnn.controller.x_p
    #     x_mean = self._pfnn.controller.x_mean
    #     x_std = self._pfnn.controller.x_std
    #     x_norm = (x-x_mean)/x_std
    #     phase = self._pfnn.character.phase
    #     return np.hstack([phase, x_norm])

    def state_imit_motion(self, sim_agent, kin_agent, motion, include_abs=True, include_rel=True):

        R_sim, p_sim = mmMath.T2Rp(sim_agent.get_facing_transform())
        R_sim_inv = R_sim.transpose()
        state_sim = self.state_body(sim_agent, None, include_com=True, return_stacked=False)
        
        state = []
        state_kin_orig = kin_agent.save_states()
        for dt in self._imit_window:
            t = basics.clamp(self._elapsed_time + dt, motion.times[0], motion.times[-1])
            kin_agent.set_pose(motion.get_pose_by_time(t), motion.get_velocity_by_time(t))
            state_kin = self.state_body(kin_agent, None, include_com=True, return_stacked=False)
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
            R_kin, p_kin = mmMath.T2Rp(kin_agent.get_facing_transform())
            state.append(np.dot(R_sim_inv, p_kin - p_sim))
            state.append(np.dot(R_sim_inv, kin_agent.get_facing_direction()))
        kin_agent.restore_states(state_kin_orig)

        return np.hstack(state)

    def state_imit_path(self, motion=None, T_ref=None):
        if motion is None: motion = self._ref_motion
        kin_agent = self._kin_agent

        state = []

        R_f_sim, p_f_sim = mmMath.T2Rp(T_ref)
        R_f_sim_inv = R_f_sim.transpose()

        state_kin_orig = kin_agent.save_states()
        for dt in self._imit_window:
            t = basics.clamp(self._elapsed_time + dt, 
                             motion.times[0],
                             motion.times[-1])
            kin_agent.set_pose(motion.get_pose_by_time(t), motion.get_velocity_by_time(t))
            R_f_kin, p_f_kin = mmMath.T2Rp(kin_agent.get_facing_transform())
            p_diff = np.dot(R_f_sim_inv, p_f_kin - p_f_sim)
            R_diff = np.dot(R_f_sim_inv, R_f_kin[:, 2])
            state.append(np.array([p_diff[0],p_diff[2],R_diff[0],R_diff[2]]))
        kin_agent.restore_states(state_kin_orig)
        return np.hstack(state)

    def reward_data(self, sim_agent=None, kin_agent=None):
        if sim_agent is None: sim_agent = self._sim_agent
        if kin_agent is None: kin_agent = self._kin_agent

        data = {}

        data['sim_root_pQvw'] = sim_agent.get_root_state()
        data['sim_link_pQvw'] = sim_agent.get_link_states()
        data['sim_joint_pv'] = sim_agent.get_joint_states()
        data['sim_facing_frame'] = sim_agent.get_facing_transform()
        data['sim_com'], data['sim_com_vel'] = sim_agent.get_com_and_com_vel()
        
        data['kin_root_pQvw'] = kin_agent.get_root_state()
        data['kin_link_pQvw'] = kin_agent.get_link_states()
        data['kin_joint_pv'] = kin_agent.get_joint_states()
        data['kin_facing_frame'] = kin_agent.get_facing_transform()
        data['kin_com'], data['kin_com_vel'] = kin_agent.get_com_and_com_vel()

        return data

    def reward_max(self):
        return 1.0

    def reward_min(self):
        return 0.0

    def return_max(self, gamma):
        assert gamma < 1.0
        return 1.0 / (1.0 - gamma)

    def return_min(self):
        return 0.0

    def reward(self, data_prev, data_next, action):

        sim_root_p, sim_root_Q, sim_root_v, sim_root_w = data_next['sim_root_pQvw']
        sim_link_p, sim_link_Q, sim_link_v, sim_link_w = data_next['sim_link_pQvw']
        sim_joint_p, sim_joint_v = data_next['sim_joint_pv']
        sim_facing_frame = data_next['sim_facing_frame']
        R_sim_f, p_sim_f = mmMath.T2Rp(sim_facing_frame)
        R_sim_f_inv = R_sim_f.transpose()
        sim_com, sim_com_vel = data_next['sim_com'], data_next['sim_com_vel']
        
        kin_root_p, kin_root_Q, kin_root_v, kin_root_w = data_next['kin_root_pQvw']
        kin_link_p, kin_link_Q, kin_link_v, kin_link_w = data_next['kin_link_pQvw']
        kin_joint_p, kin_joint_v = data_next['kin_joint_pv']
        kin_facing_frame = data_next['kin_facing_frame']
        R_kin_f, p_kin_f = mmMath.T2Rp(kin_facing_frame)
        R_kin_f_inv = R_kin_f.transpose()
        kin_com, kin_com_vel = data_next['kin_com'], data_next['kin_com_vel']

        indices = range(len(sim_link_p))

        error_pose = 0.0
        error_vel = 0.0
        error_ee = 0.0
        error_root = 0.0
        error_com = 0.0
        
        if self._rew_pose or self._rew_vel:
            for j in indices:
                joint_type = self._sim_agent.get_joint_type(j)
                if joint_type == self._pb_client.JOINT_SPHERICAL:
                    dQ = self._pb_client.getDifferenceQuaternion(sim_joint_p[j], kin_joint_p[j])
                    _, diff_pose = self._pb_client.getAxisAngleFromQuaternion(dQ)
                    diff_vel = sim_joint_v[j] - kin_joint_v[j]
                elif joint_type == self._pb_client.JOINT_FIXED:
                    continue
                else:
                    diff_pose = sim_joint_p[j] - kin_joint_p[j]
                    diff_vel = sim_joint_v[j] - kin_joint_v[j]
                if self._rew_pose:
                    error_pose += self.char_info.joint_weight[j] * np.dot(diff_pose, diff_pose)
                if self._rew_vel:
                    error_vel += self.char_info.joint_weight[j] * np.dot(diff_vel, diff_vel)
        
        if self._rew_ee:
            for j in self.char_info.end_effector_indices:
                sim_ee_local = np.dot(R_sim_f_inv, sim_link_p[j]-p_sim_f)
                kin_ee_local = np.dot(R_kin_f_inv, kin_link_p[j]-p_kin_f)
                diff_pos =  sim_ee_local - kin_ee_local
                error_ee += np.dot(diff_pos, diff_pos)
        
        if self._rew_root:
            diff_root_p = sim_root_p - kin_root_p
            
            _, diff_root_Q = self._pb_client.getAxisAngleFromQuaternion(
                self._pb_client.getDifferenceQuaternion(sim_root_Q, kin_root_Q))
            diff_root_v = sim_root_v - kin_root_v
            diff_root_w = sim_root_w - kin_root_w
            error_root = 1.0 * np.dot(diff_root_p, diff_root_p) + \
                         0.1 * np.dot(diff_root_Q, diff_root_Q) + \
                         0.01 * np.dot(diff_root_v, diff_root_v) + \
                         0.001 * np.dot(diff_root_w, diff_root_w)

        if self._rew_com:
            diff_com = sim_com - kin_com
            diff_com_vel = sim_com_vel - kin_com_vel
            error_com = 0.1 * np.dot(diff_com_vel, diff_com_vel)

        ''' Normalize Errors that can be affected by the character definition '''
        if len(indices) > 0:
            error_pose /= len(indices)
            error_vel /= len(indices)
        if len(self.char_info.end_effector_indices) > 0:
            error_ee /= len(self.char_info.end_effector_indices)

        error_scale_pose = 4*len(indices)
        error_scale_vel = 0.1*len(indices)
        error_scale_ee = 2*len(self.char_info.end_effector_indices)
        error_scale_root = 5
        error_scale_com = 10

        error = {}
        error['imit_pose'] = error_pose*error_scale_pose
        error['imit_vel']  = error_vel*error_scale_vel
        error['imit_ee']   = error_ee*error_scale_ee
        error['imit_root'] = error_root*error_scale_root
        error['imit_com']  = error_com*error_scale_com

        weight = {}

        w_sum = 0.0
        for key, val in weight.items():
            w_sum += val
        for key, val in weight.items():
            weight[key] /= w_sum

        if self._rew_mode==Env.RewardMode.Sum:
            weight['imit_pose'] = 0.5
            weight['imit_vel']  = 0.05
            weight['imit_ee']   = 0.15
            weight['imit_root'] = 0.2
            weight['imit_com']  = 0.1
            return self.reward_sum(error, weight)
        elif self._rew_mode==Env.RewardMode.Mul:
            weight['imit_pose'] = self.reward_weight_scale
            weight['imit_vel']  = self.reward_weight_scale
            weight['imit_ee']   = self.reward_weight_scale
            weight['imit_root'] = self.reward_weight_scale
            weight['imit_com']  = self.reward_weight_scale
            return self.reward_mul(error, weight)
        else:
            raise NotImplementedError

    def reward_sum(self, error, weight):
        detail = {}
        val = 0.0
        for key in list(error.keys()):
            e = math.exp(-1.0 * error[key])
            w = weight[key]
            mult = w * e
            val += mult
            detail[key] = (e, w)
            # print('%s w(%f) e(%f), diff(%f)' % (key, w, e, error[key]))
        return val, detail

    def reward_mul(self, error, weight):
        detail = {}
        val = 1.0
        for key in list(error.keys()):
            w = weight[key]
            e = math.exp(-1.0 * w * error[key])
            val *= e
            detail[key] = (e, w)
            # print('%s w(%f) e(%f), diff(%f)' % (key, w, e, error[key]))
        return val, detail

    def end_time_of_ref_motion(self):
        return self._ref_motion.times[-1]

    def check_falldown(self, agent):
        ''' check if any non-allowed body part hits the ground '''
        terminates = False
        pts = self._pb_client.getContactPoints()
        for p in pts:
            part = None
            #ignore self-collision
            if p[1] == p[2]: continue
            if p[1] == agent._body_id and p[2] == self._plane_id: part = p[3]
            if p[2] == agent._body_id and p[1] == self._plane_id: part = p[4]
            #ignore collision of other agents
            if part == None: continue
            if not self.char_info.contact_allow_map[part]: return True
        return False

    def check_root_fail(self, sim_agent, kin_agent):
        p1, Q1, _, _ = sim_agent.get_root_state()
        p2, Q2, _, _ = kin_agent.get_root_state()
        _, angle = self._pb_client.getAxisAngleFromQuaternion(
            self._pb_client.getDifferenceQuaternion(Q1, Q2))
        dist = np.linalg.norm(p2-p1)
        return angle > 0.3 * math.pi or dist > 1.0

    def is_sim_div(self, agent):
        return False

    def inspect_end_of_episode(self,
                               sim_agent,
                               kin_agent,
                               set_eoe=True):
        eoe_reason = []

        if self._et_task_complete:
            pass
        if self._et_falldown:
            check = self.check_falldown(sim_agent)
            if check: eoe_reason.append('falldown')
        if self._et_root_fail:
            check = self.check_root_fail(sim_agent, kin_agent)
            if check: eoe_reason.append('root_fail')
        if self._et_sim_div:
            check = self.is_sim_div(sim_agent)
            if check: eoe_reason.append('sim_div')
        if self._et_sim_window:
            check = self._elapsed_time - self._start_time > self._sim_window_time
            if check: eoe_reason.append('sim_window')
        if self._et_low_reward:
            check = np.mean(list(self.rew_queue)) < self._et_low_reward_thres * self.reward_max()
            if check: eoe_reason.append('low_rewards')
        if self._mode == Env.Mode.DeepMimic:
            check = self._elapsed_time >= self.end_time_of_ref_motion() + self._end_margin_time
            if check: eoe_reason.append('end_of_motion')
        elif self._mode == Env.Mode.MotionGraph:
            time_remaining = self._ref_motion.times[-1] - self._elapsed_time
            check = time_remaining < self._imit_window[-1]
            if check: eoe_reason.append('dead_end')
        
        eoe = len(eoe_reason) > 0

        if set_eoe:
            self.end_of_episode = eoe
            self.end_of_episode_reason = eoe_reason

        if self._verbose and eoe:
            print('=================EOE=================')
            print('Reason:',eoe_reason)
            print('TIME: (start:%02f ) (cur:%02f) (elapsed:%02f) (end:%02f)'\
                %(self._start_time, self._elapsed_time, self._elapsed_time-self._start_time, self.end_time_of_ref_motion()))
            print('=====================================')

        return eoe, eoe_reason

    def render_state_imit_pose(self):
        state_orig = self._kin_agent.save_states()
        num_dts = len(self._imit_window)
        for i in range(num_dts):
            dt = self._imit_window[i]
            t = basics.clamp(self._elapsed_time + dt, 
                             self._ref_motion.times[0],
                             self._ref_motion.times[-1])
            self._kin_agent.set_pose(self._ref_motion.get_pose_by_time(t),
                                     self._ref_motion.get_velocity_by_time(t))
            alpha = math.pow(10, -dt)
            bullet_render.render_model(self._pb_client, self._kin_agent._body_id, color=[0.5, 0.5, 0.5, alpha])
        self._kin_agent.restore_states(state_orig)

        # gl.glDisable(gl.GL_DEPTH_TEST)
        # gl.glDisable(gl.GL_LIGHTING)
        # s = self._kin_agent._ref_scale
        # gl.glPushMatrix()
        # gl.glScalef(s, s, s)

        # cur_pose, cur_vel = self.get_current_pose_from_motion(), self.get_current_vel_from_motion()
        # for j in cur_pose.skel.joints:
        #     R, p = mmMath.T2Rp(cur_pose.get_transform(j, local=False))
        #     v = cur_vel.get_linear_velocity(j, False, R)
        #     gl_render.render_point(p, radius=0.02/s, color=[0.8, 0.8, 0.0, 1])
        #     gl_render.render_arrow(p, p+v, D=0.005/s, color=[0, 0, 0, 1])
        #     if j.parent_joint is not None:
        #         pos_parent = mmMath.T2p(cur_pose.get_transform(j.parent_joint, local=False))
        #         gl_render.render_line(p1=pos_parent, p2=p, color=[0.5, 0.5, 0, 1])
        # gl.glPopMatrix()
        # gl.glEnable(gl.GL_DEPTH_TEST)
        # gl.glEnable(gl.GL_LIGHTING)

    def render_state_imit_path(self):
        R_f, p_f = mmMath.T2Rp(self._sim_agent.get_facing_transform())
        state_imit_path = self.state_imit_path()
        n = 4
        num_point = len(state_imit_path) // n
        px, pz, dx, dz = state_imit_path[:n]
        for i in range(1, num_point):
            px_new, pz_new, dx_new, dz_new = state_imit_path[n*i:n*i+n]
            p1 = np.dot(R_f, np.array([px, 0, pz])) + p_f
            p2 = np.dot(R_f, np.array([px_new, 0, pz_new])) + p_f
            d1 = np.dot(R_f, np.array([dx, 0, dz]))
            gl_render.render_point(p1, radius=0.03, color=[0, 0, 0, 1])
            gl_render.render_line(p1=p1, p2=p2, color=[0, 0, 0, 1])
            gl_render.render_line(p1=p1, p2=p1+0.3*d1, color=[0, 0, 1, 1])
            if i==num_point-1:
                d2 = np.dot(R_f, np.array([dx_new, 0, dz_new]))
                gl_render.render_point(p2, radius=0.03, color=[0, 0, 0, 1])
                gl_render.render_line(p1=p2, p2=p2+0.3*d2, color=[0, 0, 1, 1])
            px, pz, dx, dz = px_new, pz_new, dx_new, dz_new

    def render_pfnn(self, scale=0.009):
        options = self._pfnn.options
        trajectory = self._pfnn.trajectory
        character = self._pfnn.character

        ''' pfnn character uses centi-meter '''
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_LIGHTING)
        
        gl.glPushMatrix()
        gl.glScalef(scale, scale, scale)
        ''' Trajectory '''
        gl.glPointSize(2.0 * options.display_scale)
        gl.glBegin(gl.GL_POINTS)
        for i in range(0, trajectory.length-10):
            position_c = trajectory.positions[i]
            gl.glColor3f(trajectory.gait_jump[i], trajectory.gait_bump[i], trajectory.gait_crouch[i])
            gl.glVertex3f(position_c[0], position_c[1]+2, position_c[2]);
        gl.glEnd()
        
        gl.glPointSize(2.0 * options.display_scale)
        gl.glBegin(gl.GL_POINTS)
        for i in range(0, trajectory.length, 10):
            R, p = trajectory.rotations[i], trajectory.positions[i]
            position_c = p
            position_r = p + np.dot(R, np.array([trajectory.width, 0, 0]))
            position_l = p + np.dot(R, np.array([-trajectory.width, 0, 0]))

            gl.glColor3f(trajectory.gait_jump[i], trajectory.gait_bump[i], trajectory.gait_crouch[i])
            gl.glVertex3f(position_c[0], position_c[1]+2, position_c[2]);
            gl.glVertex3f(position_r[0], position_r[1]+2, position_r[2]);
            gl.glVertex3f(position_l[0], position_l[1]+2, position_l[2]);
        gl.glEnd()

        gl.glLineWidth(2.0 * options.display_scale)
        gl.glBegin(gl.GL_LINES)
        for i in range(0, trajectory.length, 10):
            p = trajectory.positions[i]
            d = trajectory.directions[i]
            base = p + np.array([0, 2, 0])
            side = np.cross(d, np.array([0, 1, 0]))
            side /= np.linalg.norm(side)
            fwrd = base + 15 * d
            arw0 = fwrd +  4 * side + 4 * -d
            arw1 = fwrd -  4 * side + 4 * -d
            gl.glColor3f(trajectory.gait_jump[i], trajectory.gait_bump[i], trajectory.gait_crouch[i])
            gl.glVertex3f(base[0], base[1], base[2])
            gl.glVertex3f(fwrd[0], fwrd[1], fwrd[2])
            gl.glVertex3f(fwrd[0], fwrd[1], fwrd[2])
            gl.glVertex3f(arw0[0], fwrd[1], arw0[2])
            gl.glVertex3f(fwrd[0], fwrd[1], fwrd[2])
            gl.glVertex3f(arw1[0], fwrd[1], arw1[2])
        gl.glEnd()

        # gl_render.render_transform(self._pfnn.trajectory.get_base_xform(), scale=50.0)

        if flag['kin_model']:
            ''' joint positions given from NN '''

            for i in range(character.JOINT_NUM):
                pos = character.joint_positions[i]
                gl_render.render_point(pos, radius=2, color=[0.8, 0.8, 0.0, 1.0])
                j = character.joint_parents[i]
                if j!=-1:
                    pos_parent = character.joint_positions[j]
                    gl_render.render_line(p1=pos_parent, p2=pos, color=[0.8, 0.8, 0, 1])
                # print(0.01 * pos)

            ''' joint positions computed by forward-kinamatics with rotations given from NN and IK'''

            for i in range(character.JOINT_NUM):
                pos = mmMath.T2p(character.joint_global_xform_by_ik[i])
                gl_render.render_point(pos, radius=2, color=[0.8, 0.0, 0.0, 1.0])
                j = character.joint_parents[i]
                if j!=-1:
                    pos_parent = mmMath.T2p(character.joint_global_xform_by_ik[j])
                    gl_render.render_line(p1=pos_parent, p2=pos, color=[0.8, 0, 0, 1])        

        ''' test for skeleton rendering '''

        # for i in range(character.JOINT_NUM):
        #     # xform = np.dot(character.joint_mesh_xform[i], character.link_rest_xform[i])
        #     xform = np.dot(character.joint_mesh_xform[i], character.link_global_rest_xform[i])
        #     gl_render.render_pyramid(xform, base_x=5, base_z=5, height=character.link_length[i], color=[0.5, 0.5, 0.5, 0.5])

        # for dt in [0.0, -0.25, -0.5, -0.75, -1.0]:
        #     pose = self._ref_motion.get_pose_by_time(self._elapsed_time+dt)
        #     self.render_posture(pose)g)

        gl.glPopMatrix()

        # if flag['kin_model']:
        #     state_orig = self._kin_agent.save_states()
        #     for dt in [0.0, -0.25, -0.5, -0.75, -1.0]:
        #         t = self._elapsed_time + dt
        #         alpha = math.pow(10, dt)
        #         self._kin_agent.set_pose(self._ref_motion.get_pose_by_time(t), self._ref_motion.get_velocity_by_time(t))
        #         bullet_render.render_model(self._pb_client, self._kin_agent._body_id, flag['joint'], self.char_info.end_effector_indices, color=[0.5, 0.5, 0.5, alpha])
        #     self._kin_agent.restore_states(state_orig)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)

    def render_posture(self, pose, scale=1.00):
        gl.glPushMatrix()
        gl.glScalef(scale, scale, scale)
        skel = pose.skel
        for joint in skel.joints:
            pos = mmMath.T2p(pose.get_transform(joint, local=False))
            gl_render.render_point(pos, radius=2, color=[0, 0, 0, 1])
            if joint.parent_joint is not None:
                pos_parent = mmMath.T2p(pose.get_transform(joint.parent_joint, local=False))
                gl_render.render_line(p1=pos_parent, p2=pos, color=[0, 0, 0, 1])
        gl.glPopMatrix()

    def render_target_pose(self):
        if self._target_pose is None: return
        state_orig = self._kin_agent.save_states()
        self._kin_agent.set_pose(self._target_pose)            
        bullet_render.render_model(self._pb_client, self._kin_agent._body_id, color=[0.5, 0.5, 0.0, 1.0])
        self._kin_agent.restore_states(state_orig)

    def render(self):
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        global flag

        if flag['sim_model']:
            gl.glEnable(gl.GL_DEPTH_TEST)
            if flag['shadow']:
                gl.glPushMatrix()
                d = np.array([1, 1, 1])
                d = d - mmMath.projectionOnVector(d, self.char_info.v_up_env)
                offset = 0.001*self.char_info.v_up_env
                gl.glTranslatef(offset[0], offset[1], offset[2])
                gl.glScalef(d[0], d[1], d[2])
                bullet_render.render_model(self._pb_client, 
                                           self._sim_agent._body_id, 
                                           draw_link=True, 
                                           draw_link_info=False, 
                                           draw_joint=False, 
                                           draw_joint_geom=False, 
                                           ee_indices=None, 
                                           color=[0.5,0.5,0.5,1.0],
                                           lighting=False)
                gl.glPopMatrix()
            color = np.array([85, 160, 173, 255])/255.0
            # color = np.array([28, 175, 200, 255])/255.0

            bullet_render.render_model(self._pb_client, 
                                       self._sim_agent._body_id, 
                                       draw_link=True, 
                                       draw_link_info=True, 
                                       draw_joint=flag['joint'], 
                                       draw_joint_geom=True, 
                                       ee_indices=self.char_info.end_effector_indices, 
                                       color=color)
            if flag['collision'] and self._elapsed_time > 0.0:
                gl.glPushAttrib(gl.GL_LIGHTING|gl.GL_DEPTH_TEST|gl.GL_BLEND)
                gl.glEnable(gl.GL_BLEND)
                bullet_render.render_contacts(self._pb_client, self._sim_agent._body_id)
                gl.glPopAttrib()
            if flag['com_vel']:
                p, Q, v, w = self._sim_agent.get_root_state()
                p, v = self._sim_agent.get_com_and_com_vel()
                gl_render.render_arrow(p, p+v, D=0.01, color=[0, 0, 0, 1])
            if flag['facing_frame']:
                gl.glPushAttrib(gl.GL_LIGHTING|gl.GL_DEPTH_TEST|gl.GL_BLEND)
                gl.glEnable(gl.GL_BLEND)
                gl_render.render_transform(self._sim_agent.get_facing_transform(), scale=0.5, use_arrow=True)
                gl.glPopAttrib()
        
        if flag['kin_model']:
            gl.glPushAttrib(gl.GL_LIGHTING|gl.GL_DEPTH_TEST|gl.GL_BLEND)
            bullet_render.render_model(self._pb_client, 
                                       self._kin_agent._body_id, 
                                       draw_link=True, 
                                       draw_link_info=False,
                                       draw_joint=flag['joint'], 
                                       draw_joint_geom=False, 
                                       ee_indices=self.char_info.end_effector_indices, 
                                       color=[0.5, 0.5, 0.5, 0.5])
            if flag['com_vel']:
                p, Q, v, w = self._kin_agent.get_root_state()
                p, v = self._kin_agent.get_com_and_com_vel()
                gl_render.render_arrow(p, p+v, D=0.01, color=[0.5, 0.5, 0.5, 1])
            gl.glPopAttrib()

        self._obs_manager.render()

        # if flag['kin_model']:
        #     self.render_state_imit_pose()
        #     # self.render_state_imit_path()

        # self.render_target_pose()

        if self._mode == Env.Mode.MimicPFNN:
            self.render_pfnn()

def keyboard_callback(key):
    global env, viewer
    global time_checker_auto_play
    global flag
    if env is None: return

    if key in toggle:
        flag[toggle[key]] = not flag[toggle[key]]
        print('Toggle:', toggle[key], flag[toggle[key]])
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
        print('Key[a]: auto_play:', not flag['auto_play'])
        flag['auto_play'] = not flag['auto_play']
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
    global env, viewer, flag, pi
    global time_checker_auto_play
    if env is None: return

    if allow_auto_play and flag['auto_play'] and time_checker_auto_play.get_time(restart=False) >= env._dt_con:
        action = np.zeros(env._action_info.dim)
        env.step(action)
        time_checker_auto_play.begin()

    if env._mode==Env.Mode.MimicPFNN and \
       (env._pfnn_command_type==pfnn.Runner.UserInput.Joystick or \
        env._pfnn_command_type==pfnn.Runner.UserInput.Recorded):
        command = env._pfnn.command.get()
        scale = env._pfnn.command.scale
        viewer.cam_cur.rotate(0.1*command['y_move']/scale, 0.1*command['x_move']/scale, 0.0)
        if command['zoom_o'] > 0.0:
            viewer.cam_cur.zoom(1.02)
        elif command['zoom_i'] > 0.0:
            viewer.cam_cur.zoom(0.98)
        command['target_dir'] = viewer.cam_cur.origin - viewer.cam_cur.pos
        if command['trigger_obstacle'] > 0.0:
            env.throw_obstacle()

    if flag['follow_cam']:
        p, _, _, _ = env._sim_agent.get_root_state()
        if np.allclose(env.char_info.v_up_env, np.array([0.0, 1.0, 0.0])):
            viewer.update_target_pos(p, ignore_y=True)
        elif np.allclose(env.char_info.v_up_env, np.array([0.0, 0.0, 1.0])):
            viewer.update_target_pos(p, ignore_z=True)
        else:
            raise NotImplementedError

def render_callback():
    global env, flag, tex_id_ground, file_tex_ground
    if env is None: return

    if flag['ground']:
        if tex_id_ground is None:
            tex_id_ground = gl_render.load_texture(file_tex_ground)
        gl_render.render_ground_texture(tex_id_ground,
                                        size=[40.0, 40.0], 
                                        dsize=[2.0, 2.0], 
                                        axis=kinematics.axis_to_str(env.char_info.v_up_env),
                                        origin=flag['origin'],
                                        use_arrow=True,
                                        circle_cut=True)
        # gl.glDisable(gl.GL_LIGHTING)
        # gl.glPushMatrix()
        # offset = 0.001*env.char_info.v_up_env
        # gl.glTranslatef(offset[0], offset[1], offset[2])
        # gl_render.render_disk(mmMath.I_SE3(),
        #                       r_inner=15.0,
        #                       r_outer=30.0,
        #                       slice1=32,
        #                       slice2=16,
        #                       scale=1.0,
        #                       color=[1.0, 1.0, 1.0, 1.0])
        # gl.glPopMatrix()
    if flag['fog']:
        density = 0.05;
        fogColor = [1.0, 1.0, 1.0, 1.0]
        gl.glEnable(gl.GL_FOG)
        gl.glFogi(gl.GL_FOG_MODE, gl.GL_EXP2)
        gl.glFogfv(gl.GL_FOG_COLOR, fogColor)
        gl.glFogf(gl.GL_FOG_DENSITY, density)
        gl.glHint(gl.GL_FOG_HINT, gl.GL_NICEST)
    else:
        gl.glDisable(gl.GL_FOG)

    env.render()

def overlay_callback():
    return

def str2bool(string):
    if string=="true": 
        return True
    elif string=="false": 
        return False
    else:
        raise Exception('Unknown')

def init_viewer():
    from basecode.render import glut_viewer as viewer
    from basecode.render import gl_render
    from basecode.bullet import bullet_render

    import OpenGL.GL as gl
    import OpenGL.GLU as glu
    import OpenGL.GLUT as glut

env = None
pi = None
if __name__ == '__main__':
    init_viewer()
    
    from baselines.common import cmd_util
    
    def arg_parser():
        parser = cmd_util.arg_parser()
        parser.add_argument('--env_mode', choices=['deepmimic', 'mimicpfnn', 'motiongraph'], type=str, default='deepmimic')
        parser.add_argument('--action_mode', choices=['absolute', 'relative'], type=str, default='relative')
        parser.add_argument('--ref_motion_files', action='append', help='Reference Motion File', default=['data/motion/amass/amass_hierarchy.bvh'])
        parser.add_argument('--char_info_module', type=str, default='amass_char_info.py')
        parser.add_argument('--ref_motion_scale', type=float, default=1.0)
        parser.add_argument('--sim_char_file', type=str, default='data/character/amass.urdf')
        parser.add_argument('--base_motion_file', type=str, default='data/motion/amass/amass_hierarchy.bvh')
        return parser

    # def __init__(self, 
    #              dt_sim, 
    #              dt_con, 
    #              # Mode of this environment
    #              mode="deepmimic",
    #              verbose=False,
    #              sim_window=math.inf, 
    #              end_margin=1.5,
    #              pfnn_command_type="autonomous",
    #              pfnn_command_record=False,
    #              pfnn_command_file=None,
    #              action_type="spd",
    #              action_mode="relative",
    #              action_range_min=-1.5,
    #              action_range_max=1.5,
    #              action_range_min_pol=-15,
    #              action_range_max_pol=15,
    #              ref_motion_files=None,
    #              add_noise=False,
    #              early_term=[],
    #              et_low_reward_thres=0.1,
    #              et_falldown_contactable_body=[],
    #              char_info_module=None,
    #              sim_char_file=None,
    #              base_motion_file=None,
    #              motion_graph_file=None,
    #              reward_mode="sum",
    #              ref_motion_scale=1.0,
    #              self_collision=True,
    #              ref_motion_sample='random',
    #              visualization=False,
    #              reward_weight_scale=0.5,
    #              ):
    
    print('=====Scalable Controller for Diverse Motions=====')
    
    args = basics.parse_args_by_file(arg_parser, sys.argv)
    print(args.ref_motion_files)
    
    env = Env(dt_sim=DT_SIM, 
              dt_con=DT_CON, 
              mode=args.env_mode, 
              action_mode=args.action_mode, 
              ref_motion_files=args.ref_motion_files,
              char_info_module=args.char_info_module,
              ref_motion_scale=args.ref_motion_scale,
              sim_char_file=args.sim_char_file,
              base_motion_file=args.base_motion_file,
              visualization=True,
              verbose=False
              )

    cam_origin, _, _, _ = env._sim_agent.get_root_state()

    if np.allclose(env.char_info.v_up_env, np.array([0.0, 1.0, 0.0])):
        cam_pos = cam_origin + np.array([0.0, 2.0, -3.0])
        cam_vup = np.array([0.0, 1.0, 0.0])
    elif np.allclose(env.char_info.v_up_env, np.array([0.0, 0.0, 1.0])):
        cam_pos = cam_origin + np.array([3.0, 0.0, 2.0])
        cam_vup = np.array([0.0, 0.0, 1.0])
    else:
        raise NotImplementedError

    cam = camera.Camera(pos=cam_pos,
                        origin=cam_origin, 
                        vup=cam_vup, 
                        fov=45)

    viewer.run(
        title='env',
        cam=cam,
        size=(1280, 720),
        keyboard_callback=keyboard_callback,
        render_callback=render_callback,
        overlay_callback=overlay_callback,
        idle_callback=idle_callback,
        )
