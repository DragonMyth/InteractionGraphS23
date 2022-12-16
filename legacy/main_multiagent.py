import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import yaml

import sys
import numpy as np

import gzip
import pickle
import math
import copy
import collections
import re
import random

import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from mpi4py import MPI

import policy
import pposgd
from baselines.common import cmd_util
from baselines.common import tf_util as U
from baselines import logger

import gym
from gym import core
from gym.spaces import Box
from gym.utils import seeding

from basecode.utils import basics
from basecode.math import mmMath
from basecode.utils import multiprocessing as mp

# import env_multiagent as base_env
import env_multiagent as my_env

import env_renderer as er
# import render_module as rm

SCREENSHOT_DIR = 'data/screenshot/'

cnt_screenshot = 0

time_elapsed = 0.0
reward = 0

recorder = None

time_checker_auto_play = basics.TimeChecker()
pre_sim_time = 0.0
sim_speed = collections.deque(maxlen=10)
sim_speed_avg = 0.0
val_buffer = collections.OrderedDict()
val_buffer['total'] = collections.deque(maxlen=10)

weight_experts_buffer = collections.deque(maxlen=10)

num_cluster = None
files_cluster_id = None

class SimpleRecorder(object):
    def __init__(self):
        self.data = []
    def reset(self):
        self.data = []
    def capture(self, skel, info):
        self.data.append(self._capture(skel, info))
    def _capture(self, skel, info):
        oneframe = []
        for body in skel.bodynodes:
            shapes = []
            for i in range(len(body.shapenodes)):
                shape = {}
                shape['name'] = "%s_%d"%(body.name, i)
                shape['T'] = si.get_shape_transform(skel, body.name, i).tolist()
                shape['size'] = body.shapenodes[i].shape.size().tolist()
                shape['type'] = body.shapenodes[i].shape.shape_type_name()
                shapes.append(shape)
            oneframe.append(shapes)
        return (info, oneframe)
    def save_file(self, file_name):
        print('==========capture: save.... start==========')
        if len(self.data) == 0:
            print('[Recorder] no data to write')
        else:
            with open(file_name + '.motion', 'wb') as f:
                pickle.dump(self.data, f, protocol=2)
                f.close()
        print('==========capture: save.... end============')

class HumanoidImitation(gym.Env):
    def __init__(self, env_config):
        self.base_env = my_env.Env(env_config)
        assert self.base_env._num_agent == 1

        self.seed()
        
        ob_scale = 1000.0
        dim_action = self.base_env._action_info[0].dim
        dim_state = len(self.base_env.state(idx=0))
        self.observation_space = \
            Box(-ob_scale * np.ones(dim_state),
                ob_scale * np.ones(dim_state))
        self.action_space = \
            Box(env_config['action']['range_min_pol']*np.ones(dim_action),
                env_config['action']['range_max_pol']*np.ones(dim_action))
        
        ob, rew, eoe, info = self.step(np.zeros(dim_action))
        self.reward_keys = sorted(info['rew_detail'].keys(), key=lambda x:x.lower())
        self.reset()

    def state(self):
        return self.base_env.state(idx=0)

    def reset(self, start_time=None, add_noise=None):
        self.base_env.reset(start_time, add_noise)
        return self.base_env.state(idx=0)

    def step(self, action):
        rew, info = self.base_env.step(action)
        obs = self.state()
        eoe = self.base_env.end_of_episode
        
        return obs, rew[0], eoe, info[0]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class EnvRenderer(er.EnvRenderer):
    def __init__(self, pi, **kwargs):
        self.pi = pi
        self.pi_stochastic = True
        self.time_checker_auto_play = basics.TimeChecker()
        super().__init__(**kwargs)
    def one_step(self):
        s1 = self.env.state()
        a, v = self.pi.act(self.pi_stochastic, s1)
        s2, rew, eoe, info = self.env.step(a)
    def render_callback(self):
        self.env.base_env.render(self.rm.flag, self.rm.COLORS_FOR_AGENTS)
    def idle_callback(self):
        time_elapsed = self.time_checker_auto_play.get_time(restart=False)
        if self.rm.flag['auto_play'] and time_elapsed >= self.env.base_env._dt_con:
            self.time_checker_auto_play.begin()
            self.one_step()
    def keyboard_callback(self, key):
        if key == b'r':
            s = self.env.reset()
        elif key == b' ':
            self.time_checker_auto_play.begin()
            self.one_step()

def default_cam():
    return rm.camera.Camera(pos=np.array([0.0, 3.0, 2.0]),
                            origin=np.array([0.0, 0.0, 0.0]), 
                            vup=np.array([0.0, 0.0, 1.0]), 
                            fov=45.0)

# class PPO_ENV(base_env.Env):
#     def __init__(self,
#                  fps_sim,
#                  fps_con,
#                  mode="imitation",
#                  task="heading",
#                  verbose=False,
#                  sim_window=math.inf,
#                  end_margin=1.5,
#                  state_choices=[],
#                  action_type="spd",
#                  action_mode="relative",
#                  action_range_min=-1.5,
#                  action_range_max=1.5,
#                  action_range_min_pol=-20.0,
#                  action_range_max_pol=20.0,
#                  reward_choices=[],
#                  reward_mode="sum",
#                  ref_motion_files=None,
#                  add_noise=False,
#                  early_term_choices=[],
#                  et_low_reward_thres=0.1,
#                  char_info_module=None,
#                  ref_motion_scale=1.0,
#                  sim_char_file=None,
#                  base_motion_file=None,
#                  et_falldown_contactable_body=[],
#                  self_collision=True,
#                  reward_weight_scale=1.0,
#                  base_controller=None,
#                  base_controller_input=None,
#                  base_controller_state_dim=0,
#                  base_controller_action_dim=0,
#                  base_controller_fps=0,
#                  reward_weight_keys=[],
#                  reward_weight_vals=[],
#                  num_cluster=None,
#                  cluster_id_assigned=None,
#                  ):
#         base_env.Env.__init__(self, 
#                               fps_sim=fps_sim, 
#                               fps_con=fps_con, 
#                               mode=mode,
#                               task=task,
#                               verbose=verbose, 
#                               sim_window=sim_window, 
#                               state_choices=state_choices,
#                               action_type=action_type,
#                               action_mode=action_mode,
#                               action_range_min=action_range_min,
#                               action_range_max=action_range_max,
#                               action_range_min_pol=action_range_min_pol,
#                               action_range_max_pol=action_range_max_pol,
#                               reward_choices=reward_choices,
#                               reward_mode=reward_mode,
#                               ref_motion_files=ref_motion_files,
#                               add_noise=add_noise,
#                               early_term_choices=early_term_choices,
#                               et_low_reward_thres=et_low_reward_thres,
#                               char_info_module=char_info_module,
#                               ref_motion_scale=ref_motion_scale,
#                               sim_char_file=sim_char_file,
#                               base_motion_file=base_motion_file,
#                               et_falldown_contactable_body=et_falldown_contactable_body,
#                               self_collision=self_collision,
#                               reward_weight_scale=reward_weight_scale,
#                               base_controller=base_controller,
#                               base_controller_input=base_controller_input,
#                               base_controller_state_dim=base_controller_state_dim,
#                               base_controller_action_dim=base_controller_action_dim,
#                               base_controller_fps=base_controller_fps,
#                               reward_weight_keys=reward_weight_keys,
#                               reward_weight_vals=reward_weight_vals,
#                               )
#         ob_scale = 50.0
#         dim_state = len(self.state())
#         ob_low = -ob_scale * np.ones(dim_state)
#         ob_high = ob_scale * np.ones(dim_state)

#         self.observation_space = spaces.Box(
#             low = ob_low,
#             high = ob_high,
#             dtype='float32'
#         )
#         self.action_space = spaces.Box(
#             low = action_range_min_pol * np.ones(self._dim_action),
#             high = action_range_max_pol * np.ones(self._dim_action),
#             dtype='float32'
#         )

#         # print('--------------------------------------')
#         # print('Observation:', self.observation_space)
#         # print('Action', self.action_space)
#         # print('--------------------------------------')

#         self.seed()
#         self.reward_range = (0, 1.0)
#         self.metadata = {'render.modes': []}
#         self.spec = None
#         self.rew_data_prev = self.reward_data()
#         self.rew_data_next = self.reward_data()

#         rewa_total, rew_detail = self.reward(None,
#                                              self.rew_data_prev,
#                                              self.rew_data_next, 
#                                              np.zeros(self._dim_action))
#         self.reward_keys = sorted(rew_detail.keys(), key=lambda x:x.lower())

#         self.num_cluster = num_cluster
#         self.cluster_id_assigned = cluster_id_assigned
#         self.train_experts = True
#         self.force_reset = False

#     def step(self, a):
#         reward, rew_detail = base_env.Env.step(self, a)
#         state = base_env.Env.state(self)
#         eoe = self.end_of_episode
        
#         return state, reward, eoe, rew_detail

#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]

#     def reset(self, start_time=None, add_noise=None, change_ref_motion=True, ref_motion_idx=None):
#         base_env.Env.reset(self, start_time, add_noise)
#         return base_env.Env.state(self)

def keyboard_callback(key):
    global time_elapsed
    global time_checker_auto_play
    global verbose
    global reward, gamma
    global cnt_screenshot
    global recorder
    global val_buffer, sim_speed
    global num_cluster, files_cluster_id
    global args

    env = base_env.env
    pi = base_env.policy

    if key == b'r' or key == b'R':
        print('Key[%s]: reset environment'%key)
        if key == b'r':
            s = env.reset(start_time=None, change_ref_motion=False)
        else:
            s = env.reset(start_time=0.0, change_ref_motion=True)
        a, v = pi.act(False, s)
        # print('Value:', v)
        time_elapsed = 0.0
        time_checker_auto_play.begin()
        pre_sim_time = env._elapsed_time
        sim_speed.clear()
        reward = 0.0
        for key, val in val_buffer.items():
            val.clear()
        weight_experts_buffer.clear()
        cnt_screenshot = 0
        recorder.reset()
        # ps, _, _, _ = env._sim_agent.get_link_state_choices()
        # hs = [mmMath.projectionOnVector(p, env._sim_agent._char_info.v_up_env) for p in ps]
        # print(np.max(hs), np.min(hs))
    elif key == b'>' or key == b'<':
        if key == b'>':
            idx = min(env._ref_motion_idx+1, len(env._ref_motion_all)-1)
        else:
            idx = max(env._ref_motion_idx-1, 0)
        env._ref_motion_idx = idx
        env._ref_motion = env._ref_motion_all[idx]
        if files_cluster_id is not None and len(files_cluster_id) > 0:
            print('ref_motion_idx:', idx, files_cluster_id[idx])
        else:
            print('ref_motion_idx:', idx, files_cluster_id[idx])
        keyboard_callback(b'r')
    elif key == b'a':
        print('Key[a]: auto_play:', not rm.flag['auto_play'])
        rm.flag['auto_play'] = not rm.flag['auto_play']
    elif key == b'c' or key == b'C':
        if args.cam_file is not None:
            rm.viewer.load_cam(args.cam_file)
        def save_screenshots(start_time, end_time, save_dir, ref_motion_idx=None):
            print(start_time, end_time)
            try:
                os.makedirs(save_dir, exist_ok = True)
            except OSError:
                print("Invalid Subdirectory")
                return
            cnt_screenshot = 0
            if ref_motion_idx is not None:
                s = env.reset(start_time=start_time, ref_motion_idx=ref_motion_idx)
            else:
                s = env.reset(start_time=start_time)
            while env._elapsed_time <= end_time:
                a, v = pi.act(False, s)
                s, rew, eoe, rew_detail = env.step(a)
                val_buffer['total'].append(v)
                if isinstance(pi, (policy.AdditivePolicy, 
                                   policy.AdditiveIndPolicy, 
                                   policy.MultiplicativePolicy, 
                                   policy.MultiplicativePolicy2, 
                                   policy.ValueCompositePolicy)):
                    a_v = pi.act_expert(False, s)
                    for i in range(pi.num_experts()):
                        val_buffer['expert_%d'%i].append(a_v[i][1])
                    weight_experts_buffer.append(pi.weight(s))
                name = 'screenshot_%04d'%(cnt_screenshot)
                base_env.idle_callback(allow_auto_play=False)
                rm.viewer.drawGL()
                rm.viewer.save_screen(dir=save_dir, name=name)
                print('\rTimeElapsed(%4.4f) / %s' %(env._elapsed_time, save_dir+name), end=" ")
                cnt_screenshot += 1
            print("\n")

        if args.screenshot_dir is None:
            suffix = input("Enter subdirectory for screenshot file: ")
            save_dir = "%s/%s"%(SCREENSHOT_DIR, suffix)
        else:
            save_dir = args.screenshot_dir

        if key == b'c':
            if args.screenshot_start_time is None:
                start_time = input("Enter strat time (s): ")
                try:
                   start_time = float(start_time)
                except ValueError:
                   print("That's not a number!")
                   return
            else:
                start_time = args.screenshot_start_time

            if args.screenshot_end_time is None:
                end_time = input("Enter end time (s): ")
                try:
                   end_time = float(end_time)
                except ValueError:
                   print("That's not a number!")
                   return
            else:
                end_time = args.screenshot_end_time

            save_screenshots(start_time, end_time, save_dir)
        else:
            for i in range(len(env._ref_motion_all)):
                save_dir_i = os.path.join(save_dir, env._ref_motion_file_names[i])
                save_screenshots(0.5, env._ref_motion_all[i].length(), save_dir_i, i)

    elif key == b'E' or key == b'e':
        run_simul_after_fall = (key == b'E')
        '''
        Evaluation by success/failure, and reward
        '''
        verbose = env._verbose
        env._verbose = False

        save_sim_result = args.sim_result_save_dir is not None

        if save_sim_result:
            num_saved_sim_result = {}
            for i in cluster_id_used: num_saved_sim_result[i] = 0
            
            save_motion_base = copy.deepcopy(env._ref_motion)
            save_motion_base.clear()
            save_dirs = {}
            for i in cluster_id_used:
                save_dir = os.path.join(args.sim_result_save_dir, 'cluster%d'%(i))
                os.makedirs(save_dir, exist_ok = True)
                save_dirs[i] = save_dir

        results_dict = {}

        for i in range(len(env._ref_motion_all)):
            
            filename = env._ref_motion_file_names[i]
            
            ''' Skip, if we have the result already '''
            if filename in results_dict:
                result = results_dict[filename]
                print(i, result['succ_fail'], result['rew_sum'], result['fall_time'], result['ref_motion_length'])
                continue

            cluster_id = files_cluster_id[i]
            s = env.reset(start_time=0.5, ref_motion_idx=i)

            if save_sim_result:
                if args.sim_result_save_num_per_cluster is not None:
                    if num_saved_sim_result[cluster_id] >= args.sim_result_save_num_per_cluster: continue
                save_motion = copy.deepcopy(save_motion_base)
                save_motion.add_one_frame(0.0, env._sim_agent.get_pose(save_motion.skel).data)
            
            rew_sum = 0.0
            fall_time = 0.0
            is_fallen = False
            
            while True:
                a, v = pi.act(False, s)
                s, rew, eoe, rew_detail = env.step(a)
                rew_sum += rew
                if save_sim_result: 
                    save_motion.add_one_frame(env._elapsed_time, env._sim_agent.get_pose(save_motion.skel).data)
                if 'low_rewards' in env.end_of_episode_reason:
                    if not is_fallen:
                        is_fallen = True
                        fall_time = env._elapsed_time
                    if not run_simul_after_fall: break
                if env._elapsed_time >= env._ref_motion.length():
                    if not is_fallen: fall_time = env._elapsed_time
                    break

            result = {}
            result['succ_fail'] = 'fail' if is_fallen else 'success'
            result['rew_sum'] =  rew_sum
            result['fall_time'] = env._elapsed_time
            result['ref_motion_length'] = env._ref_motion.length()
            result['success_rate'] = 0.0 if is_fallen else 1.0
            print(i, result['succ_fail'], result['rew_sum'], result['fall_time'], result['ref_motion_length'])
            results_dict[filename] = result
            
            if save_sim_result:
                save_success_only = basics.str2bool(args.sim_result_save_success_only)
                if (save_success_only and not is_fallen) or (not save_success_only):
                    save_motion.save_bvh(os.path.join(save_dirs[cluster_id], filename))
                    num_saved_sim_result[cluster_id] += 1

        env._verbose = verbose
    elif key == b'M':
        filename = 'data/temp/temp.cam.gzip'
        rm.viewer.save_cam(filename)
        print('Saved:', filename)
    elif key == b'm':
        if args.cam_file is not None:
            filename = args.cam_file
        else:
            filename = 'data/temp/temp.cam.gzip'
        rm.viewer.load_cam(filename)
        print('Loaded:', filename)
    elif key == b' ':
        state = env.state()
        action, val = pi.act(False, state)
        s, rew, eoe, rew_detail = env.step(action)
        val_buffer['total'].append(val)
        if isinstance(pi, (policy.AdditivePolicy, 
                           policy.AdditiveIndPolicy, 
                           policy.MultiplicativePolicy, 
                           policy.MultiplicativePolicy2, 
                           policy.ValueCompositePolicy)):
            a_v = pi.act_expert(False, s)
            for i in range(pi.num_experts()):
                val_buffer['expert_%d'%i].append(a_v[i][1])
            weight_experts_buffer.append(pi.weight(state))
        print('time: %f, reward: %f'%(env._elapsed_time, rew))
        print('----------------------------------')
    elif key == b'm':
        rm.flag['motion_capture'] = not rm.flag['motion_capture']
        print('[MotionCapture]', rm.flag['motion_capture'])
    elif key == b'M':
        recorder.save_file('mocap')
    elif key == b's':
        rm.flag['screenshot'] = not rm.flag['screenshot']
        print('Screenshot:', rm.flag['screenshot'])
    elif key == b'S':
        name = 'screenshot_temp'
        rm.viewer.save_screen(dir=SCREENSHOT_DIR, name=name)
    elif key == b'v':
        for i in range(pi.num_experts()):
            filename = 'data/learnable_experts/expert%d/network0'%(i)
            pi.copy_and_save_experts_orig(filename, i)
            print(filename)
    else:
        return base_env.keyboard_callback(key)
    return True

#For testing 
def make_env(seed, env_config):
    rank = cmd_util.MPI.COMM_WORLD.Get_rank()
    cmd_util.set_global_seeds(seed + 10000 * rank)
    # env = PPO_ENV(fps_sim=fps_sim, 
    #               fps_con=fps_con, 
    #               mode=env_mode,
    #               task=env_task,
    #               verbose=verbose, 
    #               sim_window=sim_window, 
    #               end_margin=end_margin,
    #               state_choices=state_choices,
    #               action_type=action_type,
    #               action_mode=action_mode,
    #               action_range_min=action_range_min,
    #               action_range_max=action_range_max,
    #               action_range_min_pol=action_range_min_pol,
    #               action_range_max_pol=action_range_max_pol,
    #               reward_choices=reward_choices,
    #               reward_mode=rew_mode,
    #               ref_motion_files=ref_motion_files,
    #               add_noise=env_noise,
    #               early_term_choices=early_term_choices,
    #               et_low_reward_thres=et_low_reward_thres,
    #               et_falldown_contactable_body=et_falldown_contactable_body,
    #               char_info_module=char_info_module,
    #               ref_motion_scale=ref_motion_scale,
    #               sim_char_file=sim_char_file,
    #               base_motion_file=base_motion_file,
    #               self_collision=self_collision,
    #               reward_weight_scale=reward_weight_scale,
    #               base_controller=base_controller,
    #               base_controller_input=base_controller_input,
    #               base_controller_state_dim=base_controller_state_dim,
    #               base_controller_action_dim=base_controller_action_dim,
    #               base_controller_fps=base_controller_fps,
    #               reward_weight_keys=reward_weight_keys,
    #               reward_weight_vals=reward_weight_vals,
    #               num_cluster=num_cluster,
    #               cluster_id_assigned=cluster_id_assigned,
    #               )
    env = HumanoidImitation(env_config)
    env = cmd_util.Monitor(env, cmd_util.os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
    env.seed(seed)
    return env

def arg_parser():
    parser = argparse.ArgumentParser()
    ''' Specification file of the expriment '''
    parser.add_argument("--spec", required=True, type=str)
    ''' Mode for running an experiment '''
    parser.add_argument("--mode", required=True, choices=['train', 'load'])
    '''  '''
    parser.add_argument("--checkpoint", type=str, default=None)
    '''  '''
    parser.add_argument("--checkpoint2", type=str, default=None)
    '''  '''
    parser.add_argument("--num_workers", type=int, default=None)
    '''  '''
    parser.add_argument("--num_gpus", type=int, default=None)
    ''' Directory where the environment and related files are stored '''
    parser.add_argument("--project_dir", type=str, default=None)
    ''' Directory where intermediate results are saved '''
    parser.add_argument("--local_dir", type=str, default=None)
    '''  '''
    parser.add_argument("--verbose", action='store_true')

    return parser

# def arg_parser():
#     parser = cmd_util.arg_parser()
#     parser.add_argument('--max_time_sec', help='Time limit for learning (min)',type=int, default=0)
#     parser.add_argument('--max_timesteps', help='# of maxinum training tuples for learning', type=int, default=int(1e9))
#     parser.add_argument('--seed', help='RNG seed', type=int, default=0)
#     parser.add_argument('--mode', help='Training mode', choices=['train', 'load', 'retrain', 'test'], type=str, default=None)
#     parser.add_argument('--env_mode', help='Env mode', choices=['imitation', 'emerge_coop', 'emerge_comp'], type=str, default='imitation')
#     parser.add_argument('--env_task', help='Training mode', choices=['heading', 'carry', 'dribble'], type=str, default='heading')
#     parser.add_argument('--rew_mode', help='Reward type', choices=['sum', 'mul'], type=str, default='mul')
#     parser.add_argument('--net', help='Loading network #', type=int, default=None)
#     parser.add_argument('--log', help='Logging method in learning', choices=['matplot', 'file', 'none'], type=str, default='file')
#     parser.add_argument('--rec_period', help='Logging save period', type=int, default=10)
#     parser.add_argument('--rec_dir', help='Logging directory', type=str, default='data/learning')
#     parser.add_argument('--verbose', choices=['true', 'false'], default='false')
#     parser.add_argument('--sim_window', help='Env time horizon', type=float, default=30.0)
#     parser.add_argument('--end_margin', help='Allow a few steps after EOE', type=float, default=1.5)
#     parser.add_argument('--gamma', help='Discount factor', type=float, default=0.95)
#     parser.add_argument('--batch_size', help='Batch size for network update', type=int, default=256)
#     parser.add_argument('--tuple_per_iter', help='# of tuples per learning iteration', type=int, default=4096*2)
#     parser.add_argument('--state_choices', action='append', choices=['body', 'imitation', 'interaction', 'task'], help='States definitions')
#     parser.add_argument('--action_type', choices=['none', 'spd', 'pd', 'cpd', 'cp', 'v'], type=str, default='spd')
#     parser.add_argument('--action_mode', choices=['relative', 'absolute'], type=str, default='absolute')
#     parser.add_argument('--action_range_min', type=float, default=-1.5)
#     parser.add_argument('--action_range_max', type=float, default=1.5)
#     parser.add_argument('--action_range_min_pol', type=float, default=-15.0)
#     parser.add_argument('--action_range_max_pol', type=float, default=15.0)
#     parser.add_argument('--reward_choices', action='append', choices=['pose', 'vel', 'ee', 'root', 'com', 'interaction'], help='States definitions')
#     parser.add_argument('--ob_filter', help='Observation Filter', choices=['true', 'false'], default='true')
#     parser.add_argument('--ref_motion_files', action='append', type=str, default=[])
#     parser.add_argument('--sim_result_save_dir', type=str, default=None)
#     parser.add_argument('--sim_result_save_success_only', choices=['true', 'false'], default="false")
#     parser.add_argument('--policy_name', help='Name scope of policy function', type=str, default="mypolicy")
#     parser.add_argument('--policy_type', choices=['mlp', 'additive', 'additive_ind', 'multiplicative', 'multiplicative2', 'valuecompsoite'], default='mlp')
#     parser.add_argument('--num_hid_size', help='Width of hidden layers', type=int, default=256)
#     parser.add_argument('--num_hid_layers', help='Depth of hidden layers', type=int, default=2)
#     parser.add_argument('--num_hid_size_gate', help='Width of hidden layers', type=int, default=128)
#     parser.add_argument('--num_hid_layers_gate', help='Depth of hidden layers', type=int, default=2)
    
#     parser.add_argument('--base_controller_input', choices=['raw', 'poses'], default='poses')
#     parser.add_argument('--base_controller_policy_name', help='Name (Base Controller)', type=str, default="mypolicy")
#     parser.add_argument('--base_controller_policy_type', help='Type (Base Controller)', type=str, default='additive')
#     parser.add_argument('--base_controller_num_hid_size', help='Width of hidden layers (Base Controller)', type=int, default=256)
#     parser.add_argument('--base_controller_num_hid_layers', help='Depth of hidden layers (Base Controller)', type=int, default=2)
#     parser.add_argument('--base_controller_num_hid_size_gate', help='Width of hidden layers (Base Controller)', type=int, default=128)
#     parser.add_argument('--base_controller_num_hid_layers_gate', help='Depth of hidden layers (Base Controller)', type=int, default=2)
#     parser.add_argument('--base_controller_weight', help='Weight (Base Controller)', type=str, default=None)
#     parser.add_argument('--base_controller_state_dim', help='Weight (Base Controller)', type=int, default=1880)
#     parser.add_argument('--base_controller_action_dim', help='Weight (Base Controller)', type=int, default=51)
#     parser.add_argument('--base_controller_fps', type=int, default=30)

#     parser.add_argument('--old_expert_names', action='append', help='Name of Old Expert Policies')
#     parser.add_argument('--old_expert_weights', action='append', help='Weights of Old Expert Policies')
#     parser.add_argument('--new_expert_names', action='append', help='Name of New expert Policies')
#     parser.add_argument('--w_new_expert_usage', help='PPO: Weight for new expert usage', type=float, default=0.0)
#     parser.add_argument('--gate_expert_alter', help='PPO: Alternate learning of gate and experts', choices=['true', 'false'], default="false")
#     parser.add_argument('--gate_expert_alter_gate_iter', help='PPO: Gate update iteration', type=int, default=10)
#     parser.add_argument('--gate_expert_alter_expert_iter', help='PPO: Experts update iteration',type=int, default=40)
#     parser.add_argument('--optim_stepsize_pol', help='PPO: Optimization stepsize of policy',type=float, default=1e-5)
#     parser.add_argument('--optim_stepsize_val', help='PPO: Optimization stepsize of value function',type=float, default=1e-3)
#     parser.add_argument('--ob_filter_gate', help='Obervation filter for gate', choices=['true', 'false'], default="true")
#     parser.add_argument('--ob_filter_old_expert', help='Obervation filter for experts', choices=['true', 'false'], default="true")
#     parser.add_argument('--ob_filter_new_expert', help='Obervation filter for beginner', choices=['true', 'false'], default="true")
#     parser.add_argument('--ob_filter_update_for_expert', help='Obervation filter update for experts', choices=['true', 'false'], default="true")
#     parser.add_argument('--trainable_gate', help='Make gate trainable', choices=['true', 'false'], default="true")
#     parser.add_argument('--trainable_old_expert', help='Train Expert Network', choices=['true', 'false'], default="false")
#     parser.add_argument('--trainable_new_expert', help='Train Beginner Network', choices=['true', 'false'], default="true")
#     parser.add_argument('--env_noise', help='Noise', choices=['true', 'false'], default="true")
#     parser.add_argument('--early_term_choices', action='append', choices=['task_complete', 'falldown', 'root_fail', 'low_reward'], help='Early termination conditions')
#     parser.add_argument('--et_low_reward_thres', help='Threshold on early termination via rewards', type=float, default=0.1)
#     parser.add_argument('--et_falldown_contactable_body', action='append', help='Contactable bodies for early termination via contact', default=[])
#     parser.add_argument('--char_info_module', action='append', type=str, default=[])
#     parser.add_argument('--ref_motion_scale', action='append', type=float, default=[])
#     parser.add_argument('--sim_char_file', action='append', type=str, default=[])
#     parser.add_argument('--base_motion_file', action='append', type=str, default=[])
#     parser.add_argument('--self_collision', choices=['true', 'false'], default="true")
#     parser.add_argument('--reward_weight_scale', type=float, default=1.0)
#     parser.add_argument('--reward_weight_keys', action='append', type=str, default=[])
#     parser.add_argument('--reward_weight_vals', action='append', type=float, default=[])
#     parser.add_argument('--render_window_w', type=int, default=1280)
#     parser.add_argument('--render_window_h', type=int, default=720)
#     parser.add_argument('--cam_pos', help='Viewer camera position', nargs='+', type=float, default=None)
#     parser.add_argument('--cam_origin', help='Viewer camera origin (target)', nargs='+', type=float, default=None)
#     parser.add_argument('--cam_fov', help='Viewer camera field of view', type=float, default=45.0)
#     parser.add_argument('--cam_follow', help='Camera follows the character or not', type=str, default='true')
#     parser.add_argument('--cam_file', help='Saved camera configuration', type=str, default=None)
#     parser.add_argument('--screenshot_dir', type=str, default=None)
#     parser.add_argument('--screenshot_start_time', type=float, default=None)
#     parser.add_argument('--screenshot_end_time', type=float, default=None)
#     parser.add_argument('--render_overlay', type=str, default='true')
#     parser.add_argument('--render_overlay_expert_weight', type=str, default='true')
#     parser.add_argument('--render_overlay_value_fn', type=str, default='true')
#     parser.add_argument('--render_overlay_basic_info', type=str, default='true')
#     parser.add_argument('--ground_w', type=float, default=30)
#     parser.add_argument('--ground_h', type=float, default=30)
#     parser.add_argument('--sim_fps', type=int, default=300)
#     parser.add_argument('--con_fps', type=int, default=30)

#     return parser

def train(args,
          spec,
          # num_cluster,
          # cluster_id_assigned,
          ):
    config = spec['config']
    hid_size = config['model']['fcnet_hiddens'][0]
    num_hid_layers = len(config['model']['fcnet_hiddens'])
    hid_size_gate = 128
    num_hid_layers_gate = 2
    U.make_session(num_cpu=1).__enter__()
    def policy_fn_mlp(name, ob_space, ac_space):
        return policy.MlpPolicy(
            name=name, independent=True,
            stochastic=None, ob=None, ob_space=ob_space, ac_space=ac_space, 
            hid_size=hid_size, num_hid_layers=num_hid_layers,
            ob_filter=config['observation_filter']=="MeanStdFilter",
            )
    # def policy_fn_valuecomposite(name, ob_space, ac_space):
    #     return policy.ValueCompositePolicy(
    #         name=name, independent=True,
    #         stochastic=None, ob=None, ob_space=ob_space, ac_space=ac_space, 
    #         hid_size_expert=num_hid_size, num_hid_layers_expert=num_hid_layers,
    #         hid_size_gate=hid_size_gate, num_hid_layers_gate=num_hid_layers_gate,
    #         old_expert_names=args.old_expert_names, 
    #         old_expert_weights=args.old_expert_weights, 
    #         new_expert_names=args.new_expert_names,
    #         ob_filter=basics.str2bool(args.ob_filter),
    #         )
    # def policy_fn_additive(name, ob_space, ac_space):
    #     return policy.AdditivePolicy(
    #         name=name, independent=True,
    #         stochastic=None, ob=None, ob_space=ob_space, ac_space=ac_space, 
    #         hid_size_expert=args.num_hid_size, num_hid_layers_expert=args.num_hid_layers,
    #         hid_size_gate=args.num_hid_size_gate, num_hid_layers_gate=args.num_hid_layers_gate,
    #         old_expert_names=args.old_expert_names, 
    #         old_expert_weights=args.old_expert_weights, 
    #         new_expert_names=args.new_expert_names,
    #         ob_filter_gate=basics.str2bool(args.ob_filter_gate),
    #         ob_filter_old_expert=basics.str2bool(args.ob_filter_old_expert),
    #         ob_filter_new_expert=basics.str2bool(args.ob_filter_new_expert),
    #         trainable_gate=basics.str2bool(args.trainable_gate),
    #         trainable_old_expert=basics.str2bool(args.trainable_old_expert),
    #         trainable_new_expert=basics.str2bool(args.trainable_new_expert),
    #         )
    # def policy_fn_additive_ind(name, ob_space, ac_space):
    #     return policy.AdditiveIndPolicy(
    #         name=name, independent=True,
    #         stochastic=None, ob=None, ob_space=ob_space, ac_space=ac_space, 
    #         hid_size_expert=args.num_hid_size, num_hid_layers_expert=args.num_hid_layers,
    #         hid_size_gate=args.num_hid_size_gate, num_hid_layers_gate=args.num_hid_layers_gate,
    #         old_expert_names=args.old_expert_names, 
    #         old_expert_weights=args.old_expert_weights, 
    #         new_expert_names=args.new_expert_names,
    #         ob_filter_gate=basics.str2bool(args.ob_filter_gate),
    #         ob_filter_old_expert=basics.str2bool(args.ob_filter_old_expert),
    #         ob_filter_new_expert=basics.str2bool(args.ob_filter_new_expert),
    #         trainable_gate=basics.str2bool(args.trainable_gate),
    #         trainable_old_expert=basics.str2bool(args.trainable_old_expert),
    #         trainable_new_expert=basics.str2bool(args.trainable_new_expert),
    #         )
    # def policy_fn_multiplicative(name, ob_space, ac_space):
    #     return policy.MultiplicativePolicy(
    #         name=name, independent=True,
    #         stochastic=None, ob=None, ob_space=ob_space, ac_space=ac_space, 
    #         hid_size_expert=args.num_hid_size, num_hid_layers_expert=args.num_hid_layers,
    #         hid_size_gate=args.num_hid_size_gate, num_hid_layers_gate=args.num_hid_layers_gate,
    #         old_expert_names=args.old_expert_names, 
    #         old_expert_weights=args.old_expert_weights, 
    #         new_expert_names=args.new_expert_names,
    #         ob_filter_gate=basics.str2bool(args.ob_filter_gate),
    #         ob_filter_old_expert=basics.str2bool(args.ob_filter_old_expert),
    #         ob_filter_new_expert=basics.str2bool(args.ob_filter_new_expert),
    #         trainable_gate=basics.str2bool(args.trainable_gate),
    #         trainable_old_expert=basics.str2bool(args.trainable_old_expert),
    #         trainable_new_expert=basics.str2bool(args.trainable_new_expert),
    #         )
    # def policy_fn_multiplicative2(name, ob_space, ac_space):
    #     return policy.MultiplicativePolicy2(
    #         name=name, independent=True,
    #         stochastic=None, ob=None, ob_space=ob_space, ac_space=ac_space, 
    #         hid_size_expert=args.num_hid_size, num_hid_layers_expert=args.num_hid_layers,
    #         hid_size_gate=args.num_hid_size_gate, num_hid_layers_gate=args.num_hid_layers_gate,
    #         old_expert_names=args.old_expert_names, 
    #         old_expert_weights=args.old_expert_weights, 
    #         new_expert_names=args.new_expert_names,
    #         ob_filter_gate=basics.str2bool(args.ob_filter_gate),
    #         ob_filter_old_expert=basics.str2bool(args.ob_filter_old_expert),
    #         ob_filter_new_expert=basics.str2bool(args.ob_filter_new_expert),
    #         trainable_gate=basics.str2bool(args.trainable_gate),
    #         trainable_old_expert=basics.str2bool(args.trainable_old_expert),
    #         trainable_new_expert=basics.str2bool(args.trainable_new_expert),
    #         )
    # def value_marginal(name, ob_space):
    #     return policy.ValueMaginal(
    #         name=name, 
    #         ob_space=ob_space,
    #         hid_size=128, num_hid_layers=2,
    #         ob_filter=False)
    def to_policy_fn(policy_type):
        if policy_type == "mlp":
            policy_fn = policy_fn_mlp
        elif policy_type == "valuecomposite":
            policy_fn = policy_fn_valuecomposite
        elif policy_type == "additive":
            policy_fn = policy_fn_additive
        elif policy_type == "additive_ind":
            policy_fn = policy_fn_additive_ind
        elif policy_type == "multiplicative":
            policy_fn = policy_fn_multiplicative
        elif policy_type == "multiplicative2":
            policy_fn = policy_fn_multiplicative2
        else:
            raise NotImplementedError
        return policy_fn

    # policy_fn = to_policy_fn(args.policy_type)
    policy_fn = policy_fn_mlp

    # if args.base_controller_weight is not None:
    #     observation_space_base = spaces.Box(
    #         low = -50 * np.ones(args.base_controller_state_dim),
    #         high = 50 * np.ones(args.base_controller_state_dim),
    #         dtype='float32'
    #     )
    #     action_space_base = spaces.Box(
    #         low = args.action_range_min_pol * np.ones(args.base_controller_action_dim),
    #         high = args.action_range_max_pol * np.ones(args.base_controller_action_dim),
    #         dtype='float32'
    #     )
    #     policy_fn_base = to_policy_fn(args.base_controller_policy_type)
    #     pi_base = policy_fn_base("pi/%s"%args.base_controller_policy_name,
    #                              observation_space_base, 
    #                              action_space_base)
    #     pi_base.load_variables(args.base_controller_weight)
    #     base_controller = len(args.sim_char_file) * [pi_base]
    #     print('Base Controller <%s:%s> is loaded'%(args.base_controller_policy_name, args.base_controller_weight))
    # else:
    #     base_controller = None

    # env = make_env(seed=args.seed,
    #                fps_sim=args.sim_fps,
    #                fps_con=args.con_fps,
    #                env_mode=args.env_mode,
    #                env_task=args.env_task,
    #                ref_motion_files=args.ref_motion_files,
    #                verbose=basics.str2bool(args.verbose),
    #                sim_window=args.sim_window,
    #                end_margin=args.end_margin,
    #                state_choices=args.state_choices,
    #                action_type=args.action_type,
    #                action_mode=args.action_mode,
    #                action_range_min=args.action_range_min,
    #                action_range_max=args.action_range_max,
    #                action_range_min_pol=args.action_range_min_pol,
    #                action_range_max_pol=args.action_range_max_pol,
    #                reward_choices=args.reward_choices,
    #                rew_mode=args.rew_mode,
    #                env_noise=basics.str2bool(args.env_noise),
    #                early_term_choices=args.early_term_choices,
    #                et_low_reward_thres=args.et_low_reward_thres,
    #                et_falldown_contactable_body=args.et_falldown_contactable_body,
    #                char_info_module=args.char_info_module,
    #                ref_motion_scale=args.ref_motion_scale,
    #                sim_char_file=args.sim_char_file,
    #                base_motion_file=args.base_motion_file,
    #                self_collision=basics.str2bool(args.self_collision),
    #                reward_weight_scale=args.reward_weight_scale,
    #                base_controller=base_controller,
    #                base_controller_input=args.base_controller_input,
    #                base_controller_state_dim=args.base_controller_state_dim,
    #                base_controller_action_dim=args.base_controller_action_dim,
    #                base_controller_fps=args.base_controller_fps,
    #                reward_weight_keys=args.reward_weight_keys,
    #                reward_weight_vals=args.reward_weight_vals,
    #                num_cluster=num_cluster,
    #                cluster_id_assigned=cluster_id_assigned,
    #                )

    ''' Create directory for logging if it does not exist '''
    rec_dir = os.path.join(spec['local_dir'], spec['name'])
    os.makedirs(rec_dir, exist_ok = True)

    ''' Save spec used for this experiment to check them in later '''
    with open(os.path.join(rec_dir,"config.yaml"), 'w') as f:
        yaml.dump(spec, f)

    env = make_env(seed=0, env_config=config['env_config'])

    batch_per_env = max(config['sgd_minibatch_size'], config['train_batch_size']//MPI.COMM_WORLD.Get_size())
    pi = pposgd.learn(env, policy_fn,
                      max_seconds=spec['stop']['time_total_s'],
                      max_timesteps=int(1e9),
                      timesteps_per_actorbatch=batch_per_env,
                      clip_param=config['clip_param'], entcoeff=0.0, vfcoeff=1.0,
                      optim_epochs=config['num_sgd_iter'],
                      optim_stepsize_pol=config['lr'],
                      optim_stepsize_val=config['lr'],
                      optim_batchsize=config['sgd_minibatch_size'],
                      gamma=config['gamma'], lam=config['lambda'], schedule='linear', adam_epsilon=1.0e-5,
                      mode=args.mode, 
                      network_number=None,
                      network_number_vmar=None, 
                      log_learning_curve="file",
                      file_record_period=spec['checkpoint_freq'],
                      file_record_dir=os.path.join(spec['local_dir'], spec['name']),
                      policy_name="%s_%s"%(spec['run'],config['env']),
                      )
    return env, pi

def idle_callback():
    global time_checker_auto_play
    global pre_sim_time, sim_speed, sim_speed_avg

    env = base_env.env
    pi = base_env.policy
    
    time_elapsed = time_checker_auto_play.get_time(restart=False)
    if rm.flag['auto_play'] and time_elapsed >= env._dt_con:
        time_checker_auto_play.begin()
        state = env.state()
        action, val = pi.act(False, state)
        s, rew, eoe, rew_detail = env.step(action)
        val_buffer['total'].append(val)
        
        if isinstance(pi, (policy.AdditivePolicy, 
                           policy.AdditiveIndPolicy, 
                           policy.MultiplicativePolicy, 
                           policy.MultiplicativePolicy2, 
                           policy.ValueCompositePolicy)):
            global weight_experts_buffer
            a_v = pi.act_expert(False, s)
            for i in range(pi.num_experts()):
                val_buffer['expert_%d'%i].append(a_v[i][1])
            weight_experts_buffer.append(pi.weight(state))

        sim_speed.append((env._elapsed_time - pre_sim_time) * rm.viewer.avg_fps)
        sim_speed_avg = np.mean(sim_speed)
        pre_sim_time = env._elapsed_time

    base_env.idle_callback(allow_auto_play=False)

def render_callback():
    base_env.render_callback()

def overlay_callback():
    if not rm.flag['overlay']: return
    global gamma, sim_speed_avg

    env = base_env.env
    pi = base_env.policy
    
    rm.gl.glPushAttrib(rm.gl.GL_LIGHTING)
    rm.gl.glDisable(rm.gl.GL_LIGHTING)
    
    w, h = rm.viewer.window_size

    graph_axis_len = 150
    origin = np.array([0.8*w, 0.3*h])
    graph_pad_len = 30

    ''' Value Function '''

    # d = list(val_buffer['total'])
    # x = [range(len(d))]
    # y = [d]
    # x_r = [(0, val_buffer['total'].maxlen-1)]
    # y_r = [(env.return_min(), env.return_max(gamma))]
    # color = [[0, 0, 0, 1]]
    # line_width = [2.0]

    # if isinstance(pi, (policy.AdditivePolicy, policy.AdditiveIndPolicy, policy.MultiplicativePolicy, policy.ValueCompositePolicy)):
    #     num_experts = pi.num_experts()
    #     for i in range(num_experts):
    #         d = list(val_buffer['expert_%d'%i])
    #         x.append(range(len(d)))
    #         y.append(d)
    #     x_r += [(0, val_buffer['total'].maxlen-1)] * num_experts
    #     y_r += [(env.return_min(), env.return_max(gamma))] * num_experts
    #     color += rm.gl_render.COLOR_SEQUENCE
    #     line_width += [2.0] * num_experts
    
    # rm.gl_render.render_graph_base_2D(origin, graph_axis_len, graph_pad_len)
    # rm.gl_render.render_graph_data_line_2D(x_data=x,
    #                                     y_data=y,
    #                                     x_range=x_r,
    #                                     y_range=y_r,
    #                                     color=color,
    #                                     line_width=line_width,
    #                                     origin=origin, 
    #                                     axis_len=graph_axis_len, 
    #                                     pad_len=graph_pad_len,
    #                                     multiple_data=True)
    
    ''' FPS '''
    rm.gl_render.render_text("FPS: %.2f"%rm.viewer.avg_fps, pos=[0.05*w, 0.9*h], font=rm.glut.GLUT_BITMAP_9_BY_15)
    rm.gl_render.render_text("Time: %.2f"%env._elapsed_time, pos=[0.05*w, 0.9*h+20], font=rm.glut.GLUT_BITMAP_9_BY_15)
    rm.gl_render.render_text("Sim Speed: %.2f"%sim_speed_avg, pos=[0.05*w, 0.9*h+40], font=rm.glut.GLUT_BITMAP_9_BY_15)

    ''' Weights of Experts '''
    global weight_experts_buffer
    if isinstance(pi, (policy.AdditivePolicy, 
                       policy.AdditiveIndPolicy, 
                       policy.MultiplicativePolicy, 
                       policy.MultiplicativePolicy2, 
                       policy.ValueCompositePolicy)):
        ''' as graph '''
        graph_axis_len, graph_pad_len = 150, 30
        origin = np.array([0.8*w, 0.8*h])
        x, y, x_r, y_r = [], [], [], []
        color, line_width = [], []

        num_experts = pi.num_experts()

        if len(weight_experts_buffer) > 0:
            weight = np.array(weight_experts_buffer)
            for i in range(num_experts):
                d = list(weight[:, i])
                x.append(range(len(d)))
                y.append(d)
            x_r += [(0, weight_experts_buffer.maxlen-1)] * num_experts
            y_r += [(0, 1)] * num_experts
            color += rm.COLORS_FOR_EXPERTS
            line_width += [2.0] * num_experts
        
        # rm.gl_render.render_graph_base_2D(origin, graph_axis_len, graph_pad_len)
        # rm.gl_render.render_graph_data_line_2D(x_data=x,
        #                                     y_data=y,
        #                                     x_range=x_r,
        #                                     y_range=y_r,
        #                                     color=color,
        #                                     line_width=line_width,
        #                                     origin=origin, 
        #                                     axis_len=graph_axis_len, 
        #                                     pad_len=graph_pad_len,
        #                                     multiple_data=True)
        
        ''' as histogram '''
        w_bar, h_bar = 150, 20
        origin = np.array([0.95*w-w_bar, 0.95*h-h_bar])
        pos = origin.copy()
        weight_cur = weight_experts_buffer[-1] if len(weight_experts_buffer) > 0 else None

        num_old_experts = pi.num_old_experts()
        for i in reversed(range(pi.num_experts())):
            if i==num_old_experts-1:
                pos += np.array([0.0, -h_bar])
            name = "E%d"%(i) if i < num_old_experts else "B%d"%(i-num_old_experts)
            w_i = weight_cur[i] if weight_cur is not None else 0.0
            rm.gl_render.render_text(name, pos=pos-np.array([25, -0.8*h_bar]), font=GLUT_BITMAP_9_BY_15)
            rm.gl_render.render_progress_bar_2D_horizontal(w_i, origin=pos, width=w_bar, height=h_bar, color_input=rm.COLORS_FOR_EXPERTS[i])
            pos += np.array([0.0, -h_bar])
    elif isinstance(pi, policy.AdditiveIndPolicy):
        weight_cur = weight_experts_buffer[-1] if len(weight_experts_buffer) > 0 else None
        if weight_cur is None:
            weight_cur = np.zeros((pi.num_experts(), env.action_space.shape[0]))
        ww, hh = 150, 150
        origin = np.array([0.95*w-ww, 0.95*h-hh])
        rm.gl_render.render_matrix(m=weight_cur, 
                                origin=origin,
                                width=ww,
                                height=hh)

    rm.gl.glPopAttrib()

    if rm.flag['screenshot']:
        global cnt_screenshot
        name = 'screenshot_interactive_%04d'%(cnt_screenshot)
        rm.viewer.save_screen(dir=SCREENSHOT_DIR, name=name)
        if cnt_screenshot%100 == 0: print(name)
        cnt_screenshot += 1

window_size = (1280, 720) #Full
# window_size = (640, 720) #2x1
# window_size = (320, 360) #4x2
# window_size = (426, 720) #3x1
# window_size = (426, 360) #3x2
# window_size = (426, 240) #3x3
# window_size = (212, 240) #6x3
# window_size = (1280, 240) #1x3
# window_size = (800, 600) # capture
# window_size = (800, 800) # capture

if __name__ == '__main__':

    args = basics.parse_args_by_file(arg_parser, sys.argv)
    
    with open(args.spec) as f:
        spec = yaml.load(f, Loader=yaml.FullLoader)

    config = spec['config']

    if args.local_dir is not None:
        spec['local_dir'] = args.local_dir
    
    if args.project_dir is not None:
        assert os.path.exists(args.project_dir)
        config['env_config']['project_dir'] = args.project_dir

    if args.checkpoint is not None:
        assert os.path.exists(args.checkpoint)
    
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
    
    if args.num_gpus is not None:
            config['num_gpus'] = args.num_gpus

    logger.configure()

    ''' 
    Unify random seed until environment launches
    After launching, each process will have different random seeds
    '''
    cmd_util.set_global_seeds(0)

    # num_cluster_used = 0
    # cluster_id_used = []
    # cluster_id_assigned = -1

    gamma = config['gamma']
    env, pi = train(args, spec, 
                    # num_cluster=num_cluster_used,
                    # cluster_id_assigned=cluster_id_assigned,
                    )

    logger.log('================================================')
    logger.log('===========ScaDive: Training Fishished==========')
    logger.log('================================================')

    '''
    If the mode is "load" or "test", we need an extra setup for visualization (e.g. OpenGL) 
    '''

    if args.mode == "load" or args.mode == "test":
        import render_module as rm
        rm.initialize()

        '''
        If the environment is loaded for visualization, let the system use more resources
        '''
        os.environ['OPENBLAS_NUM_THREADS'] = str(mp.get_num_cpus())
        os.environ['MKL_NUM_THREADS'] = str(mp.get_num_cpus())

        '''
        For composite policy, prepare value buffer for rendering
        '''
        if isinstance(pi, (policy.AdditivePolicy, 
                           policy.AdditiveIndPolicy, 
                           policy.MultiplicativePolicy, 
                           policy.MultiplicativePolicy2, 
                           policy.ValueCompositePolicy)):
            for i in range(pi.num_experts()):
                val_buffer['expert_%d'%i] = collections.deque(maxlen=10)

        cam = default_cam()
        renderer = EnvRenderer(pi=pi, env=env.env, cam=cam)
        renderer.run()

        # base_env.policy = pi
        # base_env.env = env.env

        # recorder = SimpleRecorder()

        # cam = None
        # if args.cam_file is not None:
        #     with gzip.open(args.cam_file, "rb") as f:
        #         cam = pickle.load(f)
        # else:
        #     if args.cam_origin is not None:
        #         rm.flag['follow_cam'] = False
        #         cam_origin = basics.convert_list_to_nparray(args.cam_origin)
        #     else:
        #         cam_origin = base_env.env.agent_avg_position()

        #     if np.allclose(base_env.env._v_up, np.array([0.0, 1.0, 0.0])):
        #         cam_vup = np.array([0.0, 1.0, 0.0])
        #         if args.cam_pos is not None:
        #             cam_pos = basics.convert_list_to_nparray(args.cam_pos)
        #         else:
        #             cam_pos = cam_origin + np.array([0.0, 2.0, 3.0])
        #     elif np.allclose(base_env.env._v_up, np.array([0.0, 0.0, 1.0])):
        #         cam_vup = np.array([0.0, 0.0, 1.0])
        #         if args.cam_pos is not None:
        #             cam_pos = basics.convert_list_to_nparray(args.cam_pos)
        #         else:
        #             cam_pos = cam_origin + np.array([3.0, 0.0, 2.0])
        #     else:
        #         raise NotImplementedError

        #     cam = rm.camera.Camera(pos=cam_pos,
        #                            origin=cam_origin, 
        #                            vup=cam_vup, 
        #                            fov=args.cam_fov)

        # rm.viewer.run(title=args.rec_dir,
        #               cam=cam,
        #               size=(args.render_window_w, args.render_window_h),
        #               keyboard_callback=keyboard_callback,
        #               render_callback=render_callback,
        #               overlay_callback=overlay_callback,
        #               idle_callback=idle_callback
        #               )
