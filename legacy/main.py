import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from mpi4py import MPI

import policy
import pposgd
from baselines.common import cmd_util
from baselines.common import tf_util as U
from baselines import logger

import gym
from gym import core, spaces
from gym.utils import seeding

from basecode.utils import basics
from basecode.math import mmMath
from basecode.utils import multiprocessing as mp

import env as base_env

SCREENSHOT_DIR = 'data/screenshot/'
COLORS_FOR_EXPERTS = [
    np.array([30,  120, 180, 255])/255,
    np.array([215, 40,  40,  255])/255,
    np.array([225, 120, 190, 255])/255,
    np.array([150, 100, 190, 255])/255,
    np.array([140, 90,  80,  255])/255,
    np.array([50,  160, 50,  255])/255,
    np.array([255, 125, 15,  255])/255,
    np.array([125, 125, 125, 255])/255,
    np.array([255, 0,   255, 255])/255,
    np.array([0,   255, 125, 255])/255,
    ]


flag = {}
flag['screenshot'] = False
flag['motion_capture'] = False
flag['auto_play'] = False
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

class PPO_ENV(base_env.Env):
    def __init__(self, 
                 dt_sim, 
                 dt_con, 
                 mode="mimicpfnn",
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
                 action_range_min_pol=-20.0,
                 action_range_max_pol=20.0,
                 reward_choices=[],
                 reward_mode="sum",
                 ref_motion_files=None,
                 add_noise=False,
                 early_term_choices=[],
                 et_low_reward_thres=0.1,
                 char_info_module="mimicpfnn_char_info.py",
                 ref_motion_scale=0.009,
                 sim_char_file="data/character/pfnn.urdf",
                 base_motion_file="data/motion/pfnn/pfnn_hierarchy_only.bvh",
                 motion_graph_file="data/motion/motiongraph/motion_graph_pfnn.gzip",
                 et_falldown_contactable_body=[],
                 self_collision=True,
                 ref_motion_sample="random",
                 visualization=False,
                 reward_weight_scale=1.0,
                 num_cluster=None,
                 cluster_id_assigned=None,
                 ):
        base_env.Env.__init__(self, 
                              dt_sim=dt_sim, 
                              dt_con=dt_con, 
                              mode=mode,
                              verbose=verbose, 
                              sim_window=sim_window, 
                              pfnn_command_type=pfnn_command_type, 
                              pfnn_command_record=pfnn_command_record, 
                              pfnn_command_file=pfnn_command_file,
                              state_choices=state_choices,
                              state_imit_window=state_imit_window,
                              action_type=action_type,
                              action_mode=action_mode,
                              action_range_min=action_range_min,
                              action_range_max=action_range_max,
                              action_range_min_pol=action_range_min_pol,
                              action_range_max_pol=action_range_max_pol,
                              reward_choices=reward_choices,
                              reward_mode=reward_mode,
                              ref_motion_files=ref_motion_files,
                              add_noise=add_noise,
                              early_term_choices=early_term_choices,
                              et_low_reward_thres=et_low_reward_thres,
                              char_info_module=char_info_module,
                              ref_motion_scale=ref_motion_scale,
                              sim_char_file=sim_char_file,
                              base_motion_file=base_motion_file,
                              motion_graph_file=motion_graph_file,
                              et_falldown_contactable_body=et_falldown_contactable_body,
                              self_collision=self_collision,
                              ref_motion_sample=ref_motion_sample,
                              visualization=visualization,
                              reward_weight_scale=reward_weight_scale,
                              )
        ob_scale = 50.0
        dim_state = len(self.state())
        ob_low = -ob_scale * np.ones(dim_state)
        ob_high = ob_scale * np.ones(dim_state)

        self.dim_state_body = len(self.state_body())
        self.dim_state_task = len(self.state_task())
        self.dim_state_param = len(self.state_param())

        self.observation_space = spaces.Box(
            low = ob_low,
            high = ob_high,
            dtype='float32'
        )
        self.action_space = spaces.Box(
            low = self._action_info.val_min,
            high = self._action_info.val_max,
            dtype='float32'
        )
        # print('--------------------------------------')
        # print('Observation:', self.observation_space)
        # print('Action', self.action_space)
        # print('--------------------------------------')

        ob_low_param = -ob_scale * np.ones(self.dim_state_param)
        ob_high_param = ob_scale * np.ones(self.dim_state_param)
        self.observation_space_param = spaces.Box(
            low = ob_low_param,
            high = ob_high_param,
            dtype='float32'
        )

        self.seed()
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {'render.modes': []}
        self.spec = None
        self.rew_data_prev = self.reward_data()
        self.rew_data_next = self.reward_data()

        rew_data_tmp = self.reward_data()
        rew_total, rew_detail = self.reward(rew_data_tmp,
                                           rew_data_tmp, 
                                           np.zeros(self._action_info.dim))
        self.reward_keys = sorted(rew_detail.keys(), key=lambda x:x.lower())

        self.num_cluster = num_cluster
        self.cluster_id_assigned = cluster_id_assigned
        self.train_experts = True
        self.force_reset = False

    def step(self, a):
        reward, rew_detail = base_env.Env.step(self, a)
        state = base_env.Env.state(self)
        eoe = self.end_of_episode
        
        return state, reward, eoe, rew_detail

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, start_time=None, change_ref_motion=True, ref_motion_idx=None):
        base_env.Env.reset(self, start_time, change_ref_motion, ref_motion_idx)
        # self.rew_data_prev = self.reward_data()
        # self.rew_data_next = self.reward_data()
        return base_env.Env.state(self)

    def render(self):
        base_env.Env.render(self)

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
            s = env.reset(start_time=0.5, change_ref_motion=True)
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
        # ps, _, _, _ = env._sim_agent.get_link_states()
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
        print('Key[a]: auto_play:', not flag['auto_play'])
        flag['auto_play'] = not flag['auto_play']
    elif key == b'c' or key == b'C':
        if args.cam_file is not None:
            viewer.load_cam(args.cam_file)
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
                p, _, _, _ = env._sim_agent.get_root_state()
                base_env.idle_callback(allow_auto_play=False)
                viewer.drawGL()
                viewer.save_screen(dir=save_dir, name=name)
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
        viewer.save_cam(filename)
        print('Saved:', filename)
    elif key == b'm':
        if args.cam_file is not None:
            filename = args.cam_file
        else:
            filename = 'data/temp/temp.cam.gzip'
        viewer.load_cam(filename)
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
        flag['motion_capture'] = not flag['motion_capture']
        print('[MotionCapture]', flag['motion_capture'])
    elif key == b'M':
        recorder.save_file('mocap')
    elif key == b's':
        flag['screenshot'] = not flag['screenshot']
        print('Screenshot:', flag['screenshot'])
    elif key == b'S':
        name = 'screenshot_temp'
        viewer.save_screen(dir=SCREENSHOT_DIR, name=name)
    elif key == b'v':
        for i in range(pi.num_experts()):
            filename = 'data/learnable_experts/expert%d/network0'%(i)
            pi.copy_and_save_experts_orig(filename, i)
            print(filename)
    else:
        return base_env.keyboard_callback(key)
    return True

#For testing 
def make_env(seed, 
             dt_sim,
             dt_con,
             env_mode,
             ref_motion_files,
             verbose,
             sim_window,
             end_margin,
             pfnn_command_type,
             pfnn_command_record,
             pfnn_command_file,
             state_choices,
             state_imit_window,
             action_type,
             action_mode,
             action_range_min,
             action_range_max,
             action_range_min_pol,
             action_range_max_pol,
             reward_choices,
             rew_mode,
             env_noise,
             early_term_choices,
             et_low_reward_thres,
             et_falldown_contactable_body,
             char_info_module,
             ref_motion_scale,
             sim_char_file,
             base_motion_file,
             motion_graph_file,
             self_collision,
             ref_motion_sample,
             visualization,
             reward_weight_scale,
             num_cluster,
             cluster_id_assigned,
             ):
    rank = cmd_util.MPI.COMM_WORLD.Get_rank()
    cmd_util.set_global_seeds(seed + 10000 * rank)
    env = PPO_ENV(dt_sim=dt_sim, 
                  dt_con=dt_con, 
                  mode=env_mode,
                  verbose=verbose, 
                  sim_window=sim_window, 
                  end_margin=end_margin,
                  pfnn_command_type=pfnn_command_type, 
                  pfnn_command_record=pfnn_command_record, 
                  pfnn_command_file=pfnn_command_file, 
                  state_choices=state_choices,
                  state_imit_window=state_imit_window,
                  action_type=action_type,
                  action_mode=action_mode,
                  action_range_min=action_range_min,
                  action_range_max=action_range_max,
                  action_range_min_pol=action_range_min_pol,
                  action_range_max_pol=action_range_max_pol,
                  reward_choices=reward_choices,
                  reward_mode=rew_mode,
                  ref_motion_files=ref_motion_files,
                  add_noise=env_noise,
                  early_term_choices=early_term_choices,
                  et_low_reward_thres=et_low_reward_thres,
                  et_falldown_contactable_body=et_falldown_contactable_body,
                  char_info_module=char_info_module,
                  ref_motion_scale=ref_motion_scale,
                  sim_char_file=sim_char_file,
                  base_motion_file=base_motion_file,
                  motion_graph_file=motion_graph_file,
                  self_collision=self_collision,
                  ref_motion_sample=ref_motion_sample,
                  visualization=visualization,
                  reward_weight_scale=reward_weight_scale,
                  num_cluster=num_cluster,
                  cluster_id_assigned=cluster_id_assigned,
                  )

    env = cmd_util.Monitor(env, cmd_util.os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
    env.seed(seed)
    return env

def arg_parser():
    parser = cmd_util.arg_parser()
    parser.add_argument('--max_time_sec', help='Time limit for learning (min)',type=int, default=0)
    parser.add_argument('--max_timesteps', help='# of maxinum training tuples for learning', type=int, default=int(1e9))
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--mode', help='Training mode', choices=['train', 'load', 'retrain', 'test'], type=str, default=None)
    parser.add_argument('--env_mode', help='Env mode', choices=['mimicpfnn', 'deepmimic'], type=str, default='mimicpfnn')
    parser.add_argument('--rew_mode', help='Reward type', choices=['sum', 'mul'], type=str, default='sum')
    parser.add_argument('--net', help='Loading network #', type=int, default=None)
    parser.add_argument('--log', help='Logging method in learning', choices=['matplot', 'file', 'none'], type=str, default='file')
    parser.add_argument('--rec_period', help='Logging save period', type=int, default=10)
    parser.add_argument('--rec_dir', help='Logging directory', type=str, default='data/learning')
    parser.add_argument('--verbose', choices=['true', 'false'], default='false')
    parser.add_argument('--sim_window', help='Env time horizon', type=float, default=30.0)
    parser.add_argument('--end_margin', help='Allow a few steps after EOE', type=float, default=1.5)
    parser.add_argument('--gamma', help='Discount factor', type=float, default=0.95)
    parser.add_argument('--batch_size', help='Batch size for network update', type=int, default=256)
    parser.add_argument('--tuple_per_iter', help='# of tuples per learning iteration', type=int, default=4096*2)
    parser.add_argument('--pfnn_command_type', choices=['autonomous', 'joystick', 'recorded'], type=str, default='autonomous')
    parser.add_argument('--pfnn_command_record', help='Is Recording PFNN command?', choices=['true', 'false'], default='false')
    parser.add_argument('--pfnn_command_file', help='Recorded PFNN command', type=str, default=None)
    parser.add_argument('--state_choices', action='append', choices=['body', 'imit_abs', 'imit_rel'], help='States definitions')
    parser.add_argument('--state_imit_window', type=float, action='append')
    parser.add_argument('--action_type', choices=['none', 'spd', 'pd', 'cpd', 'cp', 'v'], type=str, default='spd')
    parser.add_argument('--action_mode', choices=['relative', 'absolute'], type=str, default='absolute')
    parser.add_argument('--action_range_min', type=float, default=-1.5)
    parser.add_argument('--action_range_max', type=float, default=1.5)
    parser.add_argument('--action_range_min_pol', type=float, default=-15.0)
    parser.add_argument('--action_range_max_pol', type=float, default=15.0)
    parser.add_argument('--reward_choices', action='append', choices=['pose', 'vel', 'ee', 'root', 'com'], help='States definitions')
    parser.add_argument('--ob_filter', help='Observation Filter', choices=['true', 'false'], default='true')
    parser.add_argument('--ref_motion_files', action='append', help='Reference Motion Files')
    parser.add_argument('--ref_motion_dir', type=str, default=None)
    parser.add_argument('--ref_motion_num', type=int, default=None)
    parser.add_argument('--ref_motion_divide', choices=['true', 'false'], default="false")
    parser.add_argument('--ref_motion_cluster_info', type=str, default=None)
    parser.add_argument('--ref_motion_cluster_id', action='append', type=int, default=None)
    parser.add_argument('--ref_motion_cluster_num_sample', action='append', type=int, default=None)
    parser.add_argument('--ref_motion_cluster_sample_method', type=str, default="random")
    parser.add_argument('--ref_motion_cluster_even_sample', type=str, default="false")
    parser.add_argument('--ref_motion_cluster_even_sample_cut', choices=['true', 'false'], default="false")
    parser.add_argument('--ref_motion_shuffle', choices=['true', 'false'], default="true")
    parser.add_argument('--ref_motion_sample', choices=['random', 'adaptive'], type=str, default="random")
    parser.add_argument('--sim_result_save_dir', type=str, default=None)
    parser.add_argument('--sim_result_save_num_per_cluster', type=int, default=None)
    parser.add_argument('--sim_result_save_success_only', choices=['true', 'false'], default="false")
    parser.add_argument('--policy_name', help='Name scope of policy function', type=str, default="mypolicy")
    parser.add_argument('--policy_type', choices=['mlp', 'additive', 'additive_ind', 'multiplicative', 'multiplicative2', 'valuecompsoite'], default='mlp')
    parser.add_argument('--num_hid_size', help='Width of hidden layers', type=int, default=512)
    parser.add_argument('--num_hid_layers', help='Depth of hidden layers', type=int, default=2)
    parser.add_argument('--num_hid_size_gate', help='Width of hidden layers', type=int, default=64)
    parser.add_argument('--num_hid_layers_gate', help='Depth of hidden layers', type=int, default=2)
    parser.add_argument('--old_expert_names', action='append', help='Name of Old Expert Policies')
    parser.add_argument('--old_expert_weights', action='append', help='Weights of Old Expert Policies')
    parser.add_argument('--new_expert_names', action='append', help='Name of New expert Policies')
    parser.add_argument('--w_new_expert_usage', help='PPO: Weight for new expert usage', type=float, default=0.0)
    parser.add_argument('--gate_expert_alter', help='PPO: Alternate learning of gate and experts', choices=['true', 'false'], default="false")
    parser.add_argument('--gate_expert_alter_gate_iter', help='PPO: Gate update iteration', type=int, default=10)
    parser.add_argument('--gate_expert_alter_expert_iter', help='PPO: Experts update iteration',type=int, default=40)
    parser.add_argument('--optim_stepsize_pol', help='PPO: Optimization stepsize of policy',type=float, default=1e-5)
    parser.add_argument('--optim_stepsize_val', help='PPO: Optimization stepsize of value function',type=float, default=1e-3)
    parser.add_argument('--ob_filter_gate', help='Obervation filter for gate', choices=['true', 'false'], default="true")
    parser.add_argument('--ob_filter_old_expert', help='Obervation filter for experts', choices=['true', 'false'], default="true")
    parser.add_argument('--ob_filter_new_expert', help='Obervation filter for beginner', choices=['true', 'false'], default="true")
    parser.add_argument('--ob_filter_update_for_expert', help='Obervation filter update for experts', choices=['true', 'false'], default="true")
    parser.add_argument('--trainable_gate', help='Make gate trainable', choices=['true', 'false'], default="true")
    parser.add_argument('--trainable_old_expert', help='Train Expert Network', choices=['true', 'false'], default="false")
    parser.add_argument('--trainable_new_expert', help='Train Beginner Network', choices=['true', 'false'], default="true")
    parser.add_argument('--env_noise', help='Noise', choices=['true', 'false'], default="true")
    parser.add_argument('--early_term_choices', action='append', choices=['task_complete', 'falldown', 'root_fail', 'low_reward'], help='Early termination conditions')
    parser.add_argument('--et_low_reward_thres', help='Threshold on early termination via rewards', type=float, default=0.1)
    parser.add_argument('--et_falldown_contactable_body', action='append', help='Contactable bodies for early termination via contact', default=[])
    parser.add_argument('--char_info_module', type=str, default="mimicpfnn_char_info.py")
    parser.add_argument('--ref_motion_scale', type=float, default=0.009)
    parser.add_argument('--sim_char_file', type=str, default="data/character/pfnn.urdf")
    parser.add_argument('--base_motion_file', type=str, default="data/motion/pfnn/pfnn_hierarchy_only.bvh")
    parser.add_argument('--motion_graph_file', type=str, default="data/motion/motiongraph/motion_graph_pfnn.gzip")
    parser.add_argument('--self_collision', choices=['true', 'false'], default="true")
    parser.add_argument('--reward_weight_scale', type=float, default=0.5)
    parser.add_argument('--render_window_w', type=int, default=1280)
    parser.add_argument('--render_window_h', type=int, default=720)
    parser.add_argument('--cam_pos', help='Viewer camera position', nargs='+', type=float, default=None)
    parser.add_argument('--cam_origin', help='Viewer camera origin (target)', nargs='+', type=float, default=None)
    parser.add_argument('--cam_fov', help='Viewer camera field of view', type=float, default=45.0)
    parser.add_argument('--cam_follow', help='Camera follows the character or not', type=str, default='true')
    parser.add_argument('--cam_file', help='Saved camera configuration', type=str, default=None)
    parser.add_argument('--screenshot_dir', type=str, default=None)
    parser.add_argument('--screenshot_start_time', type=float, default=None)
    parser.add_argument('--screenshot_end_time', type=float, default=None)
    parser.add_argument('--render_overlay', type=str, default='true')
    parser.add_argument('--render_overlay_expert_weight', type=str, default='true')
    parser.add_argument('--render_overlay_value_fn', type=str, default='true')
    parser.add_argument('--render_overlay_basic_info', type=str, default='true')
    parser.add_argument('--ground_w', type=float, default=30)
    parser.add_argument('--ground_h', type=float, default=30)
    parser.add_argument('--sim_fps', type=int, default=480)
    parser.add_argument('--con_fps', type=int, default=30)

    return parser

def train(args,
          visualization,
          num_cluster,
          cluster_id_assigned,
          ):
    U.make_session(num_cpu=1).__enter__()
    def policy_fn_mlp(name, ob_space, ac_space):
        return policy.MlpPolicy(
            name=name, independent=True,
            stochastic=None, ob=None, ob_space=ob_space, ac_space=ac_space, 
            hid_size=args.num_hid_size, num_hid_layers=args.num_hid_layers,
            ob_filter=basics.str2bool(args.ob_filter),
            )
    def policy_fn_valuecomposite(name, ob_space, ac_space):
        return policy.ValueCompositePolicy(
            name=name, independent=True,
            stochastic=None, ob=None, ob_space=ob_space, ac_space=ac_space, 
            hid_size_expert=args.num_hid_size, num_hid_layers_expert=args.num_hid_layers,
            hid_size_gate=args.num_hid_size_gate, num_hid_layers_gate=args.num_hid_layers_gate,
            old_expert_names=args.old_expert_names, 
            old_expert_weights=args.old_expert_weights, 
            new_expert_names=args.new_expert_names,
            ob_filter=basics.str2bool(args.ob_filter),
            )
    def policy_fn_additive(name, ob_space, ac_space):
        return policy.AdditivePolicy(
            name=name, independent=True,
            stochastic=None, ob=None, ob_space=ob_space, ac_space=ac_space, 
            hid_size_expert=args.num_hid_size, num_hid_layers_expert=args.num_hid_layers,
            hid_size_gate=args.num_hid_size_gate, num_hid_layers_gate=args.num_hid_layers_gate,
            old_expert_names=args.old_expert_names, 
            old_expert_weights=args.old_expert_weights, 
            new_expert_names=args.new_expert_names,
            ob_filter_gate=basics.str2bool(args.ob_filter_gate),
            ob_filter_old_expert=basics.str2bool(args.ob_filter_old_expert),
            ob_filter_new_expert=basics.str2bool(args.ob_filter_new_expert),
            trainable_gate=basics.str2bool(args.trainable_gate),
            trainable_old_expert=basics.str2bool(args.trainable_old_expert),
            trainable_new_expert=basics.str2bool(args.trainable_new_expert),
            )
    def policy_fn_additive_ind(name, ob_space, ac_space):
        return policy.AdditiveIndPolicy(
            name=name, independent=True,
            stochastic=None, ob=None, ob_space=ob_space, ac_space=ac_space, 
            hid_size_expert=args.num_hid_size, num_hid_layers_expert=args.num_hid_layers,
            hid_size_gate=args.num_hid_size_gate, num_hid_layers_gate=args.num_hid_layers_gate,
            old_expert_names=args.old_expert_names, 
            old_expert_weights=args.old_expert_weights, 
            new_expert_names=args.new_expert_names,
            ob_filter_gate=basics.str2bool(args.ob_filter_gate),
            ob_filter_old_expert=basics.str2bool(args.ob_filter_old_expert),
            ob_filter_new_expert=basics.str2bool(args.ob_filter_new_expert),
            trainable_gate=basics.str2bool(args.trainable_gate),
            trainable_old_expert=basics.str2bool(args.trainable_old_expert),
            trainable_new_expert=basics.str2bool(args.trainable_new_expert),
            )
    def policy_fn_multiplicative(name, ob_space, ac_space):
        return policy.MultiplicativePolicy(
            name=name, independent=True,
            stochastic=None, ob=None, ob_space=ob_space, ac_space=ac_space, 
            hid_size_expert=args.num_hid_size, num_hid_layers_expert=args.num_hid_layers,
            hid_size_gate=args.num_hid_size_gate, num_hid_layers_gate=args.num_hid_layers_gate,
            old_expert_names=args.old_expert_names, 
            old_expert_weights=args.old_expert_weights, 
            new_expert_names=args.new_expert_names,
            ob_filter_gate=basics.str2bool(args.ob_filter_gate),
            ob_filter_old_expert=basics.str2bool(args.ob_filter_old_expert),
            ob_filter_new_expert=basics.str2bool(args.ob_filter_new_expert),
            trainable_gate=basics.str2bool(args.trainable_gate),
            trainable_old_expert=basics.str2bool(args.trainable_old_expert),
            trainable_new_expert=basics.str2bool(args.trainable_new_expert),
            )
    def policy_fn_multiplicative2(name, ob_space, ac_space):
        return policy.MultiplicativePolicy2(
            name=name, independent=True,
            stochastic=None, ob=None, ob_space=ob_space, ac_space=ac_space, 
            hid_size_expert=args.num_hid_size, num_hid_layers_expert=args.num_hid_layers,
            hid_size_gate=args.num_hid_size_gate, num_hid_layers_gate=args.num_hid_layers_gate,
            old_expert_names=args.old_expert_names, 
            old_expert_weights=args.old_expert_weights, 
            new_expert_names=args.new_expert_names,
            ob_filter_gate=basics.str2bool(args.ob_filter_gate),
            ob_filter_old_expert=basics.str2bool(args.ob_filter_old_expert),
            ob_filter_new_expert=basics.str2bool(args.ob_filter_new_expert),
            trainable_gate=basics.str2bool(args.trainable_gate),
            trainable_old_expert=basics.str2bool(args.trainable_old_expert),
            trainable_new_expert=basics.str2bool(args.trainable_new_expert),
            )
    def value_marginal(name, ob_space):
        return policy.ValueMaginal(
            name=name, 
            ob_space=ob_space,
            hid_size=128, num_hid_layers=2,
            ob_filter=False)
    
    if args.policy_type == "mlp":
        policy_fn = policy_fn_mlp
    elif args.policy_type == "valuecomposite":
        policy_fn = policy_fn_valuecomposite
    elif args.policy_type == "additive":
        policy_fn = policy_fn_additive
    elif args.policy_type == "additive_ind":
        policy_fn = policy_fn_additive_ind
    elif args.policy_type == "multiplicative":
        policy_fn = policy_fn_multiplicative
    elif args.policy_type == "multiplicative2":
        policy_fn = policy_fn_multiplicative2
    else:
        raise NotImplementedError

    env = make_env(seed=args.seed, 
                   dt_sim=1.0/args.sim_fps,
                   dt_con=1.0/args.con_fps,
                   env_mode=args.env_mode,
                   ref_motion_files=args.ref_motion_files,
                   verbose=basics.str2bool(args.verbose),
                   sim_window=args.sim_window,
                   end_margin=args.end_margin,
                   pfnn_command_type=args.pfnn_command_type,
                   pfnn_command_record=args.pfnn_command_record,
                   pfnn_command_file=args.pfnn_command_file,
                   state_choices=args.state_choices,
                   state_imit_window=args.state_imit_window,
                   action_type=args.action_type,
                   action_mode=args.action_mode,
                   action_range_min=args.action_range_min,
                   action_range_max=args.action_range_max,
                   action_range_min_pol=args.action_range_min_pol,
                   action_range_max_pol=args.action_range_max_pol,
                   reward_choices=args.reward_choices,
                   rew_mode=args.rew_mode,
                   env_noise=basics.str2bool(args.env_noise),
                   early_term_choices=args.early_term_choices,
                   et_low_reward_thres=args.et_low_reward_thres,
                   et_falldown_contactable_body=args.et_falldown_contactable_body,
                   char_info_module=args.char_info_module,
                   ref_motion_scale=args.ref_motion_scale,
                   sim_char_file=args.sim_char_file,
                   base_motion_file=args.base_motion_file,
                   motion_graph_file=args.motion_graph_file,
                   self_collision=basics.str2bool(args.self_collision),
                   ref_motion_sample=args.ref_motion_sample,
                   visualization=visualization,
                   reward_weight_scale=args.reward_weight_scale,
                   num_cluster=num_cluster,
                   cluster_id_assigned=cluster_id_assigned,
                   )
    batch_per_env = max(args.batch_size, args.tuple_per_iter//MPI.COMM_WORLD.Get_size())
    pi = pposgd.learn(env, policy_fn, value_marginal,
                      max_seconds=args.max_time_sec,
                      max_timesteps=args.max_timesteps,
                      timesteps_per_actorbatch=batch_per_env,
                      clip_param=0.2, entcoeff=0.0, vfcoeff=1.0,
                      optim_epochs=10,
                      optim_stepsize_pol=args.optim_stepsize_pol,
                      optim_stepsize_val=args.optim_stepsize_val,
                      optim_stepsize_val_mar=1e-3,
                      optim_batchsize=args.batch_size,
                      gamma=args.gamma, lam=0.95, schedule='linear', adam_epsilon=1.0e-5,
                      mode=args.mode, 
                      network_number=args.net, 
                      network_number_vmar=None, 
                      log_learning_curve=args.log,
                      file_record_period=args.rec_period,
                      file_record_dir=args.rec_dir,
                      policy_name=args.policy_name,
                      w_new_expert_usage=args.w_new_expert_usage,
                      gate_expert_alter=basics.str2bool(args.gate_expert_alter),
                      gate_expert_alter_gate_iter=args.gate_expert_alter_gate_iter,
                      gate_expert_alter_expert_iter=args.gate_expert_alter_expert_iter,
                      ob_filter_update_for_expert=basics.str2bool(args.ob_filter_update_for_expert),
            )
    #env.close()
    return env, pi

def idle_callback():
    global time_checker_auto_play
    global pre_sim_time, sim_speed, sim_speed_avg

    env = base_env.env
    pi = base_env.policy
    
    time_elapsed = time_checker_auto_play.get_time(restart=False)
    if flag['auto_play'] and time_elapsed >= env._dt_con:
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

        sim_speed.append((env._elapsed_time - pre_sim_time) * viewer.avg_fps)
        sim_speed_avg = np.mean(sim_speed)
        pre_sim_time = env._elapsed_time

    base_env.idle_callback(allow_auto_play=False)

def render_callback():
    base_env.render_callback()

def overlay_callback():
    if not base_env.flag['overlay']:
        return
    global gamma, sim_speed_avg

    env = base_env.env
    pi = base_env.policy
    
    glPushAttrib(GL_LIGHTING)
    glDisable(GL_LIGHTING)
    
    w, h = viewer.window_size

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
    #     color += gl_render.COLOR_SEQUENCE
    #     line_width += [2.0] * num_experts
    
    # gl_render.render_graph_base_2D(origin, graph_axis_len, graph_pad_len)
    # gl_render.render_graph_data_line_2D(x_data=x,
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
    gl_render.render_text("FPS: %.2f"%viewer.avg_fps, pos=[0.05*w, 0.9*h], font=GLUT_BITMAP_9_BY_15)
    gl_render.render_text("Time: %.2f"%env._elapsed_time, pos=[0.05*w, 0.9*h+20], font=GLUT_BITMAP_9_BY_15)
    gl_render.render_text("Sim Speed: %.2f"%sim_speed_avg, pos=[0.05*w, 0.9*h+40], font=GLUT_BITMAP_9_BY_15)

    ''' PFNN Info '''
    if env._mode == env.Mode.MimicPFNN:
        runner = env._pfnn
        command = runner.command.get()
        scale = runner.command.scale

        pad = 15
        w_bar, h_bar = 120, 10
        phase_radius = 60
        joy_radius = 30

        origin = np.array([0.05*w, 0.05*h])
        pos = origin.copy()
        gl_render.render_progress_circle_2D(runner.character.phase/(2*math.pi), origin=(pos[0]+phase_radius,pos[1]+phase_radius), radius=phase_radius)
        gl_render.render_text("phase", pos=(pos[0]+0.5*phase_radius+pad,pos[1]+2*phase_radius+pad), font=GLUT_BITMAP_9_BY_15)
        pos += np.array([0.0, 2*phase_radius+2*pad])
        gl_render.render_direction_input_2D((-command['x_vel'], -command['y_vel']), (scale, scale), origin=(pos[0]+joy_radius,pos[1]+joy_radius), radius=joy_radius)
        gl_render.render_text("velociy", pos=(pos[0],pos[1]+2*joy_radius+pad), font=GLUT_BITMAP_9_BY_15)
        gl_render.render_direction_input_2D((-command['x_move'], command['y_move']), (scale, scale), origin=(pos[0]+3*joy_radius+pad,pos[1]+joy_radius), radius=joy_radius)
        gl_render.render_text("direction", pos=(pos[0]+2*joy_radius+pad,pos[1]+2*joy_radius+pad), font=GLUT_BITMAP_9_BY_15)
        pos += np.array([0.0, 2*joy_radius+2*pad])
        gl_render.render_progress_bar_2D_horizontal(command['trigger_speed']/scale, origin=pos, width=w_bar, height=h_bar)
        gl_render.render_text("trigger_speed", pos=(pos[0],pos[1]+h_bar+pad), font=GLUT_BITMAP_9_BY_15)
        pos += np.array([0.0, 2*h_bar+2*pad])
        gl_render.render_progress_bar_2D_horizontal(command['trigger_crouch']/scale, origin=pos, width=w_bar, height=h_bar)
        gl_render.render_text("trigger_crouch", pos=(pos[0],pos[1]+h_bar+pad), font=GLUT_BITMAP_9_BY_15)
        pos += np.array([0.0, 2*h_bar+2*pad])
        # gl_render.render_progress_bar_2D_horizontal(command['trigger_stop']/scale, origin=pos, width=w_bar, height=h_bar)
        # gl_render.render_text("trigger_stop", pos=(pos[0],pos[1]+h_bar+pad), font=GLUT_BITMAP_9_BY_15)
        # pos += np.array([0.0, 2*h_bar+2*pad])
        # gl_render.render_progress_bar_2D_horizontal(command['trigger_jump']/scale, origin=pos, width=w_bar, height=h_bar)
        # gl_render.render_text("trigger_jump", pos=(pos[0]+w_bar+pad,pos[1]+h_bar), font=GLUT_BITMAP_9_BY_15)
        # pos += np.array([0.0, h_bar+pad])

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
            color += COLORS_FOR_EXPERTS
            line_width += [2.0] * num_experts
        
        # gl_render.render_graph_base_2D(origin, graph_axis_len, graph_pad_len)
        # gl_render.render_graph_data_line_2D(x_data=x,
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
            gl_render.render_text(name, pos=pos-np.array([25, -0.8*h_bar]), font=GLUT_BITMAP_9_BY_15)
            gl_render.render_progress_bar_2D_horizontal(w_i, origin=pos, width=w_bar, height=h_bar, color_input=COLORS_FOR_EXPERTS[i])
            pos += np.array([0.0, -h_bar])
    elif isinstance(pi, policy.AdditiveIndPolicy):
        weight_cur = weight_experts_buffer[-1] if len(weight_experts_buffer) > 0 else None
        if weight_cur is None:
            weight_cur = np.zeros((pi.num_experts(), env.action_space.shape[0]))
        ww, hh = 150, 150
        origin = np.array([0.95*w-ww, 0.95*h-hh])
        gl_render.render_matrix(m=weight_cur, 
                                origin=origin,
                                width=ww,
                                height=hh)

    glPopAttrib()

    if flag['screenshot']:
        global cnt_screenshot
        name = 'screenshot_interactive_%04d'%(cnt_screenshot)
        viewer.save_screen(dir=SCREENSHOT_DIR, name=name)
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
    
    logger.configure()
    os.makedirs(args.rec_dir, exist_ok = True)

    '''
    Save all arguments used for this experiment to check them in later
    '''
    with open(args.rec_dir+"/config.txt", 'w') as file:
        for arg in vars(args):
            file.write("%s %s\n"%(arg,getattr(args, arg)))

    ''' 
    Unify random seed until environment launches
    After launching, each process will have different random seeds
    '''
    cmd_util.set_global_seeds(0)

    num_cluster_used = 0
    cluster_id_used = []
    cluster_id_assigned = -1
    
    if args.ref_motion_dir is not None:
        if args.ref_motion_num is not None:
            files = basics.files_in_dir(args.ref_motion_dir, 
                                        ext=".bvh", 
                                        sort=True, 
                                        sample_mode="sequential", 
                                        sample_num=args.ref_motion_num)
        else:
            files = basics.files_in_dir(args.ref_motion_dir, 
                                        ext=".bvh", 
                                        sort=True)
        
        if args.ref_motion_cluster_info is not None:
            cluster_info = []
            files_filtered = []
            files_cluster_id = []

            '''
            Read cluster information
            '''
            with open(args.ref_motion_cluster_info, 'r') as file:
                for line in file:
                    l = re.split('[\t|\n|,|:| ]+', line)
                    cluster_id, rank, score, filename = int(l[0]), int(l[1]), float(l[2]), str(l[3])
                    cluster_info.append((cluster_id, rank, score, filename))

            '''
            Compute a list of files involved in each cluster
            '''
            files_in_clusters = {}
            for cluster_id, rank, score, filename in cluster_info:
                if cluster_id not in files_in_clusters: files_in_clusters[cluster_id] = []
                files_in_clusters[cluster_id].append('%s/%s'%(args.ref_motion_dir,filename))
            num_cluster = len(files_in_clusters.keys())
            
            if MPI.COMM_WORLD.Get_rank()==0:
                print('-------Motions are sampled from-------')

            if basics.str2bool(args.ref_motion_cluster_even_sample):
                '''
                This samples the same number of motions for each cluster
                '''
                num_even_sample = int(len(cluster_info) / num_cluster)
                assert num_even_sample > 0
                for cluster_id, files in files_in_clusters.items():
                    if basics.str2bool(args.ref_motion_cluster_even_sample_cut):
                        if len(files) > num_even_sample:
                            files_filtered += random.choices(files, k=num_even_sample)
                            files_cluster_id += [cluster_id]*num_even_sample
                        else:
                            files_filtered += files
                            files_cluster_id += [cluster_id]*len(files)
                    else:
                        files_filtered += random.choices(files, k=num_even_sample)
                        files_cluster_id += [cluster_id]*num_even_sample
                    num_cluster_used += 1
                    cluster_id_used.append(cluster_id)
                    if MPI.COMM_WORLD.Get_rank()==0:
                        print("cluster %d, (%d / %d)"%(cluster_id, num_even_sample, len(files)))
            else:
                '''
                This samples motions for each cluster by following user-provided arguments
                '''
                assert args.ref_motion_cluster_id is not None
                assert args.ref_motion_cluster_num_sample is not None
                assert len(args.ref_motion_cluster_id) == len(args.ref_motion_cluster_num_sample)
                for i, cluster_id in enumerate(args.ref_motion_cluster_id):
                    files = files_in_clusters[cluster_id]
                    num_sample = args.ref_motion_cluster_num_sample[i]
                    if num_sample == 0:
                        continue
                    elif num_sample < 0:
                        files_filtered += files
                        files_cluster_id += [cluster_id]*len(files)
                    else:
                        if args.ref_motion_cluster_sample_method == "random":
                            files_filtered += random.choices(files, k=num_sample)
                        elif args.ref_motion_cluster_sample_method == "top":
                            files_filtered += files[:num_sample]
                        files_cluster_id += [cluster_id]*num_sample
                    num_cluster_used += 1
                    cluster_id_used.append(cluster_id)
                    if MPI.COMM_WORLD.Get_rank()==0:
                        print("cluster[%d], (%d / %d)"%(cluster_id, num_sample, len(files)))

            files = files_filtered
        
        assert len(files) > 0

        '''
        If True, change the order of reference motions randomly
        '''
        if basics.str2bool(args.ref_motion_shuffle):
            random.shuffle(files)

        '''
        If True, divide reference motions according to the number of available MPI processes
        '''
        if basics.str2bool(args.ref_motion_divide):
            idx_divided = mp.divide_jobs_idx(len(files), MPI.COMM_WORLD.Get_size())
            idx_assigned = idx_divided[MPI.COMM_WORLD.Get_rank()]
            files = files[idx_assigned[0]:idx_assigned[1]]

        '''
        This feature is experimental, updating experts and gate alternatively by cluster information
        '''
        if basics.str2bool(args.gate_expert_alter):
            assert args.ref_motion_cluster_info is not None
            assert not basics.str2bool(args.ref_motion_shuffle)
            assert basics.str2bool(args.ref_motion_divide)
            
            num_worker_for_each_cluster = MPI.COMM_WORLD.Get_size()//num_cluster_used
            
            assert num_worker_for_each_cluster > 0
            
            cluster_id_assigned = MPI.COMM_WORLD.Get_rank()//num_worker_for_each_cluster
            
            for i in files_cluster_id[idx_assigned[0]:idx_assigned[1]]:
                assert cluster_id_assigned==i
        
        args.ref_motion_files = files
        assert len(args.ref_motion_files) > 0

    '''
    If the mode is "load" or "test", we need an extra setup for visualization (e.g. OpenGL) 
    '''
    visualization = args.mode == "load" or args.mode == "test"

    gamma = args.gamma
    env, pi = train(args,
                    visualization=visualization,
                    num_cluster=num_cluster_used,
                    cluster_id_assigned=cluster_id_assigned,
                    )

    logger.log('================================================')
    logger.log('===========ScaDive: Training Fishished==========')
    logger.log('================================================')

    if visualization:
        '''
        If the environment is loaded for visualization, let the system use more resources
        '''
        os.environ['OPENBLAS_NUM_THREADS'] = str(mp.get_num_cpus())
        os.environ['MKL_NUM_THREADS'] = str(mp.get_num_cpus())
        
        '''
        Import rendering pakages only when it is used for the visualization
        '''
        from OpenGL.GL import *
        from OpenGL.GLU import *
        from OpenGL.GLUT import *

        from basecode.render import gl_render
        from basecode.bullet import bullet_render
        from basecode.render import glut_viewer as viewer
        from basecode.render import camera

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

        base_env.policy = pi
        base_env.env = env.env
        base_env.viewer = viewer
        base_env.init_viewer()

        recorder = SimpleRecorder()

        '''
        Read viewer settings from the arguments
        '''

        base_env.flag['follow_cam'] = basics.str2bool(args.cam_follow)
        base_env.flag['overlay'] = basics.str2bool(args.render_overlay)

        cam = None
        if args.cam_file is not None:
            with gzip.open(args.cam_file, "rb") as f:
                cam = pickle.load(f)
        else:
            if args.cam_origin is not None:
                base_env.flag['follow_cam'] = False
                cam_origin = basics.convert_list_to_nparray(args.cam_origin)
            else:
                cam_origin, _, _, _ = base_env.env._sim_agent.get_root_state()

            if np.allclose(base_env.env.char_info.v_up_env, np.array([0.0, 1.0, 0.0])):
                cam_vup = np.array([0.0, 1.0, 0.0])
                if args.cam_pos is not None:
                    cam_pos = basics.convert_list_to_nparray(args.cam_pos)
                else:
                    cam_pos = cam_origin + np.array([0.0, 2.0, -3.0])
            elif np.allclose(base_env.env.char_info.v_up_env, np.array([0.0, 0.0, 1.0])):
                cam_vup = np.array([0.0, 0.0, 1.0])
                if args.cam_pos is not None:
                    cam_pos = basics.convert_list_to_nparray(args.cam_pos)
                else:
                    cam_pos = cam_origin + np.array([-3.0, 0.0, 2.0])
            else:
                raise NotImplementedError

            cam = camera.Camera(pos=cam_pos,
                                origin=cam_origin, 
                                vup=cam_vup, 
                                fov=args.cam_fov)

        viewer.run(
            title=args.rec_dir,
            cam=cam,
            size=(args.render_window_w, args.render_window_h),
            keyboard_callback=keyboard_callback,
            render_callback=render_callback,
            overlay_callback=overlay_callback,
            idle_callback=idle_callback
            )
