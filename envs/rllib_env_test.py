import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(directory))

import copy
import numpy as np

import ray
from ray.rllib.env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec

import gym
from gym.spaces import Box

from envs import env_humanoid_tag as my_env
import env_renderer as er
import render_module as rm

import rllib_model_torch as policy_model
from collections import deque

from fairmotion.core.motion import Motion
from fairmotion.data import bvh
import pickle

def get_bool_from_input(question):
    answer = input('%s [y/n]?:'%question)
    if answer == 'y' or answer == 'yes':
        answer = True
    elif answer == 'n' or answer == 'no':
        answer = False
    else:
        raise Exception('Please enter [y/n]!')
    return answer

def get_int_from_input(question):
    answer = input('%s [int]?:'%question)
    try:
       answer = int(answer)
    except ValueError:
       print("That's not an integer!")
       return
    return answer

def get_float_from_input(question):
    answer = input('%s [float]?:'%question)
    try:
       answer = float(answer)
    except ValueError:
       print("That's not a float number!")
       return
    return answer

class MultiAgentTestEnv(MultiAgentEnv):
    def __init__(self, env_config):
        # self.base_env = my_env.Env(env_config)
        # assert self.base_env._num_agent == 2
        self.env_config = env_config
        self.observation_space = []
        self.observation_space_body = []
        self.observation_space_task = []
        self.action_space = []

        self.dim_state = [500, 500]

        ob_scale = np.array(env_config.get('ob_scale', 1000.0))
        ac_scale = np.array(env_config.get('ac_scale', 3.0))
        
        for i in range(2):
            dim_state = self.dim_state[i]
            dim_state_body = 361
            dim_state_task = self.dim_state[i] - dim_state_body
            dim_action = 54
            self.observation_space.append(
                Box(low=-ob_scale * np.ones(dim_state),
                    high=ob_scale * np.ones(dim_state),
                    dtype=np.float64)
                )
            self.observation_space_body.append(
                Box(low=-ob_scale * np.ones(dim_state_body),
                    high=ob_scale * np.ones(dim_state_body),
                    dtype=np.float64)
                )
            self.observation_space_task.append(
                Box(low=-ob_scale * np.ones(dim_state_task),
                    high=ob_scale * np.ones(dim_state_task),
                    dtype=np.float64)
                )
            self.action_space.append(
                Box(low=-ac_scale*np.ones(dim_action),
                    high=ac_scale*np.ones(dim_action),
                    dtype=np.float64)
                )

        name = env_config.get('character').get('name')
        self.player1 = name[0]
        self.player2 = name[1]

    def state(self):
        return {
            self.player1: np.random.uniform(size=self.dim_state[0]),
            self.player2: np.random.uniform(size=self.dim_state[1]),
        }

    def reset(self, info={}):
        return {
            self.player1: np.random.uniform(size=self.dim_state[0]),
            self.player2: np.random.uniform(size=self.dim_state[1]),
        }

    def step(self, action_dict):
        action1 = action_dict[self.player1]
        action2 = action_dict[self.player2]

        # base_action_dict = {
        #     'target_pose': [action1, action2],
        # }
        
        # r, info_env = self.base_env.step(base_action_dict)

        obs = self.state()
        rew = {
            self.player1: np.random.uniform(0,1),
            self.player2: np.random.uniform(0,1),
        }
        done = {
            "__all__": False,
        }

        gameover = False

        info = {
            self.player1: {"gameover": gameover},
            self.player2: {"gameover": gameover},
        }

        return obs, rew, done, info

'''
TODO: DDPPO fails when the agents have different state/action dim,
might be related to optimizer?
>> MLP + DDPPO + Same State Dim -> Working
>> MLP + DDPPO + Diff State Dim -> Not Working (Killed Immediately)
>> MLP + DDPPO + Diff State Dim + Simple Optimizer -> Not Working
>> LSTM + DDPPO + Same State Dim -> Not Working (Not Responding)
>> LSTM + DDPPO + Diff State Dim -> Not Working (Killed Immediately)
>> LSTM + DDPPO + Same State Dim + No Shuffle -> Not Working
'''

# class HumanoidTag(MultiAgentEnv):
#     def __init__(self, env_config):
#         self.base_env = my_env.Env(env_config)
#         assert self.base_env._num_agent == 2
#         self.env_config = env_config
#         self.observation_space = []
#         self.observation_space_body = []
#         self.observation_space_task = []
#         self.action_space = []

#         ob_scale = np.array(env_config.get('ob_scale', 1000.0))
#         ac_scale = np.array(env_config.get('ac_scale', 3.0))
        
#         for i in range(self.base_env._num_agent):
#             dim_state = self.base_env.dim_state(i)
#             dim_state_body = self.base_env.dim_state_body(i)
#             dim_state_task = self.base_env.dim_state_task(i)
#             dim_action = self.base_env.dim_action(i)
#             self.observation_space.append(
#                 Box(low=-ob_scale * np.ones(dim_state),
#                     high=ob_scale * np.ones(dim_state),
#                     dtype=np.float64)
#                 )
#             self.observation_space_body.append(
#                 Box(low=-ob_scale * np.ones(dim_state_body),
#                     high=ob_scale * np.ones(dim_state_body),
#                     dtype=np.float64)
#                 )
#             self.observation_space_task.append(
#                 Box(low=-ob_scale * np.ones(dim_state_task),
#                     high=ob_scale * np.ones(dim_state_task),
#                     dtype=np.float64)
#                 )
#             self.action_space.append(
#                 Box(low=-ac_scale*np.ones(dim_action),
#                     high=ac_scale*np.ones(dim_action),
#                     dtype=np.float64)
#                 )

#         name = env_config.get('character').get('name')
#         self.player1 = name[0]
#         self.player2 = name[1]

#     def state(self):
#         return {
#             self.player1: self.base_env.state(idx=0),
#             self.player2: self.base_env.state(idx=1)
#         }

#     def reset(self, info={}):
#         if not self.base_env._initialized:
#             self.base_env.create()
#         self.base_env.reset(info)
#         return {
#             self.player1: self.base_env.state(idx=0),
#             self.player2: self.base_env.state(idx=1)
#         }

#     def step(self, action_dict):
#         action1 = action_dict[self.player1]
#         action2 = action_dict[self.player2]

#         base_action_dict = {
#             'target_pose': [action1, action2],
#         }
        
#         r, info_env = self.base_env.step(base_action_dict)

#         obs = self.state()
#         rew = {
#             self.player1: r[0],
#             self.player2: r[1]
#         }
#         done = {
#             "__all__": self.base_env._end_of_episode,
#         }

#         gameover = 'gameover' in info_env[0]['eoe_reason']

#         info = {
#             self.player1: {"gameover": gameover},
#             self.player2: {"gameover": gameover},
#         }

#         if self.base_env._verbose:
#             self.base_env.pretty_print_rew_info(info_env[0]['rew_info'])
#             self.base_env.pretty_print_rew_info(info_env[1]['rew_info'])
#             print(info)

#         return obs, rew, done, info

def policy_mapping_fn(share_policy, agent_id):
    if share_policy:
        return "policy"
    else:
        if agent_id=="player1": 
            return "policy_1"
        elif agent_id=="player2": 
            return "policy_2"
        else:
            raise NotImplementedError

def policies_to_train(share_policy):
    if share_policy:
        return ["policy"]
    else:
        return ["policy_1", "policy_2"]

env_cls = MultiAgentTestEnv

def config_override(spec):
    env = env_cls(spec["config"]["env_config"])

    # model_config = copy.deepcopy(spec["config"]["model"])
    # model = model_config.get("custom_model")
    # if model and model == "task_agnostic_policy_type1":
    #     model_config.get("custom_model_config").update({
    #         "observation_space_body": copy.deepcopy(env.observation_space_body[0]),
    #         "observation_space_task": copy.deepcopy(env.observation_space_task[0]),
    #     })

    share_policy = \
        spec["config"]["env_config"].get("share_policy", False)

    callbacks = MyCallbacksBase

    def gen_policy(i):
        model_config = copy.deepcopy(spec["config"]["model"])
        model = model_config.get("custom_model")
        if model and model == "task_agnostic_policy_type1":
            model_config.get("custom_model_config").update({
                "observation_space_body": copy.deepcopy(env.observation_space_body[i]),
                "observation_space_task": copy.deepcopy(env.observation_space_task[i]),
            })
        return PolicySpec(
            observation_space=copy.deepcopy(env.observation_space[i]),
            action_space=copy.deepcopy(env.action_space[i]),
            config={"model": model_config},
        )
    if share_policy:
        policies = {
            "policy": gen_policy(0),
        }       
    else:
        policies = {
            "policy_1": gen_policy(0),
            "policy_2": gen_policy(1),
        }

    del env

    config = {
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": \
                lambda agent_id: policy_mapping_fn(
                    share_policy=share_policy, agent_id=agent_id),
                # lambda agent_id, episode, worker, **kwargs: policy_mapping_fn(
                #     share_policy=share_policy, agent_id=agent_id),
            "policies_to_train": policies_to_train(share_policy),
        },
        "callbacks": callbacks,
        # "model": model_config,
    }

    return config

from typing import Dict
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks

class MyCallbacksBase(DefaultCallbacks):
    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        info = episode.last_info_for("player1")
        episode.custom_metrics["gameover"] = 1.0 if info["gameover"] else 0.0


class EnvRenderer(er.EnvRenderer):
    def __init__(self, trainers, config, **kwargs):
        # kwargs['renderer'] = 'bullet_native'
        from fairmotion.viz.utils import TimeChecker
        super().__init__(**kwargs)
        self.trainers = trainers
        self.config = config
        self.time_checker_auto_play = TimeChecker()
        self.explore = False
        self.share_policy = config["env_config"].get("share_policy", False)
        self.bgcolor=[1.0, 1.0, 1.0, 1.0]
        self.latent_random_sample = 0
        self.latent_random_sample_methods = [None, "gaussian", "uniform", "softmax", "hardmax"]
        self.cam_params = deque(maxlen=30)
        self.cam_param_offset = None
        self.options["ground_color"] = np.array([220, 140, 80.0])/255.0
        self.replay = False
        self.replay_cnt = 0
        self.data = {}
        self.reset()
    def use_default_ground(self):
        return self.env.base_env._base_env._use_default_ground
    def get_v_up_env_str(self):
        return self.env.base_env._v_up_str
    def get_pb_client(self):
        return self.env.base_env._pb_client
    def get_ground(self):
        return self.env.base_env._ground
    def get_trainer(self, idx):
        if len(self.trainers) == 1:
            return self.trainers[0]
        elif len(self.trainers) == 2:
            if idx==0:
                return self.trainers[0]
            elif idx==1:
                return self.trainers[1]
            else:
                raise Exception
        else:
            raise NotImplementedError
    def reset(self, info={}):
        if self.replay:
            self.replay_cnt = 0
            for i, agent_id in enumerate(self.agent_ids):
                if self.data[agent_id]['motion'].num_frames() == 0: continue
                motion = self.data[agent_id]['motion']
                pose = motion.get_pose_by_frame(self.replay_cnt)
                self.env.base_env._sim_agent[i].set_pose(pose)
        else:
            s = self.env.reset(info)
            self.policy_hidden_state = []
            self.agent_ids = list(s.keys())
            for i, agent_id in enumerate(self.agent_ids):
                trainer = self.get_trainer(i)
                state = trainer.get_policy(
                    policy_mapping_fn(
                        self.share_policy, agent_id)).get_initial_state()
                self.policy_hidden_state.append(state)
                motion = Motion(
                    skel=self.env.base_env._base_motion[i].skel,
                    fps=self.env.base_env._base_motion[i].fps,
                )
                self.data[agent_id] = {
                    'motion': motion,
                }
        self.cam_params.clear()
        param = self._get_cam_parameters()
        for i in range(self.cam_params.maxlen):
            self.cam_params.append(param)
    def set_pose(self):
        for i, agent_id in enumerate(self.agent_ids):
            motion = self.data[agent_id]['motion']
            pose = motion.get_pose_by_frame(self.replay_cnt)
            self.env.base_env._sim_agent[i].set_pose(pose)
    def collect_data(self, idx):
        agent_id = self.agent_ids[idx]
        sim_agent = self.env.base_env._sim_agent[idx]
        motion = self.data[agent_id]['motion']
        motion.add_one_frame(sim_agent.get_pose(motion.skel).data)
    def one_step(self):
        if self.replay:
            self.set_pose()
            self.replay_cnt = min(
                self.data[self.agent_ids[0]]['motion'].num_frames()-1,
                self.replay_cnt+1)
        else:
            s = self.env.state()
            a = {}
            for i, agent_id in enumerate(self.agent_ids):
                trainer = self.get_trainer(i)
                policy = trainer.get_policy(
                    policy_mapping_fn(self.share_policy, agent_id))
                if self.latent_random_sample:
                   action = np.random.normal(size=self.env.action_space[0].shape[0])
                   # print(action)
                else:
                    if policy.is_recurrent():
                        action, state_out, extra_fetches = trainer.compute_action(
                            s[agent_id], 
                            state=self.policy_hidden_state[i],
                            policy_id=policy_mapping_fn(self.share_policy, agent_id),
                            explore=self.explore)
                        self.policy_hidden_state[i] = state_out
                    else:
                        action = trainer.compute_action(
                            s[agent_id], 
                            policy_id=policy_mapping_fn(self.share_policy, agent_id),
                            explore=self.explore)
                a[agent_id] = action
                ''' Collect data for rendering or etc. '''
                self.collect_data(i)
            self.env.step(a)
    def extra_render_callback(self):
        self.env.base_env.render(self.rm)
    def extra_overlay_callback(self):
        w, h = self.window_size
        font = self.rm.glut.GLUT_BITMAP_9_BY_15
        h_start = h-50
        self.rm.gl_render.render_text(
            "Time: %.2f"%(self.env.base_env.get_elapsed_time()), pos=[0.05*w, h_start+20], font=font)
    def extra_idle_callback(self):
        time_elapsed = self.time_checker_auto_play.get_time(restart=False)
        if self.rm.flag['auto_play'] and time_elapsed >= self.env.base_env._dt_con:
            self.time_checker_auto_play.begin()
            self.one_step()
    def extra_keyboard_callback(self, key):
        if key == b'r':
            self.replay = False
            self.reset({'start_time': [0.0, 0.0]})
        elif key == b'R':
            self.replay = True
            self.reset()
        elif key == b' ':
            self.time_checker_auto_play.begin()
            self.one_step()
        elif key == b'e':
            self.explore = not self.explore
            print('Exploration:', self.explore)
        elif key == b's':
            save_image = get_bool_from_input("Save image")
            save_motion = get_bool_from_input("Save motion")
            save_dir = None
            if save_image or save_motion:
                ''' Read a directory for saving images and try to create it '''
                save_dir = input("Enter directory for saving: ")
                try:
                    os.makedirs(save_dir, exist_ok=True)
                except OSError:
                    print("Invalid Subdirectory")
                    return
            print('Recording the Current Scene...')
            ''' Read maximum end time '''
            end_time = get_float_from_input("Enter max end-time (sec)")
            ''' Read number of iteration '''
            num_iter = get_int_from_input("Enter num iteration")
            ''' Start each episode at zero '''
            reset_at_zero = get_bool_from_input("Always reset at 0s")
            ''' Read falldown check '''
            check_falldown = get_bool_from_input("Terminate when falldown")
            ''' Read end_of_motion check '''
            check_end_of_motion = get_bool_from_input("Terminate when reaching the end of motion")
            ''' Read end_of_motion check '''
            check_end_of_episode = get_bool_from_input("Terminate when end of episode")
            for i in range(num_iter):
                if reset_at_zero:
                    self.reset({'start_time': np.array([0.0])})
                else:
                    self.reset()
                if save_dir:
                    save_dir_i = os.path.join(save_dir, str(i))
                else:
                    save_dir_i = None
                time_elapsed = self.record_a_scene(
                    save_dir=save_dir_i, 
                    save_image=save_image,
                    save_motion=save_motion,
                    end_time=end_time, 
                    check_falldown=check_falldown, 
                    check_end_of_motion=check_end_of_motion,
                    check_end_of_episode=check_end_of_episode)
            print('Done.')
        elif key == b'c':
            agent = self.env.base_env._sim_agent[0]
            h = self.env.base_env.get_ground_height(0)
            d_face, p_face = agent.get_facing_direction_position(h)
            origin = p_face + agent._char_info.v_up_env
            pos = p_face + 4 * (agent._char_info.v_up_env - d_face)
            R_face, _ = conversions.T2Rp(agent.get_facing_transform(h))
            R_face_inv = R_face.transpose()
            origin_offset = np.dot(R_face_inv, self.cam_cur.origin - origin)
            pos_offset = np.dot(R_face_inv, self.cam_cur.pos - pos)
            self.cam_param_offset = (origin_offset, pos_offset)
            print("Set camera offset:", self.cam_param_offset)
        elif key == b'C':
            self.cam_param_offset = None
            print("Clear camera offset")
    def _get_cam_parameters(self):
        p1 = self.env.base_env._sim_agent[0].get_root_position()
        p2 = self.env.base_env._sim_agent[1].get_root_position()
        dist = np.linalg.norm(p1-p2)
        target_pos = 0.5 * (p1 + p2)
        cam_dist = np.clip(3.0 * dist, 5.0, 10.0)
        param = {
            "origin": None, 
            "pos": None, 
            "dist": cam_dist,
            "translate": {"target_pos": target_pos},
        }
        return param
    def get_cam_parameters(self, use_buffer=True):
        if use_buffer:
            param = {
                "origin": None, 
                "pos": None, 
                "dist": None,
                "translate": None,
            }
            param["dist"] = np.mean([p["dist"] for p in self.cam_params])
            param["translate"] = {
                "target_pos": np.mean([p["translate"]["target_pos"] for p in self.cam_params], axis=0)
            }
        else:
            param = self._get_cam_parameters()
        return param
    def get_elapsed_time(self):
        return self.env.base_env.get_elapsed_time()
    def record_a_scene(
        self,
        save_dir, 
        save_image,
        save_motion,
        end_time=None, 
        check_falldown=True, 
        check_end_of_motion=True,
        check_end_of_episode=True,
        verbose=True):
        if save_image or save_motion:
            assert save_dir is not None
            try:
                os.makedirs(save_dir, exist_ok = True)
            except OSError:
                print("Invalid Subdirectory")
                return
        if end_time is None or end_time <= 0.0:
            assert check_falldown or check_end_of_motion
        self.update_cam()
        cnt_screenshot = 0
        time_elapsed = 0
        if save_motion:
            motion = copy.deepcopy(self.env.base_env._base_motion[0])
            motion.clear()
        while True:
            self.one_step()
            if save_motion:
                motion.add_one_frame(
                    self.env.base_env._sim_agent[0].get_pose_data(motion.skel))
            if save_image:
                name = 'screenshot_%04d'%(cnt_screenshot)
                self.save_screen(dir=save_dir, name=name, render=True)
                if verbose:
                    print('\rsave_screen(%4.4f) / %s' % \
                        (time_elapsed, os.path.join(save_dir,name)), end=" ")
                cnt_screenshot += 1
            else:
                if verbose:
                    print('\r%4.4f' % (time_elapsed), end=" ")
            time_elapsed += self.env.base_env._dt_con
            agent_name = self.env.base_env._sim_agent[0].get_name()
            if check_falldown:
                check = False
                for i in range(self.env.base_end._num_agent):
                    if self.env.base_env.check_falldown(i):
                        check = True
                if check: break
            if check_end_of_motion:
                check = False
                for i in range(self.env.base_end._num_agent):
                    if self.env.base_env.check_end_of_motion(0):
                        check = True
                if check: break
            if check_end_of_episode:
                if self.env.base_env._end_of_episode:
                    break
            if end_time and time_elapsed > end_time:
                break
        if save_motion:
            bvh.save(
                motion, 
                os.path.join(save_dir, "motion.bvh"),
                scale=1.0, rot_order="XYZ", verbose=False)
        if verbose:
            print(" ")
        return time_elapsed

def default_cam(env):
    agent = env.base_env._sim_agent[0]
    v_up_env = agent._char_info.v_up_env
    v_up = agent._char_info.v_up
    v_face = agent._char_info.v_face
    return rm.camera.Camera(
        pos=2*(0.5*v_up-v_face),
        origin=np.array([0.0, 0.0, 0.0]), 
        vup=v_up_env, 
        fov=60.0)

