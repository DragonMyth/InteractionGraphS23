import gym
from gym.spaces import Box, Discrete, Tuple

import env_multiagent as my_env

import copy
import numpy as np

import env_renderer as er
import render_module as rm

from basecode.utils import basics

import ray
from ray.rllib.env import MultiAgentEnv
import os

class HumanoidChase(MultiAgentEnv):
    def __init__(self, env_config):
        if env_config["action"].get("use_pose_embedding"):
            option = env_config["action"].get("use_pose_embedding")
            if option == "option1":
                self.base_env = my_env.EnvWithPoseEmbedding1(env_config)
            elif option == "option2":
                self.base_env = my_env.EnvWithPoseEmbedding2(env_config)
            elif option == "option3":
                self.base_env = my_env.EnvWithPoseEmbedding3(env_config)
            else:
                raise NotImplementedError
        else:    
            self.base_env = my_env.Env(env_config)
        assert self.base_env._num_agent == 2
        
        self.observation_space = []
        self.observation_space_body = []
        self.observation_space_task = []
        self.action_space = []
        
        for i in range(self.base_env._num_agent):
            ob_scale = 1000.0
            dim_state = self.base_env.dim_state(i)
            dim_state_body = self.base_env.dim_state_body(i)
            dim_state_task = self.base_env.dim_state_task(i)
            dim_action = self.base_env.dim_action(i)
            action_range_min, action_range_max = self.base_env.action_range(i)
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
                Box(low=action_range_min,
                    high=action_range_max,
                    dtype=np.float64)
                )

        # hunter
        self.player1 = "player1"
        # prey
        self.player2 = "player2"

        self.set_timesteps(0)

    def set_timesteps(self, timesteps_total):
        self.base_env.update_learning_info({"timesteps_total": timesteps_total})

    def state(self):
        return {
            self.player1: self.base_env.state(idx=0),
            self.player2: self.base_env.state(idx=1)
        }

    def reset(self, start_time=None, add_noise=None):
        self.base_env.reset(start_time, add_noise)
        return {
            self.player1: self.base_env.state(idx=0),
            self.player2: self.base_env.state(idx=1)
        }

    def step(self, action_dict):
        action1 = action_dict[self.player1]
        action2 = action_dict[self.player2]
        
        r, info_env = self.base_env.step([action1, action2])

        obs = self.state()
        rew = {
            self.player1: r[0],
            self.player2: r[1]
        }
        done = {
            "__all__": self.base_env.end_of_episode,
        }
        touched = 'touched' in self.base_env.end_of_episode_reason
        info = {
            self.player1: {
                "info": info_env[0],
                "success": 1.0 if touched else 0.0,
            },
            self.player2: {
                "info": info_env[1],
                "success": 0.0 if touched else 1.0,
            }
        }

        # self.base_env.pretty_print_rew_detail(info_env[0]['rew_detail'])
        # self.base_env.pretty_print_rew_detail(info_env[1]['rew_detail'])
        # print(r, info_env)
        # print(action_dict)
        return obs, rew, done, info

def policy_mapping_fn(agent_id):
    if agent_id=="player1": 
        return "ppo_policy_player1"
    elif agent_id=="player2": 
        return "ppo_policy_player2"
    else:
        raise Exception(agent_id)

from typing import Dict
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks

class MyCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        winning_rate1 = episode.last_info_for("player1")["success"]
        winning_rate2 = episode.last_info_for("player2")["success"]
        episode.custom_metrics["winning_rate_player1"] = winning_rate1
        episode.custom_metrics["winning_rate_player2"] = winning_rate2
    # def on_train_result(self, trainer, result: dict, **kwargs):
    #     trainer.workers.foreach_worker(
    #         lambda ev: ev.foreach_env(
    #             lambda env: env.set_timesteps(result["timesteps_total"])))

class EnvRenderer(er.EnvRenderer):
    def __init__(self, trainer, **kwargs):
        super().__init__(**kwargs)
        self.trainer = trainer
        self.time_checker_auto_play = basics.TimeChecker()
        self.explore = False
        self.reset()
    def reset(self):
        s = self.env.reset()
        self.policy_hidden_state = []
        for agent_id in s.keys():
            state = self.trainer.get_policy(
                policy_mapping_fn(agent_id)).get_initial_state()
            self.policy_hidden_state.append(state)
    def one_step(self):
        s = self.env.state()
        a = {}
        for i, agent_id in enumerate(s.keys()):
            policy = self.trainer.get_policy(policy_mapping_fn(agent_id))
            if policy.is_recurrent():
                action, state_out, extra_fetches = self.trainer.compute_action(
                    s[agent_id], 
                    state=self.policy_hidden_state[i],
                    policy_id=policy_mapping_fn(agent_id),
                    explore=self.explore)
                self.policy_hidden_state[i] = state_out
            else:
                action = self.trainer.compute_action(
                    s[agent_id], 
                    policy_id=policy_mapping_fn(agent_id),
                    explore=self.explore)
            a[agent_id] = action
        return self.env.step(a)
    def render_callback(self):
        self.env.base_env.render(self.rm.flag, self.rm.COLORS_FOR_AGENTS)
    def idle_callback(self):
        time_elapsed = self.time_checker_auto_play.get_time(restart=False)
        if self.rm.flag['auto_play'] and time_elapsed >= self.env.base_env._dt_con:
            self.time_checker_auto_play.begin()
            self.one_step()
    def keyboard_callback(self, key):
        if key == b'r':
            self.reset()
        elif key == b' ':
            self.time_checker_auto_play.begin()
            self.one_step()
        elif key == b'e':
            self.explore = not self.explore
            print('Exploration:', self.explore)
        elif key == b'c':
            ''' Read a directory for saving images and try to create it '''
            subdir = input("Enter subdirectory for screenshot file: ")
            dir = os.path.join("data/screenshot/", subdir)
            try:
                os.makedirs(dir, exist_ok = True)
            except OSError:
                print("Invalid Subdirectory")
                return
            ''' Read maximum end time '''
            end_time = input("Enter max end-time (s): ")
            try:
               end_time = float(end_time)
            except ValueError:
               print("That's not a number!")
               return
            ''' Read number of iteration '''
            num_iter = input("Enter num iteration (int): ")
            try:
               num_iter = int(num_iter)
            except ValueError:
               print("That's not a number!")
               return
            for i in range(num_iter):
                dir_i = os.path.join(dir, str(i))
                try:
                    os.makedirs(dir_i, exist_ok = True)
                except OSError:
                    print("Invalid Subdirectory")
                    continue
                cnt_screenshot = 0
                self.env.reset()
                while True:
                    name = 'screenshot_%04d'%(cnt_screenshot)
                    self.one_step()
                    self.render()
                    self.save_screen(dir=dir_i, name=name)
                    print('\rsave_screen(%4.4f) / %s' % \
                        (self.env.base_env._elapsed_time, os.path.join(dir_i,name)), end=" ")
                    cnt_screenshot += 1
                    if self.env.base_env.end_of_episode or \
                       self.env.base_env._elapsed_time > end_time:
                        break
                print("\n")

def default_cam():
    return rm.camera.Camera(pos=np.array([12.0, 0.0, 12.0]),
                            origin=np.array([0.0, 0.0, 0.0]), 
                            vup=np.array([0.0, 0.0, 1.0]), 
                            fov=30.0)

env_cls = HumanoidChase

def config_override(spec):
    env = env_cls(spec["config"]["env_config"])
    policies = {
        "ppo_policy_player1": (None, 
                               copy.deepcopy(env.observation_space[0]), 
                               copy.deepcopy(env.action_space[0]), 
                               {}),
        "ppo_policy_player2": (None, 
                               copy.deepcopy(env.observation_space[1]), 
                               copy.deepcopy(env.action_space[1]), 
                               {}),
    }

    model_config = copy.deepcopy(spec["config"]["model"])
    model = model_config.get("custom_model")
    if model and model == "task_agnostic_policy_type1":
        model_config.get("custom_options").update({
            "observation_space_body": copy.deepcopy(env.observation_space_body[0]),
            "observation_space_task": copy.deepcopy(env.observation_space_task[0]),
        })
    
    del env

    config = {
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["ppo_policy_player1", "ppo_policy_player2"],
        },
        "callbacks": MyCallbacks,
        "model": model_config,
    }

    return config

def config_override_custom(spec):
    env = env_cls(spec["config"]["env_config"])
    policies = {
        "ppo_policy_player1": (None, 
                               copy.deepcopy(env.observation_space[0]), 
                               copy.deepcopy(env.action_space[0]), 
                               {}),
        "ppo_policy_player2": (None, 
                               copy.deepcopy(env.observation_space[1]), 
                               copy.deepcopy(env.action_space[1]), 
                               {}),
    }
    del env

    config1 = {
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["ppo_policy_player1"],
        },
        "callbacks": MyCallbacks,
    }

    config2 = {
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["ppo_policy_player2"],
        },
        "callbacks": MyCallbacks,
    }

    return config1, config2
