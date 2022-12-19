import gym
from gym.spaces import Box, Discrete, Tuple

import env_multiagent as my_env

import copy
import numpy as np

import env_renderer as er

from basecode.utils import basics

import ray
from ray.rllib.env import MultiAgentEnv
import os

class HumanoidFight(MultiAgentEnv):
    def __init__(self, env_config):
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
                    high=ob_scale * np.ones(dim_state)))
            self.observation_space_body.append(
                Box(low=-ob_scale * np.ones(dim_state_body),
                    high=ob_scale * np.ones(dim_state_body)))
            self.observation_space_task.append(
                Box(low=-ob_scale * np.ones(dim_state_task),
                    high=ob_scale * np.ones(dim_state_task)))
            self.action_space.append(
                Box(low=action_range_min,
                    high=action_range_max))
        
        self.player1 = "player1"
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
        out_of_ring1 = self.base_env.check_out_of_ring(self.base_env._sim_agent[0])
        out_of_ring2 = self.base_env.check_out_of_ring(self.base_env._sim_agent[0])
        info = {
            self.player1: {
                "info": info_env[0],
                "success": 1.0 if out_of_ring2 else 0.0,
            },
            self.player2: {
                "info": info_env[1],
                "success": 1.0 if out_of_ring1 else 0.0
            }
        }
        # print(info_env)
        # print(action_dict)
        return obs, rew, done, info

def policy_mapping_fn(agent_id):
    if agent_id=="player1": 
        return "ppo_policy"
    elif agent_id=="player2": 
        return "ppo_policy"
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
        episode.custom_metrics["winning_rate"] = np.mean([winning_rate1, winning_rate2])

    def on_train_result(self, trainer, result: dict, **kwargs):
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_timesteps(result["timesteps_total"])))

class EnvRenderer(er.EnvRenderer):
    def __init__(self, trainer, **kwargs):
        self.trainer = trainer
        self.time_checker_auto_play = basics.TimeChecker()
        super().__init__(**kwargs)
    def one_step(self):
        s1 = self.env.state()
        a = {}
        for agent_id in s1.keys():
            a[agent_id] = self.trainer.compute_action(s1[agent_id], policy_id=policy_mapping_fn(agent_id))
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
            s = self.env.reset()
        elif key == b' ':
            self.time_checker_auto_play.begin()
            self.one_step()
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
    import render_module as rm
    return rm.camera.Camera(pos=np.array([12.0, 0.0, 12.0]),
                            origin=np.array([0.0, 0.0, 0.0]), 
                            vup=np.array([0.0, 0.0, 1.0]), 
                            fov=30.0)

env_cls = HumanoidFight

def config_override(spec):
    env = env_cls(spec["config"]["env_config"])
    policies = {
        "ppo_policy": (None, 
                       copy.deepcopy(env.observation_space[0]), 
                       copy.deepcopy(env.action_space[0]), 
                       {}),
    }
    del env

    config = {
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["ppo_policy"],
        },
        "callbacks": MyCallbacks,
    }

    return config
