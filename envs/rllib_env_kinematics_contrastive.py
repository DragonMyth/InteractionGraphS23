import copy
import numpy as np
import argparse
import random

import gym
from gym.spaces import Box

import env_humanoid_kinematics_contrastive as my_env
import env_renderer as er
import render_module as rm

import os
import pickle
import gzip

from collections import deque

class HumanoidKinematicsContrastive(gym.Env):
    def __init__(self, env_config):
        self.base_env = my_env.Env(env_config)
        assert self.base_env._num_agent == 1
        
        ob_scale = 1000.0
        dim_state = self.base_env.dim_state(0)
        dim_action = self.base_env.dim_action(0)
        action_range_min, action_range_max = self.base_env.action_range(0)
        self.observation_space = \
            Box(-ob_scale * np.ones(dim_state),
                ob_scale * np.ones(dim_state),
                dtype=np.float64)
        self.action_space = \
            Box(action_range_min,
                action_range_max,
                dtype=np.float64)

    def state(self):
        return self.base_env.state(idx=0)

    def reset(self):
        self.base_env.reset()
        return self.base_env.state(idx=0)

    def step(self, action):
        rew, info = self.base_env.step([action])
        obs = self.state()
        eoe, eoe_reason = self.base_env.inspect_end_of_episode(idx=0)
        if self.base_env._verbose:
            print('Step -------------------')
            print('\tReward', rew)
            print('\tEnd of episode', eoe_reason)
        return obs, rew[0], eoe, info[0]

class EnvRenderer(er.EnvRenderer):
    def __init__(self, trainer, config, **kwargs):
        from fairmotion.viz.utils import TimeChecker
        super().__init__(**kwargs)
        self.trainer = trainer
        self.config = config
        self.time_checker_auto_play = TimeChecker()
        self.explore = False
        self.data = {}
    def one_step(self):
        s1 = self.env.state()
        a = self.trainer.compute_action(s1, explore=self.explore)
        s2, rew, eoe, info = self.env.step(a)
    def extra_render_callback(self):
        self.env.base_env.render(self.rm)
    def extra_idle_callback(self):
        time_elapsed = self.time_checker_auto_play.get_time(restart=False)
        if self.rm.flag['auto_play'] and time_elapsed >= self.env.base_env._dt:
            self.time_checker_auto_play.begin()
            self.one_step()
    def extra_keyboard_callback(self, key):
        if key == b'r':
            s = self.env.reset()
            self.data.clear()
        elif key == b'R':
            s = self.env.reset({'start_time': 0.0})
            self.data.clear()
        elif key == b' ':
            self.time_checker_auto_play.begin()
            self.one_step()
        elif key == b'e':
            self.explore = not self.explore
            print('Exploration:', self.explore)
    def get_elapsed_time(self):
        return self.env.base_env._elapsed_time

def default_cam():
    return rm.camera.Camera(pos=np.array([0.0, 3.0, 2.0]),
                            origin=np.array([0.0, 0.0, 0.0]), 
                            vup=np.array([0.0, 0.0, 1.0]), 
                            fov=60.0)

env_cls = HumanoidKinematicsContrastive

def config_override(spec):
    env = env_cls(spec["config"]["env_config"])

    model_config = copy.deepcopy(spec["config"]["model"])
    model = model_config.get("custom_model")
    if model and model == "task_agnostic_policy_type1":
        model_config.get("custom_model_config").update({
            "observation_space_body": copy.deepcopy(env.observation_space_body),
            "observation_space_task": copy.deepcopy(env.observation_space_task),
        })

    del env

    config = {
        # "callbacks": {},
        "model": model_config,
    }
    return config
