'''
python3 rllib_driver.py --mode load --spec data/spec/spec_env_contrative_test.yaml --project_dir ./
python3 rllib_driver.py --mode load --spec data/spec/spec_env_contrative_test.yaml --project_dir ./ --- checkpoint xxx

./rllib_test_interactive.sh rllib_driver.py data/spec/spec_env_contrative_test.yaml 40 15200 0 /checkpoint/xxx/tmp/ray/ct/test/
'''

import copy
import numpy as np
import argparse
import random

import gym
from gym.spaces import Box

import render_module as rm

import os
import pickle
import gzip

from collections import deque
import matplotlib.pyplot as plt

from ray.rllib.utils import try_import_torch
torch, nn = try_import_torch()

import torch_models

def gen_random_sequential_data(
    x_start, 
    x_min, 
    x_max, 
    num_iter=50, 
    scale_factor=0.05, 
    save_file=None):
    data = []
    x = x_start
    scale = scale_factor * (x_max-x_min)
    for i in range(num_iter):
        data.append(x)
        x = np.random.normal(loc=x, scale=scale)
        x = np.clip(x, x_min, x_max)
    if save_file is not None:
        with open(save_file, 'wb') as f:
            pickle.dump(data, f)
        print('Data is save:', save_file)
    return data

class Contrastive(gym.Env):
    def __init__(self, env_config):

        self.verbose = env_config.get('verbose', False)

        self.past_window_size = env_config.get('past_window_size')
        
        self.dim_state = env_config.get('dim_state')
        self.dim_action = env_config.get('dim_action')

        ob_min = env_config.get('range_min_state')
        ob_max = env_config.get('range_max_state')

        ac_min = env_config.get('range_min_action')
        ac_max = env_config.get('range_max_action')

        self.observation_space = \
            Box(ob_min * np.ones(self.dim_state),
                ob_max * np.ones(self.dim_state),
                dtype=np.float64)
        
        self.action_space = \
            Box(ac_min * np.ones(self.dim_action),
                ac_max * np.ones(self.dim_action),
                dtype=np.float64)

        self.past_trajectory = deque(maxlen=self.past_window_size)
        self.data = None
        self.dataset = None
        self.model = None
        self.model_type = None

        model = env_config.get('model')
        if model:
            self.model_type = model.get('type')
            with open(model.get('data'), 'rb') as f:
                self.data = pickle.load(f)
            with open(model.get('dataset'), 'rb') as f:
                self.dataset = pickle.load(f)
            self.model = torch.load(model.get('model'))
            self.model.eval()
            if self.verbose:
                print('-------------Loaded-------------')
                print(model.get('model'))
                print(model.get('data'))
                print(model.get('dataset'))
                print('--------------------------------')
        else:
            traj = gen_random_sequential_data(
                x_start=np.zeros(self.dim_action),
                x_min=self.action_space.low,
                x_max=self.action_space.high)
            self.data = [traj]

        self.reset()

    def state(self):
        s = []
        for i in range(self.past_window_size):
            s.append(self.past_trajectory[-1-i])
        return np.hstack(s)

    def reset(self, info={}):
        trajectory = self.data[np.random.randint(len(self.data))]
        idx = np.random.randint(0, len(trajectory)-self.past_window_size)
        self.past_trajectory.clear()
        for i in range(self.past_window_size):
            self.past_trajectory.append(trajectory[idx+i])
        return self.state()

    def inspect_eoe(self):
        return False

    def step(self, action):
        ''' Action here is just a next position '''
        self.past_trajectory.append(action)

        ''' Compute reward based on the contrastive model '''
        info = {}
        rew = 0.0
        if self.model:
            x = []
            for i in range(self.past_window_size):
                x.append(self.past_trajectory[-1-i])
            x = np.hstack(x)
            x = self.dataset.preprocess_x(x)
            x = torch.Tensor(x)

            if self.model_type == 'MSE':
                y = self.model(x).detach().numpy()
            elif self.model_type == 'CrossEntropy':
                y = F.softmax(y, dim=0).detach().numpy()[1]
            else:
                raise NotImplementedError

            rew = y

            if self.verbose:
                print('-------------Step-------------')
                print('past_trajectory: ', self.past_trajectory)  
        
        ''' Collect the next state '''
        obs = self.state()

        ''' Check end of episode '''
        eoe = self.inspect_eoe()
        
        return obs, rew, eoe, info

class EnvRenderer(object):
    class Trajectory:
        def __init__(self, ax, data, **kwargs):
            self.ax = ax
            self.data = []
            self.line, = self.ax.plot([], [], **kwargs)
            if len(data) > 0:
                self.set_data(data)
        def add_point(self, p):
            self.xs = self.xs+[p[0]]
            self.ys = self.ys+[p[1]]
            self.set_data(self.xs, self.ys)
        def set_data(self, data):
            xs, ys = [], []
            for p in data:
                xs.append(p[0])
                ys.append(p[1])
            self.line.set_data(xs, ys)
            self.data = data
    
    def __init__(self, trainer, env, **kwargs):
        self.trainer = trainer
        self.env = env
        self.explore = False
        self.data = {}

        low = self.env.action_space.low[:2]
        high = self.env.action_space.high[:2]

        self.fig = plt.figure(0, figsize=(12, 10), dpi=100)
        self.fig.canvas.set_window_title(
            "contrastive learning test for sequential data")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim((low[0], high[0]))
        self.ax.set_ylim((low[1], high[1]))
        self.ax.patch.set_facecolor('white')

        self.id_key_release = \
            self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        self.trajectory_data = []
        for data in self.env.data:
            self.trajectory_data.append(
                EnvRenderer.Trajectory(self.ax, data, color='k', linewidth=3))
        self.trajectory_cur = EnvRenderer.Trajectory(
            self.ax, self.env.past_trajectory, color='r', alpha=0.5, linewidth=5)

    def one_step(self):
        s1 = self.env.state()
        a = self.trainer.compute_action(s1, explore=self.explore)
        s2, rew, eoe, info = self.env.step(a)
        self.trajectory_cur.set_data(self.env.past_trajectory)

    def draw(self):
        self.fig.canvas.draw()

    def run(self):
        plt.show()

    def on_key_release(self, event):
        key = event.key
        if key=='escape':
            exit(0)
        elif key=='r':
            self.env.reset()
            self.trajectory_cur.set_data(self.env.past_trajectory)
            self.draw()
        elif key==' ':
            self.one_step()
            self.draw()


def default_cam():
    return rm.camera.Camera(pos=np.array([0.0, 3.0, 2.0]),
                            origin=np.array([0.0, 0.0, 0.0]), 
                            vup=np.array([0.0, 0.0, 1.0]), 
                            fov=60.0)

env_cls = Contrastive

def config_override(spec):
    model_config = copy.deepcopy(spec["config"]["model"])
    config = {
        # "callbacks": {},
        "model": model_config,
    }
    return config
