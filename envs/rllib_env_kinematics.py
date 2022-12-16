import copy
import numpy as np
import argparse
import random

import gym
from gym.spaces import Box

import env_humanoid_kinematics as my_env
import env_renderer as er
import render_module as rm

import os
import pickle
import gzip

from collections import deque

class HumanoidKinematics(gym.Env):
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
        eoe = self.base_env._end_of_episode
        return obs, rew[0], eoe, info[0]

class EnvRenderer(er.EnvRenderer):
    def __init__(self, trainer, **kwargs):
        from fairmotion.viz.utils import TimeChecker
        super().__init__(**kwargs)
        self.trainer = trainer
        self.time_checker_auto_play = TimeChecker()
        self.explore = False
        self.data = {}
    def get_pb_client(self):
        return self.env.base_env._pb_client
    def one_step(self):
        s1 = self.env.state()
        a = self.trainer.compute_action(s1, explore=self.explore)
        s2, rew, eoe, info = self.env.step(a)
    def extra_render_callback(self):
        if self.rm.flag['follow_cam']:
            p, _, _, _ = self.env.base_env._sim_agent[0].get_root_state()
            self.update_target_pos(p, ignore_z=True)
        self.env.base_env.render(self.rm)
    def extra_overlay_callback(self):
        model = self.trainer.get_policy().model
        if hasattr(model, 'gate_function'):
            expert_weights = model.gate_function()
            num_experts = model.num_experts()
            w, h = self.window_size
            w_bar, h_bar = 150, 20
            origin = np.array([0.95*w-w_bar, 0.95*h-h_bar])
            pos = origin.copy()
            for i in reversed(range(num_experts)):
                self.rm.gl_render.render_text(
                    "Expert%d"%(i), 
                    pos=pos-np.array([75, -0.8*h_bar]), 
                    font=self.rm.glut.GLUT_BITMAP_9_BY_15)
                w_i = expert_weights[0][i] if expert_weights is not None else 0.0
                self.rm.gl_render.render_progress_bar_2D_horizontal(
                    w_i, origin=pos, width=w_bar, 
                    height=h_bar, color_input=self.rm.COLORS_FOR_EXPERTS[i])
                pos += np.array([0.0, -h_bar])
        if hasattr(model, 'task_encoder_variable'):
            w, h = self.window_size
            w_size, h_size = 200, 200
            origin = np.array([50, 250])
            pos = origin.copy()
            self.rm.gl_render.render_graph_base_2D(origin=origin, axis_len=w_size, pad_len=30)            
            z_task = model.task_encoder_variable()
            z_task_range = (-1, 1)
            if z_task is not None:
                z1_cur = z_task[:,0]
                z2_cur = z_task[:,1]
                if 'z1_task' not in self.data:
                    self.data['z1_task'] = deque(maxlen=30)
                    self.data['z2_task'] = deque(maxlen=30)
                self.data['z1_task'].append(z1_cur)
                self.data['z2_task'].append(z2_cur)
                self.rm.gl_render.render_graph_data_point_2D(
                    x_data=z1_cur,
                    y_data=z2_cur,
                    x_range=z_task_range,
                    y_range=z_task_range,
                    color=[0, 0, 0, 1],
                    point_size=5.0,
                    origin=origin,
                    axis_len=w_size,
                    pad_len=30,
                )
                self.rm.gl_render.render_graph_data_line_2D(
                    x_data=self.data['z1_task'],
                    y_data=self.data['z2_task'],
                    x_range=z_task_range,
                    y_range=z_task_range,
                    color=[0, 0, 0, 1],
                    line_width=2.0,
                    origin=origin,
                    axis_len=w_size,
                    pad_len=30,
                )
                # print(z_task)
    def extra_idle_callback(self):
        time_elapsed = self.time_checker_auto_play.get_time(restart=False)
        if self.rm.flag['auto_play'] and time_elapsed >= self.env.base_env._dt_con:
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
        elif key == b'S':
            model = self.trainer.get_policy().model
            if hasattr(model, 'save_policy_weights'):
                model.save_policy_weights('data/temp/fc_policy.pt')
            if hasattr(model, 'save_weights_body_encoder'):
                model.save_weights_body_encoder('data/temp/task_agnostic_policy_body_encoder.pt')
            if hasattr(model, 'save_weights_motor_decoder'):
                model.save_weights_motor_decoder('data/temp/task_agnostic_policy_motor_decoder.pt')
        elif key == b'c':
            ''' Read a directory for saving images and try to create it '''
            subdir = input("Enter subdirectory for screenshot file: ")
            dir = os.path.join("data/screenshot/", subdir)
            try:
                os.makedirs(dir, exist_ok = True)
            except OSError:
                print("Invalid Subdirectory")
                return
            for i in range(1):
                try:
                    os.makedirs(dir, exist_ok = True)
                except OSError:
                    print("Invalid Subdirectory")
                    continue
                cnt_screenshot = 0
                while True:
                    name = 'screenshot_%04d'%(cnt_screenshot)
                    self.one_step()
                    self.save_screen(dir=dir, name=name, render=True)
                    print('\rsave_screen(%4.4f) / %s' % \
                        (self.env.base_env.get_elapsed_time(), os.path.join(dir,name)), end=" ")
                    cnt_screenshot += 1
                    if self.env.base_env._end_of_episode:
                        break
                print("\n")
        elif key == b'C':
            model = self.trainer.get_policy().model
            is_task_agnostic_policy = hasattr(model, 'task_encoder_variable')
            print('----------------------------')
            print('Extracting State-Action Pairs')
            print('----------------------------')
            ''' Read a directory for saving images and try to create it '''
            subdir = input("Enter subdirectory for saving pairs: ")
            dir = os.path.join("data/temp/", subdir)
            try:
                os.makedirs(dir, exist_ok = True)
            except OSError:
                print("Invalid Subdirectory")
                return
            data = []
            for i in range(len(self.env.base_env._ref_motion_all[0])):
                episode_data = {
                    'time': [],
                    'state': [],
                    'action': [],
                    'reward': [],
                }
                if is_task_agnostic_policy:
                    episode_data.update({
                        'z_body': [],
                        'z_task': [],
                    })
                cnt = 0
                self.env.reset({'ref_motion_id': [i], 'start_time': 0.0})
                while True:
                    s1 = self.env.state()
                    a = self.trainer.compute_action(s1, explore=self.explore)
                    s2, rew, eoe, info = self.env.step(a)
                    t = self.env.base_env.get_current_time()
                    episode_data['time'].append(t)
                    episode_data['state'].append(s1)
                    episode_data['action'].append(a)
                    episode_data['reward'].append(rew)
                    if is_task_agnostic_policy:
                        z_body = model.body_encoder_variable()[0].detach().numpy()
                        z_task = model.task_encoder_variable()[0].detach().numpy()
                        episode_data['z_body'].append(z_body)
                        episode_data['z_task'].append(z_task)
                        # print(s1)
                        # print(a)
                        # print(rew)
                        # print(t, z_body)
                        # print(t, z_task)
                        # exit(0)
                    cnt += 1
                    if cnt > 5:
                        break
                    if self.env.base_env._end_of_episode:
                        break
                    if self.env.base_env.get_current_time() >= \
                       self.env.base_env._ref_motion[0].length():
                        break
                data.append(episode_data)
                print('%d pairs were created in episode %d'%(cnt, i))
            if len(data) > 0:
                with open(os.path.join(dir, "episode_data_dict.pkl"), "wb") as file:
                    pickle.dump(data, file)

def default_cam():
    return rm.camera.Camera(pos=np.array([0.0, 3.0, 2.0]),
                            origin=np.array([0.0, 0.0, 0.0]), 
                            vup=np.array([0.0, 0.0, 1.0]), 
                            fov=60.0)

env_cls = HumanoidImitation

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
