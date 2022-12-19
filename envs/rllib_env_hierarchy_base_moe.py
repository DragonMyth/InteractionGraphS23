import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(directory))

import copy
import numpy as np

import ray
from ray.rllib.env import MultiAgentEnv

import gym
from gym.spaces import Box

import env_renderer as er
import render_module as rm

import rllib_model_torch as model

from ray.rllib.utils import try_import_torch
torch, nn = try_import_torch()

from collections import deque

import torch_models
import torch.nn.functional as F

BASE_ENV_CLS = None
CALLBACK_FUNC = None

class HumanoidHierarchyBaseMOE(MultiAgentEnv):
    def __init__(self, env_config):
        assert BASE_ENV_CLS is not None, \
            "Base environment class should be specified first"
        
        self.base_env = BASE_ENV_CLS(env_config.get('base_env_config'))
        
        fps_con = env_config.get('fps_con')
        fps_con_base_env = env_config.get('base_env_config').get('fps_con')
        assert fps_con <= fps_con_base_env and fps_con_base_env%fps_con == 0
        self.num_substep = fps_con_base_env//fps_con

        env_config.get('base_env_config')

        assert self.base_env._num_agent == 2
        
        base_model_config = env_config.get('base_model_config')
        custom_model_config = base_model_config.get('custom_model_config')
        project_dir = env_config['project_dir']
        
        for i in range(len(custom_model_config['expert_checkpoints'])):
            if custom_model_config['expert_checkpoints'][i]:
                custom_model_config['expert_checkpoints'][i] = \
                    os.path.join(
                        project_dir, 
                        custom_model_config['expert_checkpoints'][i]
                    )

        moe_type = env_config.get('moe_type', 'add')
        if moe_type == 'add':
            base_model_cls = model.MOEAddEval
        elif moe_type == 'mul':
            base_model_cls = model.MOEMulEval
        else:
            raise NotImplementedError

        self.use_dphase_action = env_config.get('use_dphase_action', True)

        self.base_model = base_model_cls(config=custom_model_config)

        ob_scale = np.array(env_config.get('ob_scale', 1000.0))
        ac_scale = np.array(env_config.get('ac_scale', 1.0))
        
        self.observation_space = []
        self.action_space = []
        
        for i in range(self.base_env._num_agent):
            dim_state = self.base_env.dim_state(i)
            if self.use_dphase_action:
                dim_action = 1 + self.base_model.num_experts()
            else:
                dim_action = self.base_model.num_experts()
            self.observation_space.append(
                Box(low=-ob_scale*np.ones(dim_state),
                    high=ob_scale*np.ones(dim_state),
                    dtype=np.float64)
                )
            self.action_space.append(
                Box(low=-ac_scale*np.ones(dim_action),
                    high=ac_scale*np.ones(dim_action),
                    dtype=np.float64)
                )

        self.player1 = "player1"
        self.player2 = "player2"

        self.data = [{} for i in range(self.base_env._num_agent)]

    def state(self):
        return {
            self.player1: self.base_env.state(idx=0),
            self.player2: self.base_env.state(idx=1),
        }

    def reset(self, info={}):
        if not self.base_env._initialized:
            self.base_env.create()
        self.base_env.reset(info)
        return self.state()

    def get_info(self):
        raise NotImplementedError

    def step(self, action_dict):
        action1 = action_dict[self.player1]
        action2 = action_dict[self.player2]


        r = np.zeros(2)
        for _ in range(self.num_substep):

            if self.use_dphase_action:
                dphase1, weight1 = action1[0], action1[1:]
                dphase2, weight2 = action2[0], action2[1:]
                x_expert1 = np.hstack([
                    self.base_env.state_body(idx=0),
                    self.base_env._phase[0]])
                x_expert2 = np.hstack([
                    self.base_env.state_body(idx=1),
                    self.base_env._phase[1]])
            else:
                weight1 = action1
                weight2 = action2
                x_expert1 = self.base_env.state_body(idx=0)
                x_expert2 = self.base_env.state_body(idx=1)

            input_dict = {
                "x_weight": np.vstack([
                    weight1, 
                    weight2]),
                "x_expert": np.vstack([
                    x_expert1,
                    x_expert2]),
            }

            '''
            Compute target poses based on weights and the experts
            '''

            target_poses = self.base_model.forward(input_dict).detach().numpy()

            if self.use_dphase_action:
                base_action_dict = {
                    'target_pose': target_poses,
                    'dphase': np.array([dphase1, dphase2]),
                }
            else:
                base_action_dict = {
                    'target_pose': target_poses,
                }

            r_i, info_env = self.base_env.step(base_action_dict)
            r += r_i

        obs = self.state()
        rew = {
            self.player1: r[0],
            self.player2: r[1]
        }
        done = {
            "__all__": self.base_env._end_of_episode,
        }

        info  = self.get_info()

        if self.base_env._verbose:
            self.base_env.pretty_print_rew_info(info_env[0]['rew_info'])
            self.base_env.pretty_print_rew_info(info_env[1]['rew_info'])

        return obs, rew, done, info

def policy_mapping_fn(agent_id, share_policy):
    if share_policy:
        return "policy_player"
    else:
        if agent_id=="player1": 
            return "policy_player1"
        elif agent_id=="player2": 
            return "policy_player2"
        else:
            raise Exception(agent_id)

def policies_to_train(share_policy):
    if share_policy:
        return ["policy_player"]
    else:
        return ["policy_player1", "policy_player2"]

env_cls = HumanoidHierarchyBaseMOE

def config_override(spec):
    env = env_cls(spec["config"]["env_config"])

    share_policy = \
        spec["config"]["env_config"].get("share_policy", False)

    if share_policy:
        policies = {
            "policy_player": (
                None, 
                copy.deepcopy(env.observation_space[0]), 
                copy.deepcopy(env.action_space[0]), 
                {}),
        }
    else:
        policies = {
            "policy_player1": (
                None, 
                copy.deepcopy(env.observation_space[0]), 
                copy.deepcopy(env.action_space[0]), 
                {}),
            "policy_player2": (
                None, 
                copy.deepcopy(env.observation_space[1]), 
                copy.deepcopy(env.action_space[1]), 
                {}),
        }

    model_config = copy.deepcopy(spec["config"]["model"])
    model = model_config.get("custom_model")
    if model and model == "task_agnostic_policy_type1":
        model_config.get("custom_model_config").update({
            "observation_space_body": 
                copy.deepcopy(env.observation_space_body[0]),
            "observation_space_task": 
                copy.deepcopy(env.observation_space_task[0]),
        })

    del env

    assert CALLBACK_FUNC is not None, \
        "callback_func should be specified first"

    config = {
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": \
                lambda agent_id : policy_mapping_fn(agent_id, share_policy),
            "policies_to_train": policies_to_train(share_policy),
        },
        "callbacks": CALLBACK_FUNC,
        "model": model_config,
    }

    return config

class EnvRenderer(er.EnvRenderer):
    def __init__(self, trainer, config, **kwargs):
        from fairmotion.viz.utils import TimeChecker
        super().__init__(**kwargs)
        self.trainer = trainer
        self.config = config
        self.time_checker_auto_play = TimeChecker()
        self.explore = False
        self.data = {}
        self.share_policy = config["env_config"].get("share_policy", False)
        self.reset()
    def get_pb_client(self):
        return self.env.base_env._pb_client
    def reset(self, info={}):
        s = self.env.reset(info)
        self.policy_hidden_state = []
        self.agent_ids = s.keys()
        for agent_id in self.agent_ids:
            state = self.trainer.get_policy(
                policy_mapping_fn(
                    agent_id, self.share_policy)).get_initial_state()
            self.policy_hidden_state.append(state)
            self.data[agent_id] = {}
    def one_step(self):
        s = self.env.state()
        a = {}
        for i, agent_id in enumerate(self.agent_ids):
            policy = self.trainer.get_policy(
                policy_mapping_fn(agent_id, self.share_policy))
            if policy.is_recurrent():
                action, state_out, extra_fetches = self.trainer.compute_action(
                    s[agent_id], 
                    state=self.policy_hidden_state[i],
                    policy_id=policy_mapping_fn(agent_id, self.share_policy),
                    explore=self.explore)
                self.policy_hidden_state[i] = state_out
            else:
                action = self.trainer.compute_action(
                    s[agent_id], 
                    policy_id=policy_mapping_fn(agent_id, self.share_policy),
                    explore=self.explore)
            a[agent_id] = action
            if 'action' not in self.data[agent_id]:
                self.data[agent_id]['action'] = deque(maxlen=30)
            else:
                self.data[agent_id]['action'].append(action)
        return self.env.step(a)
    def extra_render_callback(self):
        self.env.base_env.render(self.rm)
    def extra_overlay_callback(self):
        expert_weights = self.env.base_model.get_cur_weights()
        num_experts = self.env.base_model.num_experts()

        w, h = self.window_size
        w_bar, h_bar = 150.0, 20.0
        phase_radius = 40.0

        for i, agent_id in enumerate(self.agent_ids):
            if i==0:
                origin = np.array([20, 20])
            else:
                origin = np.array([w-2*phase_radius-20, 20])
            # self.rm.gl_render.render_progress_bar_2D_horizontal(
            #     self.env.phase[i], origin=origin, width=w_bar, 
            #     height=h_bar, color_input=[0.1, 0.1, 0.1, 1.0])
            self.rm.gl_render.render_progress_circle_2D(
                self.env.base_env._phase[i], 
                origin=(origin[0]+phase_radius,origin[1]+phase_radius), 
                radius=phase_radius,
                scale_input=0.3,
            )

        for i, agent_id in enumerate(self.agent_ids):
            if i==0:
                origin = np.array([20, h_bar*num_experts+100])
                text_offset = np.array([w_bar+5, 0.8*h_bar])
            else:
                origin = np.array([w-w_bar-20, h_bar*num_experts+100])
                text_offset = np.array([-75, 0.8*h_bar])
            pos = origin.copy()
            for j in reversed(range(num_experts)):
                self.rm.gl_render.render_text(
                    "Expert%d"%(j), 
                    pos=pos+text_offset, 
                    font=self.rm.glut.GLUT_BITMAP_9_BY_15)
                expert_weight_j = \
                    expert_weights[i][j] if expert_weights is not None else 0.0
                self.rm.gl_render.render_progress_bar_2D_horizontal(
                    expert_weight_j, origin=pos, width=w_bar, 
                    height=h_bar, color_input=self.rm.COLORS_FOR_EXPERTS[j])
                pos += np.array([0.0, -h_bar])

    def extra_idle_callback(self):
        time_elapsed = self.time_checker_auto_play.get_time(restart=False)
        if self.rm.flag['auto_play'] and time_elapsed >= self.env.base_env._dt_con:
            self.time_checker_auto_play.begin()
            self.one_step()
    def extra_keyboard_callback(self, key):
        if key == b'r':
            self.reset()
        if key == b'R':
            self.reset({'start_time': np.array([0.0, 0.0])})
        elif key == b' ':
            self.time_checker_auto_play.begin()
            self.one_step()
        elif key == b'e':
            self.explore = not self.explore
            print('Exploration:', self.explore)
        elif key == b's':
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
            ''' Read maximum end time '''
            end_time_margin = input("Enter end-time margin (s): ")
            try:
               end_time_margin = float(end_time_margin)
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
                time_eoe = None
                cnt_screenshot = 0
                # self.reset({'start_time': 0.0})
                self.reset()
                while True:
                    name = 'screenshot_%04d'%(cnt_screenshot)
                    self.one_step()
                    self.save_screen(dir=dir_i, name=name, render=True)
                    print('\rsave_screen(%4.4f) / %s' % \
                        (self.env.base_env.get_elapsed_time(), os.path.join(dir_i,name)), end=" ")
                    cnt_screenshot += 1
                    if time_eoe is None:
                        if self.env.base_env.get_elapsed_time() >= end_time:
                            time_eoe = self.env.base_env.get_elapsed_time()
                        if self.env.base_env._end_of_episode:
                            time_eoe = self.env.base_env.get_elapsed_time()
                    else:
                        if self.env.base_env.get_elapsed_time() - time_eoe >= end_time_margin:
                            break
                print("\n")
    def get_cam_target_pos(self):
        p1 = self.env.base_env._sim_agent[0].get_root_position()
        p2 = self.env.base_env._sim_agent[1].get_root_position()
        return 0.5 * (p1 + p2)
    def get_elapsed_time(self):
        return self.env.base_env.get_elapsed_time()

def default_cam():
    return rm.camera.Camera(
        pos=np.array([12.0, 0.0, 12.0]),
        origin=np.array([0.0, 0.0, 0.0]), 
        vup=np.array([0.0, 0.0, 1.0]), 
        fov=30.0)

