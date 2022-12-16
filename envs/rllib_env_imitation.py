import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(directory))

import copy
import numpy as np
import argparse
import random

import gym
from gym.spaces import Box

from envs import env_humanoid_imitation as my_env
import env_renderer as er
import render_module as rm

import pickle
import gzip

import rllib_model_torch as policy_model
from collections import deque

from ray.rllib.utils import try_import_torch
torch, nn = try_import_torch()

from fairmotion.core.motion import Pose, Motion
from fairmotion.core.velocity import MotionWithVelocity
from fairmotion.data import bvh
from fairmotion.ops import conversions

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

class HumanoidImitation(gym.Env):
    def __init__(self, env_config):
        self.base_env = my_env.Env(env_config)
        assert self.base_env._num_agent == 1
        
        ob_scale = 1000.0
        dim_state = self.base_env.dim_state(0)
        dim_state_body = self.base_env.dim_state_body(0)
        dim_state_task = self.base_env.dim_state_task(0)
        dim_action = self.base_env.dim_action(0)
        action_range_min, action_range_max = self.base_env.action_range(0)

        self.observation_space = \
            Box(-ob_scale * np.ones(dim_state),
                ob_scale * np.ones(dim_state),
                dtype=np.float64)
        self.observation_space_body = \
            Box(-ob_scale * np.ones(dim_state_body),
                ob_scale * np.ones(dim_state_body),
                dtype=np.float64)
        self.observation_space_task = \
            Box(-ob_scale * np.ones(dim_state_task),
                ob_scale * np.ones(dim_state_task),
                dtype=np.float64)
        self.action_space = \
            Box(action_range_min,
                action_range_max,
                dtype=np.float64)

    def state(self):
        return self.base_env.state(idx=0)

    def reset(self, info={}):
        if not self.base_env._initialized:
            self.base_env.create()
        self.base_env.reset(info)
        return self.base_env.state(idx=0)

    def step(self, action):
        
        # cnt = 0
        
        # if self.base_env._use_base_residual_linear_force:
        #     base_residual_linear_force = action[cnt:cnt+3]
        #     cnt += 3
        # else:
        #     base_residual_linear_force = None
        
        # if self.base_env._use_base_residual_angular_force:
        #     base_residual_angular_force = action[cnt:cnt+3]
        #     cnt += 3
        # else:
        #     base_residual_angular_force = None

        # target_pose = action[cnt:]
        
        # action_dict = {
        #     'base_residual_linear_force': base_residual_linear_force,
        #     'base_residual_angular_force': base_residual_angular_force,
        #     'target_pose': target_pose,
        # }

        rew, info = self.base_env.step([action])
        obs = self.state()
        eoe = self.base_env._end_of_episode
        if self.base_env._verbose:
            print(rew)
        return obs, rew[0], eoe, info[0]

class EnvRenderer(er.EnvRenderer):
    def __init__(self, trainers, config, **kwargs):
        from fairmotion.viz.utils import TimeChecker
        # kwargs['renderer'] = 'bullet_native'
        super().__init__(**kwargs)
        self.trainer = trainers[0]
        self.config = config
        self.time_checker_auto_play = TimeChecker()
        self.explore = False
        self.bgcolor=[1.0, 1.0, 1.0, 1.0]
        self.latent_random_sample = 0
        self.latent_random_sample_methods = [None, "gaussian", "uniform", "softmax", "hardmax"]
        self.cam_params = deque(maxlen=30)
        self.cam_param_offset = None
        self.replay = False
        self.replay_cnt = 0
        self.replay_data = {}
        self.replay_render_interval = 15
        self.replay_render_alpha = 0.5
        self.reset()
    def use_default_ground(self):
        return True
    def get_v_up_env_str(self):
        return self.env.base_env._v_up_str
    def get_ground(self):
        return self.env.base_env._ground
    def get_pb_client(self):
        return self.env.base_env._pb_client
    def reset(self, info={}):
        self.data = {
            'z_task': deque(maxlen=30),
            'gate_fn_w': deque(maxlen=30),
            'joint_data': list(),
            'link_data': list(),
            'reward': list(),
            'time': list(),
        }
        self.replay_cnt = 0
        if self.replay:
            self.set_pose()
        else:
            self.env.reset(info)
            self.policy_hidden_state = self.trainer.get_policy().get_initial_state()
            motion = Motion(
                skel=self.env.base_env._base_motion[0].skel,
                fps=self.env.base_env._base_motion[0].fps,
            )
            self.replay_data = {
                'motion': motion,
                'joint_data': list(),
                'link_data': list(),
                'others': {},
            }
        self.cam_params.clear()
        param = self._get_cam_parameters()
        for i in range(self.cam_params.maxlen):
            self.cam_params.append(param)
    def collect_replay_data(self):
        env = self.env.base_env
        sim_agent = env._sim_agent[0]
        motion = self.replay_data['motion']
        motion.add_one_frame(
            sim_agent.get_pose(motion.skel, apply_height_offset=False).data)
        joint_data, link_data = env.get_render_data(0)
        self.replay_data['joint_data'].append(joint_data)
        self.replay_data['link_data'].append(link_data)
    def set_pose(self):
        if self.replay_data['motion'].num_frames() == 0: 
            return
        motion = self.replay_data['motion']
        pose = motion.get_pose_by_frame(self.replay_cnt)
        self.env.base_env._sim_agent[0].set_pose(pose)
    def one_step(self, explore=None, collect_data=False):
        self.cam_params.append(self._get_cam_parameters())
        
        if self.replay:
            self.set_pose()
            self.replay_cnt = \
                min(self.replay_data['motion'].num_frames()-1, self.replay_cnt+1)
            return
        
        if explore is None:
            explore = self.explore
        # Run the environment
        s1 = self.env.state()
        policy = self.trainer.get_policy()
        if isinstance(policy.model, (policy_model.TaskAgnosticPolicyType1)) and \
           self.latent_random_sample != 0:
            method = self.latent_random_sample_methods[self.latent_random_sample]
            if method == 'gaussian':
                z_task = np.random.normal(
                    size=policy.model._task_encoder_output_dim)
            elif method == 'uniform':
                z_task = np.random.uniform(
                    size=policy.model._task_encoder_output_dim)
            elif method == 'softmax':
                def softmax(x, axis=None):
                    x = x - x.max(axis=axis, keepdims=True)
                    y = np.exp(x)
                    return y / y.sum(axis=axis, keepdims=True)
                z_task = np.random.normal(
                    size=policy.model._task_encoder_output_dim)
                z_task = softmax(z_task)
            elif method == 'hardmax':
                z_task = np.zeros(policy.model._task_encoder_output_dim)
                idx = np.random.choice(policy.model._task_encoder_output_dim)
                z_task[idx] = 1.0
            else:
                raise NotImplementedError
            # print(z_task)
            logits, _ = policy.model.forward_decoder(
                z_body=torch.Tensor([self.env.base_env.state_body(0)]),
                z_task=torch.Tensor([z_task]),
                state=None,
                seq_lens=None,
                state_cnt=None,
            )
            mean = logits[0][:self.env.action_space.shape[0]].detach().numpy().copy()
            if explore:
                logstd = logits[0][self.env.action_space.shape[0]:].detach().numpy().copy()
                action = np.random.normal(loc=mean, scale=np.exp(logstd))
            else:
                action = mean
            # print('z_task:', z_task)
            # print('action:', action)
        else:
            if policy.is_recurrent():
                action, state_out, extra_fetches = \
                    self.trainer.compute_single_action(
                        s1, 
                        state=self.policy_hidden_state,
                        explore=explore)
                self.policy_hidden_state = state_out
            else:
                action = self.trainer.compute_single_action(
                    s1, 
                    explore=explore)
        # Step forward
        s2, rew, eoe, info = self.env.step(action)
        self.data['reward'].append(rew)
        self.data['time'].append(self.get_elapsed_time())
        self.collect_replay_data()
        
        return s2, rew, eoe, info
    def extra_render_callback(self):
        if self.rm.get_flag('custom4'):
            if self.replay_data['motion'].num_frames() == 0: 
                return
            motion = self.replay_data['motion']
            pose = motion.get_pose_by_frame(self.replay_cnt)
            self.env.base_env._sim_agent[0].set_pose(pose)

            motion = self.replay_data['motion']

            color = self.rm.COLOR_AGENT.copy()
            color[3] = self.replay_render_alpha
            
            for i in range(0, motion.num_frames(), self.replay_render_interval):
                if motion.num_frames()-i < self.replay_render_interval:
                    break
                pose = motion.get_pose_by_frame(i)
                agent = self.env.base_env._sim_agent[0]
                agent.set_pose(pose)
                self.rm.bullet_render.render_model(
                    agent._pb_client, 
                    agent._body_id,
                    draw_link=True, 
                    draw_link_info=False, 
                    draw_joint=self.rm.flag['joint'], 
                    draw_joint_geom=False, 
                    ee_indices=agent._char_info.end_effector_indices, 
                    link_info_scale=self.rm.LINK_INFO_SCALE,
                    link_info_line_width=self.rm.LINK_INFO_LINE_WIDTH,
                    link_info_num_slice=self.rm.LINK_INFO_NUM_SLICE,
                    color=color)
        # ## Fencing
        # agent = self.env.base_env._sim_agent[0]
        # agent_color = np.array([255,  0, 0, 255])/255 
        # rm.COLORS_FOR_AGENTS[0] = {}
        # rm.COLORS_FOR_AGENTS[0]['default'] = np.array([220,  220, 220, 255])/255
        # rm.COLORS_FOR_AGENTS[0][agent._char_info.rsword_blade] = agent_color
        self.env.base_env.render(self.rm)
        ## Boxing Gloves
        # for i in range(self.env.base_env._num_agent):
        #     agent = self.env.base_env._sim_agent[i]
        #     if agent._char_info.name == "BOXER":
        #         self.rm.bullet_render.render_links(
        #             agent._pb_client,
        #             agent._body_id,
        #             link_ids=[
        #                 agent._char_info.joint_idx["lwrist"],
        #                 agent._char_info.joint_idx["rwrist"]],
        #             color=[0.1,0.1,0.1,1],
        #         )

    def extra_overlay_callback(self):
        model = self.trainer.get_policy().model
        w, h = self.window_size
        h_cur = 0
        # def draw_trajectory(
        #     data, 
        #     color_cur=[1, 0, 0, 1], 
        #     color_prev=[0.5, 0.5, 0.5, 1],
        #     w_size=150,
        #     h_size=150,
        #     pad_len=10,
        #     pos_origin=np.zeros(2),
        #     x_range=(-1.0, 1.0),
        #     y_range=(-1.0, 1.0)
        #     ):
        #     origin = pos_origin + np.array([20, h_size+20])
        #     self.rm.gl_render.render_graph_base_2D(origin=origin, axis_len=w_size, pad_len=pad_len)
        #     if len(data) > 0:
        #         action = np.vstack(data)
        #         action_x = action[:,0].tolist()
        #         action_y = action[:,1].tolist()
        #         self.rm.gl_render.render_graph_data_point_2D(
        #             x_data=[action_x[-1]],
        #             y_data=[action_y[-1]],
        #             x_range=x_range,
        #             y_range=y_range,
        #             color=color_cur,
        #             point_size=5.0,
        #             origin=origin,
        #             axis_len=w_size,
        #             pad_len=pad_len,
        #         )
        #         self.rm.gl_render.render_graph_data_line_2D(
        #             x_data=action_x,
        #             y_data=action_y,
        #             x_range=x_range,
        #             y_range=y_range,
        #             color=color_prev,
        #             line_width=2.0,
        #             origin=origin,
        #             axis_len=w_size,
        #             pad_len=pad_len,
        #         )
        # model = self.trainer.get_policy().model
        # if len(self.data['z_task']) > 0:
        #     activation_unit = model._task_encoder_activations[-1]
        #     if activation_unit == 'linear':
        #         x_range = y_range = (-10.0, 10.0)
        #     elif activation_unit == 'tanh':
        #         x_range = y_range = (-1.0, 1.0)
        #     else:
        #         raise NotImplementedError
        #     draw_trajectory(self.data['z_task'],
        #         color_cur=[1.0, 0.0, 0.0, 1.0],
        #         color_prev=[0.0, 0.0, 0.0, 0.5],
        #         pos_origin=np.array([0, h_cur]),
        #         x_range=x_range,
        #         y_range=y_range,
        #     )
        #     h_cur += 150+10
        if len(self.data['reward'])>0:
            reward = self.data['reward']
            time = self.data['time']
            x_range = (0, 10.0)
            y_range=(0,1)
            w_size=150
            h_size=150
            pad_len=10
            color_cur=[1, 0, 0, 1], 
            color_prev=[0.5, 0.5, 0.5, 1],
            pos_origin=np.zeros(2),
            origin = (pos_origin + np.array([100, h_size+100])).flatten()
            self.rm.gl_render.render_graph_base_2D(origin,w_size,pad_len)
            self.rm.gl_render.render_graph_data_point_2D(
                x_data=[time[-1]],
                y_data=[reward[-1]],
                x_range=x_range,
                y_range=y_range,
                color=[1.0, 0.0, 0.0, 1.0],
                point_size=5.0,
                origin=origin,
                axis_len=w_size,
                pad_len=pad_len,
            )

            self.rm.gl_render.render_graph_data_line_2D(
                    x_data=time,
                    y_data=reward,
                    x_range=x_range,
                    y_range=y_range,
                    color=[1.0, 0.0, 0.0,0.5],
                    line_width=2.0,
                    origin=origin,
                    axis_len=w_size,
                    pad_len=pad_len,
            )
            font = self.rm.glut.GLUT_BITMAP_9_BY_15

            self.rm.gl_render.render_text(
                "Reward: %.4f"%(reward[-1]), pos=[100,100], font=font)
        env = self.env.base_env
        font = self.rm.glut.GLUT_BITMAP_9_BY_15
        ref_motion_name = env._ref_motion_file_names[0][env._ref_motion_idx[0]]

        self.rm.gl_render.render_text(
            "File: %s"%ref_motion_name, pos=[0.05*w, 0.05*h], font=font)


        # ''' Expert Weights '''
        # w_bar, h_bar = 150.0, 20.0
        # if len(self.data['gate_fn_w']) > 0:
        #     expert_weights = self.data['gate_fn_w'][-1]
        #     num_experts = len(expert_weights)
        #     origin = np.array([20, h_cur+h_bar*num_experts])
        #     text_offset = np.array([w_bar+5, 0.8*h_bar])
        #     pos = origin.copy()
        #     for j in reversed(range(num_experts)):
        #         self.rm.gl_render.render_text(
        #             "Expert%d"%(j), 
        #             pos=pos+text_offset, 
        #             font=self.rm.glut.GLUT_BITMAP_9_BY_15)
        #         expert_weight_j = \
        #             expert_weights[j] if expert_weights is not None else 0.0
        #         self.rm.gl_render.render_progress_bar_2D_horizontal(
        #             expert_weight_j, origin=pos, width=w_bar, 
        #             height=h_bar, color_input=self.rm.COLORS_FOR_EXPERTS[j])
        #         pos += np.array([0.0, -h_bar])
        #     h_cur += (h_bar*num_experts+10)
        # radius = 80
        # origin = np.array([40, h_cur+40]) + radius
        # self.rm.gl_render.render_expert_weights_circle(
        #     weights=self.data['gate_fn_w'],
        #     radius=radius,
        #     origin=origin,
        #     # scale_weight=25.0,
        #     scale_weight=5.0,
        #     color_experts=self.rm.COLORS_FOR_EXPERTS,
        # )
    def extra_idle_callback(self):
        time_elapsed = self.time_checker_auto_play.get_time(restart=False)
        if self.rm.flag['auto_play'] and time_elapsed >= self.env.base_env._dt_con:
            self.time_checker_auto_play.begin()
            self.one_step(collect_data=True)
    def extra_keyboard_callback(self, key):
        if key == b'r':
            print("Reset w/o replay")
            self.replay = False
            s = self.env.reset()
            self.reset()
        elif key == b'R':
            time = get_float_from_input("Enter start time")
            self.reset({'start_time': np.array([time])})
        elif key == b'p':
            print("Reset w/ replay")
            self.replay = True
            self.reset()
        elif key == b']':
            if self.replay:
                self.replay_cnt = \
                    min(self.replay_data[0]['motion'].num_frames()-1, self.replay_cnt+1)
                self.set_pose()
        elif key == b'[':
            if self.replay:
                self.replay_cnt = max(0, self.replay_cnt-1)
                self.set_pose()
        elif key == b'+':
            self.replay_render_interval = \
                min(90, self.replay_render_interval+5)
            print('replay_render_interval', self.replay_render_interval)
        elif key == b'-':
            self.replay_render_interval = \
                max(5, self.replay_render_interval-5)
            print('replay_render_interval', self.replay_render_interval)
        elif key == b'>':
            self.replay_render_alpha = \
                min(1.0, self.replay_render_alpha+0.1)
            print('replay_render_alpha', self.replay_render_alpha)
        elif key == b'<':
            self.replay_render_alpha = \
                max(0.1, self.replay_render_alpha-0.1)
            print('replay_render_alpha', self.replay_render_alpha)
        elif key == b' ':
            self.time_checker_auto_play.begin()
            self.one_step()
        elif key == b'e':
            self.explore = not self.explore
            print('Exploration:', self.explore)
        elif key == b'E':
            model = self.trainer.get_policy().model
            exp_std = get_float_from_input("Exploration Std")
            assert exp_std >= 0.0
            model.set_exploration_std(exp_std)
        elif key == b'w':
            print('Save Model Weights...')
            model = self.trainer.get_policy().model
            if hasattr(model, 'save_weights'):
                model.save_weights('data/temp/policy_weights.pt')
            if hasattr(model, 'save_weights_body_encoder'):
                model.save_weights_body_encoder('data/temp/task_agnostic_policy_body_encoder.pt')
            if hasattr(model, 'save_weights_task_encoder'):
                model.save_weights_task_encoder('data/temp/task_agnostic_policy_task_encoder.pt')
            if hasattr(model, 'save_weights_motor_decoder'):
                model.save_weights_motor_decoder('data/temp/task_agnostic_policy_motor_decoder.pt')
            if hasattr(model, 'save_weights_seperate'):
                model.save_weights_seperate(
                    file_gate='data/temp/imitation_policy_gate',
                    file_experts='data/temp/imitation_policy_expert',
                    file_helpers='data/temp/imitation_policy_helper')
            print('Done.')
        elif key == b's' or key == b'S':
            save_image = get_bool_from_input("Save image")
            save_motion = get_bool_from_input("Save motion")
            save_motion_only_success = False
            if save_motion:
                save_motion_only_success = get_bool_from_input("Save success motion only")
            save_dir = None
            if save_image or save_motion:
                ''' Read a directory for saving images and try to create it '''
                save_dir = input("Enter directory for saving: ")
                try:
                    os.makedirs(save_dir, exist_ok=True)
                except OSError:
                    print("Invalid Subdirectory")
                    return
            if key == b's':
                print('Recording the Current Scene...')
                ''' Read maximum end time '''
                end_time = get_float_from_input("Enter max end-time (sec)")
                ''' Read number of iteration '''
                num_iter = get_int_from_input("Enter num iteration")
                ''' Start each episode at zero '''
                reset_at_fixed_time = get_bool_from_input("Always reset at a fixed time")
                if reset_at_fixed_time:
                    start_time = get_float_from_input("Enter start time")
                ''' Read falldown check '''
                check_falldown = get_bool_from_input("Terminate when falldown")
                ''' Read end_of_motion check '''
                check_end_of_motion = get_bool_from_input("Terminate when reaching the end of motion")
                for i in range(num_iter):
                    if reset_at_fixed_time:
                        self.reset({'start_time': np.array([start_time])})
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
                        save_motion_only_success=save_motion_only_success,
                        end_time=end_time, 
                        check_falldown=check_falldown, 
                        check_end_of_motion=check_end_of_motion)
                print('Done.')
            elif key == b'S':
                print('Recording the Entire Motions...')
                for i in range(len(self.env.base_env._ref_motion_all[0])):
                    self.reset({
                        'start_time': np.array([0.0]),
                        'ref_motion_id': [i],
                        })
                    if save_dir:
                        save_dir_i = os.path.join(save_dir, str(i))
                    else:
                        save_dir_i = None
                    self.record_a_scene(
                        save_dir=save_dir_i,
                        save_image=save_image,
                        save_motion=save_motion)
                print('Done.')
        elif key == b'?':
            filename = os.path.join(save_dir_i, "replay.pkl")
            pickle.dump(self.replay_data, open(filename, "wb"))
            print(filename)
        elif key == b'/':
            print('Load Replay Data...')
            name = input("Enter data file: ")
            with open(name, "rb") as f:
                self.replay_data = pickle.load(f)
            self.replay = True
            self.reset()
            print('Done.')
        elif key == b'g':
            ''' Read maximum end time '''
            end_time = get_float_from_input("Enter max end-time (sec)")
            ''' Read number of iteration '''
            run_every_motion = get_bool_from_input("Run for every ref_motion?")
            ''' Read number of iteration '''
            num_iter = get_int_from_input("Enter num iteration")
            ''' Start each episode at zero '''
            reset_at_zero = get_bool_from_input("Always reset at 0s")
            ''' Verbose '''
            verbose = get_bool_from_input("Verbose")
            ''' Result file '''
            result_file = input("Result file:")
            ''' Save Motion '''
            save_motion = get_bool_from_input("Save motion")
            save_motion_dir = None
            if save_motion:
                save_motion_dir = input("Enter directory for saving: ")
                try:
                    os.makedirs(save_motion_dir, exist_ok=True)
                except OSError:
                    print("Invalid Subdirectory")
                    return

            while True:
                if self.latent_random_sample_methods[self.latent_random_sample] == 'gaussian':
                    break
                self.latent_random_sample = (self.latent_random_sample+1)%len(self.latent_random_sample_methods)

            motor_decoders = [None]
            motor_decoders = [
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-29-55/TrainModel_51ebc_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-09-30_19-29-56/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-29-55/TrainModel_51ebc_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-09-30_19-29-56/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-29-55/TrainModel_51ebc_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-09-30_19-29-56/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-29-55/TrainModel_51ebc_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-09-30_19-29-56/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-29-55/TrainModel_51ebc_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-09-30_19-29-56/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-29-55/TrainModel_51ebc_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-09-30_19-29-56/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-29-55/TrainModel_51ebc_00006_6_FP_cyc_coeff=0.1,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-09-30_19-29-56/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-29-55/TrainModel_51ebc_00007_7_FP_cyc_coeff=0.3,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-09-30_19-29-56/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-29-55/TrainModel_51ebc_00008_8_FP_cyc_coeff=1.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-09-30_19-29-56/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-30-51/TrainModel_73402_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-09-30_19-30-51/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-30-51/TrainModel_73402_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-09-30_19-30-52/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-30-51/TrainModel_73402_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-09-30_19-30-52/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-30-51/TrainModel_73402_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-09-30_19-30-52/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-30-51/TrainModel_73402_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-09-30_19-30-52/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-30-51/TrainModel_73402_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-09-30_19-30-52/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-30-51/TrainModel_73402_00006_6_FP_cyc_coeff=0.1,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-09-30_19-30-52/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-30-51/TrainModel_73402_00007_7_FP_cyc_coeff=0.3,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-09-30_19-30-52/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-09-30_19-30-51/TrainModel_73402_00008_8_FP_cyc_coeff=1.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-09-30_19-30-52/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",

                # lookaded=4
                # "/home/jungdam/ray_results/TrainModel_2021-10-01_23-44-13/TrainModel_02a22_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-01_23-44-13/checkpoint_000800/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-01_23-44-13/TrainModel_02a22_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-01_23-44-13/checkpoint_000800/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-01_23-44-13/TrainModel_02a22_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-01_23-44-13/checkpoint_000800/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-01_23-44-13/TrainModel_02a22_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-01_23-44-13/checkpoint_000800/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-01_23-44-13/TrainModel_02a22_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-01_23-44-13/checkpoint_000800/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-01_23-44-13/TrainModel_02a22_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-01_23-44-13/checkpoint_000800/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-01_23-44-13/TrainModel_02a22_00006_6_FP_cyc_coeff=0.1,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-01_23-44-13/checkpoint_000800/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-01_23-44-13/TrainModel_02a22_00007_7_FP_cyc_coeff=0.3,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-01_23-44-13/checkpoint_000800/task_agnostic_policy_motor_decoder.pt",

                # lookaded=2
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_01-08-07/TrainModel_e53b9_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-03_01-08-07/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_01-08-07/TrainModel_e53b9_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-03_01-08-07/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_01-08-07/TrainModel_e53b9_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-03_01-08-07/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_01-08-07/TrainModel_e53b9_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-03_01-08-07/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_01-08-07/TrainModel_e53b9_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-03_01-08-07/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_01-08-07/TrainModel_e53b9_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-03_01-08-07/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_01-08-07/TrainModel_e53b9_00006_6_FP_cyc_coeff=0.1,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-03_01-08-07/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_01-08-07/TrainModel_e53b9_00007_7_FP_cyc_coeff=0.3,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-03_01-08-07/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",

                # Only one episode per motion
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_16-59-19/TrainModel_c7552_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-03_16-59-20/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_16-59-19/TrainModel_c7552_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-03_16-59-20/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_16-59-19/TrainModel_c7552_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-03_16-59-20/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_16-59-19/TrainModel_c7552_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-03_16-59-20/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_16-59-19/TrainModel_c7552_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-03_16-59-20/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_16-59-19/TrainModel_c7552_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-03_16-59-20/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_16-59-19/TrainModel_c7552_00006_6_FP_cyc_coeff=0.1,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-03_16-59-20/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_16-59-19/TrainModel_c7552_00007_7_FP_cyc_coeff=0.3,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-03_16-59-20/checkpoint_001000/task_agnostic_policy_motor_decoder.pt",

                # use_a_gt=False
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_12-47-37/TrainModel_9ddaf_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-03_12-47-38/checkpoint_000900/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_12-47-37/TrainModel_9ddaf_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-03_12-47-38/checkpoint_000900/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_12-47-37/TrainModel_9ddaf_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-03_12-47-38/checkpoint_000900/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_12-47-37/TrainModel_9ddaf_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-03_12-47-38/checkpoint_000900/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_12-47-37/TrainModel_9ddaf_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-03_12-47-38/checkpoint_000900/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_12-47-37/TrainModel_9ddaf_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-03_12-47-38/checkpoint_000900/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_12-47-37/TrainModel_9ddaf_00006_6_FP_cyc_coeff=0.1,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-03_12-47-38/checkpoint_000900/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-03_12-47-37/TrainModel_9ddaf_00007_7_FP_cyc_coeff=0.3,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-03_12-47-38/checkpoint_000900/task_agnostic_policy_motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-10-04_02-39-46/TrainModel_dd861_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-04_02-39-46/checkpoint_000300/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_02-39-46/TrainModel_dd861_00001_1_FP_cyc_coeff=0.005,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-04_02-39-46/checkpoint_000300/task_agnostic_policy_motor_decoder.pt",

                # exp 0.01
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-13-47/TrainModel_95549_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-04_16-13-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-13-47/TrainModel_95549_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-04_16-13-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-13-47/TrainModel_95549_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-04_16-13-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-13-47/TrainModel_95549_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-04_16-13-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-13-47/TrainModel_95549_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-04_16-13-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-13-47/TrainModel_95549_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-04_16-13-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.1
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-09-12/TrainModel_f0e49_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-04_16-09-12/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-09-12/TrainModel_f0e49_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-04_16-09-12/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-09-12/TrainModel_f0e49_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-04_16-09-12/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-09-12/TrainModel_f0e49_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-04_16-09-12/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-09-12/TrainModel_f0e49_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-04_16-09-12/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-09-12/TrainModel_f0e49_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-04_16-09-12/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.05
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-15-17/TrainModel_ca800_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-04_16-15-17/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-15-17/TrainModel_ca800_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-04_16-15-17/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-15-17/TrainModel_ca800_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-04_16-15-17/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-15-17/TrainModel_ca800_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-04_16-15-17/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-15-17/TrainModel_ca800_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-04_16-15-17/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_16-15-17/TrainModel_ca800_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-04_16-15-17/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.01, a_gt_true
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-06-31/TrainModel_79f86_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-04_21-06-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-06-31/TrainModel_79f86_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-04_21-06-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-06-31/TrainModel_79f86_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-04_21-06-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-06-31/TrainModel_79f86_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-04_21-06-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-06-31/TrainModel_79f86_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-04_21-06-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-06-31/TrainModel_79f86_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-04_21-06-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.05, a_gt_true
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-09-13/TrainModel_da4be_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-04_21-09-13/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-09-13/TrainModel_da4be_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-04_21-09-13/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-09-13/TrainModel_da4be_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-04_21-09-13/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-09-13/TrainModel_da4be_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-04_21-09-13/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-09-13/TrainModel_da4be_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-04_21-09-13/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-09-13/TrainModel_da4be_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-04_21-09-13/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.1, a_gt_true
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-09-47/TrainModel_ef0f3_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-04_21-09-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-09-47/TrainModel_ef0f3_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-04_21-09-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-09-47/TrainModel_ef0f3_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-04_21-09-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-09-47/TrainModel_ef0f3_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-04_21-09-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-09-47/TrainModel_ef0f3_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-04_21-09-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_21-09-47/TrainModel_ef0f3_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-04_21-09-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.1, a_gt=true, gtPhySim, win=10, skip=0.5
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_20-47-24/TrainModel_ce7c4_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-04_20-47-24/checkpoint_000300/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_20-47-24/TrainModel_ce7c4_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-04_20-47-24/checkpoint_000300/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_20-47-24/TrainModel_ce7c4_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-04_20-47-24/checkpoint_000300/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_20-47-24/TrainModel_ce7c4_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-04_20-47-24/checkpoint_000300/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_20-47-24/TrainModel_ce7c4_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-04_20-47-24/checkpoint_000300/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-04_20-47-24/TrainModel_ce7c4_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-04_20-47-24/checkpoint_000300/task_agnostic_policy_motor_decoder.pt",

                # exp 0.05, a_gt=true, gtPhySim, win=5, skip=1.0
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_00-04-30/TrainModel_5775a_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-05_00-04-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_00-04-30/TrainModel_5775a_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-05_00-04-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_00-04-30/TrainModel_5775a_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-05_00-04-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_00-04-30/TrainModel_5775a_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-05_00-04-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_00-04-30/TrainModel_5775a_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-05_00-04-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_00-04-30/TrainModel_5775a_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-05_00-04-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.05, a_gt=false, gtPhySim, win=5, skip=1.0
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_00-04-30/TrainModel_5775a_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-05_00-04-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_00-04-30/TrainModel_5775a_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-05_00-04-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_00-04-30/TrainModel_5775a_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-05_00-04-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_00-04-30/TrainModel_5775a_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-05_00-04-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_00-04-30/TrainModel_5775a_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-05_00-04-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_00-04-30/TrainModel_5775a_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-05_00-04-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.05, a_gt=false, gtPhySim, win=5, skip=1.0, vae=0.1
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_01-24-42/TrainModel_8b1cb_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-05_01-24-42/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_01-24-42/TrainModel_8b1cb_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-05_01-24-42/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_01-24-42/TrainModel_8b1cb_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-05_01-24-42/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_01-24-42/TrainModel_8b1cb_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-05_01-24-42/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_01-24-42/TrainModel_8b1cb_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-05_01-24-42/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-05_01-24-42/TrainModel_8b1cb_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-05_01-24-42/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.1 a_gt=true, win=10, skip=0.5, vae=0.1, gamma 0.95->0.99
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-42-50/TrainModel_1018d_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-06_11-42-50/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-42-50/TrainModel_1018d_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-06_11-42-51/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-42-50/TrainModel_1018d_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_11-42-51/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-42-50/TrainModel_1018d_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_11-42-51/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-42-50/TrainModel_1018d_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-06_11-42-51/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-42-50/TrainModel_1018d_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-06_11-42-51/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.1 a_gt=true, win=10, skip=0.5, vae=1.0, gamma 0.95->0.99
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-42-50/TrainModel_1018d_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-06_11-42-51/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-42-50/TrainModel_1018d_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-06_11-42-51/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-42-50/TrainModel_1018d_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_11-42-51/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-42-50/TrainModel_1018d_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_11-42-51/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-42-50/TrainModel_1018d_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_11-42-51/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-42-50/TrainModel_1018d_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_11-42-51/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.1 a_gt=false, win=10, skip=0.5, vae=0.1, 1.0, gamma 0.95->0.99
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-10-33/TrainModel_8da34_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-06_11-10-34/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-10-33/TrainModel_8da34_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-06_11-10-34/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-10-33/TrainModel_8da34_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_11-10-34/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-10-33/TrainModel_8da34_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_11-10-34/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-10-33/TrainModel_8da34_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-06_11-10-34/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-10-33/TrainModel_8da34_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-06_11-10-34/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-10-33/TrainModel_8da34_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-06_11-10-34/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-10-33/TrainModel_8da34_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-06_11-10-34/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-10-33/TrainModel_8da34_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_11-10-34/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-10-33/TrainModel_8da34_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_11-10-34/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-10-33/TrainModel_8da34_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_11-10-34/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_11-10-33/TrainModel_8da34_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_11-10-34/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.1 a_gt=false, win=5, skip=1.0, vae=0.1, 1.0, gamma 0.95->0.99
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_10-40-47/TrainModel_652a8_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-06_10-40-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_10-40-47/TrainModel_652a8_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-06_10-40-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_10-40-47/TrainModel_652a8_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_10-40-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_10-40-47/TrainModel_652a8_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_10-40-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_10-40-47/TrainModel_652a8_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-06_10-40-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_10-40-47/TrainModel_652a8_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-06_10-40-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_10-40-47/TrainModel_652a8_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-06_10-40-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_10-40-47/TrainModel_652a8_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-06_10-40-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_10-40-47/TrainModel_652a8_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_10-40-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_10-40-47/TrainModel_652a8_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_10-40-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_10-40-47/TrainModel_652a8_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_10-40-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-06_10-40-47/TrainModel_652a8_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-06_10-40-48/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.1 a_gt=false, win=10, skip=0.5, vae=2.0, 5.0, gamma 0.95->0.99
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-51/TrainModel_53ccb_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-07_09-41-51/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-51/TrainModel_53ccb_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-07_09-41-51/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-51/TrainModel_53ccb_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_09-41-51/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-51/TrainModel_53ccb_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_09-41-52/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-51/TrainModel_53ccb_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-07_09-41-52/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-51/TrainModel_53ccb_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-07_09-41-52/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-51/TrainModel_53ccb_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-07_09-41-52/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-51/TrainModel_53ccb_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-07_09-41-52/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-51/TrainModel_53ccb_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_09-41-52/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-51/TrainModel_53ccb_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_09-41-52/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-51/TrainModel_53ccb_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_09-41-52/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-51/TrainModel_53ccb_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_09-41-52/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.1 a_gt=false, win=10, skip=0.5, vae=2.0, 5.0, gamma 0.95->0.99, MD_depth=3
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-37/TrainModel_4b587_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-07_09-41-37/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-37/TrainModel_4b587_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-07_09-41-37/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-37/TrainModel_4b587_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_09-41-37/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-37/TrainModel_4b587_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_09-41-37/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-37/TrainModel_4b587_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-07_09-41-37/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-37/TrainModel_4b587_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-07_09-41-37/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-37/TrainModel_4b587_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-07_09-41-37/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-37/TrainModel_4b587_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-07_09-41-37/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-37/TrainModel_4b587_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_09-41-37/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-37/TrainModel_4b587_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_09-41-37/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-37/TrainModel_4b587_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_09-41-37/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_09-41-37/TrainModel_4b587_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_09-41-37/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.1 a_gt=true, win=10, skip=0.5, vae=2.0, 5.0, gamma 0.98, MD_depth=2
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-02/TrainModel_8b8c7_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-07_22-15-02/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-02/TrainModel_8b8c7_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-07_22-15-02/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-02/TrainModel_8b8c7_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_22-15-02/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-02/TrainModel_8b8c7_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_22-15-02/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-02/TrainModel_8b8c7_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-07_22-15-02/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-02/TrainModel_8b8c7_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-07_22-15-02/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-02/TrainModel_8b8c7_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-07_22-15-02/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-02/TrainModel_8b8c7_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-07_22-15-02/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-02/TrainModel_8b8c7_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_22-15-02/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-02/TrainModel_8b8c7_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_22-15-02/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-02/TrainModel_8b8c7_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_22-15-02/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-02/TrainModel_8b8c7_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_22-15-02/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",

                # exp 0.1 a_gt=true, win=10, skip=0.5, vae=2.0, 5.0, gamma 0.995, MD_depth=2
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-38/TrainModel_a13e4_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-07_22-15-38/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-38/TrainModel_a13e4_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-07_22-15-39/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-38/TrainModel_a13e4_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_22-15-39/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-38/TrainModel_a13e4_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_22-15-39/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-38/TrainModel_a13e4_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-07_22-15-39/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-38/TrainModel_a13e4_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-07_22-15-39/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-38/TrainModel_a13e4_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-07_22-15-39/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-38/TrainModel_a13e4_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-07_22-15-39/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-38/TrainModel_a13e4_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_22-15-39/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-38/TrainModel_a13e4_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_22-15-39/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-38/TrainModel_a13e4_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_22-15-39/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-07_22-15-38/TrainModel_a13e4_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=2,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-07_22-15-39/checkpoint_000400/task_agnostic_policy_motor_decoder.pt",

                # exp 0.1 a_gt=true, win=10, skip=0.25, vae=2.0, 5.0, gamma 0.95, MD_depth=3
                # "/home/jungdam/ray_results/TrainModel_2021-10-08_02-59-25/TrainModel_45fa5_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-08_02-59-25/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-08_02-59-25/TrainModel_45fa5_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-08_02-59-25/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-08_02-59-25/TrainModel_45fa5_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-08_02-59-25/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-08_02-59-25/TrainModel_45fa5_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-08_02-59-25/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-08_02-59-25/TrainModel_45fa5_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-08_02-59-25/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-08_02-59-25/TrainModel_45fa5_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-08_02-59-25/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-08_02-59-25/TrainModel_45fa5_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-08_02-59-25/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-08_02-59-25/TrainModel_45fa5_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-08_02-59-25/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-08_02-59-25/TrainModel_45fa5_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-08_02-59-25/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-08_02-59-25/TrainModel_45fa5_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-08_02-59-25/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-08_02-59-25/TrainModel_45fa5_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-08_02-59-25/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-08_02-59-25/TrainModel_45fa5_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-08_02-59-25/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.1 a_gt=true,false win=10, skip=0.25, vae=0.1, 1.0, gamma 0.95, MD_depth=3
                # "/home/jungdam/ray_results/TrainModel_2021-10-09_14-49-35/TrainModel_a5e9c_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-09_14-49-35/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-09_14-49-35/TrainModel_a5e9c_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-09_14-49-35/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-09_14-49-35/TrainModel_a5e9c_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-09_14-49-35/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-09_14-49-35/TrainModel_a5e9c_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-09_14-49-35/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-09_14-49-35/TrainModel_a5e9c_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-09_14-49-35/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-09_14-49-35/TrainModel_a5e9c_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-09_14-49-35/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-09_14-49-35/TrainModel_a5e9c_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-09_14-49-35/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-09_14-49-35/TrainModel_a5e9c_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-09_14-49-35/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-09_14-49-35/TrainModel_a5e9c_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-09_14-49-35/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-09_14-49-35/TrainModel_a5e9c_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-09_14-49-35/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-09_14-49-35/TrainModel_a5e9c_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-09_14-49-35/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-09_14-49-35/TrainModel_a5e9c_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-09_14-49-35/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.1 a_gt=true, win=10, skip=0.25, vae=2.0, gamma 0.95, MD_depth=3
                # "/home/jungdam/ray_results/TrainModel_2021-10-10_17-54-50/TrainModel_b1a74_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-10_17-54-51/checkpoint_000500//task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-10_17-54-50/TrainModel_b1a74_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-10_17-54-51/checkpoint_000500//task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-10_17-54-50/TrainModel_b1a74_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-10_17-54-51/checkpoint_000500//task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-10_17-54-50/TrainModel_b1a74_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-10_17-54-51/checkpoint_000500//task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-10_17-54-50/TrainModel_b1a74_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-10_17-54-51/checkpoint_000500//task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-10_17-54-50/TrainModel_b1a74_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-10_17-54-51/checkpoint_000500//task_agnostic_policy_motor_decoder.pt",

                # exp 0.1 a_gt=true, win=5, skip=0.2, vae=2.0, gamma 0.95, MD_depth=3
                # "/home/jungdam/ray_results/TrainModel_2021-10-11_23-49-40/TrainModel_6d7c9_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-11_23-49-40/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-11_23-49-40/TrainModel_6d7c9_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-11_23-49-40/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-11_23-49-40/TrainModel_6d7c9_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-11_23-49-40/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-11_23-49-40/TrainModel_6d7c9_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-11_23-49-40/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-11_23-49-40/TrainModel_6d7c9_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-11_23-49-40/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-11_23-49-40/TrainModel_6d7c9_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-11_23-49-40/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # exp 0.1 a_gt=false, win=5, skip=0.2, vae=2.0, gamma 0.95, MD_depth=3
                # "/home/jungdam/ray_results/TrainModel_2021-10-11_23-49-40/TrainModel_6d7c9_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-11_23-49-40/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-11_23-49-40/TrainModel_6d7c9_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-11_23-49-40/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-11_23-49-40/TrainModel_6d7c9_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-11_23-49-40/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-11_23-49-40/TrainModel_6d7c9_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-11_23-49-40/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-11_23-49-40/TrainModel_6d7c9_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-11_23-49-40/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-11_23-49-40/TrainModel_6d7c9_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-11_23-49-40/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.1 a_gt=true, win=5, skip=0.1, vae=2.0, gamma 0.95, MD_depth=3
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_10-15-10/TrainModel_cf17c_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-12_10-15-10/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_10-15-10/TrainModel_cf17c_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-12_10-15-10/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_10-15-10/TrainModel_cf17c_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-12_10-15-10/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_10-15-10/TrainModel_cf17c_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-12_10-15-10/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_10-15-10/TrainModel_cf17c_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-12_10-15-10/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_10-15-10/TrainModel_cf17c_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-12_10-15-10/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # exp 0.1 a_gt=false, win=5, skip=0.1, vae=2.0, gamma 0.95, MD_depth=3
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_10-15-10/TrainModel_cf17c_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-12_10-15-10/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_10-15-10/TrainModel_cf17c_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-12_10-15-10/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_10-15-10/TrainModel_cf17c_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-12_10-15-10/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_10-15-10/TrainModel_cf17c_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-12_10-15-10/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_10-15-10/TrainModel_cf17c_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-12_10-15-10/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_10-15-10/TrainModel_cf17c_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-12_10-15-10/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # iter=20, exp 0.1 a_gt=true, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_14-59-00/TrainModel_75bfd_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-12_14-59-00/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_14-59-00/TrainModel_75bfd_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-12_14-59-00/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_14-59-00/TrainModel_75bfd_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-12_14-59-00/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_14-59-00/TrainModel_75bfd_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-12_14-59-00/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_14-59-00/TrainModel_75bfd_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-12_14-59-00/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_14-59-00/TrainModel_75bfd_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-12_14-59-00/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # iter=20, exp 0.1 a_gt=false, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-24-23/TrainModel_ea8e0_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-12_20-24-23/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-24-23/TrainModel_ea8e0_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-12_20-24-23/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-24-23/TrainModel_ea8e0_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-12_20-24-23/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-24-23/TrainModel_ea8e0_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-12_20-24-23/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-24-23/TrainModel_ea8e0_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-12_20-24-23/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-24-23/TrainModel_ea8e0_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-12_20-24-23/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # exp 0.05 a_gt=true, win=5, skip=0.1, vae=2.0, gamma 0.95, MD_depth=3
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-44-47/TrainModel_c3e87_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-12_20-44-47/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-44-47/TrainModel_c3e87_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-12_20-44-47/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-44-47/TrainModel_c3e87_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-12_20-44-47/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-44-47/TrainModel_c3e87_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-12_20-44-47/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-44-47/TrainModel_c3e87_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-12_20-44-47/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-44-47/TrainModel_c3e87_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-12_20-44-47/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # exp 0.05 a_gt=false, win=5, skip=0.1, vae=2.0, gamma 0.95, MD_depth=3
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-44-47/TrainModel_c3e87_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-12_20-44-47/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-44-47/TrainModel_c3e87_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-12_20-44-47/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-44-47/TrainModel_c3e87_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-12_20-44-47/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-44-47/TrainModel_c3e87_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-12_20-44-47/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-44-47/TrainModel_c3e87_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-12_20-44-47/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-12_20-44-47/TrainModel_c3e87_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-12_20-44-47/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # iter=50, exp 0.05 a_gt=true, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=pseudo
                # "/home/jungdam/ray_results/TrainModel_2021-10-13_08-36-41/TrainModel_37883_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-13_08-36-41/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-13_08-36-41/TrainModel_37883_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-13_08-36-41/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-13_08-36-41/TrainModel_37883_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-13_08-36-41/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-13_08-36-41/TrainModel_37883_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-13_08-36-41/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-13_08-36-41/TrainModel_37883_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-13_08-36-41/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-13_08-36-41/TrainModel_37883_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-13_08-36-41/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # iter=50, exp 0.05 a_gt=false, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=pseudo
                # "/home/jungdam/ray_results/TrainModel_2021-10-13_08-36-41/TrainModel_37883_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-13_08-36-41/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-13_08-36-41/TrainModel_37883_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-13_08-36-41/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-13_08-36-41/TrainModel_37883_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-13_08-36-41/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-13_08-36-41/TrainModel_37883_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-13_08-36-41/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-13_08-36-41/TrainModel_37883_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-13_08-36-41/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-13_08-36-41/TrainModel_37883_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-13_08-36-41/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # iter=50, exp 0.05 a_gt=true, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=real
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_09-42-46/TrainModel_9d749_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-14_09-42-46/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_09-42-46/TrainModel_9d749_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-14_09-42-46/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_09-42-46/TrainModel_9d749_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-14_09-42-46/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_09-42-46/TrainModel_9d749_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-14_09-42-46/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_09-42-46/TrainModel_9d749_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-14_09-42-46/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_09-42-46/TrainModel_9d749_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-14_09-42-46/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # iter=50, exp 0.05 a_gt=false, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=real
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_09-42-46/TrainModel_9d749_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-14_09-42-46/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_09-42-46/TrainModel_9d749_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-14_09-42-46/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_09-42-46/TrainModel_9d749_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-14_09-42-46/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_09-42-46/TrainModel_9d749_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-14_09-42-46/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_09-42-46/TrainModel_9d749_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-14_09-42-46/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_09-42-46/TrainModel_9d749_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-14_09-42-46/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # iter=1, exp 0.02 a_gt=false, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=real, lookahead=1
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_20-03-31/TrainModel_55359_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-14_20-03-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_20-03-31/TrainModel_55359_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-14_20-03-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_20-03-31/TrainModel_55359_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-14_20-03-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_20-03-31/TrainModel_55359_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-14_20-03-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_20-03-31/TrainModel_55359_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-14_20-03-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_20-03-31/TrainModel_55359_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-14_20-03-31/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # iter=1, exp 0.02 a_gt=false, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=real, lookahead=4
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-06-12/TrainModel_16f90_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-14_21-06-12/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-06-12/TrainModel_16f90_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-14_21-06-12/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-06-12/TrainModel_16f90_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-14_21-06-12/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-06-12/TrainModel_16f90_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-14_21-06-12/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-06-12/TrainModel_16f90_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-14_21-06-12/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-06-12/TrainModel_16f90_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-14_21-06-12/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # # iter=100, exp 0.05 a_gt=true, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=real
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_14-42-42/TrainModel_d36cc_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-15_14-42-42/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_14-42-42/TrainModel_d36cc_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-15_14-42-42/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_14-42-42/TrainModel_d36cc_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_14-42-42/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_14-42-42/TrainModel_d36cc_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_14-42-42/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_14-42-42/TrainModel_d36cc_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-15_14-42-42/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_14-42-42/TrainModel_d36cc_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-15_14-42-42/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # # iter=100, exp 0.05 a_gt=false, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=real
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_14-42-42/TrainModel_d36cc_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-15_14-42-42/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_14-42-42/TrainModel_d36cc_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-15_14-42-42/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_14-42-42/TrainModel_d36cc_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_14-42-42/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_14-42-42/TrainModel_d36cc_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_14-42-43/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_14-42-42/TrainModel_d36cc_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_14-42-43/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_14-42-42/TrainModel_d36cc_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_14-42-43/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # # iter=100, exp 0.05 a_gt=true, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=real
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_22-50-35/TrainModel_2619d_00000_0_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_22-50-35/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_22-50-35/TrainModel_2619d_00002_2_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_22-50-36/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_22-50-35/TrainModel_2619d_00004_4_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_22-50-36/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_22-50-35/TrainModel_2619d_00006_6_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_22-50-36/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_22-50-35/TrainModel_2619d_00008_8_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_22-50-36/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_22-50-35/TrainModel_2619d_00010_10_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-16_22-50-36/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_22-50-35/TrainModel_2619d_00012_12_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-16_22-50-36/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_22-50-35/TrainModel_2619d_00014_14_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-16_22-50-36/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # iter=100, exp 0.05 a_gt=false, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=real
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_22-50-35/TrainModel_2619d_00001_1_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_22-50-36/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_22-50-35/TrainModel_2619d_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_22-50-36/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_22-50-35/TrainModel_2619d_00005_5_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_22-50-36/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_22-50-35/TrainModel_2619d_00007_7_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_22-50-36/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_22-50-35/TrainModel_2619d_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_22-50-36/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_22-50-35/TrainModel_2619d_00011_11_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-16_22-50-36/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_22-50-35/TrainModel_2619d_00013_13_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-16_22-50-36/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_22-50-35/TrainModel_2619d_00015_15_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-16_22-50-36/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # PFNN

                # iter=1, exp 0.05 a_gt=true, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=real, lookahead=1
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-21-18/TrainModel_329ce_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-14_21-21-18/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-21-18/TrainModel_329ce_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-14_21-21-18/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-21-18/TrainModel_329ce_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-14_21-21-18/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-21-18/TrainModel_329ce_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-14_21-21-18/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-21-18/TrainModel_329ce_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-14_21-21-18/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-21-18/TrainModel_329ce_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-14_21-21-18/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # iter=1, exp 0.05 a_gt=false, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=real, lookahead=1
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-21-18/TrainModel_329ce_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-14_21-21-18/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-21-18/TrainModel_329ce_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-14_21-21-18/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-21-18/TrainModel_329ce_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-14_21-21-18/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-21-18/TrainModel_329ce_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-14_21-21-18/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-21-18/TrainModel_329ce_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-14_21-21-18/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-14_21-21-18/TrainModel_329ce_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-14_21-21-18/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # # iter=5, exp 0.05 a_gt=true, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=real, lookahead=1
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_01-43-01/TrainModel_c2816_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-15_01-43-01/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_01-43-01/TrainModel_c2816_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-15_01-43-01/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_01-43-01/TrainModel_c2816_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_01-43-01/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_01-43-01/TrainModel_c2816_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_01-43-01/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_01-43-01/TrainModel_c2816_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-15_01-43-01/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_01-43-01/TrainModel_c2816_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-15_01-43-01/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # # iter=5, exp 0.05 a_gt=false, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=real, lookahead=1
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_01-43-01/TrainModel_c2816_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-15_01-43-01/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_01-43-01/TrainModel_c2816_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-15_01-43-01/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_01-43-01/TrainModel_c2816_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_01-43-01/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_01-43-01/TrainModel_c2816_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_01-43-01/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_01-43-01/TrainModel_c2816_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_01-43-01/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_01-43-01/TrainModel_c2816_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_01-43-01/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # # iter=10, exp 0.05 a_gt=true, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=real, lookahead=1
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_11-10-58/TrainModel_19dbe_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-15_11-10-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_11-10-58/TrainModel_19dbe_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-15_11-10-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_11-10-58/TrainModel_19dbe_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_11-10-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_11-10-58/TrainModel_19dbe_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_11-10-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_11-10-58/TrainModel_19dbe_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-15_11-10-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_11-10-58/TrainModel_19dbe_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-10-15_11-10-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # # iter=10, exp 0.05 a_gt=false, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=real, lookahead=1
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_11-10-58/TrainModel_19dbe_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-10-15_11-10-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_11-10-58/TrainModel_19dbe_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-15_11-10-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_11-10-58/TrainModel_19dbe_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_11-10-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_11-10-58/TrainModel_19dbe_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_11-10-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_11-10-58/TrainModel_19dbe_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_11-10-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-15_11-10-58/TrainModel_19dbe_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-15_11-10-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",

                # iter=5, exp 0.05 a_gt=true, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=real, lookahead=1
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00000_0_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00001_1_FP_cyc_coeff=0.002,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00002_2_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00003_3_FP_cyc_coeff=0.004,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00004_4_FP_cyc_coeff=0.005,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00005_5_FP_cyc_coeff=0.006,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00005_5_FP_cyc_coeff=0.006,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00006_6_FP_cyc_coeff=0.007,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00007_7_FP_cyc_coeff=0.008,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00008_8_FP_cyc_coeff=0.009,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # iter=5, exp 0.05 a_gt=false, win=1000, skip=1000, vae=2.0, gamma 0.95, MD_depth=3, FP=real, lookahead=1
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00009_9_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00010_10_FP_cyc_coeff=0.002,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00011_11_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00012_12_FP_cyc_coeff=0.004,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00013_13_FP_cyc_coeff=0.005,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00014_14_FP_cyc_coeff=0.006,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00015_15_FP_cyc_coeff=0.007,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00016_16_FP_cyc_coeff=0.008,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-10-16_01-46-58/TrainModel_7a224_00017_17_FP_cyc_coeff=0.009,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-10-16_01-46-58/checkpoint_000500/task_agnostic_policy_motor_decoder.pt",


                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-19_16-20-48/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-19_16-20-56/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_16-21-03/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_16-21-11/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-19_16-21-20/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-19_16-21-28/checkpoint_000500/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-19_16-21-36/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-19_16-21-44/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_16-21-52/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_16-22-00/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_16-22-08/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_16-22-16/checkpoint_000500/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00012_12_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-19_16-22-24/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00013_13_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-19_16-22-33/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00014_14_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-19_16-22-41/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00015_15_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-19_16-22-50/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00016_16_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_16-22-58/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00017_17_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_16-23-06/checkpoint_000500/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00018_18_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-19_16-23-15/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00019_19_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-19_16-23-23/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00020_20_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-19_16-23-33/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00021_21_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-19_16-23-41/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00022_22_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_16-23-49/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00023_23_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_16-23-59/checkpoint_000500/motor_decoder.pt",

                ## PFNN latent dim test
                # "/home/jungdam/ray_results/TrainModel_2022-01-25_18-27-55/TrainModel_91cb2_00000_0_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-25_18-27-55/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-25_18-27-55/TrainModel_91cb2_00001_1_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-25_18-28-11/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-25_18-27-55/TrainModel_91cb2_00002_2_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-25_18-28-28/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-25_18-27-55/TrainModel_91cb2_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-25_18-28-43/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-25_18-27-55/TrainModel_91cb2_00004_4_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-25_18-28-58/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-25_18-27-55/TrainModel_91cb2_00005_5_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-25_18-29-17/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-25_18-27-55/TrainModel_91cb2_00006_6_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-25_18-29-33/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-25_18-27-55/TrainModel_91cb2_00007_7_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-25_18-29-47/checkpoint_000500/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2022-01-25_18-38-23/TrainModel_07e23_00000_0_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-25_18-38-23/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-25_18-38-23/TrainModel_07e23_00001_1_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-25_18-38-30/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-25_18-38-23/TrainModel_07e23_00002_2_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-25_18-38-38/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-25_18-38-23/TrainModel_07e23_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-25_18-38-45/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-25_18-38-23/TrainModel_07e23_00004_4_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-25_18-38-52/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-25_18-38-23/TrainModel_07e23_00005_5_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-25_18-39-00/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-25_18-38-23/TrainModel_07e23_00006_6_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-25_18-39-07/checkpoint_000500/motor_decoder.pt",
                "/home/jungdam/ray_results/TrainModel_2022-01-25_18-38-23/TrainModel_07e23_00007_7_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-25_18-39-14/checkpoint_000500/motor_decoder.pt",

                # Boxing, No Variation
                # "/home/jungdam/ray_results/TrainModel_2022-01-02_19-19-20/TrainModel_f0f95_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2022-01-02_19-19-20/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-02_19-19-20/TrainModel_f0f95_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2022-01-02_19-19-27/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-02_19-19-20/TrainModel_f0f95_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-02_19-19-35/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-02_19-19-20/TrainModel_f0f95_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-02_19-19-42/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-02_19-19-20/TrainModel_f0f95_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2022-01-02_19-19-48/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-02_19-19-20/TrainModel_f0f95_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2022-01-02_19-19-55/checkpoint_000500/motor_decoder.pt",

                # AIST exp03

                # AIST exp05

                # "/home/jungdam/ray_results/TrainModel_2021-12-17_02-40-25/TrainModel_993be_00018_18_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-17_02-40-26/checkpoint_001000/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-17_02-40-25/TrainModel_993be_00019_19_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-17_02-40-26/checkpoint_001000/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-17_02-40-25/TrainModel_993be_00020_20_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-17_02-40-26/checkpoint_001000/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-17_02-40-25/TrainModel_993be_00021_21_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-17_02-40-26/checkpoint_001000/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-17_02-40-25/TrainModel_993be_00022_22_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-17_02-40-26/checkpoint_001000/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-17_02-40-25/TrainModel_993be_00023_23_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-17_02-40-26/checkpoint_001000/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-17_19-00-26/TrainModel_a67e7_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-17_19-00-26/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-17_19-00-26/TrainModel_a67e7_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-17_19-00-39/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-17_19-00-26/TrainModel_a67e7_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-17_19-00-52/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-17_19-00-26/TrainModel_a67e7_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-17_19-01-07/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-17_19-00-26/TrainModel_a67e7_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-17_19-01-21/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-17_19-00-26/TrainModel_a67e7_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-17_19-01-34/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-17_19-00-26/TrainModel_a67e7_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-17_19-01-47/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-17_19-00-26/TrainModel_a67e7_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-17_19-02-00/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-17_19-00-26/TrainModel_a67e7_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-17_19-02-14/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-17_19-00-26/TrainModel_a67e7_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-17_19-02-29/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-17_19-00-26/TrainModel_a67e7_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-17_19-02-43/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-17_19-00-26/TrainModel_a67e7_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-17_19-02-57/checkpoint_000500/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-18_23-01-50/TrainModel_6503e_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-18_23-01-51/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_23-01-50/TrainModel_6503e_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-18_23-01-51/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_23-01-50/TrainModel_6503e_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-18_23-01-51/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_23-01-50/TrainModel_6503e_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-18_23-01-51/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_23-01-50/TrainModel_6503e_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-18_23-01-51/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_23-01-50/TrainModel_6503e_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-18_23-01-51/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_23-01-50/TrainModel_6503e_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-18_23-01-51/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_23-01-50/TrainModel_6503e_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-18_23-01-51/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_23-01-50/TrainModel_6503e_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-18_23-01-51/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_23-01-50/TrainModel_6503e_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-18_23-01-51/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_23-01-50/TrainModel_6503e_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-18_23-01-51/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_23-01-50/TrainModel_6503e_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-18_23-01-51/checkpoint_000500/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00042_42_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-18_22-44-11/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00043_43_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-18_22-44-26/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00044_44_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-18_22-44-41/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00045_45_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-18_22-44-56/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00046_46_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-18_22-45-14/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00047_47_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-18_22-45-30/checkpoint_000300/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00024_24_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-18_22-39-22/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00025_25_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-18_22-39-39/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00026_26_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-18_22-39-54/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00027_27_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-18_22-40-08/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00028_28_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-18_22-40-25/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00029_29_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-18_22-40-39/checkpoint_000300/motor_decoder.pt",

                # # vae_z_coeff 0.2, a_gt false
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00030_30_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-18_22-40-58/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00031_31_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-18_22-41-12/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00032_32_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-18_22-41-28/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00033_33_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-18_22-41-47/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00034_34_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-18_22-42-06/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00035_35_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-18_22-42-21/checkpoint_000500/motor_decoder.pt",
                # # vae_z_coeff 0.02, a_gt false
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00018_18_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-18_22-37-50/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00019_19_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-18_22-38-06/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00020_20_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-18_22-38-21/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00021_21_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-18_22-38-37/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00022_22_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-18_22-38-52/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-18_22-33-36/TrainModel_983f9_00023_23_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-18_22-39-07/checkpoint_000500/motor_decoder.pt",

                # # AIST exp03
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-28-55/TrainModel_24a76_00018_18_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-19_09-31-32/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-28-55/TrainModel_24a76_00019_19_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-19_09-31-41/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-28-55/TrainModel_24a76_00020_20_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-19_09-31-51/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-28-55/TrainModel_24a76_00021_21_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-19_09-31-59/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-28-55/TrainModel_24a76_00022_22_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_09-32-08/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-28-55/TrainModel_24a76_00023_23_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_09-32-17/checkpoint_000500/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-28-55/TrainModel_24a76_00030_30_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-19_09-33-19/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-28-55/TrainModel_24a76_00031_31_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-19_09-33-27/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-28-55/TrainModel_24a76_00032_32_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-19_09-33-37/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-28-55/TrainModel_24a76_00033_33_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-19_09-33-45/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-28-55/TrainModel_24a76_00034_34_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_09-33-55/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-28-55/TrainModel_24a76_00035_35_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_09-34-05/checkpoint_000500/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-24_08-33-11/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-33-17/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-33-23/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-33-28/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-24_08-33-34/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-24_08-33-39/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-24_08-33-45/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-33-52/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-33-58/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-34-03/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-34-09/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-34-14/checkpoint_000500/motor_decoder.pt",
                
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00012_12_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-24_08-34-21/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00013_13_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-24_08-34-27/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00014_14_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-34-32/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00015_15_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-34-38/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00016_16_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-34-44/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00017_17_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-34-49/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00018_18_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-24_08-34-56/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00019_19_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-24_08-35-02/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00020_20_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-35-07/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00021_21_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-35-13/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00022_22_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-35-19/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-33-11/TrainModel_2f172_00023_23_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-35-25/checkpoint_000500/motor_decoder.pt",

                # AIST exp04
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-30-05/TrainModel_4e0a1_00018_18_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-19_09-32-42/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-30-05/TrainModel_4e0a1_00019_19_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-19_09-32-50/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-30-05/TrainModel_4e0a1_00020_20_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-19_09-32-58/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-30-05/TrainModel_4e0a1_00021_21_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-19_09-33-07/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-30-05/TrainModel_4e0a1_00022_22_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_09-33-16/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-30-05/TrainModel_4e0a1_00023_23_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_09-33-26/checkpoint_000500/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-30-05/TrainModel_4e0a1_00030_30_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-19_09-34-29/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-30-05/TrainModel_4e0a1_00031_31_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-19_09-34-39/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-30-05/TrainModel_4e0a1_00032_32_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-19_09-34-48/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-30-05/TrainModel_4e0a1_00033_33_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-19_09-34-56/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-30-05/TrainModel_4e0a1_00034_34_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_09-35-06/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_09-30-05/TrainModel_4e0a1_00035_35_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-19_09-35-14/checkpoint_000500/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-24_08-34-12/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-34-18/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-34-23/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-34-29/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-24_08-34-35/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-24_08-34-40/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-24_08-34-47/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-34-53/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-34-59/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-35-04/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-35-11/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-35-18/checkpoint_000500/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00012_12_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-24_08-35-24/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00013_13_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-24_08-35-29/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00014_14_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-35-35/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00015_15_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-35-41/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00016_16_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-35-48/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00017_17_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-35-54/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00018_18_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-24_08-35-59/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00019_19_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-24_08-36-05/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00020_20_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-36-12/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00021_21_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-36-18/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00022_22_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-36-24/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-12/TrainModel_53572_00023_23_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-36-30/checkpoint_000500/motor_decoder.pt",

                # # AIST exp05
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-24_08-34-59/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-35-04/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-35-10/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-35-15/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-24_08-35-20/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-24_08-35-20/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-24_08-35-20/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-35-26/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-35-31/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-35-36/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-35-42/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-35-47/checkpoint_000500/motor_decoder.pt",
                
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00012_12_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-24_08-35-52/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00013_13_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-24_08-35-58/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00014_14_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-36-03/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00015_15_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-36-08/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00016_16_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-36-13/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00017_17_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-36-19/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00018_18_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-24_08-36-24/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00019_19_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-24_08-36-29/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00020_20_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-36-34/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00021_21_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_08-36-34/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00022_22_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-36-34/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_08-34-59/TrainModel_6f8ac_00023_23_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_08-36-41/checkpoint_000500/motor_decoder.pt",

                ## AIST 3+4+5
                # "/home/jungdam/ray_results/TrainModel_2022-01-21_20-41-50/TrainModel_9d1b0_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2022-01-21_20-41-50/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-21_20-41-50/TrainModel_9d1b0_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2022-01-21_20-42-08/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-21_20-41-50/TrainModel_9d1b0_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-21_20-42-27/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-21_20-41-50/TrainModel_9d1b0_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-21_20-42-47/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-21_20-41-50/TrainModel_9d1b0_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2022-01-21_20-43-06/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-21_20-41-50/TrainModel_9d1b0_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2022-01-21_20-43-26/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-21_20-41-50/TrainModel_9d1b0_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2022-01-21_20-43-46/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-21_20-41-50/TrainModel_9d1b0_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2022-01-21_20-44-07/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-21_20-41-50/TrainModel_9d1b0_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-21_20-44-28/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-21_20-41-50/TrainModel_9d1b0_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-21_20-44-49/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-21_20-41-50/TrainModel_9d1b0_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-21_20-45-10/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-21_20-41-50/TrainModel_9d1b0_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-21_20-45-31/checkpoint_000600/motor_decoder.pt",


                # Boxing
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-24_20-45-14/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_20-45-19/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_20-45-26/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_20-45-31/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-24_20-45-38/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-24_20-45-43/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-24_20-45-49/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_20-45-57/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_20-46-02/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_20-46-09/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_20-46-15/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_20-46-21/checkpoint_000500/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00012_12_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-24_20-46-28/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00013_13_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-24_20-46-34/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00014_14_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_20-46-40/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00015_15_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_20-46-46/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00016_16_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_20-46-52/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00017_17_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_20-46-58/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00018_18_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-24_20-47-04/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00019_19_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff_2021-12-24_20-47-10/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00020_20_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_20-47-16/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00021_21_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-24_20-47-23/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00022_22_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_20-47-29/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-24_20-45-13/TrainModel_72fe0_00023_23_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-24_20-47-36/checkpoint_000500/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-28_18-43-49/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-28_18-44-00/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-44-13/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-44-23/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-28_18-44-33/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-28_18-44-47/checkpoint_000500/motor_decoder.pt",
                
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-28_18-45-04/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-28_18-45-17/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-45-29/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-45-42/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-45-54/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-46-08/checkpoint_000500/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-28_18-43-49/checkpoint_000800/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-28_18-44-00/checkpoint_000800/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-44-13/checkpoint_000800/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-44-23/checkpoint_000800/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-28_18-44-33/checkpoint_000800/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-28_18-44-47/checkpoint_000800/motor_decoder.pt",
                
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-28_18-45-04/checkpoint_000800/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-28_18-45-17/checkpoint_000800/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-45-29/checkpoint_000800/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-45-42/checkpoint_000800/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-45-54/checkpoint_000800/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-46-08/checkpoint_000800/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-28_18-45-04/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-28_18-45-17/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-45-29/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-45-42/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-45-54/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-46-08/checkpoint_000300/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-28_18-45-04/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-28_18-45-17/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-45-29/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-45-42/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-45-54/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-46-08/checkpoint_000600/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-28_18-45-04/checkpoint_001000/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-28_18-45-17/checkpoint_001000/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-45-29/checkpoint_001000/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-45-42/checkpoint_001000/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-45-54/checkpoint_001000/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-28_18-43-49/TrainModel_26aec_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-28_18-46-08/checkpoint_001000/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-29_14-35-08/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-29_14-35-20/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-35-32/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-35-46/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-29_14-35-59/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-29_14-36-13/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-29_14-36-29/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-29_14-36-42/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-36-56/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-37-10/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-37-22/checkpoint_000300/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-37-35/checkpoint_000300/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-29_14-35-08/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-29_14-35-20/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-35-32/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-35-46/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-29_14-35-59/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-29_14-36-13/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-29_14-36-29/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-29_14-36-42/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-36-56/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-37-10/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-37-22/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-37-35/checkpoint_000600/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-29_14-35-08/checkpoint_000900/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-29_14-35-20/checkpoint_000900/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-35-32/checkpoint_000900/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-35-46/checkpoint_000900/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-29_14-35-59/checkpoint_000900/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-29_14-36-13/checkpoint_000900/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2021-12-29_14-36-29/checkpoint_000900/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-29_14-36-42/checkpoint_000900/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-36-56/checkpoint_000900/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-37-10/checkpoint_000900/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-37-22/checkpoint_000900/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-29_14-35-08/TrainModel_939d6_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2021-12-29_14-37-35/checkpoint_000900/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00018_18_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2021-12-19_16-23-15/checkpoint_000500/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2021-12-19_16-20-47/TrainModel_ae165_00021_21_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2021-12-19_16-23-41/checkpoint_000500/motor_decoder.pt",



                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-57-30/TrainModel_31335_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2022-01-07_14-57-30/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-57-30/TrainModel_31335_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2022-01-07_14-57-43/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-57-30/TrainModel_31335_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-07_14-57-55/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-57-30/TrainModel_31335_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-07_14-58-08/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-57-30/TrainModel_31335_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2022-01-07_14-58-21/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-57-30/TrainModel_31335_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2022-01-07_14-58-34/checkpoint_000600/motor_decoder.pt",
                
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-57-30/TrainModel_31335_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2022-01-07_14-58-47/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-57-30/TrainModel_31335_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2022-01-07_14-59-01/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-57-30/TrainModel_31335_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-07_14-59-14/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-57-30/TrainModel_31335_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-07_14-59-27/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-57-30/TrainModel_31335_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-07_14-59-42/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-57-30/TrainModel_31335_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-07_14-59-56/checkpoint_000600/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-59-15/TrainModel_6f95d_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2022-01-07_14-59-15/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-59-15/TrainModel_6f95d_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2022-01-07_14-59-27/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-59-15/TrainModel_6f95d_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-07_14-59-41/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-59-15/TrainModel_6f95d_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-07_14-59-54/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-59-15/TrainModel_6f95d_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2022-01-07_15-00-08/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-59-15/TrainModel_6f95d_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2022-01-07_15-00-22/checkpoint_000600/motor_decoder.pt",
                
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-59-15/TrainModel_6f95d_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2022-01-07_15-00-35/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-59-15/TrainModel_6f95d_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2022-01-07_15-00-47/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-59-15/TrainModel_6f95d_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-07_15-01-02/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-59-15/TrainModel_6f95d_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-07_15-01-15/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-59-15/TrainModel_6f95d_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-07_15-01-29/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-07_14-59-15/TrainModel_6f95d_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-07_15-01-43/checkpoint_000600/motor_decoder.pt",

                # PFNN3

                # 0.02

                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-43-51/TrainModel_e1bbf_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2022-01-14_23-43-51/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-43-51/TrainModel_e1bbf_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2022-01-14_23-44-04/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-43-51/TrainModel_e1bbf_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-14_23-44-18/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-43-51/TrainModel_e1bbf_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-14_23-44-32/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-43-51/TrainModel_e1bbf_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2022-01-14_23-44-45/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-43-51/TrainModel_e1bbf_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2022-01-14_23-44-58/checkpoint_000600/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-43-51/TrainModel_e1bbf_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2022-01-14_23-45-13/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-43-51/TrainModel_e1bbf_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2022-01-14_23-45-28/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-43-51/TrainModel_e1bbf_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-14_23-45-42/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-43-51/TrainModel_e1bbf_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-14_23-45-56/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-43-51/TrainModel_e1bbf_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-14_23-46-12/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-43-51/TrainModel_e1bbf_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-14_23-46-26/checkpoint_000600/motor_decoder.pt",

                # 0.05

                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-45-06/TrainModel_0eb5f_00000_0_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2022-01-14_23-45-06/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-45-06/TrainModel_0eb5f_00001_1_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2022-01-14_23-45-20/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-45-06/TrainModel_0eb5f_00002_2_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-14_23-45-33/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-45-06/TrainModel_0eb5f_00003_3_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-14_23-45-47/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-45-06/TrainModel_0eb5f_00004_4_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2022-01-14_23-46-00/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-45-06/TrainModel_0eb5f_00005_5_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1._2022-01-14_23-46-14/checkpoint_000600/motor_decoder.pt",

                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-45-06/TrainModel_0eb5f_00006_6_FP_cyc_coeff=0.0,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1.0_2022-01-14_23-46-28/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-45-06/TrainModel_0eb5f_00007_7_FP_cyc_coeff=0.0003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=_2022-01-14_23-46-42/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-45-06/TrainModel_0eb5f_00008_8_FP_cyc_coeff=0.001,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-14_23-46-55/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-45-06/TrainModel_0eb5f_00009_9_FP_cyc_coeff=0.003,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-14_23-47-10/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-45-06/TrainModel_0eb5f_00010_10_FP_cyc_coeff=0.01,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-14_23-47-24/checkpoint_000600/motor_decoder.pt",
                # "/home/jungdam/ray_results/TrainModel_2022-01-14_23-45-06/TrainModel_0eb5f_00011_11_FP_cyc_coeff=0.03,FP_depth=2,FP_width=1024,MD_depth=3,MD_type=mlp,MD_width=512,TE_cond=abs,a_rec_coeff=1_2022-01-14_23-47-41/checkpoint_000600/motor_decoder.pt",
            ]   
            print("----------------------------")
            print("latent_random_sample:", self.latent_random_sample_methods[self.latent_random_sample])
            print("exploration:", self.explore)
            print("----------------------------")

            with open(result_file, 'w') as f:
                f.write("------------------VAE MD Performace Test------------------\n")

                if run_every_motion:
                    ref_motions = np.arange(len(self.env.base_env._ref_motion_all[0]))
                else:
                    ref_motions = [0]

                ''' Use fixed start-time value for comparison '''
                start_times_all = []
                for i in range(len(ref_motions)):
                    if reset_at_zero:
                        start_times = np.zeros(num_iter)
                    else:
                        self.reset({'ref_motion_id': [0]})
                        start_times = np.linspace(
                            start=0.0,
                            stop=self.env.base_env._ref_motion[i].length(),
                            num=num_iter)
                        print("start_times:", start_times)
                    start_times_all.append(start_times)

                f.writelines([md+"\n" for md in motor_decoders])

                for k, md in enumerate(motor_decoders):
                    if md:
                        model = self.trainer.get_policy().model
                        model.load_weights_motor_decoder(md)
                    print("Loaded:", md)
                    time_elapsed = []
                    for i in ref_motions:
                        if verbose:
                            print('ref_motion[%d]'%(i))
                        for j in range(num_iter):
                            save_motion_name = "motion_%d_%d_%d.bvh" % (k, i, j)
                            self.reset({
                                'ref_motion_id':[i], 
                                'start_time': [start_times_all[i][j]]})
                            time_elapsed.append(self.record_a_scene(
                                save_dir=save_motion_dir, 
                                save_image=False,
                                save_motion=save_motion,
                                save_motion_name=save_motion_name,
                                end_time=end_time, 
                                check_falldown=True, 
                                check_end_of_motion=False, 
                                verbose=verbose))
                            # print(time_elapsed)
                    msg = "%4.4f\t%4.4f" % (np.mean(time_elapsed), np.std(time_elapsed))
                    print(msg)
                    f.write("%s\n"%msg)
                    f.flush()
                f.close()
            print("----------------------------")
        elif key == b'l':
            file = input("Enter Checkpoint:")
            if os.path.isfile(file):
                model = self.trainer.get_policy().model
                model.load_weights_motor_decoder(file)
                print(file)
        elif key == b'L':
            model = self.trainer.get_policy().model
            is_task_agnostic_policy = hasattr(model, 'task_encoder_variable')
            print('----------------------------')
            print('Extracting State-Action Pairs')
            print('----------------------------')
            ''' Read a directory for saving images and try to create it '''
            output_dir = input("Enter output dir: ")
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError:
                print("Invalid Subdirectory")
                return
            iter_per_episode = get_int_from_input("Iteration Per Episode")
            window_size = get_float_from_input("Window Size (sec)")
            stride = get_float_from_input("Stride (sec)")
            state_type = input("State Type: ")
            exp_std = get_float_from_input("Exploration Std")
            assert state_type in ["facing", "facing_R6_h", "root_R6_h"]
            assert exp_std >= 0.0
            model.set_exploration_std(exp_std)
            def state_body_custom(type):
                return self.env.base_env.state_body(
                    idx=0, 
                    agent="sim", 
                    type=type, 
                    return_stacked=True)
            data = {
                'iter_per_episode': iter_per_episode,
                'dim_state': self.env.base_env.dim_state(0),
                'dim_state_body': len(state_body_custom(state_type)),
                'dim_state_task': self.env.base_env.dim_state_task(0),
                'dim_action': self.env.base_env.dim_action(0),
                'episodes': [],
                'exp_std': exp_std,
            }
            base_env = self.env.base_env
            for i in range(len(self.env.base_env._ref_motion_all[0])):
                for j in range(iter_per_episode):
                    cnt_per_trial = 0
                    time_start = -window_size + stride
                    while True:
                        print("\rtime_start: ", time_start, end='')
                        episode = {
                            'time': [],
                            'state': [],
                            'action': [],
                            'action_gt': [],
                            'reward': [],
                            'state_body': [],
                            'state_task': [],
                        }
                        if is_task_agnostic_policy:
                            episode.update({
                                'z_body': [],
                                'z_task': [],
                            })
                        self.env.reset({
                            'ref_motion_id': [i], 
                            'start_time': np.array([max(0.0, time_start)])}
                        )
                        time_elapsed = max(0.0, time_start) - time_start
                        falldown = False
                        cnt_per_window = 0
                        while True:
                            s1 = self.env.state()
                            s1_body = state_body_custom(state_type)
                            s1_task = base_env.state_task(0)
                            a = self.trainer.compute_single_action(s1, explore=True)
                            a_gt = self.trainer.compute_single_action(s1, explore=False)
                            s2, rew, eoe, info = self.env.step(a)
                            t = base_env.get_current_time()
                            time_elapsed += base_env._dt_con
                            episode['time'].append(t)
                            episode['state'].append(s1)
                            episode['action'].append(a)
                            episode['action_gt'].append(a_gt)
                            episode['reward'].append(rew)
                            episode['state_body'].append(s1_body)
                            episode['state_task'].append(s1_task)
                            if is_task_agnostic_policy:
                                z_body = model.body_encoder_variable()[0].detach().numpy()
                                z_task = model.task_encoder_variable()[0].detach().numpy()
                                episode['z_body'].append(z_body)
                                episode['z_task'].append(z_task)
                            cnt_per_window += 1
                            ''' 
                            The policy output might not be ideal 
                            when no future reference motion exists.
                            '''
                            if base_env._ref_motion[0].length()-base_env.get_current_time() \
                               <= base_env._imit_window[-1]:
                                break
                            if time_elapsed >= window_size:
                                break
                            if self.env.base_env._end_of_episode:
                                falldown = True
                                break
                        ''' Include only successful (not falling) episodes '''
                        if not falldown:
                            data['episodes'].append(episode)
                            time_start += stride
                            cnt_per_trial += cnt_per_window
                            ''' End if not enough frames remain '''
                            if base_env._ref_motion[0].length() < time_start + stride:
                                break
                        else:
                            print("\r******FALLDOWN****** Retrying...")
                    print('\n%d pairs were created in %d-th trial of episode%d'%(cnt_per_trial, j, i))
            output_file = os.path.join(
                output_dir, 
                "data_iter=%d,winsize=%.2f,stride=%.2f,state_type=%s,exp_std=%.2f.pkl"%(iter_per_episode, window_size, stride, state_type, exp_std))
            with open(output_file, "wb") as file:
                pickle.dump(data, file)
                print("Saved:", file)
        elif key == b'j':
            print('Save Current Render Data...')
            posfix = input("Enter prefix for the file name:")
            name_joint_data = os.path.join(
                "data/temp", "joint_data_" + posfix +".pkl")
            name_link_data = os.path.join(
                "data/temp", "link_data_" + posfix +".pkl")
            pickle.dump(
                self.data['joint_data'], 
                open(name_joint_data, "wb"))
            pickle.dump(
                self.data['link_data'], 
                open(name_link_data, "wb"))
            print('Done.')
        elif key==b'5':
            if self.rm.flag['kin_model']:
                for i in range(self.env.base_env._num_agent):
                    self.env.base_env._kin_agent[i].change_visual_color([0, 0, 0.5, 1])
            else:
                for i in range(self.env.base_env._num_agent):
                    self.env.base_env._kin_agent[i].change_visual_color([0, 0, 0.5, 0])
        elif key == b'q':
            self.latent_random_sample = (self.latent_random_sample+1)%len(self.latent_random_sample_methods)
            print("latent_random_sample:", self.latent_random_sample_methods[self.latent_random_sample])
        elif key == b'x':
            model = self.trainer.get_policy().model
            exp_std = get_float_from_input("Exploration Std")
            assert exp_std >= 0.0
            model.set_exploration_std(exp_std)
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
    # def get_cam_parameters(self, use_buffer=True):
    #     v_up_env_str = self.get_v_up_env_str()
    #     param = {
    #         "translate": {
    #             "target_pos": self.env.base_env._sim_agent[0].get_root_position(),
    #             "ignore_y": v_up_env_str=="y",
    #             "ignore_z": v_up_env_str=="z",
    #         },
    #     }
    #     return param
    def _get_cam_parameters(self, apply_offset=True):
        param = {
            "origin": None, 
            "pos": None, 
            "dist": None,
            "translate": None,
        }
        
        agent = self.env.base_env._sim_agent[0]
        h = self.env.base_env.get_ground_height(0)
        d_face, p_face = agent.get_facing_direction_position(h)
        origin = p_face + agent._char_info.v_up_env

        if self.rm.get_flag("follow_cam") == "pos+rot":
            pos = p_face + 2 * (agent._char_info.v_up_env - d_face)
        else:
            pos = self.cam_cur.pos + (origin - self.cam_cur.origin)
        
        if apply_offset and self.cam_param_offset is not None:
            if self.rm.get_flag("follow_cam") == "pos+rot":
                R_face, _ = conversions.T2Rp(agent.get_facing_transform(h))
                pos += np.dot(R_face, self.cam_param_offset[1])
                origin += np.dot(R_face, self.cam_param_offset[0])
        
        param["origin"] = origin
        param["pos"] = pos
        
        return param
    def get_cam_parameters(self, use_buffer=True):
        if use_buffer:
            param = {
                "origin": np.mean([p["origin"] for p in self.cam_params], axis=0), 
                "pos": np.mean([p["pos"] for p in self.cam_params], axis=0), 
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
        save_motion_name="motion.bvh",
        save_motion_only_success=False,
        save_replay_name="replay.pkl",
        end_time=None, 
        check_falldown=True, 
        check_end_of_motion=True,
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
                if self.env.base_env.check_falldown(0):
                    break
            if check_end_of_motion:
                if self.env.base_env.check_end_of_motion(0):
                    break
            if end_time and time_elapsed >= end_time:
                break
        if save_motion:
            save = True
            if end_time and save_motion_only_success:
                save = time_elapsed >= end_time
            if save:
                bvh.save(
                    motion, 
                    os.path.join(save_dir, save_motion_name),
                    scale=1.0, rot_order="XYZ", verbose=False)
                filename = os.path.join(save_dir, save_replay_name)
                pickle.dump(self.replay_data, open(filename, "wb"))
        if verbose:
            print(" ")
        return time_elapsed

def default_cam(env):
    agent = env.base_env._sim_agent[0]
    # R, p = conversions.T2Rp(agent.get_facing_transform(0.0))    
    v_up_env = agent._char_info.v_up_env
    v_up = agent._char_info.v_up
    v_face = agent._char_info.v_face
    origin = np.zeros(3)
    return rm.camera.Camera(
        pos=3*(v_up+v_face),
        origin=origin, 
        vup=v_up_env, 
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
