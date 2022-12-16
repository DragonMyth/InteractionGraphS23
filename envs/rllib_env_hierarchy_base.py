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

from ray.rllib.utils import try_import_torch
torch, nn = try_import_torch()

import rllib_model_torch as policy_model
from collections import deque

import torch_models
import torch.nn.functional as F

from fairmotion.core.motion import Motion
from fairmotion.data import bvh
import pickle

BASE_ENV_CLS = None
CALLBACK_FUNC = None

class HumanoidHierarchyBase(MultiAgentEnv):
    def __init__(self, env_config):
        assert BASE_ENV_CLS is not None, \
            "Base environment class should be specified first"
        
        self.base_env = BASE_ENV_CLS(env_config.get('base_env_config'))
        
        base_model_config = env_config.get('base_model_config')
        custom_model_config = base_model_config.get('custom_model_config')
        project_dir = env_config['project_dir']
        if 'body_encoder_load_weights' in custom_model_config:
            custom_model_config['body_encoder_load_weights'] = \
                os.path.join(project_dir, custom_model_config['body_encoder_load_weights'])
        if 'motor_decoder_load_weights' in custom_model_config:
            custom_model_config['motor_decoder_load_weights'] = \
                os.path.join(project_dir, custom_model_config['motor_decoder_load_weights'])

        dim_state = self.base_env.dim_state(0)
        dim_action = custom_model_config.get('task_encoder_output_dim')

        ob_scale = 1000.0

        base_dim_state_body = self.base_env.dim_state_body(0)
        base_dim_state_task = dim_action
        base_dim_state = base_dim_state_body + base_dim_state_task
        base_dim_action = self.base_env.dim_action(0)
        base_action_range_min, base_action_range_max = \
            self.base_env.action_range(0)
        
        base_observation_space = \
            Box(low=-ob_scale * np.ones(base_dim_state),
                high=ob_scale * np.ones(base_dim_state),
                dtype=np.float64)
        base_observation_space_body = \
            Box(low=-ob_scale * np.ones(base_dim_state_body),
                high=ob_scale * np.ones(base_dim_state_body),
                dtype=np.float64)
        base_observation_space_task = \
            Box(low=-ob_scale * np.ones(base_dim_state_task),
                high=ob_scale * np.ones(base_dim_state_task),
                dtype=np.float64)
        base_action_space = \
            Box(low=base_action_range_min,
                high=base_action_range_max,
                dtype=np.float64)

        custom_model_config.update({
            "observation_space_body": copy.deepcopy(base_observation_space_body),
            "observation_space_task": copy.deepcopy(base_observation_space_task),
        })

        self.base_model = model.TaskAgnosticPolicyType1(
            obs_space=base_observation_space,
            action_space=base_action_space,
            num_outputs=base_dim_action*2,
            model_config=base_model_config,
            name="base_model"
            )
        self.base_model_state = [
            [self.base_model.get_initial_state()],
            [self.base_model.get_initial_state()],
        ]
        self.base_model_data = [
            {'gate_fn_w': deque(maxlen=30)},
            {'gate_fn_w': deque(maxlen=30)},
        ]
        
        self.observation_space = []
        self.action_space = []

        l = env_config['base_model_config']['custom_model_config']['task_encoder_layers'][-1]
        
        if l['type'] == 'lstm':
            activation = l['output_activation']
        elif l['type'] == 'fc':
            activation = l['activation']
        else:
            raise NotImplementedError
        
        if activation == "linear":
            action_range_scale = 10.0
        elif activation == "tanh":
            action_range_scale = 1.0
        else:
            raise NotImplementedError
        
        for i in range(self.base_env._num_agent):
            self.observation_space.append(
                Box(low=-ob_scale*np.ones(dim_state),
                    high=ob_scale*np.ones(dim_state),
                    dtype=np.float64)
                )
            self.action_space.append(
                Box(low=-action_range_scale*np.ones(dim_action),
                    high=action_range_scale*np.ones(dim_action),
                    dtype=np.float64)
                )

        self.player1 = "player1"
        self.player2 = "player2"

        self.data = [{} for i in range(self.base_env._num_agent)]

        ''' Load conditional probability of trajectories in the task embbeding space '''
        self.task_embedding_prob_model = None
        prob_model_config = env_config.get('task_embedding_prob_model_config')
        if prob_model_config:
            # Prepare a model
            prob_model = torch_models.Classifier(
                torch_models.FCNN,
                size_in=prob_model_config.get('size_in'),
                size_out=prob_model_config.get('size_out'),
                hiddens=prob_model_config.get('hiddens'),
                activations=prob_model_config.get('activations'),
                init_weights=prob_model_config.get('init_weights'),
                init_bias=prob_model_config.get('init_bias'))
            # Load weights
            load_weights = prob_model_config.get('load_weights')
            if load_weights:
                print('load_weights: ', load_weights)
                dict_weights_loaded = torch.load(load_weights)
                prob_model.load_state_dict(dict_weights_loaded)
                prob_model.eval()
            self.task_embedding_prob_model = prob_model
            for i in range(self.base_env._num_agent):
                self.data[i]['prev_task_embedding'] = deque(maxlen=10)

    def state(self):
        return {
            self.player1: self.base_env.state(idx=0),
            self.player2: self.base_env.state(idx=1),
        }

    def reset(self, info={}):
        if not self.base_env._initialized:
            self.base_env.create()
        self.base_env.reset(info)
        self.base_model_state = [
            [self.base_model.get_initial_state()],
            [self.base_model.get_initial_state()],
        ]
        self.base_model_data = [
            {'gate_fn_w': deque(maxlen=30)},
            {'gate_fn_w': deque(maxlen=30)},
        ]
        if self.task_embedding_prob_model:
            for i in range(self.base_env._num_agent):
                self.data[i]['prev_task_embedding'].clear()
        return self.state()

    def get_info(self):
        raise NotImplementedError

    def step(self, action_dict):
        action = [
            action_dict[self.player1],
            action_dict[self.player2],
        ]
        state_body = [
            self.base_env.state_body(idx=0),
            self.base_env.state_body(idx=1),
        ]
        logits = [None, None]
        base_action = [None, None]

        for i in range(2):
            logits[i], _ = self.base_model.forward_decoder(
                z_body=torch.Tensor([state_body[i]]),
                z_task=torch.Tensor([action[i]]),
                state=self.base_model_state[i],
                seq_lens=np.ones(1),
                state_cnt=0,
            )
            base_action[i] = logits[i][0].detach().numpy()[:self.base_env.dim_action(i)]
            if self.base_model._cur_motor_decoder_expert_weights is not None:
                self.base_model_data[i]['gate_fn_w'].append(
                    self.base_model._cur_motor_decoder_expert_weights)

        base_action_dict = {
            'target_pose': base_action,
        }

        r, info_env = self.base_env.step(base_action_dict)

        obs = self.state()
        rew = {
            self.player1: r[0],
            self.player2: r[1]
        }
        done = {
            "__all__": self.base_env._end_of_episode,
        }

        info  = self.get_info()

        # print(info)

        if self.base_env._verbose:
            self.base_env.pretty_print_rew_info(info_env[0]['rew_info'])
            self.base_env.pretty_print_rew_info(info_env[1]['rew_info'])

        if self.task_embedding_prob_model:
            task_embedding_queues = [
                self.data[0]['prev_task_embedding'], 
                self.data[1]['prev_task_embedding'],
            ]
            actions = [
                action1,
                action2,
            ]
            for q, a in zip(task_embedding_queues, actions):
                if len(q) < q.maxlen:
                    for i in range(q.maxlen):
                        q.append(a)
                q.append(a)
                z = torch.Tensor([np.hstack([q[i] for i in range(-4, 0)])])
                y = self.task_embedding_prob_model(z)
                print(y)
                y = F.softmax(y, dim=0).detach().numpy()
                print(y)
                print('--------------------')

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

env_cls = HumanoidHierarchyBase

def config_override(spec):
    env = env_cls(spec["config"]["env_config"])

    share_policy = \
        spec["config"]["env_config"].get("share_policy", False)

    if share_policy:
        policies = {
            "policy_player": (None, 
                           copy.deepcopy(env.observation_space[0]), 
                           copy.deepcopy(env.action_space[0]), 
                           {}),
        }
    else:
        policies = {
            "policy_player1": (None, 
                           copy.deepcopy(env.observation_space[0]), 
                           copy.deepcopy(env.action_space[0]), 
                           {}),
            "policy_player2": (None, 
                           copy.deepcopy(env.observation_space[1]), 
                           copy.deepcopy(env.action_space[1]), 
                           {}),
        }

    model_config = copy.deepcopy(spec["config"]["model"])
    model = model_config.get("custom_model")
    if model and model == "task_agnostic_policy_type1":
        model_config.get("custom_model_config").update({
            "observation_space_body": copy.deepcopy(env.observation_space_body[0]),
            "observation_space_task": copy.deepcopy(env.observation_space_task[0]),
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
    def __init__(self, trainers, config, **kwargs):
        from fairmotion.viz.utils import TimeChecker
        super().__init__(**kwargs)
        self.trainers = trainers
        self.config = config
        self.time_checker_auto_play = TimeChecker()
        self.explore = True
        self.latent_random_sample = False
        self.share_policy = config["env_config"].get("share_policy", False)
        self.cam_target_pos = deque(maxlen=30)
        self.cam_dist = deque(maxlen=30)
        self.replay = False
        self.replay_cnt = 0
        self.data = {}
        self.reset()
    def get_pb_client(self):
        return self.env.base_env._pb_client
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
                        agent_id, self.share_policy)).get_initial_state()
                self.policy_hidden_state.append(state)
                motion = Motion(
                    skel=self.env.base_env._base_motion[i].skel,
                    fps=self.env.base_env._base_motion[i].fps,
                )
                self.data[agent_id] = {
                    'motion': motion,
                    'z_task': list(),
                    'gate_fn_w': list(),
                    'joint_data': list(),
                    'link_data': list(),
                }
        self.cam_target_pos.clear()
        self.cam_dist.clear()
        pos, dist = self.compute_cam_pos_dist()
        for i in range(self.cam_target_pos.maxlen):
            self.cam_target_pos.append(pos)
            self.cam_dist.append(dist)
    def collect_data(self, idx):
        agent_id = self.agent_ids[idx]
        sim_agent = self.env.base_env._sim_agent[idx]

        motion = self.data[agent_id]['motion']
        motion.add_one_frame(sim_agent.get_pose(motion.skel).data)

        joint_data, link_data = \
            self.env.base_env.get_render_data(idx)
        self.data[agent_id]['joint_data'].append(joint_data)
        self.data[agent_id]['link_data'].append(link_data)

        trainer = self.get_trainer(idx)
        policy = trainer.get_policy(
            policy_mapping_fn(agent_id, self.share_policy))
        if isinstance(policy.model, (policy_model.TaskAgnosticPolicyType1)):
            if policy.model._motor_decoder_type == "mlp":
                self.data[agent_id]['z_task'].append(
                    policy.model._cur_task_encoder_variable)
            elif policy.model._motor_decoder_type == "moe":
                self.data[agent_id]['z_task'].append(
                    policy.model._cur_task_encoder_variable)
                self.data[agent_id]['gate_fn_w'].append(
                    policy.model._cur_motor_decoder_expert_weights[0].detach().numpy().copy())
            else:
                raise NotImplementedError
        elif isinstance(policy.model, (policy_model.MOEPolicyBase)):
            self.data[agent_id]['gate_fn_w'].append(
                policy.model.gate_function()[0].detach().numpy().copy())
    def set_pose(self):
        for i, agent_id in enumerate(self.agent_ids):
            motion = self.data[agent_id]['motion']
            pose = motion.get_pose_by_frame(self.replay_cnt)
            self.env.base_env._sim_agent[i].set_pose(pose)
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
                    policy_mapping_fn(agent_id, self.share_policy))
                if self.latent_random_sample:
                   action = np.random.normal(size=self.env.action_space[0].shape[0])
                   # print(action)
                else:
                    if policy.is_recurrent():
                        action, state_out, extra_fetches = trainer.compute_action(
                            s[agent_id], 
                            state=self.policy_hidden_state[i],
                            policy_id=policy_mapping_fn(agent_id, self.share_policy),
                            explore=self.explore)
                        self.policy_hidden_state[i] = state_out
                    else:
                        action = trainer.compute_action(
                            s[agent_id], 
                            policy_id=policy_mapping_fn(agent_id, self.share_policy),
                            explore=self.explore)
                a[agent_id] = action
                ''' Collect data for rendering or etc. '''
                self.collect_data(i)
            self.env.step(a)
        ''' Record distance between the agents for camera update '''
        pos, dist = self.compute_cam_pos_dist()
        self.cam_target_pos.append(pos)
        self.cam_dist.append(dist)
    def extra_render_callback(self):
        self.env.base_env.render(self.rm)
    def extra_overlay_callback(self):
        w, h = self.window_size
        h_cur = 0
        radius = 80
        for i, agent_id in enumerate(self.agent_ids):
            if i==0:
                origin = np.array([20, h_cur+20]) + radius
            else:
                origin = np.array([w-2*radius-20, h_cur+20]) + radius
            if self.replay:
                weights = self.data[agent_id]['gate_fn_w'][:self.replay_cnt]
            else:
                weights = self.data[agent_id]['gate_fn_w']
            self.rm.gl_render.render_expert_weights_circle(
                weights=weights,
                radius=radius,
                origin=origin,
                scale_weight=20.0,
                color_experts=self.rm.COLORS_FOR_EXPERTS,
            )
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
        elif key == b'g':
            self.replay = False
            self.reset({
                'ref_motion_idx': [0, 0],
                'start_time': [0.0, 0.0],
                'player_pos': [
                    np.array([1.5, 0.0]),
                    np.array([-1.5, 0.0])]
                }
            )
        elif key == b' ':
            self.time_checker_auto_play.begin()
            self.one_step()
        elif key == b'e':
            self.explore = not self.explore
            print('Exploration:', self.explore)
        elif key == b'v':
            self.env.base_env._verbose = not self.env.base_env._verbose
            print('Verbose:', self.env.base_env._verbose)
        elif key == b']':
            if self.replay:
                self.replay_cnt = min(
                    self.data[self.agent_ids[0]]['motion'].num_frames()-1,
                    self.replay_cnt+1)
                self.set_pose()
        elif key == b'[':
            if self.replay:
                self.replay_cnt = max(0, self.replay_cnt-1)
                self.set_pose()
        elif key == b'w':
            for i, agent_id in enumerate(self.agent_ids):
                trainer = self.get_trainer(i)
                policy = trainer.get_policy(
                    policy_mapping_fn(agent_id, self.share_policy))
                break
            model = policy.model
            print('Save Model Weights...')
            model.save_weights_task_encoder('data/temp/task_agnostic_policy_task_encoder.pt')
            model.save_weights_body_encoder('data/temp/task_agnostic_policy_body_encoder.pt')
            model.save_weights_motor_decoder('data/temp/task_agnostic_policy_motor_decoder.pt')
        elif key == b'l':
            print('Save Current Render Data...')
            for agent_id in self.agent_ids:
                name_joint_data = os.path.join(
                    "data/temp", "joint_data_" + str(agent_id)+".pkl")
                name_link_data = os.path.join(
                    "data/temp", "link_data_" + str(agent_id)+".pkl")
                pickle.dump(
                    self.data[agent_id]['joint_data'], 
                    open(name_joint_data, "wb"))
                pickle.dump(
                    self.data[agent_id]['link_data'], 
                    open(name_link_data, "wb"))
            print('Done.')
        elif key == b'L':
            print('Load Data...')
            name = input("Enter data file: ")
            with open(name, "rb") as f:
                self.data = pickle.load(f)
            self.replay = True
            self.reset()
            print('Done.')
        elif key == b'S':
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
            if self.replay:
                self.run_one_episode(
                    save_dir=dir, 
                    end_time=end_time,
                    is_random_init=False, 
                    is_save_screen=True, 
                    is_save_data=False)
            else:
                ''' Read number of iteration '''
                num_iter = input("Enter num iteration (int): ")
                try:
                   num_iter = int(num_iter)
                except ValueError:
                   print("That's not a number!")
                   return
                ''' Save screen '''
                is_save_screen = -1
                while is_save_screen!=1 and is_save_screen!=0:
                    try:
                        is_save_screen = int(input("Save Screen? (1/0)"))
                    except ValueError:
                        continue
                ''' Random Initialization '''
                is_random_init = -1
                while is_random_init!=1 and is_random_init!=0:
                    try:
                        is_random_init = int(input("Random Initialization? (1/0)"))
                    except ValueError:
                        continue
                for i in range(num_iter):
                    dir_i = os.path.join(dir, str(i))
                    self.run_one_episode(
                        save_dir=dir_i, 
                        end_time=end_time,
                        is_random_init=is_random_init, 
                        is_save_screen=is_save_screen, 
                        is_save_data=True)
        elif key == b'E':
            print('Evaluation ...')
            for i in range(20):
                time_elapsed = 0.0
                self.reset()
                while True:
                    self.one_step()
                    time_elapsed += self.env.base_env._dt_con
                    if self.env.base_env._end_of_episode:
                        break
                score = self.env.base_env._game_remaining_score
                print(score[0], score[1])
            print('Done.')
        elif key == b'q':
            self.latent_random_sample = not self.latent_random_sample
            print("latent_random_sample:", self.latent_random_sample)


    def run_one_episode(self, 
        save_dir, end_time, is_random_init, is_save_screen, is_save_data):
        try:
            os.makedirs(save_dir, exist_ok = True)
        except OSError:
            print("Invalid Directory", save_dir)
            return
        cnt_screenshot = 0
        eoe = False
        eoe_margin = 1.0
        eoe_elapsed = 0.0
        time_elapsed = 0.0
        # self.reset({'start_time': 0.0})
        if is_random_init:
            self.reset()
        else:
            self.reset({
                'ref_motion_idx': [0, 0],
                'start_time': [0.0, 0.0],
                'player_pos': [
                    np.array([1.5, 0.0]),
                    np.array([-1.5, 0.0])]
                }
            )
        self.update_cam()
        while True:
            name = 'screenshot_%04d'%(cnt_screenshot)
            self.one_step()
            time_elapsed += self.env.base_env._dt_con
            print('\rSimulation Processed for (%4.4f)' % time_elapsed, end=" ")
            if is_save_screen:
                self.save_screen(dir=save_dir, name=name, render=True, save_alpha_channel=True)
                cnt_screenshot += 1
            if self.replay:
                eoe = time_elapsed >= end_time or\
                    self.replay_cnt==self.data[self.agent_ids[0]]['motion'].num_frames()-1
                if eoe: break
            else:
                if eoe:
                    eoe_elapsed += self.env.base_env._dt_con
                else:
                    eoe = self.env.base_env._end_of_episode or \
                        time_elapsed >= end_time
                if eoe_elapsed >= eoe_margin: break
        print("Simulation End")
        if is_save_data:
            name = os.path.join(save_dir, "data.pickle")
            pickle.dump(self.data, open(name,"wb"))
            print("Data was saved:", name)
        print("Done.")
    def compute_cam_pos_dist(self):
        p1 = self.env.base_env._sim_agent[0].get_root_position()
        p2 = self.env.base_env._sim_agent[1].get_root_position()
        dist = np.linalg.norm(p1-p2)
        target_pos = 0.5 * (p1 + p2)
        cam_dist = np.clip(3.0 * dist, 5.0, 10.0)
        return target_pos, cam_dist
    def get_cam_parameters(self, use_buffer=True):
        return np.mean(self.cam_target_pos, axis=0), np.mean(self.cam_dist)
    def get_elapsed_time(self):
        return self.env.base_env.get_elapsed_time()

# def default_cam():
#     return rm.camera.Camera(
#         pos=np.array([12.0, 0.0, 12.0]),
#         origin=np.array([0.0, 0.0, 0.0]), 
#         vup=np.array([0.0, 0.0, 1.0]), 
#         fov=30.0)

def default_cam():
    return rm.camera.Camera(
        pos=np.array([7.0, 7.0, 7.0]),
        origin=np.array([0.0, 0.0, 0.0]), 
        vup=np.array([0.0, 0.0, 1.0]), 
        fov=30.0)

class EnvRenderer1(er.EnvRenderer):
    def __init__(self, trainers, config, **kwargs):
        from fairmotion.viz.utils import TimeChecker
        super().__init__(**kwargs)
        self.trainers = trainers
        self.config = config
        self.time_checker_auto_play = TimeChecker()
        self.explore = False
        self.data = {}
        self.share_policy = config["env_config"].get("share_policy", False)
        self.reset()
    def get_pb_client(self):
        return self.env.base_env._pb_client
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
        s = self.env.reset(info)
        self.policy_hidden_state = []
        self.agent_ids = s.keys()
        for agent_id in self.agent_ids:
            state = self.trainer.get_policy(
                policy_mapping_fn(
                    agent_id, self.share_policy)).get_initial_state()
            self.policy_hidden_state.append(state)
            self.data[agent_id] = {
                'action': deque(maxlen=30),
                'base_model_state': deque(maxlen=30),
                'gate_fn_w': deque(maxlen=30),
            }
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
            self.data[agent_id]['action'].append(action)
        # print(a)
        self.env.step(a)
        for i, agent_id in enumerate(self.agent_ids):
            if len(self.env.base_model_data[i]['gate_fn_w']) > 0:
                self.data[agent_id]['gate_fn_w'].append(
                    self.env.base_model_data[i]['gate_fn_w'][-1].detach().numpy().copy())
            if self.env.base_model_state[i][0] is None or \
               len(self.env.base_model_state[i][0]) == 0:
                continue
            # print(self.env.base_model_state[i])
            # print(self.env.base_model_state[i][0])
            self.data[agent_id]['base_model_state'].append(
                self.env.base_model_state[i][0][0].detach().numpy().copy())
    def extra_render_callback(self):
        self.env.base_env.render(self.rm)
    def extra_overlay_callback(self):
        w, h = self.window_size
        h_cur = 20
        def draw_trajectory(
            data, 
            color_cur=[1, 0, 0, 1], 
            color_prev=[0.5, 0.5, 0.5, 1],
            pos_origin=np.zeros(2)):
            w_size, h_size = 150, 150
            pad_len = 10
            if i==0:
                origin = pos_origin + np.array([20, h_size+20])
            else:
                origin = pos_origin + np.array([w-w_size-20, h_size+20])
            self.rm.gl_render.render_graph_base_2D(origin=origin, axis_len=w_size, pad_len=pad_len)
            x_range = (self.env.action_space[0].low[0], self.env.action_space[0].high[0])
            y_range = (self.env.action_space[0].low[1], self.env.action_space[0].high[1])
            if len(data) > 0:
                action = np.vstack(data)
                action_x = action[:,0].tolist()
                action_y = action[:,1].tolist()
                self.rm.gl_render.render_graph_data_point_2D(
                    x_data=[action_x[-1]],
                    y_data=[action_y[-1]],
                    x_range=x_range,
                    y_range=y_range,
                    color=color_cur,
                    point_size=5.0,
                    origin=origin,
                    axis_len=w_size,
                    pad_len=pad_len,
                )
                self.rm.gl_render.render_graph_data_line_2D(
                    x_data=action_x,
                    y_data=action_y,
                    x_range=x_range,
                    y_range=y_range,
                    color=color_prev,
                    line_width=2.0,
                    origin=origin,
                    axis_len=w_size,
                    pad_len=pad_len,
                )
        for i, agent_id in enumerate(self.agent_ids):
            if len(self.data[agent_id]['action']) > 0:
                draw_trajectory(self.data[agent_id]['action'],
                    color_cur=[1.0, 0.0, 0.0, 1.0],
                    color_prev=[1.0, 0.0, 0.0, 0.5],
                )
            if len(self.data[agent_id]['base_model_state']) > 0:
                draw_trajectory(
                    self.data[agent_id]['base_model_state'],
                    color_cur=[0.0, 0.0, 1.0, 1.0],
                    color_prev=[0.0, 0.0, 1.0, 0.5],
                )
            if i==1: h_cur += 150+10
        ''' Expert Weights '''
        w_bar, h_bar = 150.0, 20.0
        for i, agent_id in enumerate(self.agent_ids):
            if len(self.data[agent_id]['gate_fn_w']) > 0:
                expert_weights = self.data[agent_id]['gate_fn_w'][-1]
                num_experts = len(expert_weights[0])
                if i==0:
                    origin = np.array([20, h_cur+h_bar*num_experts])
                    text_offset = np.array([w_bar+5, 0.8*h_bar])
                else:
                    origin = np.array([w-w_bar-20, h_cur+h_bar*num_experts])
                    text_offset = np.array([-75, 0.8*h_bar])
                pos = origin.copy()
                for j in reversed(range(num_experts)):
                    self.rm.gl_render.render_text(
                        "Expert%d"%(j), 
                        pos=pos+text_offset, 
                        font=self.rm.glut.GLUT_BITMAP_9_BY_15)
                    expert_weight_j = \
                        expert_weights[0][j] if expert_weights is not None else 0.0
                    self.rm.gl_render.render_progress_bar_2D_horizontal(
                        expert_weight_j, origin=pos, width=w_bar, 
                        height=h_bar, color_input=self.rm.COLORS_FOR_EXPERTS[j])
                    pos += np.array([0.0, -h_bar])
                if i==1: h_cur += (h_bar*num_experts+10)
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


