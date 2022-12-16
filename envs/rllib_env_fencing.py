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

from env import env_humanoid_fencing as my_env
import env_renderer as er
import render_module as rm

import rllib_model_torch as policy_model
from collections import deque

from fairmotion.core.motion import Motion
from fairmotion.data import bvh
from fairmotion.ops import conversions
import pickle

class HumanoidFencing(MultiAgentEnv):
    def __init__(self, env_config):
        self.base_env = my_env.Env(env_config)
        assert self.base_env._num_agent == 2

        self.env_config = env_config
        
        self.observation_space = []
        self.observation_space_body = []
        self.observation_space_task = []
        self.action_space = []

        ob_scale = np.array(env_config.get('ob_scale', 1000.0))
        ac_scale = np.array(env_config.get('ac_scale', 1.0))
        
        for i in range(self.base_env._num_agent):
            dim_state = self.base_env.dim_state(i)
            dim_state_body = self.base_env.dim_state_body(i)
            dim_state_task = self.base_env.dim_state_task(i)
            dim_action = self.base_env.dim_action(i)
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

    def reset(self, info={}):
        if not self.base_env._initialized:
            self.base_env.create()
        self.base_env.reset(info)
        return {
            self.player1: self.base_env.state(idx=0),
            self.player2: self.base_env.state(idx=1)
        }

    def step(self, action_dict):
        action1 = action_dict[self.player1]
        action2 = action_dict[self.player2]

        base_action_dict = {
            'target_pose': [action1, action2],
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

        if self.base_env._game_type == "first_touch":
            player1_win = 'player1_win' in self.base_env._end_of_episode_reason
            player2_win = 'player2_win' in self.base_env._end_of_episode_reason
        else:
            raise NotImplementedError
        info = {
            self.player1: {
                "win": 1.0 if player1_win else 0.0,
            },
            self.player2: {
                "win": 1.0 if player2_win else 0.0,
            }
        }

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

env_cls = HumanoidFencing

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

    callbacks = MyCallbacksBase

    if spec["config"]["model"]["custom_model"] == "task_agnostic_policy_type1":
        mc = spec["config"]["model"]["custom_model_config"]
        training_schedule = mc.get("training_schedule")
        if training_schedule:
            method = \
                training_schedule.get("method", "pingpong")
            decoder_learn_start = \
                training_schedule["decoder_learn_start"]
            decoder_learn_lengh = \
                training_schedule["decoder_learn_lengh"]
            encoder_learn_lengh = \
                training_schedule["encoder_learn_lengh"]
            callbacks = lambda: MyCallbacksWithSchedule(
                    share_policy=share_policy,
                    learn_method=method,
                    decoder_learn_start=decoder_learn_start,
                    decoder_learn_lengh=decoder_learn_lengh,
                    encoder_learn_lengh=encoder_learn_lengh,
                )

    config = {
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": \
                lambda agent_id : policy_mapping_fn(agent_id, share_policy),
            "policies_to_train": policies_to_train(share_policy),
        },
        "callbacks": callbacks,
        "model": model_config,
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
        episode.custom_metrics["player1_score"] = \
            episode.last_info_for("player1")["win"]
        episode.custom_metrics["player2_score"] = \
            episode.last_info_for("player2")["win"]

class MyCallbacksWithSchedule(MyCallbacksBase):
    def __init__(
        self, 
        share_policy, 
        learn_method,
        decoder_learn_start,
        decoder_learn_lengh,
        encoder_learn_lengh):
        super().__init__()
        self.share_policy = share_policy
        self.learn_method = learn_method
        self.decoder_learn_start = decoder_learn_start
        self.decoder_learn_lengh = decoder_learn_lengh
        self.encoder_learn_lengh = encoder_learn_lengh
        self.learn_encoder = False
        self.learn_cnt = 0
    def on_train_result(self, trainer, result: dict, **kwargs):
        def set_learnable(ev, policy_name, task_encoder, body_encoder, motor_decoder):
            model = ev.get_policy(policy_mapping_fn(policy_name, self.share_policy)).model
            model.set_learnable_task_encoder(task_encoder)
            model.set_learnable_body_encoder(task_encoder)
            model.set_learnable_motor_decoder(motor_decoder)

        if trainer._iteration < self.decoder_learn_start:
            task_encoder = body_encoder = True
            motor_decoder = False
        else:
            self.learn_cnt += 1
            learn_cnt_max = self.encoder_learn_lengh if self.learn_encoder else self.decoder_learn_lengh

            if self.learn_cnt >= learn_cnt_max:
                self.learn_encoder = not self.learn_encoder
                self.learn_cnt = 0
            
            if self.learn_encoder:
                task_encoder = body_encoder = True
                motor_decoder = False
            else:
                if self.learn_method == "pingpong":
                    task_encoder = body_encoder = False
                    motor_decoder = True
                elif self.learn_method == "e2e":
                    task_encoder = body_encoder = True
                    motor_decoder = True
                else:
                    raise NotImplementedError

        # print('**********************************')
        # print(trainer._iteration)
        # print(task_encoder, body_encoder, motor_decoder)
        # print('**********************************')

        trainer.workers.foreach_worker(
            lambda ev: set_learnable(
                ev, "player1", task_encoder, body_encoder, motor_decoder))
        if not self.share_policy:
            trainer.workers.foreach_worker(
                lambda ev: set_learnable(
                    ev, "player2", task_encoder, body_encoder, motor_decoder))

class EnvRenderer(er.EnvRenderer):
    def __init__(self, trainers, config, **kwargs):
        from fairmotion.viz.utils import TimeChecker
        kwargs["ground_tex_file"] = "data/image/grid3.png"
        super().__init__(**kwargs)
        self.trainers = trainers
        self.config = config
        self.time_checker_auto_play = TimeChecker()
        self.explore = True
        self.share_policy = config["env_config"].get("share_policy", False)
        self.cam_target_pos = deque(maxlen=30)
        self.cam_dist = deque(maxlen=30)
        self.replay = False
        self.replay_cnt = 0
        self.data = {}
        self.cam_info = []
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
        self.cam_info.clear()
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
            # if self.replay_cnt == 52:
            #     self.env.base_env._opponent_touched[1] = True
            # if self.replay_cnt == 39:
            #     self.env.base_env._opponent_touched[1] = True
            # if self.replay_cnt == 49:
            #     self.env.base_env._opponent_touched[0] = True
            # if self.replay_cnt == 123:
            #     self.env.base_env._opponent_touched[0] = True
        else:
            if self.env.base_env._end_of_episode: return
            s = self.env.state()
            a = {}
            for i, agent_id in enumerate(self.agent_ids):
                trainer = self.get_trainer(i)
                policy = trainer.get_policy(
                    policy_mapping_fn(agent_id, self.share_policy))
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
        h_cur += 20
        for i, agent_id in enumerate(self.agent_ids):
            if i==0:
                origin = np.array([20, h_cur]) + radius
            else:
                origin = np.array([w-2*radius-20, h_cur]) + radius
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
            if i==1: h_cur += radius + 20

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
                    is_save_data=False,
                    is_save_cam_info=True)
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
                        is_save_data=True,
                        is_save_cam_info=True)
    def run_one_episode(
        self, 
        save_dir, 
        end_time, 
        is_random_init, 
        is_save_screen, 
        is_save_data, 
        is_save_cam_info):
        try:
            os.makedirs(save_dir, exist_ok = True)
        except OSError:
            print("Invalid Directory", save_dir)
            return
        cnt_screenshot = 0
        eoe = False
        eoe_margin = 0.5
        eoe_elapsed = 0.0
        time_elapsed = 0.0
        # self.reset({'start_time': 0.0})
        if is_random_init:
            self.reset()
        else:
            self.reset({
                'ref_motion_idx': [0, 0],
                'start_time': [0.0, 0.0],
                }
            )
        self.update_cam()
        while True:
            name = 'screenshot_%04d'%(cnt_screenshot)
            self.one_step()
            time_elapsed += self.env.base_env._dt_con
            print('\rSimulation Processed for (%4.4f)' % time_elapsed, end=" ")
            if is_save_cam_info:
                R = self.cam_cur.get_cam_rotation()
                R = np.dot(R, conversions.Ay2R(np.pi))
                self.cam_info.append((
                    conversions.R2Q(R),
                    self.cam_cur.pos))
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
        if is_save_cam_info:
            name = os.path.join(save_dir, "cam_info.pickle")
            pickle.dump(self.cam_info, open(name,"wb"))
            print("CamInfo was saved:", name)
        print("Done.")
    def compute_cam_pos_dist(self):
        p1 = self.env.base_env._sim_agent[0].get_root_position()
        p2 = self.env.base_env._sim_agent[1].get_root_position()
        dist = np.linalg.norm(p1-p2)
        target_pos = 0.5 * (p1 + p2)
        cam_dist = np.clip(2.5 * dist, 5.0, 8.0)
        return target_pos, cam_dist
    def get_cam_parameters(self, use_buffer=True):
        return np.mean(self.cam_target_pos, axis=0), np.mean(self.cam_dist)
    def get_elapsed_time(self):
        return self.env.base_env.get_elapsed_time()

def default_cam():
    return rm.camera.Camera(pos=np.array([8.0, 0.0, 6.0]),
                            origin=np.array([0.0, 0.0, 0.0]), 
                            vup=np.array([0.0, 0.0, 1.0]), 
                            fov=30.0)


