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

from envs import env_humanoid_boxing_solo as my_env
import env_renderer as er
import render_module as rm

import rllib_model_torch as policy_model
from collections import deque

from ray.rllib.utils import try_import_torch
torch, nn = try_import_torch()

from fairmotion.core.motion import Motion
from fairmotion.data import bvh
from fairmotion.ops import conversions
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

class HumanoidBoxingSolo(gym.Env):
    def __init__(self, env_config):
        self.base_env = my_env.Env(env_config)
        
        ob_scale = np.array(env_config.get('ob_scale', 1000.0))
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
        action_dict = {
            'target_pose': [action],
        }
        rew, info_env = self.base_env.step(action_dict)
        obs = self.state()
        eoe = self.base_env._end_of_episode
        if self.base_env._verbose:
            self.base_env.pretty_print_rew_info(info_env[0]['rew_info'])
            print(info_env)
            print("cur_hit_source", self.base_env._cur_hit_source)
            print("_hit_force", self.base_env._hit_force)
        info = {
            "hit_the_target": info_env[0]["hit_the_target"],
        }
        return obs, rew[0], eoe, info

from typing import Dict
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks

class MyCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        info = episode.last_info_for()
        episode.custom_metrics["hit_the_target"] = info["hit_the_target"]

class EnvRenderer(er.EnvRenderer):
    def __init__(self, trainers, config, **kwargs):
        # kwargs['renderer'] = 'bullet_native'
        from fairmotion.viz.utils import TimeChecker
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
        if not self.env.base_env._base_env._use_default_ground:
            self.options["ground_color"] = np.array([220, 140, 80.0])/255.0
        self.replay = False
        self.replay_cnt = 0
        self.replay_data = [{} for i in range(self.env.base_env._num_agent)]
        self.reset()
    def use_default_ground(self):
        return self.env.base_env._base_env._use_default_ground
    def get_v_up_env_str(self):
        return self.env.base_env._v_up_str
    def get_pb_client(self):
        return self.env.base_env._pb_client
    def get_ground(self):
        return self.env.base_env._ground
    def reset(self, info={}):
        self.replay_cnt = 0
        if self.replay:
            self.set_pose()
            env = self.env.base_env
            for i in range(env._num_agent):
                env.update_sensors(i, reset=True)
        else:    
            s = self.env.reset(info)
            self.policy_hidden_state = self.trainer.get_policy().get_initial_state()
            env = self.env.base_env
            for i in range(env._num_agent):
                ''' Setup replay data '''
                motion = Motion(
                    skel=env._base_motion[i].skel,
                    fps=env._base_motion[i].fps,
                )
                self.replay_data[i] = {
                    'motion': motion,
                    'joint_data': list(),
                    'link_data': list(),
                    'others': {},
                }
                ''' Setup render data '''
                data = self.replay_data[i]['others']
                data["root_R"] = []
                data["root_p"] = []
                # goal_type = env._goal_type
                # if goal_type == "pos":
                #     p = env._sim_agent[i].get_facing_position(env.get_ground_height(i))
                #     data["target_p"] = [p]
                # elif goal_type == "vel":
                #     p = env._sim_agent[i].get_facing_position(env.get_ground_height(i))
                #     _, v = env._sim_agent[i].get_com_and_com_vel()
                #     v = env._sim_agent[i].project_to_ground(v)
                #     data["face_p"] = [p]
                #     data["com_v"] = [v]
                #     data["target_v"] = [env._target_vel.copy()]
                # elif goal_type == "path":
                #     p = env._sim_agent[i].get_facing_position(env.get_ground_height(i))
                #     _, v = env._sim_agent[i].get_com_and_com_vel()
                #     v = env._sim_agent[i].project_to_ground(v)
                #     v /= np.linalg.norm(v)
                #     data["face_p"] = [p]
                #     data["com_v"] = [v]
                #     data["target_p"] = [env._target_pos[0].copy()]
                #     data["target_d"] = [env._target_dir[0].copy()]
        self.cam_params.clear()
        param = self._get_cam_parameters()
        for i in range(self.cam_params.maxlen):
            self.cam_params.append(param)
    def collect_replay_data(self):
        env = self.env.base_env
        for i in range(env._num_agent):
            sim_agent = env._sim_agent[i]
            motion = self.replay_data[i]['motion']
            motion.add_one_frame(sim_agent.get_pose(motion.skel).data)
            joint_data, link_data = env.get_render_data(i)
            self.replay_data[i]['joint_data'].append(joint_data)
            self.replay_data[i]['link_data'].append(link_data)

            data = self.replay_data[i]['others']
            R, p = conversions.T2Rp(env._sim_agent[i].get_root_transform())
            data["root_R"].append(R)
            data["root_p"].append(p)
            # goal_type = env._goal_type
            # if goal_type == "pos":
            #     data["target_p"].append(env._target_pos.copy())
            # elif goal_type == "vel":
            #     p = env._sim_agent[i].get_facing_position(env.get_ground_height(i))
            #     _, v = env._sim_agent[i].get_com_and_com_vel()
            #     v = env._sim_agent[i].project_to_ground(v)
            #     data["face_p"].append(p)
            #     data["com_v"].append(v)
            #     data["target_v"].append(env._target_vel.copy())
            # elif goal_type == "path":
            #     p = env._sim_agent[i].get_facing_position(env.get_ground_height(i))
            #     _, v = env._sim_agent[i].get_com_and_com_vel()
            #     v = env._sim_agent[i].project_to_ground(v)
            #     v /= np.linalg.norm(v)
            #     data["face_p"].append(p)
            #     data["com_v"].append(v)
            #     data["target_p"].append(env._target_pos[0].copy())
            #     data["target_d"].append(env._target_dir[0].copy())
    def set_pose(self):
        for i in range(self.env.base_env._num_agent):
            if self.replay_data[i]['motion'].num_frames() == 0: 
                continue
            motion = self.replay_data[i]['motion']
            pose = motion.get_pose_by_frame(self.replay_cnt)
            self.env.base_env._sim_agent[i].set_pose(pose)
    def one_step(self, explore=None):
        self.cam_params.append(self._get_cam_parameters())
        
        if self.replay:
            self.set_pose()
            self.replay_cnt = \
                min(self.replay_data[0]['motion'].num_frames()-1, self.replay_cnt+1)
            env = self.env.base_env
            for i in range(env._num_agent):
                env.update_sensors(i, reset=False)
            return
        
        if explore is None:
            explore = self.explore
        
        ''' Compute an action '''
        
        s1 = self.env.state()
        policy = self.trainer.get_policy()
        
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
        
        ''' Run the environment '''
        
        s2, rew, eoe, info = self.env.step(action)

        self.collect_replay_data()
        # self.cam_params.append(self._get_cam_parameters())
        return s2, rew, eoe, info
    def extra_render_callback(self):
        def render_trajectory(
            points, color, scale=1.0, line_width=1.0, point_size=1.0):
            self.rm.gl.glColor(color)
            self.rm.gl.glLineWidth(line_width)
            self.rm.gl.glBegin(self.rm.gl.GL_LINE_STRIP)
            for p in points:
                self.rm.gl.glVertex3d(p[0], p[1], p[2])
            self.rm.gl.glEnd()
        def render_arrow_2D(p1, p2, diameter=0.05, color=[0.0, 0.0, 0.0, 1.0]):
            self.rm.gl.glDisable(self.rm.gl.GL_LIGHTING)
            self.rm.gl.glPushMatrix()
            self.rm.gl.glScalef(1.0, 0.1, 1.0)
            self.rm.gl_render.render_arrow(p1, p2, D=diameter, color=color)
            self.rm.gl.glPopMatrix()
        self.env.base_env.render(self.rm)
        if self.rm.flag["root_trajectory"]:
            for i in range(self.env.base_env._num_agent):
                data = self.replay_data[i]['others']
                root_positions = data["root_p"][0:self.replay_cnt]
                if len(root_positions) > 0:
                    render_trajectory(
                        root_positions, [0.0, 0.0, 0.0, 0.8], scale=1.0, line_width=5.0, point_size=1.0)
        if self.replay:
            goal_type = self.env.base_env._goal_type
            for i in range(self.env.base_env._num_agent):
                data = self.replay_data[i]['others']
                if goal_type == "pos":
                    target_positions = data["target_p"][0:self.replay_cnt]
                    # if len(target_positions) > 0:
                    #     prev_target = target_positions[0]
                    #     for i in range(len(target_positions)):
                    #         if i==0:
                    #             draw = True
                    #         else:
                    #             draw = np.linalg.norm(target_positions[i]-prev_target) > 1e-02
                    #         if draw:
                    #             prev_target = target_positions[i]
                    #             self.env.base_env.render_target_pos(self.rm, target_positions[i])
                    if len(target_positions) > 1:
                        self.env.base_env.render_target_pos(self.rm, target_positions[-1])
                elif goal_type == "vel":
                    p_face = data['face_p'][self.replay_cnt]
                    v_com = data['com_v'][self.replay_cnt]
                    v_target = data['target_v'][self.replay_cnt]
                    render_arrow_2D(p_face, p_face+v_com, color=[0.0, 0.0, 0.0, 1.0])
                    render_arrow_2D(p_face, p_face+v_target, color=[1.0, 0.0, 0.0, 0.8])
                elif goal_type == "path":
                    p_face = data['face_p'][self.replay_cnt]
                    p_target = data['target_p'][self.replay_cnt]
                    d_target = data['target_d'][self.replay_cnt]
                    v_com = data['com_v'][self.replay_cnt]
                    render_arrow_2D(p_target, p_target+d_target, color=[1, 0, 0, 1])
                    render_arrow_2D(p_face, p_face+v_com, color=[0.0, 0.0, 0.0, 1.0])
    def extra_overlay_callback(self):
        if self.rm.flag['overlay_text']:
            w, h = self.window_size
            font = self.rm.glut.GLUT_BITMAP_9_BY_15
            h_start = 50
            self.rm.gl_render.render_text(
                "Time: %.2f"%(self.env.base_env.get_elapsed_time()), pos=[0.05*w, h_start+20], font=font)
    def extra_idle_callback(self):
        time_elapsed = self.time_checker_auto_play.get_time(restart=False)
        if self.rm.flag['auto_play'] and time_elapsed >= self.env.base_env._dt_con:
            self.time_checker_auto_play.begin()
            self.one_step()
    def extra_keyboard_callback(self, key):
        if key == b'r':
            print("Reset w/o replay")
            self.replay = False
            self.reset()
        elif key == b'R':
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
        elif key == b' ':
            self.time_checker_auto_play.begin()
            self.one_step()
        elif key == b'e':
            self.explore = not self.explore
            print('Exploration:', self.explore)
        elif key == b'w':
            filename = input("[Policy Weights] Enter filename for saving: ")
            policy = self.trainer.get_policy()
            policy.model.save_weights(filename)
            print("Done.")
        elif key == b'W':
            filename = input("[Policy Weights] Enter filename for loading: ")
            policy = self.trainer.get_policy()
            policy.model.load_weights(filename)
            print("Done.")
        elif key == b's':
            save_image = get_bool_from_input("Save image")
            save_motion = get_bool_from_input("Save motion")
            save_replay = get_bool_from_input("Save replay data")
            save_dir = None
            if save_image or save_motion or save_replay:
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
            ''' Read end of episode '''
            check_eoe = get_bool_from_input("Check EOE")
            for i in range(num_iter):
                if reset_at_zero:
                    self.reset({'start_time': np.array([0.0])})
                else:
                    self.reset()
                if save_dir:
                    save_dir_i = os.path.join(save_dir, str(i))
                else:
                    save_dir_i = None
                if save_dir_i:
                    try:
                        os.makedirs(save_dir_i, exist_ok=True)
                    except OSError:
                        print("Invalid Subdirectory")
                        return
                time_elapsed = self.record_a_scene(
                    save_dir=save_dir_i, 
                    save_image=save_image,
                    save_motion=save_motion,
                    end_time=end_time, 
                    check_falldown=check_falldown, 
                    check_end_of_motion=check_end_of_motion,
                    check_end_of_episode=check_eoe)
                if save_replay:
                    if self.env.base_env._goal_type=="maze":
                        print("Maze:num_cell_visited", self.env.base_env._maze_num_visited)
                        if self.env.base_env._maze_num_visited >= 60:
                            filename = os.path.join(save_dir, "replay_%d.pkl"%(i))
                            pickle.dump(self.replay_data, open(filename, "wb"))
                            print(filename)
                    else:
                        filename = os.path.join(save_dir_i, "replay.pkl")
                        pickle.dump(self.replay_data, open(filename, "wb"))
                        print(filename)
            print('Done.')
        elif key == b'S':
            print('Load Replay Data...')
            name = input("Enter data file: ")
            with open(name, "rb") as f:
                self.replay_data = pickle.load(f)
            self.replay = True
            self.reset()
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
            R_face, _ = conversions.T2Rp(agent.get_facing_transform(h))
            origin += np.dot(R_face, self.cam_param_offset[0])
            pos += np.dot(R_face, self.cam_param_offset[1])
        
        param["origin"] = origin
        param["pos"] = pos
        
        return param
    def get_cam_parameters(self, use_buffer=True):
        if use_buffer:
            param = {
                "origin": None, 
                "pos": None, 
                "dist": None,
                "translate": None,
            }
            param["origin"] = np.mean([p["origin"] for p in self.cam_params], axis=0)
            param["pos"] = np.mean([p["pos"] for p in self.cam_params], axis=0)
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
        check_end_of_episode=False,
        verbose=True):
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
    return rm.camera.Camera(
        pos=np.array([0, 5, 5]),
        origin=np.zeros(3),
        vup=np.array([0, 0, 1]), 
        fov=60.0)

env_cls = HumanoidBoxingSolo

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
        "callbacks": MyCallbacks,
        "model": model_config,
    }
    return config
