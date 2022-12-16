import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(directory))

import copy
import numpy as np

from fairmotion.utils import utils, constants
from fairmotion.ops import conversions, motion, math

import motion_utils

from env import env_humanoid_with_sensors_base

from collections import deque

class Env(env_humanoid_with_sensors_base.Env):
    def __init__(self, config):
        super().__init__(config)

    def create(self):
        self._prev_opponent_pos = [np.zeros(3) for i in range(self._num_agent)]
        self._prev_opponent_vel = [np.zeros(3) for i in range(self._num_agent)]

    def callback_reset_after(self, info):
        for i in range(self._num_agent):
            self._prev_opponent_pos[i][:] = 0.0    
            self._prev_opponent_vel[i][:] = 0.0
        super().callback_reset_after(info)

    def get_state_by_key(self, idx, key):
        state = []
        
        if key=="body":
            state.append(self.state_body(idx, "sim"))
        elif key=="vision" or key=="vision_label":
            sensors = self._sensors[idx]
            sensors_config = self._config["character"]["sensors"][idx]
            for name in sensors.keys():
                sensor = sensors[name]
                sensor_config = sensors_config[name]
                sensor_type = sensor_config["type"]
                if sensor_type == "vision":
                    if key=="vision":
                        for s in sensor:
                            state.append(s["ray_dist"])
                    elif key=="vision_label":
                        for s in sensor:
                            state.append(self.get_idx_op(idx)==np.array(s["hit_obj_idx"]))
                            # print(idx, self.get_idx_op(idx)==np.array(s["hit_obj_idx"]))
                    else:
                        raise Exception
        elif key=="height_map":
            sensors = self._sensors[idx]
            sensors_config = self._config["character"]["sensors"][idx]
            for name in sensors.keys():
                sensor = sensors[name]
                sensor_config = sensors_config[name]
                sensor_type = sensor_config["type"]
                if sensor_type == "height_map":
                    state.append(np.hstack([s["height_rel"] for s in sensor]))
        elif key=="opponent_com_pos_vel_always" or \
             key=="opponent_com_pos_vel_when_visible" or \
             key=="opponent_com_pos_vel_when_visible_with_memory":
            
            if key=="opponent_com_pos_vel_when_visible":
                visible = self.check_opponent_visible(idx)
            elif key=="opponent_com_pos_vel_when_visible_with_memory":
                visible = self.check_opponent_visible(idx)
            else:
                visible = True
            
            if visible:
                idx_op = self.get_idx_op(idx)
                p_com_op, v_com_op = self._sim_agent[idx_op].get_com_and_com_vel()
                p_com_op = self._sim_agent[idx_op].project_to_ground(p_com_op)
                v_com_op = self._sim_agent[idx_op].project_to_ground(v_com_op)
                self._prev_opponent_pos[idx] = p_com_op
                self._prev_opponent_vel[idx] = v_com_op
            else:
                if key=="opponent_com_pos_vel_when_visible":
                    self._prev_opponent_pos[idx] = None
                    self._prev_opponent_vel[idx] = None
            
            if self._prev_opponent_pos[idx] is not None:
                R_face, p_face = conversions.T2Rp(
                    self._sim_agent[idx].get_facing_transform(self.get_ground_height(idx)))
                R_face_inv = R_face.transpose()
                diff_pos = self._prev_opponent_pos[idx] - p_face
                diff_pos_len = np.linalg.norm(diff_pos)
                v_com_op = self._prev_opponent_vel[idx]
                state.append(np.dot(R_face_inv, diff_pos/diff_pos_len))
                state.append(np.array([diff_pos_len]))
                state.append(np.dot(R_face_inv, v_com_op))
            else:
                state.append(np.zeros(3))
                state.append(np.array([0.0]))
                state.append(np.zeros(3))
        else:
            raise NotImplementedError(key)

        return np.hstack(state)

    def reward_data(self, idx):
        data = {}
        idx_op = self.get_idx_op(idx)
        agent1 = self._sim_agent[idx]
        agent2 = self._sim_agent[idx_op]
        p1 = agent1.get_facing_position(self.get_ground_height(idx))
        p2 = agent2.get_facing_position(self.get_ground_height(idx_op))

        data['falldown'] = self._base_env.check_falldown(idx)
        data['alive'] = not data['falldown']
        data['hit_the_wall'] = self.check_hit_the_wall(agent1)
        data['gameover'] = self.check_gameover()
        data['distance'] = np.linalg.norm(p2-p1)
        data['opponent_visible'] = self.check_opponent_visible(idx)

        return data

    def reward_max(self):
        return 1.0
    
    def reward_min(self):
        return 0.0

    def get_task_error(self, idx, data_prev, data_next, action_dict):
        error = {}

        sim_agent = self._sim_agent[idx]
        char_info = sim_agent._char_info

        data = data_next[idx]

        if self.exist_rew_fn_subterm(idx, "falldown"):
            error['falldown'] = 1.0 if data['falldown'] else 0.0
        if self.exist_rew_fn_subterm(idx, "alive"):
            error['alive'] = 1.0 if data['alive'] else 0.0
        if self.exist_rew_fn_subterm(idx, "hit_the_wall"):
            error['hit_the_wall'] = 1.0 if data['hit_the_wall'] else 0.0
        if self.exist_rew_fn_subterm(idx, "gameover"):
            error["gameover"] = 1.0 if data['gameover'] else 0.0
        if self.exist_rew_fn_subterm(idx, "delta_dist_always"):
            dist = data["distance"]
            dist_prev = data_prev[idx]["distance"]
            error["delta_dist_always"] = dist_prev - dist
        if self.exist_rew_fn_subterm(idx, "delta_dist_when_visible_asymetric"):
            visible = data["opponent_visible"]
            if visible:
                dist = data["distance"]
                dist_prev = data_prev[idx]["distance"]
                error["delta_dist_when_visible_asymetric"] = dist_prev - dist
            else:
                error["delta_dist_when_visible_asymetric"] = 0.0
        if self.exist_rew_fn_subterm(idx, "dist_always"):
            dist = data["distance"]
            error["dist_always"] = np.dot(dist, dist)
        if self.exist_rew_fn_subterm(idx, "dist_always_clip"):
            dist = data["distance"]
            if dist <= 5.0:
                error["dist_always_clip"] = np.dot(dist, dist)
            else:
                error["dist_always_clip"] = 1e5
        if self.exist_rew_fn_subterm(idx, "dist_when_visible_asymetric"):
            visible = data["opponent_visible"]
            if visible:
                dist = data["distance"]
                error["dist_when_visible_asymetric"] = np.dot(dist, dist)
            else:
                error["dist_when_visible_asymetric"] = 1e5
        if self.exist_rew_fn_subterm(idx, "tanh3"):
            dist = data["distance"]
            error["tanh3"] = np.tanh(dist-3)
        if self.exist_rew_fn_subterm(idx, "tanh5"):
            dist = data["distance"]
            error["tanh5"] = np.tanh(dist-5)

        return error

    def inspect_end_of_episode(self):
        eoe_reason = super().inspect_end_of_episode()
        if 'gameover' in self._early_term_choices:
            check = self.check_gameover()
            if check: eoe_reason.append('gameover')
        return eoe_reason

    def check_gameover(self):
        return self._base_env.check_collision(
            self._sim_agent[0]._body_id, self._sim_agent[1]._body_id)

    def get_idx_op(self, idx):
        if idx==0:
            return 1
        elif idx==1:
            return 0
        else:
            raise Exception('Invalid index')

    def check_opponent_visible(self, idx):
        idx_op = self.get_idx_op(idx)
        sensors = self._sensors[idx]
        sensors_config = self._config["character"]["sensors"][idx]
        visible = False
        for name in sensors.keys():
            sensor = sensors[name]
            sensor_config = sensors_config[name]
            sensor_type = sensor_config["type"]
            if sensor_type == "vision":
                visible = visible or idx_op in sensor[-1]["hit_obj_idx"]
            if visible:
                break
        return visible

    def render_sensors(self, rm, idx):
        idx_op = self.get_idx_op(idx)
        sensors = self._sensors[idx]
        sensors_config = self._config["character"]["sensors"][idx]
        for name in sensors.keys():
            sensor = sensors[name]
            sensor_config = sensors_config[name]
            sensor_type = sensor_config["type"]
            if sensor_type == "vision":
                ray_start = sensor[-1]["ray_start"]
                ray_end = sensor[-1]["ray_end"]
                hit_obj_idx = sensor[-1]["hit_obj_idx"]
                for i in range(len(ray_start)):
                    visible = hit_obj_idx[i] == idx_op
                    color = [1.0, 0.0, 0.0, 1.0] if visible else [0.0, 1.0, 0.0, 0.5]
                    rm.gl_render.render_line(
                        ray_start[i], ray_end[i], color=color, line_width=2.0)

    def render(self, rm):
        super().render(rm)
