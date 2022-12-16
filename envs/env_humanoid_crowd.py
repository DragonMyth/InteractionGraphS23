import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(directory))

import copy
import numpy as np

from fairmotion.utils import utils
from fairmotion.ops import conversions, motion, math

import motion_utils

from envs import env_humanoid_with_sensors_base

from collections import deque

class Env(env_humanoid_with_sensors_base.Env):
    def __init__(self, config):
        super().__init__(config)

    def create(self):
        self._goal_reached = np.zeros(self._num_agent, dtype=bool)
        launching_places = self._config["goal"]["launching_places"]
        assert len(launching_places)==self._num_agent
        self._target_pos = []
        for i in range(self._num_agent):
            _, p = self.sample_launching_place(launching_places[i])
            self._target_pos.append(p)

    def callback_reset_after(self, info):
        super().callback_reset_after(info)
        self._goal_reached[:] = False

    def callback_step_after(self, action_dict, infos):
        super().callback_step_after(action_dict, infos)
        for i in range(self._num_agent):
            self.update_goal_reached(i)
            infos[i]["num_goal_reached"] = np.sum(self._goal_reached)

    def get_state_by_key(self, idx, key):
        state = []
        
        if key=="body":
            state.append(self.state_body(idx, "sim"))
        elif key=="goal":
            state.append(self.state_goal(idx))
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
        else:
            raise NotImplementedError(key)

        return np.hstack(state)

    def state_goal(self, idx):
        state = []
        
        sim_agent = self._sim_agent[idx]
        R_sim, p_sim = conversions.T2Rp(
            sim_agent.get_facing_transform(self.get_ground_height(idx)))
        R_sim_inv = R_sim.transpose()

        v = self._target_pos[idx] - p_sim
        dist = np.linalg.norm(v)
        state.append(np.dot(R_sim_inv, v/dist))
        state.append(np.array([dist]))        

        return np.hstack(state)

    def reward_data(self, idx):
        data = {}
        sim_agent = self._sim_agent[idx]

        data['falldown'] = self._base_env.check_falldown(idx)
        data['hit_the_wall'] = self.check_hit_the_wall(sim_agent)
        data['task_complete'] = self.check_task_complete()
        data['hit_others'] = self.check_hit_others(idx)

        d, p = sim_agent.get_facing_direction_position(self.get_ground_height(idx))
        data['facing_dir'] = d
        data['facing_pos'] = p
        data['target_pos'] = self._target_pos[idx].copy()

        data['sim_joint_pv'] = self._sim_agent[idx].get_joint_states()

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

        facing_dir = data["facing_dir"]
        facing_pos = data["facing_pos"]
        target_pos = data["target_pos"]

        if self.exist_rew_fn_subterm(idx, "falldown"):
            error['falldown'] = 1.0 if data['falldown'] else 0.0
        if self.exist_rew_fn_subterm(idx, "hit_the_wall"):
            error['hit_the_wall'] = 1.0 if data['hit_the_wall'] else 0.0
        if self.exist_rew_fn_subterm(idx, "hit_others"):
            error['hit_others'] = 1.0 if data['hit_others'] else 0.0
        if self.exist_rew_fn_subterm(idx, "advance_to_goal"):
            facing_pos_prev = data_prev[idx]["facing_pos"]
            dist = np.linalg.norm(facing_pos-target_pos)
            dist_prev = np.linalg.norm(facing_pos_prev-target_pos)
            error["advance_to_goal"] = dist_prev - dist
        if self.exist_rew_fn_subterm(idx, "dist_to_goal"):
            dist = np.linalg.norm(target_pos - facing_pos)
            error["dist_to_goal"] = dist
        if self.exist_rew_fn_subterm(idx, "goal_reached"):
            dist = np.linalg.norm(target_pos - facing_pos)
            reached = dist < self._config["goal"]["dist_reaching"]
            error["goal_reached"] = 1.0 if reached else 0.0
        if self.exist_rew_fn_subterm(idx, 'energy_consumption'):
            p_prev, v_prev = data_prev[idx]['sim_joint_pv']
            p_next, v_next = data_next[idx]['sim_joint_pv']
            dv = np.hstack(v_next)-np.hstack(v_prev)
            error['energy_consumption'] = np.dot(dv, dv)

        return error

    def inspect_end_of_episode(self):
        eoe_reason = super().inspect_end_of_episode()
        if 'task_complete' in self._early_term_choices:
            check = self.check_task_complete()
            if check: eoe_reason.append('task_complete')
        return eoe_reason

    def update_goal_reached(self, idx):
        sim_agent = self._sim_agent[idx]
        d, p = sim_agent.get_facing_direction_position(self.get_ground_height(idx))
        dist = np.linalg.norm(self._target_pos[idx] - p)
        self._goal_reached[idx] = dist < self._config["goal"]["dist_reaching"]

    def check_task_complete(self):
        return np.sum(self._goal_reached)==self._num_agent

    def check_hit_others(self, idx):
        sim_agent = self._sim_agent[idx]
        for i in range(self._num_agent):
            if i==idx:
                continue
            other_agent = self._sim_agent[i]
            if self._base_env.check_collision(sim_agent._body_id, other_agent._body_id):
                return True
        return False

    # def render_sensors(self, rm, idx):
    #     idx_op = self.get_idx_op(idx)
    #     sensors = self._sensors[idx]
    #     sensors_config = self._config["character"]["sensors"][idx]
    #     for name in sensors.keys():
    #         sensor = sensors[name]
    #         sensor_config = sensors_config[name]
    #         sensor_type = sensor_config["type"]
    #         if sensor_type == "vision":
    #             ray_start = sensor[-1]["ray_start"]
    #             ray_end = sensor[-1]["ray_end"]
    #             hit_obj_idx = sensor[-1]["hit_obj_idx"]
    #             for i in range(len(ray_start)):
    #                 visible = hit_obj_idx[i] == idx_op
    #                 color = [1.0, 0.0, 0.0, 1.0] if visible else [0.0, 1.0, 0.0, 0.5]
    #                 rm.gl_render.render_line(
    #                     ray_start[i], ray_end[i], color=color, line_width=2.0)

    def render(self, rm):
        super().render(rm)

        if rm.get_flag('custom2'):
            for i in range(self._num_agent):
                rm.gl.glEnable(rm.gl.GL_LIGHTING)
                rm.gl_render.render_point(
                    self._target_pos[i], radius=0.2, color=rm.COLORS_FOR_AGENTS[i])
