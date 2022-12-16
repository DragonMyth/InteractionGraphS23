import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(directory))

import copy
import numpy as np

from fairmotion.utils import utils, constants
from fairmotion.ops import conversions, motion, math

import motion_utils

from envs import env_humanoid_with_sensors_base

from collections import deque

class Env(env_humanoid_with_sensors_base.Env):
    def __init__(self, config):
        super().__init__(config)

    def create(self):
        self._goal_type = self._config['goal']['type']
        
        if self._goal_type=="pos" or self._goal_type=="vel" or self._goal_type=="path":
            wall_offset = [0.0, 0.0, 0.0]
            self.update_goal(init=True)
        elif self._goal_type=="maze":
            self._ray_height_offset = -2.0
            wall_offset = [0.0, self._ray_height_offset, 0.0]
            self.init_maze()
            self._visited_info = []
        else:
            raise NotImplementedError

    def update_goal(self, init=False):
        sim_agent = self._sim_agent[0]
        char_info = sim_agent._char_info

        if self._goal_type=="pos":
            places = self._config["goal"]["launching_places"]
            R_face, p_face = conversions.T2Rp(
                sim_agent.get_facing_transform(self.get_ground_height(0)))
            R, p = self.sample_launching_place(places, R_face, p_face)
            self._target_pos = p
        elif self._goal_type=="vel":
            ''' Sample target direction and velocity randomly '''
            if init:
                v_dir = sim_agent.get_facing_direction()
                v_mag = np.random.uniform(
                    self._config["goal"]["v_mag_min"], 
                    self._config["goal"]["v_mag_max"])
            else:
                angle = np.random.uniform(0, 2*np.pi)
                v_dir = np.dot(
                    conversions.A2R(angle*char_info.v_up_env), 
                    char_info.v_ax1_env)
                v_mag = np.random.uniform(
                    self._config["goal"]["v_mag_min"], 
                    self._config["goal"]["v_mag_max"])
            
            self._target_vel_dir = v_dir
            self._target_vel_mag = v_mag
            self._target_vel = v_mag * v_dir

            ''' Buffer for averaging the velocity of the character '''
            self._time_left_to_complete = np.random.uniform(
                self._config["goal"]["time_to_complete_min"], 
                self._config["goal"]["time_to_complete_max"])
        elif self._goal_type=="path":
            if init:
                ax1 = char_info.v_ax1_env
                ax2 = char_info.v_ax2_env
                N = 1200
                K = 5
                A = int(self._config["goal"]["radius"])
                b = int(self._config["goal"]["speed"])
                self._target_stride = int(self._config["goal"]["future_input_stride"])
                self._target_path_pos = []
                self._target_path_dir = []
                self._target_path_len = N
                self._target_path_pos_render = []
                for i in range(N):
                    t = 2*np.pi*i/float(N)
                    c1 = A * np.sin(b*t)
                    c2 = A * np.sin(b*t)*np.cos(b*t)
                    c1_dot = A * b * np.cos(b*t)
                    c2_dot = A * b * (np.cos(b*t)*np.cos(b*t) - np.sin(b*t)*np.sin(b*t))
                    p = c1 * ax1 + c2 * ax2
                    v = c1_dot * ax1 + c2_dot * ax2
                    v = v / np.linalg.norm(v)
                    self._target_path_pos.append(p)
                    self._target_path_dir.append(v)
                    if i%K==0:
                        self._target_path_pos_render.append(p)
                self._target_path_cnt = 0
            S = self._target_stride
            self._target_pos = [
                self._target_path_pos[(self._target_path_cnt)%self._target_path_len],
                self._target_path_pos[(self._target_path_cnt+1*S)%self._target_path_len],
                self._target_path_pos[(self._target_path_cnt+2*S)%self._target_path_len],
                self._target_path_pos[(self._target_path_cnt+3*S)%self._target_path_len],
            ]
            self._target_dir = [
                self._target_path_dir[(self._target_path_cnt)%self._target_path_len],
                self._target_path_dir[(self._target_path_cnt+1*S)%self._target_path_len],
                self._target_path_dir[(self._target_path_cnt+2*S)%self._target_path_len],
                self._target_path_dir[(self._target_path_cnt+3*S)%self._target_path_len],
            ]
        else:
            raise NotImplementedError

    def init_maze(self):
        N = self._config["goal"]["num_buckets"]
        D = self._config["goal"]["map_size"]
        self._maze_visited = np.zeros((N[0], N[1]), dtype=bool)
        self._maze_buckets = []
        self._maze_num_visited = 0
        self._maze_num_visited_max = \
            int(N[0]*N[1]*self._config["goal"]["coverage_ratio_for_completion"])
        for i in range(N[0]):
            b = []
            for j in range(N[1]):
                x = float(i)/N[0]*D[0] - 0.5*D[0] + 0.5*D[0]/N[0]
                z = float(j)/N[1]*D[1] - 0.5*D[1] + 0.5*D[1]/N[1]
                b.append(np.array([x, 0.5, z]))
            self._maze_buckets.append(b)

    def update_maze(self):
        ''' 
        Indices were clamped to prepare divergence of simulation 
        '''
        N = self._config["goal"]["num_buckets"]
        D = self._config["goal"]["map_size"]
        p = self._sim_agent[0].get_facing_position(0)
        i = max(0, min(int(N[0] * p[0]/D[0] + 0.5*N[0]), N[0]-1))
        j = max(0, min(int(N[1] * p[2]/D[1] + 0.5*N[1]), N[1]-1))
        if not self._maze_visited[i][j]:
            self._maze_visited[i][j] = True
            self._maze_num_visited += 1

    def check_visited_maze(self, p):
        ''' 
        Indices were clamped to prepare divergence of simulation 
        '''
        N = self._config["goal"]["num_buckets"]
        D = self._config["goal"]["map_size"]
        i = max(0, min(int(N[0] * p[0]/D[0] + 0.5*N[0]), N[0]-1))
        j = max(0, min(int(N[1] * p[2]/D[1] + 0.5*N[1]), N[1]-1))
        return self._maze_visited[i][j]
    
    def check_new_visit_maze(self):
        p = self._sim_agent[0].get_facing_position(0)
        return not self.check_visited_maze(p)

    def callback_reset_prev(self, info):
        super().callback_reset_prev(info)

    def callback_reset_after(self, info):
        ''' Give the initial target vel corresponding to the facing dir '''
        if self._goal_type=="pos":
            self.update_goal(init=True)
        elif self._goal_type=="vel":
            self.update_goal(init=True)
        elif self._goal_type=="path":
            self.update_goal(init=True)
        elif self._goal_type=="maze":
            self.init_maze()
        else:
            raise NotImplementedError

        ''' Buffer for averaging the velocity of the character '''
        self._vel_buffer = deque([], maxlen=self._fps_con)

        super().callback_reset_after(info)

    def callback_step_end(self, action_dict, infos):
        if self._goal_type=="pos":
            p = self._sim_agent[0].get_facing_position(self.get_ground_height(0))
            dist = np.linalg.norm(self._target_pos - p)
            if dist < self._config["goal"]["dist_reaching"]:
                self.update_goal()
        elif self._goal_type=="vel":
            self._time_left_to_complete -= self._dt_con
            if self._time_left_to_complete <= 0.0:
                self.update_goal()
        elif self._goal_type=="path":
            self._target_path_cnt = (self._target_path_cnt+1)%self._target_path_len
            self.update_goal()
        elif self._goal_type=="maze":
            self.update_maze()
        else:
            raise NotImplementedError

    def inspect_end_of_episode(self):
        eoe_reason = super().inspect_end_of_episode()
        if 'task_complete' in self._early_term_choices:
            check = self.check_task_complete()
            if check: eoe_reason.append('task_complete')
        return eoe_reason

    def check_task_complete(self):
        if self._goal_type=="pos":
            return False
        elif self._goal_type=="vel":
            return False
        elif self._goal_type=="path":
            return False
        elif self._goal_type=="maze":
            return self._maze_num_visited >= self._maze_num_visited_max
        else:
            raise NotImplementedError

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
        elif key=="gps":
            sim_agent = self._sim_agent[idx]
            d_face, p_face = \
                sim_agent.get_facing_direction_position(self.get_ground_height(idx))
            state.append(d_face)
            state.append(p_face)
        else:
            raise NotImplementedError

        return np.hstack(state)

    def state_goal(self, idx):
        state = []
        
        sim_agent = self._sim_agent[idx]
        R_sim, p_sim = conversions.T2Rp(
            sim_agent.get_facing_transform(self.get_ground_height(idx)))
        R_sim_inv = R_sim.transpose()

        if self._goal_type=="pos":
            v = self._target_pos - p_sim
            dist = np.linalg.norm(v)
            state.append(np.dot(R_sim_inv, v/dist))
            state.append(np.array([dist]))
        elif self._goal_type=="vel":
            p_com, v_com = sim_agent.get_com_and_com_vel()
            v_com_ground = sim_agent.project_to_ground(v_com)
            state.append(np.dot(R_sim_inv, self._target_vel))
            state.append(np.dot(R_sim_inv, self._target_vel-v_com_ground))
        elif self._goal_type=="path":
            for i in range(len(self._target_pos)):
                state.append(np.dot(R_sim_inv, self._target_pos[i]-p_sim))
                state.append(np.dot(R_sim_inv, self._target_dir[i]))
        elif self._goal_type=="maze":
            state.append(self.get_visitation_map(idx))
        else:
            raise NotImplementedError

        return np.hstack(state)

    def get_visitation_map(self, idx):
        sim_agent = self._sim_agent[idx]

        h_face = self.get_ground_height(idx)
        R_face = conversions.T2R(sim_agent.get_facing_transform(h_face))
        d_face, p_face = sim_agent.get_facing_direction_position(h_face)

        width = 9
        height = 9
        N_w = 5
        N_h = 5
        forward_dist = 0.0

        visited = []

        ww = np.linspace(-0.5*width, 0.5*width, num=N_w, endpoint=True)
        hh = np.linspace(-0.5*height, 0.5*height, num=N_h, endpoint=True)
        self._visited_info = []
        for w in ww:
            data = []
            for h in hh:
                p = p_face + forward_dist * d_face + np.dot(R_face, np.array([w, 0, h]))
                visited.append(self.check_visited_maze(p))
                self._visited_info.append((p, self.check_visited_maze(p)))
        return visited

    def check_goal_visible(self, idx):
        assert self._goal_type=="pos"
        p_face = self._sim_agent[idx].get_facing_position(self.get_ground_height(idx))
        p_face[1] += 1.4
        p_target = self._target_pos + np.array([0, 0.3, 0])
        d1 = np.linalg.norm(p_target-p_face)
        res = self._pb_client.rayTest(p_face, p_target)[0]
        return res[2] > 0.99

    def check_goal_visible_pos(self, idx):
        assert self._goal_type=="pos"
        p_face = self._sim_agent[idx].get_facing_position(self.get_ground_height(idx))
        p_face[1] += 1.4
        p_target = self._target_pos + np.array([0, 0.3, 0])
        d1 = np.linalg.norm(p_target-p_face)
        res = self._pb_client.rayTest(p_face, p_target)[0]
        return p_face, p_target

    def collect_sensor_data(self, idx, key):
        return super().collect_sensor_data(idx, key)

    def reward_data(self, idx):
        data = {}
        sim_agent = self._sim_agent[idx]

        data['falldown'] = self._base_env.check_falldown(idx)
        data['hit_the_wall'] = self.check_hit_the_wall(sim_agent)
        data['task_complete'] = self.check_task_complete()

        if self._goal_type=="pos":
            _, v_com = sim_agent.get_com_and_com_vel()
            d, p = sim_agent.get_facing_direction_position(self.get_ground_height(idx))
            data['v_com'] = v_com
            data['facing_dir'] = d
            data['facing_pos'] = p
            data['target_pos'] = self._target_pos.copy()
            data['goal_visible'] = self.check_goal_visible(idx)
        elif self._goal_type=="vel":
            _, v_com = sim_agent.get_com_and_com_vel()
            data['v_com_ground'] = sim_agent.project_to_ground(v_com)
            data['facing_dir'] = sim_agent.get_facing_direction()
            data['target_vel_dir'] = self._target_vel_dir.copy()
            data['target_vel_mag'] = self._target_vel_mag
        elif self._goal_type=="path":
            d, p = sim_agent.get_facing_direction_position(self.get_ground_height(idx))
            data['facing_dir'] = d
            data['facing_pos'] = p
            data['target_pos'] = self._target_pos[0].copy()
            data['target_dir'] = self._target_dir[0].copy()
        elif self._goal_type=="maze":
            data['new_visit'] = self.check_new_visit_maze()
        else:
            raise NotImplementedError        

        return data

    def reward_max(self):
        return np.inf
    
    def reward_min(self):
        return -np.inf

    def get_task_error(self, idx, data_prev, data_next, action_dict):
        error = {}

        sim_agent = self._sim_agent[idx]
        char_info = sim_agent._char_info

        data = data_next[idx]

        if self.exist_rew_fn_subterm(idx, "falldown"):
            error['falldown'] = 1.0 if data['falldown'] else 0.0
        if self.exist_rew_fn_subterm(idx, "hit_the_wall"):
            error['hit_the_wall'] = 1.0 if data['hit_the_wall'] else 0.0
        if self.exist_rew_fn_subterm(idx, "task_complete"):
            error['task_complete'] = 1.0 if data['task_complete'] else 0.0

        if self._goal_type=="pos":
            v_com = data["v_com"]
            facing_dir = data["facing_dir"]
            facing_pos = data["facing_pos"]
            target_pos = data["target_pos"]
            goal_visible = data["goal_visible"]
            if self.exist_rew_fn_subterm(idx, "vel_to_goal"):
                v_dir = target_pos - facing_pos
                v_dir /= np.linalg.norm(v_dir)
                v_com_proj_len = np.linalg.norm(math.projectionOnVector(v_com, v_dir))
                error["vel_to_goal"] = max(
                    self._config["goal"]["v_target_min"]-v_com_proj_len, 
                    0.0)
            if self.exist_rew_fn_subterm(idx, "advance_to_goal"):
                facing_pos_prev = data_prev[idx]["facing_pos"]
                dist = np.linalg.norm(facing_pos-target_pos)
                dist_prev = np.linalg.norm(facing_pos_prev-target_pos)
                error["advance_to_goal"] = dist_prev - dist
            if self.exist_rew_fn_subterm(idx, "advance_to_goal_when_visible"):
                facing_pos_prev = data_prev[idx]["facing_pos"]
                dist = np.linalg.norm(facing_pos-target_pos)
                dist_prev = np.linalg.norm(facing_pos_prev-target_pos)
                advancement = dist_prev - dist
                if advancement < 0.0:
                    error["advance_to_goal_when_visible"] = advancement
                else:
                    if goal_visible:
                        error["advance_to_goal_when_visible"] = advancement
                    else:
                        error["advance_to_goal_when_visible"] = 0.0
            if self.exist_rew_fn_subterm(idx, "goal_reached"):
                dist = np.linalg.norm(target_pos - facing_pos)
                reached = dist < self._config["goal"]["dist_reaching"]
                error["goal_reached"] = 1.0 if reached else 0.0
        elif self._goal_type=="vel":
            v_com_ground = data["v_com_ground"]
            facing_dir = data["facing_dir"]
            target_vel_dir = data["target_vel_dir"]
            target_vel_mag = data["target_vel_mag"]
            l_v_com_ground = np.linalg.norm(v_com_ground)
            if self.exist_rew_fn_subterm(idx, "v_com_dir"):
                if l_v_com_ground >= 0.2:
                    v_dir = v_com_ground / l_v_com_ground
                else:
                    v_dir = facing_dir
                diff = target_vel_dir - v_dir
                error["v_com_dir"] = np.dot(diff, diff)
            if self.exist_rew_fn_subterm(idx, "v_com_mag"):
                diff = target_vel_mag - l_v_com_ground
                error["v_com_mag"] = np.dot(diff, diff)
        elif self._goal_type=="path":
            facing_dir = data["facing_dir"]
            facing_pos = data["facing_pos"]
            target_dir = data["target_dir"]
            target_pos = data["target_pos"]
            if self.exist_rew_fn_subterm(idx, "pos"):
                diff = target_pos - facing_pos
                error["pos"] = np.dot(diff, diff)
            if self.exist_rew_fn_subterm(idx, "dir"):
                diff = target_dir - facing_dir
                error["dir"] = np.dot(diff, diff)
        elif self._goal_type=="maze":
            if data["new_visit"]:
                error["cover"] = 1.0
            else:
                error["cover"] = 0.0
        else:
            raise NotImplementedError

        return error

    def render_target_pos(self, rm, p, height=1.5):
        rm.gl.glEnable(rm.gl.GL_LIGHTING)
        rm.gl_render.render_point(p, radius=0.2, color=[1.0, 0.0, 0.0, 0.8])
        if self._v_up_str=="y":
            height = 1.5
            pp = p + np.array([0.0, 0.5*height, 0.0])
            rm.gl_render.render_cylinder(
                conversions.Rp2T(conversions.Ax2R(0.5*np.pi), pp), 
                height, 0.1, color=[1.0, 0, 0, 0.8])
        else:
            raise NotImplementedError

    def render(self, rm):
        super().render(rm)
        agent = self._sim_agent[0]

        p_face = agent.get_facing_position(self.get_ground_height(0))

        def render_arrow_2D(p1, p2, diameter=0.05, color=[0.0, 0.0, 0.0, 1.0]):
            rm.gl.glDisable(rm.gl.GL_LIGHTING)
            rm.gl.glPushMatrix()
            rm.gl.glScalef(1.0, 0.1, 1.0)
            rm.gl_render.render_arrow(p1, p2, D=diameter, color=color)
            rm.gl.glPopMatrix()

        if self._goal_type=="pos":
            if rm.get_flag('custom2'):
                self.render_target_pos(rm, self._target_pos.copy())
        elif self._goal_type=="vel":
            if rm.get_flag('custom2'):
                _, v = agent.get_com_and_com_vel()
                v = agent.project_to_ground(v)
                render_arrow_2D(p_face, p_face+v, color=[0.0, 0.0, 0.0, 1.0])
                render_arrow_2D(p_face, p_face+self._target_vel, color=[1.0, 0.0, 0.0, 0.8])
        elif self._goal_type=="path":
            def render_path(
                data, color=[0.0, 0.0, 0.0], line_width=1.0, closed=False):
                rm.gl.glDisable(rm.gl.GL_LIGHTING)
                rm.gl.glColor(color)
                rm.gl.glLineWidth(line_width)
                if closed:
                    rm.gl.glBegin(rm.gl.GL_LINE_LOOP)
                else:
                    rm.gl.glBegin(rm.gl.GL_LINE_STRIP)
                for p in data:
                    rm.gl.glVertex3d(p[0], p[1]+0.01, p[2])
                rm.gl.glEnd()
            render_path(
                self._target_path_pos_render, 
                color=[1.0, 0.0, 0.0, 1.0],
                line_width=5, 
                closed=True)
            if rm.get_flag('custom2'):
                render_arrow_2D(
                    self._target_pos[0], 
                    self._target_pos[0]+self._target_dir[0],
                    color=[1, 0, 0, 1])
                _, v = agent.get_com_and_com_vel()
                v = agent.project_to_ground(v)
                v /= np.linalg.norm(v)
                render_arrow_2D(p_face, p_face+v, color=[0.0, 0.0, 0.0, 1.0])
        elif self._goal_type=="maze":
            if rm.get_flag('custom2'):
                rm.gl.glEnable(rm.gl.GL_LIGHTING)
                for i in range(len(self._maze_buckets)):
                    for j in range(len(self._maze_buckets[i])):
                        p = self._maze_buckets[i][j]
                        color = [1.0, 0.0, 0.0] if self._maze_visited[i][j] else [0.5, 0.5, 0.5]
                        rm.gl_render.render_point(p, radius=0.2, color=color)
            if rm.get_flag('custom3'):
                for p, visited in self._visited_info:
                    color = [0, 0, 1] if visited else [0, 0, 0]
                    rm.gl_render.render_point(p, radius=0.1, color=color)
        else:
            raise NotImplementedError
