import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(directory))

import copy
import numpy as np

from fairmotion.utils import utils, constants
from fairmotion.ops import conversions, motion, math

import motion_utils
from bullet import bullet_utils as bu

from envs import env_humanoid_with_sensors_base

from collections import deque

import random

class Env(env_humanoid_with_sensors_base.Env):
    def __init__(self, config):
        super().__init__(config)

    def create(self):
        self._physical_target = self.create_physical_target()
        if self._physical_target:
            self._physical_target_init_state = \
                bu.get_state_all(self._pb_client, self._physical_target)
        # self._goal_type = self._config['goal']['type']
        
        # if self._goal_type=="pos" or self._goal_type=="vel" or self._goal_type=="path":
        #     wall_offset = [0.0, 0.0, 0.0]
        #     self.update_goal(init=True)
        # else:
        #     raise NotImplementedError

        ''' extempore or stream '''
        self._goal_type = self._config["goal"].get("type", "extempore")

        self._hit_source = \
            self._config["goal"].get("hit_source", ["lwrist", "rwrist"])
        self._hit_source_idx = \
            [self._sim_agent[0]._char_info.joint_idx[n] for n in self._hit_source]
        self._target_change_pos_when_hit = \
            self._config["goal"].get("change_pos_when_hit", False)
        self._min_hit_force = \
            self._config["goal"].get("min_hit_force", 5.0)
        self._idle_start_time = \
            self._config["goal"].get("idle_start_time", 0.1)

        self._target_buffer = deque()
        self.fill_target_buffer(self._target_buffer)

        self._stream_lookahead = self._config["goal"].get("stream_lookahead", 3)
        assert self._stream_lookahead >= 1

        self.reset_target(change_pos=True)

        assert self._target_reset_time > self._idle_start_time

    def fill_target_buffer(self, target_buffer, num=100):
        for _ in range(num):
            reset_time = np.random.uniform(
                self._config["goal"].get("reset_time_min", 1.0),
                self._config["goal"].get("reset_time_max", 1.0))
            hit_source = random.choice(self._hit_source)
            places = self._config["goal"]["launching_places"]
            R, p = self.sample_launching_place(places)
            p[2] = self._config["goal"]["height"]
            target_buffer.append((reset_time, hit_source, p))

    def get_wall_urdf_file(self):
        return None

    def create_physical_target(self):
        file = self._config['goal'].get('physical_target_urdf_file')
        if file is None: return None

        target = self._pb_client.loadURDF(file, [0, 1, 2], useFixedBase=True)
        # for lid in range(-1, self._pb_client.getNumJoints(wall)):
        #     self._pb_client.setCollisionFilterPair(target, self._ground, lid, -1, False)

        num_joint = self._pb_client.getNumJoints(target)

        # Disable the initial motor control
        for j in range(num_joint):
            self._pb_client.changeDynamics(
                target, 
                j, 
                jointDamping=0.5,
                linearDamping=0.5, 
                angularDamping=0.5,
                )
            self._pb_client.setJointMotorControl2(
                target,
                j, 
                self._pb_client.POSITION_CONTROL, 
                targetVelocity=0, 
                force=0)
            self._pb_client.setJointMotorControlMultiDof(
                target,
                j, 
                self._pb_client.POSITION_CONTROL,
                targetPosition=[0,0,0,1], 
                targetVelocity=[0,0,0], 
                positionGain=0, 
                velocityGain=0, 
                force=[0,0,0])
            
        return target

    def reset_target(self, change_pos=False):
        if len(self._target_buffer) <= 10:
            self.fill_target_buffer(self._target_buffer)
        reset_time, hit_source, p = self._target_buffer.popleft()
        self._target_reset_time = reset_time
        self._cur_hit_source = hit_source
        self._target_hit_elapsed = 0.0
        if change_pos:
            self._target_pos = p
            if self._physical_target:
                p, _, _, _ = bu.get_base_pQvw(self._pb_client, self._physical_target)
                p_new = [self._target_pos[0],self._target_pos[1],p[2]]
                init_state = self._physical_target_init_state.copy()
                init_state[0] = p_new
                bu.set_state_all(self._pb_client, self._physical_target, init_state)

    def callback_reset_prev(self, info):
        super().callback_reset_prev(info)

    def callback_reset_after(self, info):
        self.reset_target(change_pos=True)
        self._num_hit_during_episode = 0
        self._hit_force = self.compute_hit_force()
        super().callback_reset_after(info)

    def callback_step_end(self, action_dict, infos):
        super().callback_step_end(action_dict, infos)
        
        if self._target_hit_elapsed >= self._target_reset_time:
            self.reset_target(
                change_pos=self._target_change_pos_when_hit)
        
        self._hit_force = self.compute_hit_force()

        if self._target_hit_elapsed == 0.0:
            cur_hit_force = self._hit_force[self._cur_hit_source]
            if max(cur_hit_force) >= self._min_hit_force:
                self._target_hit_elapsed = self._dt_con
        else:
            self._target_hit_elapsed += self._dt_con

        if self._target_hit_elapsed == self._dt_con:
            self._num_hit_during_episode += 1
        
        infos[0]["hit_the_target"] = self._num_hit_during_episode

    def compute_hit_force(self):
        hit_force = {
            "lwrist": [0.0],
            "rwrist": [0.0],
        }
        '''
        When a physical target exists, we check real collision.
        Otherwise, it is measured by the distance.
        '''
        sim_agent = self._sim_agent[0]
        if self._physical_target:
            pts_l = self._pb_client.getContactPoints(
                bodyA=sim_agent._body_id, 
                bodyB=self._physical_target,
                linkIndexA=sim_agent._char_info.joint_idx["lwrist"])
            for p in pts_l:
                hit_force["lwrist"].append(p[9])
            pts_r = self._pb_client.getContactPoints(
                bodyA=sim_agent._body_id, 
                bodyB=self._physical_target,
                linkIndexA=sim_agent._char_info.joint_idx["rwrist"])
            for p in pts_r:
                hit_force["rwrist"].append(p[9])
        else:
            ps, Qs, vs, ws = sim_agent.get_link_states([
                sim_agent._char_info.lwrist,
                sim_agent._char_info.rwrist,
                ])
            if "lwrist" in self._hit_source:
                dist = np.linalg.norm(self._target_pos - ps[0])
                if dist < self._config["goal"]["dist_target_hit"]:
                    hit_force["lwrist"].append(self._min_hit_force)
            if "rwrist" in self._hit_source:
                dist = np.linalg.norm(self._target_pos - ps[1])
                if dist < self._config["goal"]["dist_target_hit"]:
                    hit_force["rwrist"].append(self._min_hit_force)
        return hit_force

    def inspect_end_of_episode(self):
        eoe_reason = super().inspect_end_of_episode()
        if 'task_complete' in self._early_term_choices:
            check = self.check_task_complete()
            if check: eoe_reason.append('task_complete')
        return eoe_reason

    def check_task_complete(self):
        return False

    def is_idle(self):
        return self._target_hit_elapsed >= self._idle_start_time

    def is_first_hit(self):
        return self._target_hit_elapsed == self._dt_con

    def get_state_by_key(self, idx, key):
        state = []
        
        if key=="body":
            state.append(self.state_body(idx, "sim"))
        elif key=="goal":
            state.append(self.state_goal(idx))
        else:
            raise NotImplementedError

        return np.hstack(state)

    def state_goal(self, idx):
        state = []
        
        sim_agent = self._sim_agent[idx]
        R_root, p_root = conversions.T2Rp(sim_agent.get_root_transform())
        R_root_inv = R_root.transpose()

        if self._physical_target:
            p, Q, v, w = bu.get_link_pQvw(self._pb_client, self._physical_target)
            if self._goal_type == "extempore":
                state.append(np.dot(R_root_inv, p[0]-p_root))
                state.append(np.dot(R_root_inv, v[0]))
                state.append(self._cur_hit_source=="lwrist")
                state.append(self._cur_hit_source=="rwrist")
            elif self._goal_type == "stream":
                ''' Currently moving target '''
                state.append(np.dot(R_root_inv, p[0]-p_root))
                state.append(np.dot(R_root_inv, v[0]))
                ''' Current and future target patterns '''
                buffer = [(self._target_reset_time, self._cur_hit_source, self._target_pos)]
                for i in range(self._stream_lookahead-1):
                    buffer.append(self._target_buffer[i])
                t = self._target_reset_time
                for reset_time, hit_source, p in buffer:
                    state.append(np.dot(R_root_inv, p-p_root))
                    state.append(hit_source=="lwrist")
                    state.append(hit_source=="rwrist")
                    state.append(t-self._target_hit_elapsed)
                    t += reset_time
            else:
                raise NotImplementedError
        else:
            state.append(np.dot(R_root_inv, self._target_pos-p_root))
        state.append(self.is_idle())

        return np.hstack(state)

    def reward_data(self, idx):
        data = {}
        sim_agent = self._sim_agent[idx]

        data['falldown'] = self._base_env.check_falldown(idx)
        data['hit_the_wall'] = self.check_hit_the_wall(sim_agent)
        data['task_complete'] = self.check_task_complete()
        data['first_hit'] = self.is_first_hit()
        data['idle'] = self.is_idle()

        d, p = sim_agent.get_facing_direction_position(self.get_ground_height(idx))
        data['facing_dir'] = d
        data['facing_pos'] = p
        data['target_pos'] = self._target_pos.copy()

        ps, Qs, vs, ws = sim_agent.get_link_states(self._hit_source_idx)

        for i, n in enumerate(self._hit_source):
            data[n] = (ps[i], vs[i])

        p_root, Q_root, v_root, w_root = sim_agent.get_root_state()

        data['root_p'] = p_root
        data['root_v'] = v_root

        if self._physical_target:
            cur_hit_force = self._hit_force[self._cur_hit_source]
            data['hit_force_max'] = max(cur_hit_force)
            data['hit_force_max_all'] = [max(self._hit_force[s]) for s in self._hit_source]

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
        if self.exist_rew_fn_subterm(idx, "hit_the_wall"):
            error['hit_the_wall'] = 1.0 if data['hit_the_wall'] else 0.0
        if self.exist_rew_fn_subterm(idx, "task_complete"):
            error['task_complete'] = 1.0 if data['task_complete'] else 0.0

        target_pos = data['target_pos']
        target_pos_on_ground = np.array([target_pos[0],target_pos[1],0])
        facing_pos = data["facing_pos"]
        dist_to_target = np.linalg.norm(target_pos_on_ground - facing_pos)
        
        if dist_to_target > self._config["goal"]["phase1_dist_reached"]:
            if self.exist_rew_fn_subterm(idx, "phase1_dist_to_target"):
                error['phase1_dist_to_target'] = dist_to_target*dist_to_target
            if self.exist_rew_fn_subterm(idx, "phase2_dist_to_target_L"):
                error['phase2_dist_to_target_L'] = 1.0e5
            if self.exist_rew_fn_subterm(idx, "phase2_vel_to_target_L"):
                error['phase2_vel_to_target_L'] = 0.0
            if self.exist_rew_fn_subterm(idx, "phase2_dist_to_target_R"):
                error['phase2_dist_to_target_R'] = 1.0e5
            if self.exist_rew_fn_subterm(idx, "phase2_vel_to_target_R"):
                error['phase2_vel_to_target_R'] = 0.0
            if self.exist_rew_fn_subterm(idx, "phase2_keep_distance"):
                error['phase2_keep_distance'] = 1.0e5
            if self.exist_rew_fn_subterm(idx, "phase2_facing_dir_alignment"):
                error['phase2_facing_dir_alignment'] = 1.0e5
            if self.exist_rew_fn_subterm(idx, "hit_the_target"):
                error['hit_the_target'] = 1.0 if data['first_hit'] else 0.0
            if self.exist_rew_fn_subterm(idx, "hit_force_max"):
                error['hit_force_max'] = 0.0
            if self.exist_rew_fn_subterm(idx, "hit_during_idle"):
                error['hit_during_idle'] = 0.0
        else:
            root_vel = data['root_v']
            v_dir = (target_pos_on_ground - facing_pos)/dist_to_target
            if self.exist_rew_fn_subterm(idx, "phase1_dist_to_target"):
                error['phase1_dist_to_target'] = 1.0e5
            if self.exist_rew_fn_subterm(idx, "phase2_dist_to_target_L"):
                lwrist_pos = data['lwrist'][0]
                diff = target_pos-lwrist_pos
                error['phase2_dist_to_target_L'] = np.dot(diff, diff)
            if self.exist_rew_fn_subterm(idx, "phase2_vel_to_target_L"):
                lwrist_vel = data['lwrist'][1]
                error['phase2_vel_to_target_L'] = \
                    np.clip(0.5*np.dot(lwrist_vel, v_dir), 0.0, 1.0)
            if self.exist_rew_fn_subterm(idx, "phase2_dist_to_target_R"):
                rwrist_pos = data['rwrist'][0]
                diff = target_pos-rwrist_pos
                error['phase2_dist_to_target_R'] = np.dot(diff, diff)
            if self.exist_rew_fn_subterm(idx, "phase2_vel_to_target_R"):
                rwrist_vel = data['rwrist'][1]
                error['phase2_vel_to_target_R'] = \
                    np.clip(0.5*np.dot(rwrist_vel, v_dir), 0.0, 1.0)
            if self.exist_rew_fn_subterm(idx, "phase2_keep_distance"):
                diff = dist_to_target - self._config["goal"]["phase2_keep_distance"]
                error['phase2_keep_distance'] = diff*diff
            if self.exist_rew_fn_subterm(idx, "phase2_facing_dir_alignment"):
                d, p = data['facing_dir'], data['facing_pos']
                l = target_pos_on_ground - p
                l /= np.linalg.norm(l)
                diff = np.dot(d, l) - 1.0
                error['phase2_facing_dir_alignment'] = diff * diff
            if self.exist_rew_fn_subterm(idx, "hit_the_target"):
                error['hit_the_target'] = 1.0 if data['first_hit'] else 0.0
            if self.exist_rew_fn_subterm(idx, "hit_force_max"):
                if data['first_hit']:
                    hit_force_max = data['hit_force_max']
                    f = np.clip(
                        hit_force_max, 
                        0, 
                        self._config["goal"].get("max_hit_force", 200.0))
                    if f < self._config["goal"].get("min_hit_force", 5.0):
                        f = 0.0
                    scale_force = self._config["goal"].get("scale_force", 0.01)
                    error['hit_force_max'] = np.clip(scale_force*f, 0, 1)
                else:
                    error['hit_force_max'] = 0.0
            if self.exist_rew_fn_subterm(idx, "hit_during_idle"):
                if data['idle']:
                    hit_force_max = data['hit_force_max']
                    f = np.clip(
                        hit_force_max, 
                        0, 
                        self._config["goal"].get("max_hit_force", 200.0))
                    scale_force = self._config["goal"].get("scale_force", 0.01)
                    error['hit_during_idle'] = np.clip(scale_force*f, 0, 1)
                else:
                    error['hit_during_idle'] = 0.0
        
        return error

    def render_target_pos(self, rm, p, height=1.5):
        rm.gl.glEnable(rm.gl.GL_LIGHTING)
        rm.gl_render.render_point(
            p, 
            radius=self._config["goal"]["dist_target_hit"], 
            color=[1.0, 0.0, 0.0, 0.8])
        height = self._config["goal"]["height"]
        pp = np.array([p[0], p[1], 0.5*height])
        rm.gl_render.render_cylinder(
            conversions.Rp2T(conversions.Ax2R(0.0*np.pi), pp), 
            height, 
            0.5*self._config["goal"]["dist_target_hit"], 
            color=[1.0, 0, 0, 0.8])
        rm.gl_render.render_circle(
            conversions.p2T(np.array([p[0], p[1], 0.0])),
            r=self._config["goal"]["phase1_dist_reached"],
            slice=64,
            scale=1.0,
            line_width=5.0,
            color=[0, 0, 0],
            draw_plane="xy",
        )
        phase2_keep_distance = self._config["goal"].get("phase2_keep_distance")
        if phase2_keep_distance:
            rm.gl_render.render_circle(
                conversions.p2T(np.array([p[0], p[1], 0.0])),
                r=phase2_keep_distance,
                slice=64,
                scale=1.0,
                line_width=5.0,
                color=[0, 0, 0],
                draw_plane="xy",
            )

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

        if self._physical_target:
            if self.is_idle():
                color = [0.3,0.3,0.3,1.0]
            elif self._cur_hit_source == "lwrist":
                color = [0.9,0.0,0.3,1.0]
            elif self._cur_hit_source == "rwrist":
                color = [0.3,0.3,0.9,1.0]

            color = [0.3,0.3,0.3,1.0]

            rm.bullet_render.render_model(
                self._pb_client, 
                self._physical_target,
                draw_link=True, 
                draw_link_info=True, 
                draw_joint=False, 
                draw_joint_geom=False, 
                link_info_scale=1.0,
                link_info_color=[0, 0, 0, 1],
                link_info_line_width=2.0,
                color=color,
                lighting=True)
        else:
            self.render_target_pos(rm, self._target_pos.copy())
