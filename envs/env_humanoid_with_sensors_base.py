import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(directory))

import copy
import numpy as np

from fairmotion.utils import utils, constants
from fairmotion.ops import conversions, motion, math

import motion_utils
from envs import env_humanoid_base
from collections import deque

class Env(env_humanoid_base.Env):
    def __init__(self, config):
        super().__init__(config)

        self._initialized = False
        self._start_time = np.zeros(self._num_agent)

        self._wall = self.create_wall()

        self._sensors = []
        config_sensor = self._config['character'].get('sensors')
        if config_sensor:
            for i in range(self._num_agent):
                sensors = {}
                for name in config_sensor[i].keys():
                    num_history = config_sensor[i][name]['num_history']
                    sensors[name] = deque([], maxlen=num_history)
                self._sensors.append(sensors)

        for i in range(self._num_agent):
            self.update_sensors(i, reset=True)

        project_dir = self._config['project_dir']
        ref_motion_db = self._config['character'].get('ref_motion_db')
        ref_motion_file = motion_utils.collect_motion_files(project_dir, ref_motion_db)
        
        ''' 
        Load reference motion. This is only used for initialize the character.
        '''

        self._ref_motion_all = []
        self._ref_motion_file_names = []
        for i in range(self._num_agent):
            ref_motion_all, ref_motion_file_names = \
                motion_utils.load_motions(
                    ref_motion_file[i], 
                    None,
                    self._sim_agent[i]._char_info,
                    self._verbose)
            self._ref_motion_all.append(ref_motion_all)
            self._ref_motion_file_names.append(ref_motion_file_names)

        self.create()

        self.reset({'add_noise': False})

        self._initialized = True

    def create(self):
        raise NotImplementedError

    def get_wall_urdf_file(self):
        environment_file = self._config['character'].get('environment_file')
        if environment_file:
            return environment_file[0]
        else:
            return None

    def create_wall(self):
        wall_file = self.get_wall_urdf_file()
        wall = None
        if wall_file:
            wall = self._pb_client.loadURDF(
                wall_file, [0, 0, 0], useFixedBase=True)
            for lid in range(-1, self._pb_client.getNumJoints(wall)):
                self._pb_client.setCollisionFilterPair(wall, self._ground, lid, -1, False)
        return wall

    def sample_launching_place(self, places, R_face=None, p_face=None):
        place = np.random.choice(places)
        if place["type"]=="box":
            p = np.random.uniform(place["p_min"], place["p_max"])
            p = np.array([p[0], 0, p[1]])
        elif place["type"]=="circle":
            angle = np.random.uniform(0.0, 2*np.pi)
            radius = np.random.uniform(place["radius_min"], place["radius_max"])
            p = np.array(place["p_center"]) + radius*np.array([np.cos(angle), np.sin(angle)])
            if self._v_up_str=="y":
                p = np.array([p[0], 0, p[1]])
            elif self._v_up_str=="z":
                p = np.array([p[0], p[1], 0])
            else:
                raise NotImplementedError
        elif place["type"]=="fan":
            angle = place.get("angle")
            assert angle[0] <= angle[1]
            angle = np.random.uniform(angle[0], angle[1])
            radius = np.random.uniform(place["radius_min"], place["radius_max"])
            p = np.array(place["p_center"]) + radius*np.array([np.cos(angle), np.sin(angle)])
            if self._v_up_str=="y":
                p = np.array([p[0], 0, p[1]])
            elif self._v_up_str=="z":
                p = np.array([p[0], p[1], 0])
            else:
                raise NotImplementedError

        rotation = place.get("rotation")
        if rotation:
            assert rotation[0] <= rotation[1]
            angle = np.random.uniform(rotation[0], rotation[1])
            if self._v_up_str=="y":
                R = conversions.Ay2R(angle)
            elif self._v_up_str=="z":
                R = conversions.Az2R(angle)
            else:
                raise NotImplementedError
        else:
            R = constants.eye_R()

        mode = place.get("mode", "absolute")
        if mode == "absolute":
            pass
        elif mode == "relative":
            ''' 
            In the relative mode, 
            R and p should be computed from facing_transform
            '''
            R = np.dot(R_face, R)
            p = p_face + np.dot(R_face, p)
        else:
            raise NotImplementedError

        h = self.get_ground_height_at([p])[0]
        p = p + h*self._v_up

        return R, p

    def collect_sensor_data(self, idx, config):
        sensor_type = config["type"]
        if sensor_type=="vision":
            return self.get_vision(idx, config)
        if sensor_type=="height_map":
            return self.get_height_map(idx, config)
        return Exception("Undefined sensor", config)

    def update_sensors(self, idx, reset=True):
        if len(self._sensors) == 0:
            return
        sensors = self._sensors[idx]
        sensors_config = self._config["character"]["sensors"][idx]
        for name in sensors.keys():
            data = self.collect_sensor_data(idx, sensors_config[name])
            sensors[name].append(data)
            if reset:
                for _ in range(sensors[name].maxlen-1):
                    sensors[name].append(sensors[name][-1])

    def get_vision(self, idx, config):
        sim_agent = self._sim_agent[idx]
        
        vision_rays = []
        hit_object_idx = []

        agent_radius = config["agent_radius"]
        num_ray = config["num_ray"]
        max_dist_ray = config["max_dist_ray"]
        angle = np.radians(config["angle_ray"])
        angle_offset = np.radians(config["angle_ray_offset"])
        height_offset = config["height_offset"]
        attachment = config["attachment"]

        if attachment=="none":
            d_face, p_face = \
                sim_agent.get_facing_direction_position(0)
        elif attachment=="ground":
            d_face, p_face = \
                sim_agent.get_facing_direction_position(self.get_ground_height(0))
        else:
            raise NotImplementedError
        ray_start = []
        ray_end = []
        for i in range(num_ray):
            theta = -0.5 * angle + angle * i/float(num_ray-1) + angle_offset
            R = conversions.A2R(theta * self._v_up)
            d = np.dot(R, d_face)
            rs = p_face + height_offset * self._v_up + agent_radius * d
            re = p_face + height_offset * self._v_up + max_dist_ray * d
            ray_start.append(rs)
            ray_end.append(re)
        res = self._pb_client.rayTestBatch(ray_start, ray_end)
        for i in range(len(ray_start)):
            ray_end[i] = ray_start[i] + res[i][2]*(ray_end[i]-ray_start[i])
            vision_rays.append(res[i][2]*max_dist_ray)
            hit_object_idx.append(res[i][0])

        data = {
            "ray_dist": vision_rays,
            "hit_obj_idx": hit_object_idx,
            "ray_start": ray_start,
            "ray_end": ray_end,
        }

        return data

    def get_height_map(self, idx, config):
        sim_agent = self._sim_agent[idx]

        h_face = self.get_ground_height(idx)
        R_face = conversions.T2R(sim_agent.get_facing_transform(h_face))
        d_face, p_face = sim_agent.get_facing_direction_position(h_face)

        width = config["width"]
        height = config["height"]
        N_w = config["num_w"]
        N_h = config["num_h"]
        forward_dist = config["forward_dist"]

        ps = []
        ps_rel = []
        ghs = []
        ghs_rel = []

        ww = np.linspace(-0.5*width, 0.5*width, num=N_w, endpoint=True)
        hh = np.linspace(-0.5*height, 0.5*height, num=N_h, endpoint=True)
        for w in ww:
            for h in hh:
                p = p_face + forward_dist * d_face + np.dot(R_face, np.array([w, 0, h]))
                ps.append(p)
                ps_rel.append(p.copy())
        ghs = self.get_ground_height_at(ps)
        for i, gh in enumerate(ghs):
            ps[i][1] = gh
            ps_rel[i][1] = gh-h_face
            ghs_rel.append(gh-h_face)

        data = {
            "height_abs": ghs,
            "height_rel": ghs_rel,
            "pos_on_the_ground": ps,
        }

        return data

    def callback_reset_prev(self, info):
        ''' Choose a reference motion randomly whenever reset '''
        
        self._ref_motion = \
            self.sample_ref_motion(info.get('ref_motion_id'))
        
        ''' Choose a start time for the current reference motion '''
        
        if 'start_time' in info.keys():
            self._start_time = np.array(info.get('start_time'))
            assert self._start_time.shape[0] == self._num_agent
        else:
            for i in range(self._num_agent):
                self._start_time[i] = \
                    np.random.uniform(0.0, self._ref_motion[i].length())

    def callback_reset_after(self, info):
        for i in range(self._num_agent):
            self.update_sensors(i, reset=True)

    def callback_step_after(self, action_dict, infos):
        for i in range(self._num_agent):
            self.update_sensors(i, reset=False)

    def print_log_in_step(self):
        if self._verbose and self._end_of_episode:
            print('=================EOE=================')
            print('Reason:', self._end_of_episode_reason)
            print('TIME: (start:%s) (elapsed:%02f) (time_after_eoe: %02f)'\
                %(self._start_time,
                  self.get_elapsed_time(),
                  self._time_elapsed_after_end_of_episode))
            print('=====================================')

    def compute_init_pose_vel(self, info):
        ''' This performs reference-state-initialization (RSI) '''
        init_poses, init_vels = [], []

        for i in range(self._num_agent):
            ''' Set the state of simulated agent by using the state of reference motion '''
            init_pose = self._ref_motion[i].get_pose_by_time(self._start_time[i])
            init_vel = self._ref_motion[i].get_velocity_by_time(self._start_time[i])
            launching_places = self._config["character"].get("launching_places")
            if launching_places:
                ''' Adjust the height of pose based on the current height map '''
                init_pose = copy.deepcopy(init_pose)
                R, p = conversions.T2Rp(init_pose.get_root_transform())
                h = self._sim_agent[i]._ref_scale * self.get_ground_height_at([p])[0]
                if self._v_up_str=="y":
                    p_face = np.array([p[0], h, p[2]])
                elif self._v_up_str=="z":
                    p_face = np.array([p[0], p[1], h])
                else:
                    raise NotImplementedError
                dR_face_launch, p_face_launch = self.sample_launching_place(launching_places[i])
                R = np.dot(dR_face_launch, R)
                p += p_face_launch - p_face
                init_pose.set_root_transform(conversions.Rp2T(R, p), local=False)
            init_poses.append(init_pose)
            init_vels.append(init_vel)
        return init_poses, init_vels

    def get_state_by_key(self, idx, key):
        raise NotImplementedError

    def state_body(self, 
                   idx, 
                   agent="sim", 
                   type=None, 
                   return_stacked=True):
        if agent == "sim":
            agent = self._sim_agent[idx]
        elif agent == "kin":
            agent = self._kin_agent[idx]
        else:
            raise NotImplementedError
        return self._state_body(idx, agent, type, return_stacked)

    def state_task(self, idx):
        sc = self._state_choices.copy()
        sc.remove('body')
        return self.state(idx, sc)

    def check_hit_the_wall(self, agent):
        if self._wall:
            return self._base_env.check_collision(agent._body_id, self._wall)
        else:
            return False

    def inspect_end_of_episode(self):
        eoe_reason = super().inspect_end_of_episode()
        if 'hit_the_wall' in self._early_term_choices:
            check = self.check_hit_the_wall(self._sim_agent[0])
            if check: eoe_reason.append('[%s] hit_the_wall'%self._sim_agent[0]._name)
        return eoe_reason

    def sample_ref_motion(self, indices=None):
        ref_indices = []
        ref_motions = []
        for i in range(self._num_agent):
            if indices is None:
                idx = np.random.randint(len(self._ref_motion_all[i]))
            else:
                idx = indices[i]
            ref_indices.append(idx)
            ref_motions.append(self._ref_motion_all[i][idx])
        if self._verbose:
            print("Ref. motions selected:", ref_indices)
        return ref_motions

    def render_sensors(self, rm, idx):
        rm.gl.glDisable(rm.gl.GL_LIGHTING)
        if len(self._sensors)==0:
            return
        sensors = self._sensors[idx]
        sensors_config = self._config["character"]["sensors"][idx]
        for name in sensors.keys():
            sensor = sensors[name]
            sensor_config = sensors_config[name]
            sensor_type = sensor_config["type"]
            if sensor_type == "vision":
                offset = -0.5 * self._v_up
                rm.gl.glPushMatrix()
                rm.gl.glTranslatef(offset[0],offset[1],offset[2])
                ray_start = sensor[-1]["ray_start"]
                ray_end = sensor[-1]["ray_end"]
                for rs, re in zip(ray_start, ray_end):
                    rm.gl_render.render_line(
                        rs, re,
                        color=[0.0, 1.0, 0.0, 0.8],
                        line_width=2.0)
                rm.gl.glPopMatrix()
            elif sensor_type == "height_map":
                pos_on_the_ground = sensor[-1]["pos_on_the_ground"]
                for p in pos_on_the_ground:
                    rm.gl_render.render_point(p, radius=0.05, color=[0, 1, 0])
            else:
                raise NotImplementedError(sensor_type)

    def render(self, rm):
        super().render(rm)
        if self._wall:
            rm.bullet_render.render_model(
                self._pb_client, 
                self._wall,
                draw_link=True, 
                draw_link_info=False, 
                draw_joint=False, 
                draw_joint_geom=False, 
                link_info_scale=1.0,
                link_info_color=[0, 0, 0, 1],
                link_info_line_width=3.0,
                color=[0.8,0.8,0.8,1.0],
                lighting=True)
        if rm.get_flag('custom1'):
            for i in range(self._num_agent):
                self.render_sensors(rm, i)
