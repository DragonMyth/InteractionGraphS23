import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(directory))

import pickle
import copy

import numpy as np
from collections import deque

from fairmotion.utils import utils, constants
from fairmotion.ops import math, conversions, motion, quaternion
from fairmotion.data import bvh
from scipy.spatial import distance

import motion_utils

from envs import env_humanoid_base

# from motion_plausibility import scorer
# model = scorer.ZScorer("/checkpoint/dgopinath/models/z_task/5_2_0005_ce/60.model")

class Env(env_humanoid_base.Env):
    def __init__(self, config):
        super().__init__(config)

        ''' Setup for Games '''

        DEFAULT_GAME_CONFIG = {
            "type": "first_touch",
            "min_touch_force": 1.0,
            "max_force_clip": 100.0,
            "init_score": 100.0,
            "duration": 30.0,
            "touch_interval": 0.5,
            "init_dir_random_degree": 0.0,
            "init_dist_random_min": 0.7,
            "init_dist_random_max": 3.0,
            "too_close_duration": 3.0,
            "too_close_distance": 0.4,
            "out_of_arena_duration": 5.0,
            "arena_width": 4.7,
            "arena_outer_margin": 1.05,
            "target_links": {
                "lowerneck": 1.0,
                "upperneck": 1.0,
                "lowerback": 1.0,
                "upperback": 1.0,
                "chest": 1.0,
                "lclavicle": 1.0,
                "rclavicle": 1.0,
            },
        }

        config_game = DEFAULT_GAME_CONFIG.copy()
        config_game_custom = self._config.get('game')
        if config_game_custom:
            config_game.update(config_game_custom)

        assert self._num_agent == 2
        
        self._initialized = False
        self._ref_motion_all = None
        self._ref_motion = None
        self._start_time = np.zeros(self._num_agent)
        self._timeover = False
        self._gameover = False
        self._game_result = "none"

        self._knee_correction = self._config.get('knee_correction', False)

        ''' Setup Arena '''

        environment_file = self._config['character'].get('environment_file')
        self._rope = None
        if environment_file:
            tf_rope = conversions.R2Q(constants.eye_R())
            self._rope = self._pb_client.loadURDF(
                environment_file[0], [0.0, 0.0, 0.0], tf_rope, useFixedBase=True)
        
        # self._arena_size = [6.0, 6.0]
        # outer_margin = 1.2
        self._arena_size = [
            config_game.get('arena_width'), 
            config_game.get('arena_width')]
        arena_outer_margin = config_game.get('arena_outer_margin')
        
        # 8 corners of the arena
        w_x, w_y = self._arena_size[0], self._arena_size[1]
        self._arena_h = 0.0005
        self._arena_boundaries = [
            np.array([0.5*w_x, 0.5*w_y, self._arena_h]),
            np.array([0.5*w_x, -0.5*w_y, self._arena_h]),
            np.array([-0.5*w_x, 0.5*w_y, self._arena_h]),
            np.array([-0.5*w_x, -0.5*w_y, self._arena_h]),
        ]
        self._arena_boundaries_outer = [
            arena_outer_margin*self._arena_boundaries[0],
            arena_outer_margin*self._arena_boundaries[1],
            arena_outer_margin*self._arena_boundaries[2],
            arena_outer_margin*self._arena_boundaries[3],
        ]
        self._arena_boundaries_outer[0][2] = 0.5*self._arena_h
        self._arena_boundaries_outer[1][2] = 0.5*self._arena_h
        self._arena_boundaries_outer[2][2] = 0.5*self._arena_h
        self._arena_boundaries_outer[3][2] = 0.5*self._arena_h
        self._arena_info = [
            {
                "T_ref": conversions.Rp2T(
                    conversions.Az2R(0.0), 
                    np.array([-0.5*w_x, -0.5*w_y, self._arena_h])),
                "R_ref": conversions.Az2R(0.0),
                "p_ref": np.array([0, -0.5*w_y, self._arena_h]),
                "R_ref_inv": conversions.Az2R(0.0).transpose(),
            },
            {   
                "T_ref": conversions.Rp2T(
                    conversions.Az2R(np.pi), 
                    np.array([0.5*w_x, 0.5*w_y, self._arena_h])),
                "R_ref": conversions.Az2R(np.pi),
                "p_ref": np.array([0, 0.5*w_y, self._arena_h]),
                "R_ref_inv": conversions.Az2R(np.pi).transpose(),
            },
        ]

        self._phase = np.zeros(self._num_agent)
        self._dphase_default = \
            self._config.get('dphase_default', 0.0) * \
            np.ones(self._num_agent)

        self._contact_info = [
            np.zeros(self._sim_agent[0]._num_link, dtype=bool),
            np.zeros(self._sim_agent[1]._num_link, dtype=bool),
        ]

        ''' The type of the game '''
        self._game_type = config_game.get('type')
        ''' The minium force required to be considered as a valid touch '''
        self._game_min_touch_force = config_game.get('min_touch_force')
        ''' The contact force is cliped by this value  '''
        self._game_max_force_clip = config_game.get('max_force_clip')
        ''' The duration of the game '''
        self._game_duration = config_game.get('duration')
        ''' The initial score that the agents have when the game starts  '''
        self._game_init_score = config_game.get('init_score')
        ''' The game restarts if the players are too close for this duration  '''
        self._game_too_close_duration = config_game.get('too_close_duration')
        self._game_too_close_distance = config_game.get('too_close_distance')
        self._game_too_close_elapsed = 0.0
        self._game_out_of_arena_duration = config_game.get('out_of_arena_duration')
        self._game_out_of_arena_elapsed = np.zeros(self._num_agent)
        ''' 
        The remaining score of the agents currently have.
        If the score becomes to zero, the game ends.
        '''
        self._game_remaining_score = \
            np.array([self._game_init_score, self._game_init_score])
        '''
        How much score was changed for each step of the game
        '''
        self._game_remaining_score_change = np.zeros(2)
        ''' A minumum time between getting a socre by touching '''
        self._game_touch_interval = config_game.get('touch_interval')
        ''' Whether the agent can get a point by touching the opponent '''
        self._game_touch_avail = np.ones(self._num_agent, dtype=bool)
        ''' How long the agent should wait until it can get a next point '''
        self._game_touch_avail_remaining = np.zeros(self._num_agent, dtype=int)

        self._game_init_dir_random_rad = \
            np.radians(config_game.get('init_dir_random_degree'))
        self._game_init_dist_random_min = \
            config_game.get('init_dist_random_min')
        self._game_init_dist_random_max = \
            config_game.get('init_dist_random_max')

        self._target_links = {}
        for k, v in config_game.get('target_links').items():
            link_idx = self._sim_agent[0]._char_info.joint_idx[k]
            link_weight = v
            self._target_links[link_idx] = link_weight

        assert self._game_type in ['total_score']
        assert self._game_duration > 0.0
        assert self._game_min_touch_force >= 0.0
        assert len(self._target_links) > 0
        
        if self._game_type=='total_score':
            assert self._game_touch_interval >= 0.0
            assert self._game_init_score > 0

        self._cur_touch_info = [{} for i in range(self._num_agent)]

        ''' Motion Plausibility Model '''
        motion_scorer = self._config.get('motion_scorer')
        self._motion_scorers = []
        if motion_scorer:
            self._motion_scorers.append(
                scorer.ZScorer(motion_scorer[i]))

        ref_motion_embedding_replay = \
            self._config['character'].get('ref_motion_embedding_replay', [])
        if len(ref_motion_embedding_replay) > 0:
            for file in ref_motion_embedding_replay:
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                    self._ref_motion_embedding_replay.append(data)

        if self._config.get('lazy_creation'):
            if self._verbose:
                print('The environment was created in a lazy fashion.')
                print('The function \"create\" should be called before it')
            return

        self.create()

    def create(self):
        project_dir = self._config['project_dir']
        ref_motion_db = self._config['character'].get('ref_motion_db')

        ''' Load Reference Motion '''

        if ref_motion_db is not None:
            ref_motion_file = motion_utils.collect_motion_files(project_dir, ref_motion_db)
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

        ''' Setup for Contrastive Reward '''

        config_contrastive = self._config.get('contrastive')
        self._contrastive_models = [None for i in range(self._num_agent)]
        
        if config_contrastive:
            assert self._ref_motion_all is not None
            assert len(self._ref_motion_all[i]) > 0
            self._past_window_size = config_contrastive.get('past_window_size')
            self._prev_poses = [[] for i in range(self._num_agent)]
            
            from motion_plausibility import scorer
            models = config_contrastive.get('model')
            for i in range(self._num_agent):
                if models[i]:
                    self._contrastive_models[i] = \
                        scorer.PlausibilityScorer(path=models[i])

        ''' Should call reset after all setups are done '''

        self.reset({'add_noise': False})

        self._initialized = True

        if self._verbose:
            print('----- Humanoid Boxing Environment Created -----')
            for i in range(self._num_agent):
                print('[Agent%d]: state(%d) and action(%d)' \
                      %(i, len(self.state(i)), self._action_space[i].dim))
            print('-------------------------------')

    def callback_reset_prev(self, info):
        if self._ref_motion_all is not None:
            ''' Choose a reference motion randomly whenever reset '''
            if 'ref_motion_idx' in info.keys():
                self._ref_motion = \
                    self.sample_ref_motion(info.get('ref_motion_idx'))
            else:
                self._ref_motion = self.sample_ref_motion()
            
            if 'start_time' in info.keys():
                self._start_time = info.get('start_time')
                assert len(self._start_time) == self._num_agent
            else:
                for i in range(self._num_agent):
                    self._start_time[i] = \
                        np.random.uniform(0.0, self._ref_motion[i].length())

        self._phase[:] = 0.0

    def callback_reset_after(self, info):
        self._timeover = False
        self._gameover = False
        self._game_result = "none"

        for i in range(self._num_agent):
            self._cur_touch_info[i] = {}

        self._game_touch_avail[:] = True
        self._game_touch_avail_remaining[:] = 0
        self._game_remaining_score[:] = self._game_init_score
        self._game_remaining_score_change[:] = 0.0
        self._game_too_close_elapsed = 0.0
        self._game_out_of_arena_elapsed[:] = 0.0

        # if self._ref_motion_all is not None:
        #     for i in range(self._num_agent):
        #         cur_time = self.get_current_time(i)
        #         self._kin_agent[i].set_pose(
        #             self._ref_motion[i].get_pose_by_time(cur_time),
        #             self._ref_motion[i].get_velocity_by_time(cur_time)
        #         )
        
        def place_players(p1=None, p2=None, x_range=2.0, y_range=2.0):
            if p1 is None or p2 is None:
                while True:
                    p1 = np.random.uniform(
                        np.array([-x_range, -y_range]), np.array([x_range, y_range]))
                    p2 = np.random.uniform(
                        np.array([-x_range, -y_range]), np.array([x_range, y_range]))
                    d = np.linalg.norm(p1-p2)
                    if self._game_init_dist_random_min <= d and \
                       self._game_init_dist_random_max >= d:
                        break
            p_mid = 0.5 * (p1 + p2)

            dd1 = np.zeros(3)
            dd2 = np.zeros(3)

            dd1[:2] = (p_mid - p1)
            dd2[:2] = (p_mid - p2)

            dd1 = dd1 / np.linalg.norm(dd1)
            dd2 = dd2 / np.linalg.norm(dd2)
            
            fd1 = self._sim_agent[0].get_facing_direction()
            fd2 = self._sim_agent[1].get_facing_direction()

            ''' Add a bit of randomness on the directions '''

            fd1 = np.dot(conversions.Az2R(
                np.random.uniform(
                    -self._game_init_dir_random_rad,
                    self._game_init_dir_random_rad)),
                fd1)
            fd2 = np.dot(conversions.Az2R(
                np.random.uniform(
                    -self._game_init_dir_random_rad,
                    self._game_init_dir_random_rad)),
                fd2)

            self._sim_agent[0].set_root_transform(
                np.dot(
                    conversions.R2T(math.R_from_vectors(fd1, dd1)), 
                    self._sim_agent[0].get_root_transform()))
            self._sim_agent[1].set_root_transform(
                np.dot(
                    conversions.R2T(math.R_from_vectors(fd2, dd2)), 
                    self._sim_agent[1].get_root_transform()))

            fp1 = self._sim_agent[0].get_facing_position(0.0)
            fp2 = self._sim_agent[1].get_facing_position(0.0)

            self._sim_agent[0].set_root_transform(
                np.dot(
                    conversions.p2T(np.array([-fp1[0]+p1[0], -fp1[1]+p1[1], 0.0])),
                    self._sim_agent[0].get_root_transform()))
            self._sim_agent[1].set_root_transform(
                np.dot(
                    conversions.p2T(np.array([-fp2[0]+p2[0], -fp2[1]+p2[1], 0.0])),
                    self._sim_agent[1].get_root_transform()))

        if 'player_pos' in info.keys():
            pos = info.get('player_pos')
            place_players(p1=pos[0], p2=pos[1])
        else:
            place_players()

        for i in range(self._num_agent):
            if self._contrastive_models[i] is not None:
                self._prev_poses[i].clear()
                for j in range(self._past_window_size[i], -1, -1):
                    time = max(0.0, self._start_time[i] - j * self._dt_con)
                    self._prev_poses[i].append(
                        self._ref_motion[i].get_pose_by_time(time)
                    )

        self.update_contact_info()

    def callback_step_after(self, action_dict, infos):
        ''' Increase phase variables '''
        self._phase += self._dphase_default
        if 'dphase' in action_dict:
            self._phase += action_dict['dphase']

        for i in range(self._num_agent):
            while not(0.0 <= self._phase[i] < 1.0):
                if self._phase[i] >= 1.0: self._phase[i] -= 1.0
                if self._phase[i] < 0.0: self._phase[i] += 1.0

        ''' Contact Info Update '''
        self.update_contact_info()

        ''' Check time over '''
        if not self._timeover:
            self._timeover = self.get_elapsed_time()>self._game_duration
        
        ''' Update the current state of the game '''
        if not self._gameover:
            game_remaining_score_prev = self._game_remaining_score.copy()
            
            for i in range(self._num_agent):
                ''' Check if the agent physically is touching the opponent '''
                self._cur_touch_info[i] = self.check_opponent_touched(i)

            for i in range(self._num_agent):
                if self._game_touch_avail[i]:                
                    idx_op = self.get_idx_op(i)
                    if len(self._cur_touch_info[i])>0:
                        f = np.clip(
                                np.max(self._cur_touch_info[i]['normal_force']), 
                                0.0, 
                                self._game_max_force_clip)
                        ''' Decrease the opponent's score when a valid touch occurs '''
                        self._game_remaining_score[idx_op] = \
                            max(0, self._game_remaining_score[idx_op]-f)
                        ''' Disable next touch for a wile if touch_interval was set '''
                        if self._game_touch_interval > 0.0:
                            self._game_touch_avail[i] = False
                            self._game_touch_avail_remaining[i] = \
                                    int(self._game_touch_interval / self._dt_con)
                else:
                    self._game_touch_avail_remaining[i] = \
                        max(self._game_touch_avail_remaining[i]-1, 0)
                    self._game_touch_avail[i] = (self._game_touch_avail_remaining[i]==0)
            
            ''' Recored score changes for an efficient reward computation '''
            self._game_remaining_score_change = \
                self._game_remaining_score - game_remaining_score_prev

            ''' Check the duration when two players are too close '''
            p1 = self._sim_agent[0].get_facing_position(self.get_ground_height(0))
            p2 = self._sim_agent[1].get_facing_position(self.get_ground_height(1))
            dist = np.linalg.norm(p1-p2)
            if dist <= self._game_too_close_distance:
                self._game_too_close_elapsed += self._dt_con
            else:
                self._game_too_close_elapsed = 0.0
            
            ''' Check the duration of out-of-arena '''
            for i in range(self._num_agent):
                if self.check_out_of_arena(i):
                    self._game_out_of_arena_elapsed[i] += self._dt_con
                else:
                    self._game_out_of_arena_elapsed[i] = 0.0
            
            # print('------------------------------')
            # print('Score: ', self._game_remaining_score)
            # print('Score change: ', self._game_remaining_score_change)
            # print('Touch avail: ', self._game_touch_avail)
            # print('Touch avail remaining: ', self._game_touch_avail_remaining)
            # print('EOE:', self._end_of_episode_reason)
            # print('------------------------------')

        ''' Check if the current state of the game '''
        if not self._gameover:
            self._game_result = self.judge_game()
            self._gameover = self._game_result in ['p1', 'p2', 'draw']

        ''' Collect poses if contrastive model is used '''
        for i in range(self._num_agent):
            if self._contrastive_models[i] is not None:
                self._prev_poses[i].append(
                    self._sim_agent[i].get_pose(self._base_motion[i].skel)
                )

        return infos

    def print_log_in_step(self):
        super().print_log_in_step()
        if self._verbose:
            print('score:', self._game_remaining_score)
            print('score_change:', self._game_remaining_score_change)
    
    def compute_init_pose_vel(self, info):
        if self._ref_motion_all is not None:
            ''' This performs reference-state-initialization (RSI) '''
            init_poses, init_vels = [], []
            
            for i in range(self._num_agent):
                ## self.get_elapsed_time should not be used here
                cur_time = self._start_time[i]
                ''' Set the state of simulated agent by using the state of reference motion '''
                cur_pose = self._ref_motion[i].get_pose_by_time(cur_time)
                cur_vel = self._ref_motion[i].get_velocity_by_time(cur_time)
                ''' Add noise to the state if necessary '''
                if info.get('add_noise'):
                    cur_pose, cur_vel = \
                        self._base_env.add_noise_to_pose_vel(
                            self._sim_agent[i], cur_pose, cur_vel)
                init_poses.append(cur_pose)
                init_vels.append(cur_vel)
            return init_poses, init_vels
        else:
            return super().compute_init_pose_vel(info)

    def compute_target_pose(self, idx, action):
        target_pose = super().compute_target_pose(idx, action)
        if self._knee_correction:
            sim_agent = self._sim_agent[idx]
            char_info = sim_agent._char_info
            for name in ['lankle', 'rankle', 'lknee', 'rknee']:
                joint_idx = char_info.joint_idx[name]
                Q_target, p_target = conversions.T2Qp(
                    target_pose.get_transform(char_info.bvh_map[joint_idx], local=True))
                Q_closest, _ = quaternion.Q_closest(
                    Q_target, 
                    np.array([0.0, 0.0, 0.0, 1.0]),
                    np.array([1.0, 0.0, 0.0]),
                    )
                    # sim_agent.get_joint_axis(joint_idx))
                target_pose.set_transform(
                    char_info.bvh_map[joint_idx], 
                    conversions.Qp2T(Q_closest, p_target),
                    do_ortho_norm=False,
                    local=True)
        return target_pose

    def get_state_by_key(self, idx, key):
        state = []

        h = self.get_ground_height(idx)
        idx_op = self.get_idx_op(idx)
        agent = self._sim_agent[idx]
        agent_op = self._sim_agent[idx_op]

        if key=='body':
            state.append(self.state_body(idx))
        elif key=='phase_linear':
            state.append(self._phase[idx])
        elif key=='phase_complex':
            theta = 2 * np.pi * self._phase[idx]
            state.append(np.array([np.cos(theta), np.sin(theta)]))
        elif key=='game_basic':
            T_face = agent.get_facing_transform(h)
            R_face, p_face = conversions.T2Rp(T_face)
            R_face_inv = R_face.transpose()

            ## Own Location and Direction

            R_arena_1_inv = self._arena_info[idx]['R_ref_inv']
            p_arena_1 = self._arena_info[idx]['p_ref']

            d1, p1 = agent.get_facing_direction_position(h)

            state.append(np.dot(R_arena_1_inv, p1-p_arena_1)[:2])
            state.append(np.dot(R_arena_1_inv, d1))

            ## Opponent body state w.r.t. facing xform

            state.append(
                self._state_body_raw(
                    idx_op,
                    agent_op,
                    T_ref=T_face, 
                    include_com=False, 
                    include_p=True, 
                    include_Q=False, 
                    include_v=True, 
                    include_w=False, 
                    include_R6=False,
                    include_root=False,
                    include_root_height=False,
                    return_stacked=True)
            )

            ## Opponent's targets/gloves w.r.t. the player's gloves

            ps_glove_1, Rs_glove_1, vs_glove_1, ws_glove_1 = \
                self.get_glove_states(idx)
            ps_glove_2, Rs_glove_2, vs_glove_2, ws_glove_2 = \
                self.get_glove_states(idx_op)

            ps_target_1, Qs_target_1, vs_target_1, ws_target_1 = \
                agent.get_link_states(self._target_links.keys())
            ps_target_2, Qs_target_2, vs_target_2, ws_target_2 = \
                agent_op.get_link_states(self._target_links.keys())

            for i in range(len(Rs_glove_1)):
                R_glove_inv_1 = Rs_glove_1[i].transpose()
                p_glove_1 = ps_glove_1[i]
                v_glove_1 = vs_glove_1[i]
                for p_target_2, v_target_2 in zip(ps_target_2, vs_target_2):
                    state.append(np.dot(R_glove_inv_1, p_target_2-p_glove_1))
                    state.append(np.dot(R_glove_inv_1, v_target_2-v_glove_1))
                for p_glove_2, v_glove_2 in zip(ps_glove_2, vs_glove_2):
                    state.append(np.dot(R_glove_inv_1, p_glove_2-p_glove_1))
                    state.append(np.dot(R_glove_inv_1, v_glove_2-v_glove_1))

            ## Opponent's gloves w.r.t. the agent's targets

            Rs_target_1 = conversions.Q2R(Qs_target_1)
            
            for i in range(len(Rs_target_1)):
                R_target_inv_1 = Rs_target_1[i].transpose()
                p_target_1 = ps_target_1[i]
                v_target_1 = vs_target_1[i]
                for p_glove_2, v_glove_2 in zip(ps_glove_2, vs_glove_2):
                    state.append(np.dot(R_target_inv_1, p_glove_2-p_target_1))
                    state.append(np.dot(R_target_inv_1, v_glove_2-v_target_1))
        elif key=='contact_state':
            state.append(self._contact_info[idx].astype(float))
            state.append(self._contact_info[idx_op].astype(float))
        elif key=='game_score':
            state.append(
                self._game_remaining_score[idx])
            state.append(
                self._game_remaining_score[idx_op])
        elif key=='game_touch_avail':
            game_touch_avail = self._game_touch_avail.astype(float)
            game_touch_avail_remaining = self._game_touch_avail_remaining.astype(float)
            state.append(
                game_touch_avail[idx])
            state.append(
                game_touch_avail[idx_op])
            state.append(
                game_touch_avail_remaining[idx])
            state.append(
                game_touch_avail_remaining[idx_op])
        elif key=='remaining_time':
            state.append(
                max(0.0, self._game_duration-self.get_elapsed_time()))
        # elif key=='task_body':
        #     state.append(self.state_body(idx))
        else:
            raise NotImplementedError

        return np.hstack(state)

    def state_body(self, 
                   idx, 
                   type=None, 
                   return_stacked=True):
        agent = self._sim_agent[idx]
        return self._state_body(idx, agent, type, return_stacked)

    def state_task(self, idx):
        sc = self._state_choices.copy()
        sc.remove('body')
        return self.state(idx, sc)
    
    def reward_data(self, idx):
        data = {}

        if self.exist_rew_fn_subterm(idx, 'upright_upper_body'):
            ps, Qs, vs, ws = \
                self._sim_agent[idx].get_link_states([
                    self._sim_agent[idx]._char_info.lowerback,
                    self._sim_agent[idx]._char_info.upperback,
                    self._sim_agent[idx]._char_info.chest,
                    ])
            Rs = conversions.Q2R(Qs)
            data['sim_upper_body_vup'] = Rs[...,:,1]
        
        if self.exist_rew_fn_subterm(idx, 'energy_consumption_by_distance') or \
           self.exist_rew_fn_subterm(idx, 'joint_limit'):
            data['sim_joint_pv'] = self._sim_agent[idx].get_joint_states()
        
        if self.exist_rew_fn_subterm(idx, 'stay_near_opponent'):
            data['sim_com'], data['sim_com_vel'] = self._sim_agent[idx].get_com_and_com_vel()
        
        if self.exist_rew_fn_subterm(idx, 'falldown'):
            data['falldown'] = self._base_env.check_falldown(idx)
        
        if self.exist_rew_fn_subterm(idx, 'too_close_pos'):
            data['too_close_pos'] = self.is_too_close_pos()
        
        if self.exist_rew_fn_subterm(idx, 'dist_to_targets'):
            data['dist_to_targets'] = self.dist_to_targets(idx)
        
        data['sim_facing_dir'], data['sim_facing_pos'] = \
            self._sim_agent[idx].get_facing_direction_position(self.get_ground_height(idx))
        data['game_result'] = self.judge_game()
        data['score_change'] = self._game_remaining_score_change[idx]

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
        data_op = data_next[self.get_idx_op(idx)]

        dist = np.linalg.norm(data_op['sim_facing_pos']-data['sim_facing_pos'])
        dist_clip = max(0.0, dist-1.0)
        # rew_scale_by_dist = np.exp(-50.0*np.dot(dist_clip,dist_clip))
        rew_scale_by_dist = 1.0 - np.exp(-50.0*np.dot(dist_clip,dist_clip))

        if self.exist_rew_fn_subterm(idx, 'joint_limit'):
            # print(idx)
            ''' This assumes that all joints are spherical '''
            # target_pose = action_dict['target_pose'][idx].reshape(-1, 3)
            # target_angles = np.linalg.norm(target_pose, axis=1)

            p_next, _ = data_next[idx]['sim_joint_pv']
            Qs_joint = np.hstack(p_next).reshape(-1,4)
            joint_angles = np.linalg.norm(conversions.Q2A(Qs_joint), axis=1)
            # print(joint_angles)
            # print(joint_angles[2], joint_angles[5])
            # print(p_next)
            # print(np.vstack(p_next))
            # print(conversions.Q2A(np.array(p_next)))
            # joint_angles = np.array(conversions.Q2A(np.array(p_next)))
            # joint_angles = np.linalg.norm(conversions.Q2A(p_next), axis=1)
            # print(p_next)
            # print(joint_angles)
            # print(target_angles - 0.7*np.pi)
            # print(np.maximum(0.0, target_angles - 0.7*np.pi))
            error['joint_limit'] = 1.0
                # np.sum(np.maximum(0.0, target_angles - 0.7*np.pi)) / len(target_angles)

        if self.exist_rew_fn_subterm(idx, 'upright_upper_body'):
            diff = data['sim_upper_body_vup'] - char_info.v_up_env
            error['upright_upper_body'] = \
                rew_scale_by_dist * np.mean(np.sum(diff**2, axis=1))

        if self.exist_rew_fn_subterm(idx, 'energy_consumption_by_distance'):
            p_prev, v_prev = data_prev[idx]['sim_joint_pv']
            p_next, v_next = data_next[idx]['sim_joint_pv']
            dv = np.hstack(v_next)-np.hstack(v_prev)

            # error['energy_consumption_by_distance'] = \
            #     max(0.0, np.dot(dv, dv) - (5000.0*scale+100))

            error['energy_consumption_by_distance'] = \
                rew_scale_by_dist * max(0.0, np.dot(dv, dv)-100)
            
            # if idx==0: 
            #     print(dist, error['energy_consumption_by_distance'])

        if self.exist_rew_fn_subterm(idx, 'constant'):
            error['constant'] = 1.0

        if self.exist_rew_fn_subterm(idx, 'stay_near_opponent'):
            p_com = data['sim_com']
            p_com_op = data_op['sim_com']
            diff = self._sim_agent[idx].project_to_ground(p_com_op-p_com)
            error['stay_near_opponent'] = np.dot(diff, diff)

        if self.exist_rew_fn_subterm(idx, 'facing_dir_alignment'):
            d, p = data['sim_facing_dir'], data['sim_facing_pos']
            p_op = data_op['sim_facing_pos']
            l = p_op - p
            l /= np.linalg.norm(l)
            diff = np.dot(d, l) - 1.0
            error['facing_dir_alignment'] = diff * diff

        if self.exist_rew_fn_subterm(idx, 'dist_to_targets'):
            error['dist_to_targets'] = data['dist_to_targets']

        if self.exist_rew_fn_subterm(idx, 'dist_to_gloves'):
            error['dist_to_gloves'] = data_op['dist_to_targets']

        if self.exist_rew_fn_subterm(idx, 'out_of_arena'):
            p = data['sim_facing_pos']
            error['out_of_arena'] = 1.0 if self.check_out_of_arena(idx) else 0.0

        if self.exist_rew_fn_subterm(idx, 'falldown'):
            error['falldown'] = 1.0 if data['falldown'] else 0.0

        if self.exist_rew_fn_subterm(idx, 'too_close_pos'):
            error['too_close_pos'] = 1.0 if data['too_close_pos'] else 0.0

        if self.exist_rew_fn_subterm(idx, 'draw'):
            error['draw'] = 1.0 if data['game_result']=='draw' else 0.0

        if self.exist_rew_fn_subterm(idx, 'take_score'):
            error['take_score'] = float(np.abs(data_op['score_change']))

        if self.exist_rew_fn_subterm(idx, 'lose_score'):
            error['lose_score'] = float(np.abs(data['score_change']))

        if self.exist_rew_fn_subterm(idx, 'imitation_by_contrastive'):
            score = self._contrastive_models[idx].evaluate(
                self._prev_poses[idx][-self._past_window_size[idx]-1:-1], 
                self._prev_poses[idx][-1])
            error['imitation_by_contrastive'] = score

        return error
    
    def inspect_end_of_episode(self):
        eoe_reason = super().inspect_end_of_episode()
        if 'timeover' in self._early_term_choices:
            if self._timeover:
                eoe_reason.append("timeover")
        if 'gameover' in self._early_term_choices:
            res = self.judge_game()
            if res=='p1':
                eoe_reason.append('player1_win')
            elif res=='p2':
                eoe_reason.append('player2_win')
            elif res=='draw':
                eoe_reason.append('draw')
        if 'out_of_arena' in self._early_term_choices:
            check = self.is_out_of_arena()
            if check: eoe_reason.append('out_of_arena')
        if 'out_of_facing_dir' in self._early_term_choices:
            if self.is_out_of_facing_dir(0):
                eoe_reason.append('out_of_facing_dir_1')
            if self.is_out_of_facing_dir(1):
                eoe_reason.append('out_of_facing_dir_2')
        if 'too_close_pos' in self._early_term_choices:
            if self.is_too_close_pos():
                eoe_reason.append('too_close_pos')

        return eoe_reason

    def judge_game(self):
        if self._game_type=='total_score':
            return self.judge_game_by_total_score()
        else:
            raise NotImplementedError

    def judge_game_by_total_score(self):
        if self._timeover:
            if self._game_remaining_score[0] > self._game_remaining_score[1]:
                return 'p1'
            elif self._game_remaining_score[0] < self._game_remaining_score[1]:
                return 'p2'
            else:
                return 'draw'
        else:
            if self._game_remaining_score[0] == 0 and \
               self._game_remaining_score[1] > 0:
                return 'p1'
            elif self._game_remaining_score[1] == 0 and \
                 self._game_remaining_score[0] > 0:
                return 'p2'
            else:
                return 'none'

    def is_too_close_pos(self):
        return self._game_too_close_elapsed >= self._game_too_close_duration

    def dist_to_targets(self, idx):
        idx_op = self.get_idx_op(idx)
        agent = self._sim_agent[idx]
        agent_op = self._sim_agent[idx_op]        

        ps_glove, Rs_glove, vs_glove, ws_glove = \
            self.get_glove_states(idx)
        ps_target, Qs_target, vs_target, ws_target = \
            agent_op.get_link_states(self._target_links.keys())

        return np.min(
            distance.cdist(ps_glove, ps_target, 'euclidean').flatten())

    def is_out_of_arena(self):
        check1 = self._game_out_of_arena_elapsed[0] \
            >= self._game_out_of_arena_duration
        check2 = self._game_out_of_arena_elapsed[1] \
            >= self._game_out_of_arena_duration
        return check1 or check2

    def check_out_of_arena(self, idx):
        p = self._sim_agent[idx].get_root_position()
        x1, y1, z1 = self._arena_boundaries[0]
        x2, y2, z2 = self._arena_boundaries[3]
        px, py, pz = p
        return not(x2 <= px <= x1) or not(y2 <= py <= y1)

    def update_contact_info(self):
        self._contact_info[0][:] = False
        self._contact_info[1][:] = False

        pts = self._pb_client.getContactPoints(
            bodyA=self._sim_agent[0]._body_id, 
            bodyB=self._sim_agent[1]._body_id)
        for p in pts:
            self._contact_info[0][p[3]+1] = True
            self._contact_info[1][p[4]+1] = True

    def check_opponent_touched(self, idx):
        '''
        This returns TRUE if any hand touched the opponent
        '''
        pts_l = self._pb_client.getContactPoints(
            bodyA=self._sim_agent[idx]._body_id, 
            bodyB=self._sim_agent[self.get_idx_op(idx)]._body_id, 
            linkIndexA=self._sim_agent[idx]._char_info.joint_idx["lwrist"])
        pts_r = self._pb_client.getContactPoints(
            bodyA=self._sim_agent[idx]._body_id, 
            bodyB=self._sim_agent[self.get_idx_op(idx)]._body_id, 
            linkIndexA=self._sim_agent[idx]._char_info.joint_idx["rwrist"])

        pts = pts_l + pts_r

        touch_info = {}

        for p in pts:
            if p[4] in self._target_links.keys() and \
               p[9] >= self._game_min_touch_force:
                if len(touch_info)==0:
                    touch_info['link_idx'] = []
                    touch_info['normal_dir'] = []
                    touch_info['normal_force'] = []
                touch_info['link_idx'].append(p[4])
                touch_info['normal_dir'].append(p[7])
                touch_info['normal_force'].append(p[9])

        return touch_info

    def get_idx_op(self, idx):
        if idx==0:
            return 1
        elif idx==1:
            return 0
        else:
            raise Exception('Invalid index')

    def get_glove_states(self, idx):
        p, Q, v, w = self._sim_agent[idx].get_link_states(
            [self._sim_agent[idx]._char_info.joint_idx["lwrist"],
             self._sim_agent[idx]._char_info.joint_idx["rwrist"]])
        R = conversions.Q2R(Q)
        return p, R, v, w

    def get_current_time(self, idx):
        return self._start_time[idx] + self.get_elapsed_time()

    def sample_ref_motion(self, indices=None):
        ref_indices = []
        ref_motions = []
        for i in range(self._num_agent):
            if indices is not None:
                idx = indices[i]
            else:
                idx = np.random.randint(len(self._ref_motion_all[i]))
            ref_indices.append(idx)
            ref_motions.append(self._ref_motion_all[i][idx])
        if self._verbose:
            print('Ref. motions selected: ', ref_indices)
        return ref_motions

    def render(self, rm):
        super().render(rm)
        ''' Arena '''
        if self._rope:
            rm.bullet_render.render_model(
                self._pb_client, 
                self._rope, 
                draw_link=True, 
                draw_link_info=True, 
                draw_joint=False, 
                draw_joint_geom=False, 
                link_info_line_width=2.0,
                color=[0.6, 0.6, 0.6, 1.0])
        size = self._arena_size
        dsize = (2.0, 2.0)
        rm.gl_render.render_quad(
            self._arena_boundaries[0],
            self._arena_boundaries[1],
            self._arena_boundaries[3],
            self._arena_boundaries[2],
            tex_id=rm.tex_id_ground,
            tex_param1=[0, 0],
            tex_param2=[size[0] / dsize[0], 0],
            tex_param3=[size[0] / dsize[0], size[1] / dsize[1]],
            tex_param4=[0, size[1] / dsize[0]],
        )
        rm.gl_render.render_quad(
            self._arena_boundaries_outer[0],
            self._arena_boundaries_outer[1],
            self._arena_boundaries_outer[3],
            self._arena_boundaries_outer[2],
            color=[0.3, 0.3, 0.3, 1])
        # ''' Gloves '''
        # for i in range(self._num_agent):
        #     rm.bullet_render.render_links(
        #         self._sim_agent[i]._pb_client,
        #         self._sim_agent[i]._body_id,
        #         link_ids=[
        #             self._sim_agent[i]._char_info.joint_idx["lwrist"],
        #             self._sim_agent[i]._char_info.joint_idx["rwrist"]],
        #         color=[0.1,0.1,0.1,1],
        #     )
        ''' Arena Reference '''
        if rm.flag['custom1']:
            for i in range(self._num_agent):
                rm.gl_render.render_transform(
                    self._arena_info[i]['T_ref'],
                    scale=1.0,
                    line_width=1.0,
                    point_size=0.05,
                    render_pos=True,
                    render_ori=[True, True, True],
                    color_pos=[0, 0, 0, 1],
                    color_ori=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    use_arrow=False)
            for i in range(self._num_agent):
                # Touched link
                agent_op = self._sim_agent[self.get_idx_op(i)]
                if len(self._cur_touch_info[i])>0:
                    rm.bullet_render.render_links(
                        pb_client=agent_op._pb_client,
                        model=agent_op._body_id,
                        color=[0.7, 0.7, 0.7, 1.0],
                        draw_link_info=False,
                        link_info_scale=1.5,
                        link_filter=[self._cur_touch_info[i]['link_idx']],
                    )

        # ps, Qs, vs, ws = \
        #     self._sim_agent[0].get_link_states([
        #         self._sim_agent[0]._char_info.lowerback,
        #         self._sim_agent[0]._char_info.upperback,
        #         self._sim_agent[0]._char_info.chest,
        #         ])
        # Rs = conversions.Q2R(Qs)
        # for R, p in zip(Rs, ps):
        #     T = conversions.Rp2T(R, p)
        #     rm.gl_render.render_transform(
        #         T,
        #         scale=1.0,
        #         line_width=1.0,
        #         point_size=0.05,
        #         render_pos=True,
        #         render_ori=[True, True, True],
        #         color_pos=[0, 0, 0, 1],
        #         color_ori=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        #         use_arrow=False)

if __name__ == '__main__':

    import env_renderer as er
    import render_module as rm
    import argparse
    from fairmotion.viz.utils import TimeChecker

    rm.initialize()
    
    def arg_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True, type=str)
        return parser

    class EnvRenderer(er.EnvRenderer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.time_checker_auto_play = TimeChecker()
            self.reset()
        def reset(self):
            self.env.reset()
        def one_step(self):
            # a = np.zeros(100)
            self.env.step()
        def extra_render_callback(self):
            if self.rm.flag['follow_cam']:
                p, _, _, _ = env._sim_agent[0].get_root_state()
                self.rm.viewer.update_target_pos(p, ignore_z=True)
            self.env.render(self.rm)
        def extra_idle_callback(self):
            time_elapsed = self.time_checker_auto_play.get_time(restart=False)
            if self.rm.flag['auto_play'] and time_elapsed >= self.env._dt_con:
                self.time_checker_auto_play.begin()
                self.one_step()
        def extra_keyboard_callback(self, key):
            if key == b'r':
                self.reset()
            elif key == b'O':
                size = np.random.uniform(0.1, 0.3, 3)
                p, Q, v, w = self.env._agent[0].get_root_state()
                self.env._obs_manager.throw(p, size=size)
    
    print('=====Humanoid Fencing Environment=====')
    
    args = arg_parser().parse_args()

    env = Env(args.config)

    cam = rm.camera.Camera(pos=np.array([12.0, 0.0, 12.0]),
                           origin=np.array([0.0, 0.0, 0.0]), 
                           vup=np.array([0.0, 0.0, 1.0]), 
                           fov=30.0)

    renderer = EnvRenderer(env=env, cam=cam)
    renderer.run()
