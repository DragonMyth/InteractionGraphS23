import os
import pickle
import copy

import numpy as np
from collections import deque

from fairmotion.utils import utils
from fairmotion.ops import conversions, motion
from fairmotion.data import bvh
from scipy.spatial import distance

import motion_utils

import env_humanoid_base

class Env(env_humanoid_base.Env):
    def __init__(self, config):
        super().__init__(config)

        assert self._num_agent == 2
        
        self._initialized = False
        self._ref_motion_all = None
        self._ref_motion = None
        self._start_time = np.zeros(self._num_agent)
        self._timeover = False
        self._gameover = False
        self._game_result = "none"

        ''' Setup Arena '''
        
        self._arena_size = [2.0, 12.0]
        # 8 corners of the arena
        w_x, w_y = self._arena_size[0], self._arena_size[1]
        self._arena_h = 0.001
        self._arena_boundaries = [
            np.array([0.5*w_x, 0.5*w_y, self._arena_h]),
            np.array([0.5*w_x, -0.5*w_y, self._arena_h]),
            np.array([-0.5*w_x, 0.5*w_y, self._arena_h]),
            np.array([-0.5*w_x, -0.5*w_y, self._arena_h]),
        ]
        outer_margin = 0.1
        self._arena_boundaries_outer = [
            self._arena_boundaries[0] + \
                np.array([outer_margin, outer_margin, -0.5*self._arena_h]),
            self._arena_boundaries[1] + \
                np.array([outer_margin, -outer_margin, -0.5*self._arena_h]),
            self._arena_boundaries[2] + \
                np.array([-outer_margin, outer_margin, -0.5*self._arena_h]),
            self._arena_boundaries[3] + \
                np.array([-outer_margin, -outer_margin, -0.5*self._arena_h]),
        ]
        self._arena_info = [
            {
                "T_ref": conversions.Rp2T(
                    conversions.Az2R(0.0), 
                    np.array([0, -0.5*w_y, self._arena_h])),
                "R_ref": conversions.Az2R(0.0),
                "p_ref": np.array([0, -0.5*w_y, self._arena_h]),
                "R_ref_inv": conversions.Az2R(0.0).transpose(),
            },
            {   
                "T_ref": conversions.Rp2T(
                    conversions.Az2R(np.pi), 
                    np.array([0, 0.5*w_y, self._arena_h])),
                "R_ref": conversions.Az2R(np.pi),
                "p_ref": np.array([0, 0.5*w_y, self._arena_h]),
                "R_ref_inv": conversions.Az2R(np.pi).transpose(),
            },
        ]

        ''' Setup for Games '''

        DEFAULT_GAME_CONFIG = {
            "type": "first_touch",
            "min_touch_force": 1.0,
            "init_score": 100,
            "duration": 30.0,
            "touch_interval": 0.0,
            "counter_touch_allow_length": 0.3,
            "target_links": [
                "lowerback",
                "upperback",
                "chest",
                "lclavicle",
                "rclavicle",
            ],
        }

        config_game = DEFAULT_GAME_CONFIG.copy()
        config_game_custom = self._config.get('game')
        if config_game_custom:
            config_game.update(config_game_custom)

        ''' The type of the game '''
        self._game_type = config_game.get('type')
        ''' The minum force required to be considered as a valid touch '''
        self._game_min_touch_force = config_game.get('min_touch_force')
        ''' The duration of the game '''
        self._game_duration = config_game.get('duration')
        ''' The initial score that the agents have when the game starts  '''
        self._game_init_score = config_game.get('init_score')
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
        
        ''' 
        After the agent touch the opponent, that touch can be ignored 
        if the opponent makes a counter touch within 
        '''
        self._game_touch_counter = np.zeros(self._num_agent)
        self._game_counter_touch_allow_length = \
            config_game.get('counter_touch_allow_length')

        self._target_links = set()
        for l in config_game.get('target_links'):
            self._target_links.add(self._sim_agent[0]._char_info.joint_idx[l])

        assert self._game_type in ['first_touch', 'total_score']
        assert self._game_duration > 0.0
        assert self._game_min_touch_force >= 0.0
        assert len(self._target_links) > 0
        
        if self._game_type=='first_touch':
            assert self._game_touch_interval == 0.0
        if self._game_type=='total_score':
            assert self._game_touch_interval >= 0.0
            assert self._game_init_score > 0

        self._cur_touch_info = [{} for i in range(self._num_agent)]
        self._opponent_touched = np.zeros(2, dtype=bool)

        if config.get('lazy_creation'):
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

        ''' Should call reset after all setups are done '''

        self.reset({'add_noise': False})

        self._initialized = True

        if self._verbose:
            print('----- Humanoid Fencing Environment Created -----')
            for i in range(self._num_agent):
                print('[Agent%d]: state(%d) and action(%d)' \
                      %(i, len(self.state(i)), self._action_space[i].dim))
            print('-------------------------------')

    def callback_reset_prev(self, info):
        if self._ref_motion_all is not None:
            ''' Choose a reference motion randomly whenever reset '''
            self._ref_motion = self.sample_ref_motion()
            if 'start_time' in info.keys():
                self._start_time = info.get('start_time')
                assert len(self._start_time) == self._num_agent
            else:
                for i in range(self._num_agent):
                    self._start_time[i] = \
                        np.random.uniform(0.0, self._ref_motion[i].length())

    def callback_reset_after(self, info):
        self._timeover = False
        self._gameover = False
        self._game_result = "none"

        for i in range(self._num_agent):
            self._cur_touch_info[i] = {}
        self._opponent_touched[:] = False

        self._game_touch_counter[:] = 0.0
        self._game_touch_avail[:] = True
        self._game_touch_avail_remaining[:] = 0
        self._game_remaining_score[:] = self._game_init_score
        self._game_remaining_score_change[:] = 0.0

        self._game_counter_touch_occurred = False

    def callback_step_after(self, action_dict, infos):        
        ''' Check time over '''
        if not self._timeover:
            self._timeover = self.get_elapsed_time()>self._game_duration
        
        ''' Update the current state of the game '''
        if not self._gameover:
            game_remaining_score_prev = self._game_remaining_score.copy()
            
            for i in range(self._num_agent):
                ''' Check if the agent physically is touching the opponent '''
                self._cur_touch_info[i] = self.check_opponent_touched(i)
                self._opponent_touched[i] = \
                    self._opponent_touched[i] or len(self._cur_touch_info[i]) > 0
            
            touch = [len(self._cur_touch_info[0])>0, len(self._cur_touch_info[1])>0]
            
            if self._game_type=="first_touch":
                if self._game_touch_counter[0] > 0:
                    if not touch[1]:
                        self._game_touch_counter[0] += self._dt_con
                        if self._game_touch_counter[0] >= self._game_counter_touch_allow_length:
                            self._game_result = "p1"
                    else:
                        self._game_result = "draw"
                elif self._game_touch_counter[1] > 0:
                    if not touch[0]:
                        self._game_touch_counter[1] += self._dt_con
                        if self._game_touch_counter[1] >= self._game_counter_touch_allow_length:
                            self._game_result = "p2"
                    else:
                        self._game_result = "draw"
                else:
                    if touch[0] and not touch[1]:
                        self._game_touch_counter[0] += self._dt_con
                    elif not touch[0] and touch[1]:
                        self._game_touch_counter[1] += self._dt_con
                    elif touch[0] and touch[1]:
                        self._game_result = "draw"
            elif self._game_type=="total_score":
                raise NotImplementedError
            else:
                raise NotImplementedError

            # for i in range(self._num_agent):
            #     if self._game_touch_avail[i]:                
            #         idx_op = self.get_idx_op(i)
            #         is_touched_by_op = len(self._cur_touch[idx_op])
                    
            #         ''' 
            #         If a valid touch happens in the past, increase the counder until 
            #         the opponent touches the agent
            #         '''
            #         if self._game_touch_counter[i] > 0:
            #             if is_touched_by_op:
            #                 self._game_touch_counter[i] = 0
            #                 self._game_counter_touch_occurred = True
            #             else:
            #                 self._game_touch_counter[i] += 1
            #         else:
            #             ''' Simultaneous touch is not considered as a valid touch '''
            #             exclusive_touch = self._cur_touch[i] and not is_touched_by_op
            #             if exclusive_touch:
            #                 self._game_touch_counter[i] += 1

            #         is_valid_touch_occur = \
            #             self._game_touch_counter[i] > self._game_touch_counter_max
                    
            #         if is_valid_touch_occur:
            #             ''' Decrease the opponent's score when a valid touch occurs '''
            #             self._game_remaining_score[idx_op] = \
            #                 max(0, self._game_remaining_score[idx_op]-1)
            #             ''' Disable next touch for a wile if touch_interval was set '''
            #             if self._game_touch_interval > 0.0:
            #                 self._game_touch_avail[i] = False
            #                 self._game_touch_avail_remaining[i] = \
            #                         int(self._game_touch_interval / self._dt_con)
            #                 self._game_touch_counter[i] = 0
            #                 self._old_touch_info[i] = self._cur_touch_info[i]
            #     else:
            #         self._game_touch_avail_remaining[i] = \
            #             max(self._game_touch_avail_remaining[i]-1, 0)
            #         self._game_touch_avail[i] = (self._game_touch_avail_remaining[i]==0)
            
            # ''' Recored score changes for an efficient reward computation '''
            # self._game_remaining_score_change = \
            #     self._game_remaining_score - game_remaining_score_prev
            
            # print('------------------------------')
            # print('Score: ', self._game_remaining_score)
            # print('Score change: ', self._game_remaining_score_change)
            # print('Touch avail: ', self._game_touch_avail)
            # print('Touch avail remaining: ', self._game_touch_avail_remaining)
            # print('Touch counter: ', self._game_touch_counter)
            # print('------------------------------')

        ''' Check if the current state of the game '''
        if not self._gameover:
            self._gameover = self._game_result in ['p1', 'p2', 'draw']

        return infos

    def print_log_in_step(self):
        super().print_log_in_step()
        if self._verbose:
            print('touch_counter:', self._game_touch_counter)
            print('touch_occurred:', (self._game_touch_counter > 0).astype(float))
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

    def get_state_by_key(self, idx, key):
        state = []

        h = self.get_ground_height()
        idx_op = self.get_idx_op(idx)
        agent = self._sim_agent[idx]
        agent_op = self._sim_agent[idx_op]

        if key=='body':
            state.append(
                self._state_body(
                    self._sim_agent[idx],
                    T_ref=None, 
                    include_com=True, 
                    include_p=True, 
                    include_Q=True, 
                    include_v=True, 
                    include_w=True, 
                    return_stacked=True,
                )
            )
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
                self._state_body(
                    agent_op,
                    T_ref=T_face, 
                    include_com=False, 
                    include_p=True, 
                    include_Q=False, 
                    include_v=True, 
                    include_w=False, 
                    return_stacked=True)
            )

            ## Opponent's targets/sword w.r.t. the player's sword
            ps_sword_1, Rs_sword_1, vs_sword_1, ws_sword_1 = \
                self.get_sword_state_samples(idx)
            ps_sword_2, Rs_sword_2, vs_sword_2, ws_sword_2 = \
                self.get_sword_state_samples(idx_op)

            ps_target_1, Qs_target_1, vs_target_1, ws_target_1 = \
                agent.get_link_states(self._target_links)
            ps_target_2, Qs_target_2, vs_target_2, ws_target_2 = \
                agent_op.get_link_states(self._target_links)

            for i in range(len(Rs_sword_1)):
                R_sword_inv_1 = Rs_sword_1[i].transpose()
                p_sword_1 = ps_sword_1[i]
                v_sword_1 = vs_sword_1[i]
                for p_target_2, v_target_2 in zip(ps_target_2, vs_target_2):
                    state.append(np.dot(R_sword_inv_1, p_target_2-p_sword_1))
                    state.append(np.dot(R_sword_inv_1, v_target_2-v_sword_1))
                for p_sword_2, v_sword_2 in zip(ps_sword_2, vs_sword_2):
                    state.append(np.dot(R_sword_inv_1, p_sword_2-p_sword_1))
                    state.append(np.dot(R_sword_inv_1, v_sword_2-v_sword_1))

            ## Opponent's sword w.r.t. the agent's targets

            Rs_target_1 = conversions.Q2R(Qs_target_1)
            
            for i in range(len(Rs_target_1)):
                R_target_inv_1 = Rs_target_1[i].transpose()
                p_target_1 = ps_target_1[i]
                v_target_1 = vs_target_1[i]
                for p_sword_2, v_sword_2 in zip(ps_sword_2, vs_sword_2):
                    state.append(np.dot(R_target_inv_1, p_sword_2-p_target_1))
                    state.append(np.dot(R_target_inv_1, v_sword_2-v_target_1))
        elif key=='game_score':
            state.append(
                self._game_remaining_score[idx])
            state.append(
                self._game_remaining_score[idx_op])
        elif key=='game_touch_counter':
            game_touch_occurred = (self._game_touch_counter > 0).astype(float)
            state.append(
                self._game_touch_counter[idx])
            state.append(
                self._game_touch_counter[idx_op])
            state.append(
                game_touch_occurred[idx])
            state.append(
                game_touch_occurred[idx_op])
        elif key=='game_touch_counter_scaled':
            game_touch_counter = 100*self._game_touch_counter
            game_touch_occurred = 100*(self._game_touch_counter > 0).astype(float)
            state.append(
                game_touch_counter[idx])
            state.append(
                game_touch_counter[idx_op])
            state.append(
                game_touch_occurred[idx])
            state.append(
                game_touch_occurred[idx_op])
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
        elif key=='task_body':
            state.append(
                self._state_body(
                    agent,
                    T_ref=None, 
                    include_com=True, 
                    include_p=True, 
                    include_Q=True, 
                    include_v=True, 
                    include_w=True, 
                    return_stacked=True)
            )
        else:
            raise NotImplementedError

        return np.hstack(state)

    def state_body(self, idx):
        return self.get_state_by_key(idx, 'body')

    def state_task(self, idx):
        sc = self._state_choices.copy()
        sc.remove('body')
        return self.state(idx, sc)
    
    def reward_data(self, idx):
        data = {}

        # data['sim_link_pQvw'] = self._sim_agent[idx].get_link_states()
        if self.exist_rew_fn_subterm(idx, 'energy_consumption_by_distance'):
            data['sim_joint_pv'] = self._sim_agent[idx].get_joint_states()
        
        if self.exist_rew_fn_subterm(idx, 'stay_near_opponent'):
            data['sim_com'], data['sim_com_vel'] = self._sim_agent[idx].get_com_and_com_vel()    
        
        if self.exist_rew_fn_subterm(idx, 'falldown'):
            data['falldown'] = self._base_env.check_falldown(self._sim_agent[idx])
        
        if self.exist_rew_fn_subterm(idx, 'out_of_facing_dir'):
            data['out_of_facing_dir'] = self.is_out_of_facing_dir(idx)
        
        if self.exist_rew_fn_subterm(idx, 'too_close_pos'):
            data['too_close_pos'] = self.is_too_close_pos()
        
        if self.exist_rew_fn_subterm(idx, 'dist_to_targets'):
            data['dist_to_targets'] = self.dist_to_targets(idx)
        
        data['sim_facing_dir'], data['sim_facing_pos'] = \
            self._sim_agent[idx].get_facing_direction_position(self.get_ground_height())
        data['game_result'] = str(self._game_result)
        # data['score_change'] = self._game_remaining_score_change.copy()

        return data
    
    def reward_max(self):
        return 1.0
    
    def reward_min(self):
        return 0.0
    
    def get_task_error(self, idx, data_prev, data_next, action):
        error = {}

        sim_agent = self._sim_agent[idx]
        char_info = sim_agent._char_info

        data = data_next[idx]
        data_op = data_next[self.get_idx_op(idx)]

        dist = np.linalg.norm(data_op['sim_facing_pos']-data['sim_facing_pos'])
        dist_clip = max(0.0, dist-1.0)
        # rew_scale_by_dist = np.exp(-50.0*np.dot(dist_clip,dist_clip))
        rew_scale_by_dist = 1.0 - np.exp(-50.0*np.dot(dist_clip,dist_clip))

        if self.exist_rew_fn_subterm(idx, 'upright_upper_body'):
            diff = data['sim_upper_body_vup'] - char_info.v_up_env
            error['upright_upper_body'] = \
                rew_scale_by_dist * np.mean(np.sum(diff**2, axis=1))

        if self.exist_rew_fn_subterm(idx, 'energy_consumption_by_distance'):
            p_prev, v_prev = data_prev[idx]['sim_joint_pv']
            p_next, v_next = data_next[idx]['sim_joint_pv']
            dv = np.hstack(v_next)-np.hstack(v_prev)

            error['energy_consumption_by_distance'] = \
                rew_scale_by_dist * max(0.0, np.dot(dv, dv)-100)            

        if self.exist_rew_fn_subterm(idx, 'constant'):
            error['constant'] = 1.0

        if self.exist_rew_fn_subterm(idx, 'stay_near_opponent'):
            p_com = data['sim_com']
            p_com_op = data_op['sim_com']
            diff = self._sim_agent[idx].project_to_ground(p_com_op-p_com)
            error['stay_near_opponent'] = np.dot(diff, diff)

        if self.exist_rew_fn_subterm(idx, 'dist_to_targets'):
            error['dist_to_targets'] = data['dist_to_targets']

        if self.exist_rew_fn_subterm(idx, 'dist_to_sword'):
            error['dist_to_sword'] = data_op['dist_to_targets']

        if self.exist_rew_fn_subterm(idx, 'out_of_arena'):
            p = data['sim_facing_pos']
            error['out_of_arena'] = 1.0 if self.is_out_of_arena(p) else 0.0

        if self.exist_rew_fn_subterm(idx, 'falldown'):
            error['falldown'] = 1.0 if data['falldown'] else 0.0

        if self.exist_rew_fn_subterm(idx, 'out_of_facing_dir'):
            error['out_of_facing_dir'] = 1.0 if data['out_of_facing_dir'] else 0.0

        if self.exist_rew_fn_subterm(idx, 'too_close_pos'):
            error['too_close_pos'] = 1.0 if data['too_close_pos'] else 0.0

        if self.exist_rew_fn_subterm(idx, 'win'):
            if idx==0:
                error['win'] = 1.0 if data['game_result']=='p1' else 0.0
            else:
                error['win'] = 1.0 if data['game_result']=='p2' else 0.0

        if self.exist_rew_fn_subterm(idx, 'lose'):
            if idx==0:
                error['lose'] = 1.0 if data['game_result']=='p2' else 0.0
            else:
                error['lose'] = 1.0 if data['game_result']=='p1' else 0.0

        if self.exist_rew_fn_subterm(idx, 'draw'):
            error['draw'] = 1.0 if data['game_result']=='draw' else 0.0

        return error
    
    def inspect_end_of_episode(self):
        eoe_reason = super().inspect_end_of_episode()
        if 'timeover' in self._early_term_choices:
            if self._timeover:
                eoe_reason.append("timeover")
        if 'gameover' in self._early_term_choices:
            if self._game_result=='p1':
                eoe_reason.append('player1_win')
            elif self._game_result=='p2':
                eoe_reason.append('player2_win')
            elif self._game_result=='draw':
                eoe_reason.append('draw')
        if 'out_of_arena' in self._early_term_choices:
            for i in range(self._num_agent):
                check = self.is_out_of_arena(self._sim_agent[i].get_root_position())
                if check: eoe_reason.append('[%s] out_of_arena'%self._sim_agent[i].get_name())
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
        if self._game_type=='first_touch':
            raise NotImplementedError
        elif self._game_type=='total_score':
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
        h = self.get_ground_height()
        p1 = self._sim_agent[0].get_facing_position(h)
        p2 = self._sim_agent[1].get_facing_position(h)
        return np.linalg.norm(p1-p2) < 0.5

    def dist_to_targets(self, idx):
        idx_op = self.get_idx_op(idx)
        agent = self._sim_agent[idx]
        agent_op = self._sim_agent[idx_op]        

        ps_sword, Rs_sword, vs_sword, ws_sword = \
            self.get_sword_state_samples(idx)
        ps_target, Qs_target, vs_target, ws_target = \
            agent_op.get_link_states(self._target_links)

        return np.min(
            distance.cdist(ps_sword, ps_target, 'euclidean').flatten())

    def is_out_of_facing_dir(self, idx):
        R_arena_inv = self._arena_info[idx]['R_ref_inv']

        h = self.get_ground_height()
        d = self._sim_agent[idx].get_facing_direction(h)
        d_local = np.dot(R_arena_inv, d)

        v_ref = np.array([-0.5, 0.8, 0.0])
        v_ref /= np.linalg.norm(v_ref)

        # print(idx, np.arccos(np.dot(d_local, v_ref)))

        return np.arccos(np.dot(d_local, v_ref)) >= 0.5*np.pi

    def is_out_of_arena(self, p):
        x1, y1, z1 = self._arena_boundaries[0]
        x2, y2, z2 = self._arena_boundaries[3]
        px, py, pz = p
        return not(x2 <= px <= x1) or not(y2 <= py <= y1)

    def get_ground_height(self):
        return 0.0

    def check_opponent_touched(self, idx):
        '''
        This returns TRUE if the sword touched the opponent
        '''
        pts = self._pb_client.getContactPoints(
            bodyA=self._sim_agent[idx]._body_id, 
            bodyB=self._sim_agent[self.get_idx_op(idx)]._body_id, 
            linkIndexA=self._sim_agent[idx]._char_info.joint_idx["rsword_blade"])

        touch_info = {}

        for p in pts:
            if p[4] in self._target_links and \
               p[9] >= self._game_min_touch_force:
               touch_info['link_idx'] = p[4]
               touch_info['normal_dir'] = p[7]
               touch_info['normal_force'] = p[9]
               break

        return touch_info

    def get_idx_op(self, idx):
        if idx==0:
            return 1
        elif idx==1:
            return 0
        else:
            raise Exception('Invalid index')

    def get_sword_state_tip(self, idx):
        ps, Rs, vs, ws = self.get_sword_state_samples(idx, [-0.3])
        return ps[0], Rs[0], vs[0], ws[0]

    def get_sword_state_samples(self, idx, offsets=[-0.3, -0.1, 0.1]):
        p, Q, v, w = self._sim_agent[idx].get_link_states(
            [self._sim_agent[idx]._char_info.joint_idx["rsword_blade"]])
        R = conversions.Q2R(Q)
        ps = []
        Rs = []
        vs = []
        ws = []
        for o in offsets:
            rr = np.array([o, 0.0, 0.0])
            vv = v + np.cross(w, rr)
            ps.append(p + np.dot(R, rr))
            Rs.append(R)
            vs.append(vv)
            ws.append(w)
        return ps, Rs, vs, ws

    def get_current_time(self, idx):
        return self._start_time[idx] + self.get_elapsed_time()

    def sample_ref_motion(self):
        ref_indices = []
        ref_motions = []
        for i in range(self._num_agent):
            idx = np.random.randint(len(self._ref_motion_all[i]))
            ref_indices.append(idx)
            ref_motions.append(self._ref_motion_all[i][idx])
        if self._verbose:
            print('Ref. motions selected: ', ref_indices)
        return ref_motions

    def render(self, rm):
        ## TODO: make a seperate render-config file for each environment
        for i in range(self._num_agent):
            agent = self._sim_agent[i]
            agent_op = self._sim_agent[(i+1)%self._num_agent]
            if i==0:
                agent_color = np.array([255,  0, 0, 255])/255 
            else:
                agent_color = np.array([0,  0, 255, 255])/255
            # Base color
            rm.COLORS_FOR_AGENTS[i] = {}
            rm.COLORS_FOR_AGENTS[i]['default'] = np.array([220,  220, 220, 255])/255
            rm.COLORS_FOR_AGENTS[i][agent._char_info.rsword_blade] = agent_color
            # Sword tip
            p, R, _, _ = self.get_sword_state_tip(i)
            T = conversions.Rp2T(R, p)
            rm.gl_render.render_sphere(T=T, r=0.02, color=agent_color)
        super().render(rm)
        ''' Fencing Play Stage '''
        color = []
        for i in range(self._num_agent):
            if self._opponent_touched[i]:
                if i==0:
                    color.append(np.array([255,  0, 0, 200])/255)
                else:
                    color.append(np.array([0,  0, 255, 200])/255)
            else:
                color.append(np.array([50,  50, 50, 200])/255)
        alpha = 0.2
        rm.gl_render.render_quad(
            self._arena_boundaries_outer[0],
            self._arena_boundaries_outer[1],
            (1-alpha)*self._arena_boundaries_outer[1]+alpha*self._arena_boundaries_outer[3],
            (1-alpha)*self._arena_boundaries_outer[0]+alpha*self._arena_boundaries_outer[2],
            color=color[0])
        rm.gl_render.render_quad(
            self._arena_boundaries_outer[3],
            self._arena_boundaries_outer[2],
            (1-alpha)*self._arena_boundaries_outer[2]+alpha*self._arena_boundaries_outer[0],
            (1-alpha)*self._arena_boundaries_outer[3]+alpha*self._arena_boundaries_outer[1],
            color=color[1])
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
                agent = self._sim_agent[i]
                agent_op = self._sim_agent[(i+1)%self._num_agent]
                # Sword tip
                p, R, _, _ = self.get_sword_state_tip(i)
                T = conversions.Rp2T(R, p)
                rm.gl_render.render_sphere(T=T, r=0.02, color=[1,0,0])
                # Sword sample points
                pos, _, vel, _ = self.get_sword_state_samples(i)
                for p, v in zip(pos, vel):
                    rm.gl_render.render_sphere(
                        T=conversions.p2T(p), r=0.015, color=[0,0,0])
                    rm.gl_render.render_line(p, p+v, color=[0.5, 0.5, 0.0, 1.0], line_width=1.0)
                # Sword-body lines
                p_sword_2, _, v_sword_2, _ = self.get_sword_state_samples((i+1)%self._num_agent)
                ps1, Qs1, vs1, ws1 = \
                    agent.get_link_states(self._target_links)
                for a in p_sword_2:
                    for b in ps1:
                        rm.gl_render.render_line(
                            a, b, color=[0.0, 0.8, 0.0, 1.0], line_width=1.0)
                # Touched link
                for old_touch_info, cur_touch_info in zip(self._old_touch_info, self._cur_touch_info):
                    if old_touch_info is not None:
                        rm.bullet_render.render_links(
                            pb_client=agent_op._pb_client,
                            model=agent_op._body_id,
                            link_ids=[old_touch_info['link_idx']],
                            color=[0.3, 0.3, 0.3, 1.0],
                            draw_link_info=False
                        )
                    if cur_touch_info is not None:
                        rm.bullet_render.render_links(
                            pb_client=agent_op._pb_client,
                            model=agent_op._body_id,
                            link_ids=[cur_touch_info['link_idx']],
                            color=[0.7, 0.7, 0.7, 1.0],
                            draw_link_info=False
                        )

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
