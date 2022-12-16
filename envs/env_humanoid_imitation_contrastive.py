import os
import numpy as np

from fairmotion.utils import utils
from fairmotion.ops import conversions

import motion_utils

import env_humanoid_base

from motion_plausibility import scorer

class Env(env_humanoid_base.Env):
    def __init__(self, config):
        super().__init__(config)
        
        self._ref_motion = self._base_motion
        self._imit_window = [0.05, 0.15]
        self._start_time = np.zeros(self._num_agent)

        project_dir = config['project_dir']
        ref_motion_db = config['character'].get('ref_motion_db')
        ref_motion_file = \
            motion_utils.collect_motion_files(project_dir, ref_motion_db)
        plausibility_model_file = \
            config['plausibility_model'].get('file')
        plausibility_model_type = \
            config['plausibility_model'].get('model_type')
        plausibility_feature_type = \
            config['plausibility_model'].get('feature_type')
        plausibility_model_past_window_size = \
            config['plausibility_model'].get('past_window_size')
        plausibility_model_skip_frames = \
            config['plausibility_model'].get('skip_frames')
        if project_dir:
            for i in range(len(plausibility_model_file)):
                plausibility_model_file[i] = os.path.join(
                    project_dir, plausibility_model_file[i])
        
        ''' 
        Load reference motions, 
        these will only be used for setting the initial state of the agent
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

        '''
        Load motion plausibility (by contrastive learning) model
        '''

        self._plausibility_models = []
        for i in range(self._num_agent):
            self._plausibility_models.append(
                scorer.ScorerFactory.get_scorer(
                    path=plausibility_model_file[i],
                    model_type=plausibility_model_type[i],
                    feature_type=plausibility_feature_type[i],
                )
            )

        self._past_window_sizes = plausibility_model_past_window_size
        self._skip_frames = plausibility_model_skip_frames

        self._prev_poses = [[] for i in range(self._num_agent)]

        ''' Should call reset after all setups are done '''

        self.reset({'add_noise': False})

        if self._verbose:
            print('----- HumanoidImitationContrastive Environment Created -----')
            for i in range(self._num_agent):
                print('[Agent%d]: state(%d) and action(%d)' \
                      %(i, len(self.state(i)), self._action_space[i].dim))
            print('-------------------------------')

    def callback_reset_prev(self, info):
        
        ''' Choose a reference motion randomly whenever reset '''
        
        self._ref_motion = \
            self.sample_ref_motion(info.get('ref_motion_id'))
        
        ''' Choose a start time for the current reference motion '''
        
        if 'start_time' in info.keys():
            self._start_time = info.get('start_time')
            assert len(self._start_time) == self._num_agent
        else:
            for i in range(self._num_agent):
                self._start_time[i] = \
                    np.random.uniform(0.0, self._ref_motion[i].length())

        ''' 
        Remove saved previous poses and 
        set them by using a newly sampled motion
        '''

        for i in range(self._num_agent):
            self._prev_poses[i].clear()
            for j in range(-self._past_window_sizes[i]*self._skip_frames[i]-1, 0):
                time = max(0.0, self._start_time[i] + j * self._dt_con)
                self._prev_poses[i].append(
                    self._ref_motion[i].get_pose_by_time(time))

    def callback_step_after(self, action_dict, infos):
        ''' Add pose of the currently simulated agent '''
        for i in range(self._num_agent):
            self._prev_poses[i].append(
                self._sim_agent[i].get_pose(self._base_motion[i].skel))

    def print_log_in_step(self):
        if self._verbose and self._end_of_episode:
            print('=================EOE=================')
            print('Reason:', self._end_of_episode_reason)
            print('TIME: (start:%02f) (elapsed:%02f) (time_after_eoe: %02f)'\
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
            ''' Add noise to the state if necessary '''
            if info.get('add_noise'):
                init_pose, init_vel = \
                    self._base_env.add_noise_to_pose_vel(
                        self._sim_agent[i], init_pose, init_vel)
            init_poses.append(init_pose)
            init_vels.append(init_vel)
        return init_poses, init_vels

    def get_state_by_key(self, idx, key):
        state = []

        ref_motion = self._ref_motion[idx] 
        time = self.get_ref_motion_time()
        
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
        elif key=='prev_poses':
            s = []
            for i in range(
                -self._past_window_sizes[idx]*self._skip_frames[idx]-1, 
                -self._skip_frames[idx],
                self._skip_frames[idx]):
                pose = self._prev_poses[idx][i]
                s.append(conversions.R2A(pose.rotations())[1:])
            state.append(np.array(s).flatten())
        elif key=='model_feature':
            prev_poses = []
            for i in range(
                -self._past_window_sizes[idx]*self._skip_frames[idx]-1, 
                -self._skip_frames[idx],
                self._skip_frames[idx]):
                prev_poses.append(self._prev_poses[idx][i])
            cur_pose = self._prev_poses[idx][-1]
            prev_data, cur_data = \
                self._plausibility_models[idx].featurizer.featurize(
                    prev_poses, cur_pose)
            state.append(prev_data.flatten().shape)
            state.append(cur_data.flatten().shape)
        else:
            raise NotImplementedError

        return np.hstack(state)
    
    def reward_data(self, idx):
        data = {}

        data['sim_root_pQvw'] = self._sim_agent[idx].get_root_state()
        data['sim_link_pQvw'] = self._sim_agent[idx].get_link_states()
        data['sim_joint_pv'] = self._sim_agent[idx].get_joint_states()
        data['sim_facing_frame'] = self._sim_agent[idx].get_facing_transform(self.get_ground_height())
        data['sim_com'], data['sim_com_vel'] = self._sim_agent[idx].get_com_and_com_vel()
        data['falldown'] = self._base_env.check_falldown(self._sim_agent[idx])

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

        if self.exist_rew_fn_subterm(idx, 'plausibility_score'):
            prev_poses = []
            for i in range(
                -self._past_window_sizes[idx]*self._skip_frames[idx]-1, 
                -self._skip_frames[idx],
                self._skip_frames[idx]):
                prev_poses.append(self._prev_poses[idx][i])
            cur_pose = self._prev_poses[idx][-1]
            score = self._plausibility_models[idx].evaluate(
                prev_poses,
                cur_pose)
            error['plausibility_score'] = score

        if self.exist_rew_fn_subterm(idx, 'falldown'):
            error['falldown'] = 1.0 if data['falldown'] else 0.0

        return error

    def inspect_end_of_episode(self):
        eoe_reason = super().inspect_end_of_episode()
        if "ref_motion_end" in self._early_term_choices:
            cur_time = self.get_current_time()
            for i in range(self._num_agent):
                check = cur_time[i] >= self._ref_motion[i].length()
                if check: eoe_reason.append('[%s] end_of_motion'%self._sim_agent[i].get_name())
        return eoe_reason

    def get_ground_height(self):
        return 0.0

    def get_ref_motion_time(self):
        cur_time = self.get_current_time()
        return cur_time

    def get_current_time(self):
        return self._start_time + self.get_elapsed_time()

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
            print('Ref. motions selected: ', ref_indices)
        return ref_motions

    def get_phase(self, motion, elapsed_time, mode='linear', **kwargs):
        if mode == 'linear':
            return elapsed_time / motion.length()
        elif mode == 'trigon':
            period = kwargs.get('period', 1.0)
            theta = 2*np.pi * elapsed_time / period
            return np.array([np.cos(theta), np.sin(theta)])
        else:
            raise NotImplementedError

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
            if self.rm.flag['auto_play'] and time_elapsed >= self.env._dt_act:
                self.time_checker_auto_play.begin()
                self.one_step()
        def extra_keyboard_callback(self, key):
            if key == b'r':
                self.reset()
            elif key == b'O':
                size = np.random.uniform(0.1, 0.3, 3)
                p, Q, v, w = self.env._agent[0].get_root_state()
                self.env._obs_manager.throw(p, size=size)
    
    print('=====Humanoid Imitation Environment=====')
    
    args = arg_parser().parse_args()

    env = Env(args.config)

    cam = rm.camera.Camera(pos=np.array([12.0, 0.0, 12.0]),
                           origin=np.array([0.0, 0.0, 0.0]), 
                           vup=np.array([0.0, 0.0, 1.0]), 
                           fov=30.0)

    renderer = EnvRenderer(env=env, cam=cam)
    renderer.run()
