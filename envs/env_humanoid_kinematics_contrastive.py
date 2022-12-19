'''
python3 env_humanoid_contrastive.py

press the space key for proceeding one-step
press 'a' for proceeding continuously
press 'r' for reset the environment
'''


import os

''' 
This forces the environment to use only 1 cpu when running.
This is helpful to launch multiple environment simulatenously.
'''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np

from fairmotion.ops import conversions
import env_humanoid_kinematics
import copy
from collections import deque

class Env(env_humanoid_kinematics.Env):
    def __init__(self, config):
        DEFAULT_CONFIG = {
            'fps': 30,
            'past_window_size': 15,
            'char_info_module': ['amass_char_info.py'],
            'sim_char_file': ['data/character/amass.urdf'],
            'ref_motion_scale': [1.0],
            'base_motion_file': ['data/motion/amass/amass_hierarchy.bvh'],
            'ref_motion_file': [['data/motion/amass/amass_hierarchy.bvh']],
            'plausibility_model': ['data/etc/pretrained_model/fencing/contrastive/aa_15_2_0005/40.model'],
            'plausibility_type': 'mlp',
            'feature_type': 'rotation',
            'low_reward_thres': 0.1,
            'verbose': False,
        }
        config_updated = DEFAULT_CONFIG
        config_updated.update(copy.deepcopy(config))

        project_dir         = config_updated.get('project_dir')
        char_info_module    = config_updated.get('char_info_module')
        sim_char_file       = config_updated.get('sim_char_file')
        base_motion_file    = config_updated.get('base_motion_file')
        ref_motion_file     = config_updated.get('ref_motion_file')
        plausibility_model  = config_updated.get('plausibility_model')
        plausibility_type   = config_updated.get('plausibility_type')
        feature_type        = config_updated.get('feature_type')

        ''' Append project dir to given file paths '''
        if project_dir:
            for i in range(len(char_info_module)):
                char_info_module[i] = os.path.join(project_dir, char_info_module[i])
                sim_char_file[i]    = os.path.join(project_dir, sim_char_file[i])
                base_motion_file[i] = os.path.join(project_dir, base_motion_file[i])
                for j in range(len(ref_motion_file[i])):
                    ref_motion_file[i][j] = os.path.join(project_dir, ref_motion_file[i][j])
                plausibility_model[i] = os.path.join(project_dir, plausibility_model[i])

        super().__init__(
            fps=config_updated.get('fps'),
            past_window_size=config_updated.get('past_window_size'),
            char_info_module=char_info_module,
            sim_char_file=sim_char_file,
            ref_motion_scale=config_updated.get('ref_motion_scale'),
            base_motion_file=base_motion_file,
            ref_motion_file=ref_motion_file,
            verbose=config_updated.get('verbose'),
        )
        
        from motion_plausibility import scorer
        self._plausibility_models = []
        for i in range(self._num_agent):
            self._plausibility_models.append(
                scorer.ScorerFactory.get_scorer(
                    path=plausibility_model[i],
                    model_type=plausibility_type,
                    feature_type=feature_type,
                )
            )
        self._feature_type = feature_type
        self._low_reward_thres = config_updated.get('low_reward_thres')
        self._rew_queue = self._num_agent * [None]
        for i in range(self._num_agent):
            self._rew_queue[i] = deque(maxlen=int(1.0/self._dt))

        self.reset()

    def inspect_end_of_episode(self, idx):
        eoe_reason = []
        if np.mean(list(self._rew_queue[idx])) \
            < self._low_reward_thres * self.reward_max():
            eoe_reason.append('low_reward')
        if self._cur_time[idx] >= self._cur_ref_motion[idx].length():
            eoe_reason.append('ref_motion_end')
        if self._elapsed_time >= 5.0:
            eoe_reason.append('time_elapsed')
        return len(eoe_reason) > 0, eoe_reason

    def reset(self):
        super().reset()
        for i in range(self._num_agent):
            self._rew_queue[i].clear()
            for j in range(self._rew_queue[i].maxlen):
                self._rew_queue[i].append(self.reward_max())

    def collect_step_info(self):
        return [{} for i in range(self._num_agent)]
    
    def reward(self, idx):
        prev_poses []
        for i in range(
            -self._past_window_size*self._skip_frames-1, 
            -self._skip_frames,
            self._skip_frames):
            prev_poses.append(self._prev_poses[idx][i])
        cur_pose = self._prev_poses[idx][-1]
        score = self._plausibility_models[idx].evaluate(
            prev_poses, 
            cur_pose)
        return score

    def step(self, actions):
        rews, info = super().step(actions)
        for i in range(self._num_agent):
            self._rew_queue[i].append(rews[i])
        return rews, info
    
    def state(self, idx):
        if self._feature_type == 'rotation':
            s = []
            # for i in range(-self._past_window_size, 0):
            for i in range(
                -self._past_window_size*self._skip_frames-1, 
                0,
                self._skip_frames):
                pose = self._prev_poses[idx][i]
                s.append(conversions.R2A(pose.rotations())[1:])
            return np.array(s).flatten()
        elif self._feature_type == 'facing':
            prev_poses []
            for i in range(
                -self._past_window_size*self._skip_frames-1, 
                -self._skip_frames,
                self._skip_frames):
                prev_poses.append(self._prev_poses[idx][i])
            cur_pose = self._prev_poses[idx][-1]
            prev_state, cur_state = (
                self._plausibility_models[idx].featurizer.featurize(
                    prev_poses,
                    cur_pose,
                )
            )
            mean = self._plausibility_models[idx].featurizer.mean
            std = self._plausibility_models[idx].featurizer.std
            prev_state = prev_state * std + mean
            return prev_state.flatten()
        else:
            return None

    def reward_max(self):
        return 1.0

    def reward_min(self):
        return 0.0

if __name__ == '__main__':

    import env_renderer as er
    import render_module as rm
    from fairmotion.viz.utils import TimeChecker

    rm.initialize()

    class EnvRenderer(er.EnvRenderer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.time_checker_auto_play = TimeChecker()
            self.reset()
        def reset(self):
            self.env.reset()
        def one_step(self):
            a = np.random.uniform(-0.5, 0.5, size=100)
            self.env.state(0)
            self.env.step([a])
            print('reward:', self.env.reward(0))
        def extra_render_callback(self):
            self.env.render(self.rm)
        def extra_idle_callback(self):
            time_elapsed = self.time_checker_auto_play.get_time(restart=False)
            if self.rm.flag['auto_play'] and time_elapsed >= self.env._dt:
                self.time_checker_auto_play.begin()
                self.one_step()
        def extra_keyboard_callback(self, key):
            if key == b'r':
                self.reset()
            elif key == b' ':
                self.one_step()
    
    print('=====Motion Tracking Controller=====')

    env = Env(config={})

    cam = rm.camera.Camera(
        pos=np.array([12.0, 0.0, 12.0]),
        origin=np.array([0.0, 0.0, 0.0]), 
        vup=np.array([0.0, 0.0, 1.0]), 
        fov=30.0,
    )

    renderer = EnvRenderer(env=env, cam=cam)
    renderer.run()
