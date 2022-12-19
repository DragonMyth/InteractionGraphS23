import ray
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
from typing import Dict

import numpy as np
import render_module as rm

import rllib_env_hierarchy_base as env
import env_humanoid_fencing

class HumanoidFencingHierarchy(env.HumanoidHierarchyBase):
    def __init__(self, env_config):
        super().__init__(env_config)

    def get_info(self):
        if self.base_env._game_type == "first_touch":
            player1_win = 1.0 if 'player1_win' in self.base_env._end_of_episode_reason else 0.0
            player2_win = 1.0 if 'player2_win' in self.base_env._end_of_episode_reason else 0.0
        elif self.base_env._game_type == "total_score":
            score = self.base_env._game_remaining_score
            init_score = self.base_env._game_init_score
            player1_win = score[0]/init_score
            player2_win = score[1]/init_score
        else:
            raise NotImplementedError
        info = {
            self.player1: {
                # "info": info_env[0],
                "win": player1_win,
            },
            self.player2: {
                # "info": info_env[1],
                "win": player2_win,
            }
        }
        return info

def policy_mapping_fn(agent_id, share_policy):
    return env.policy_mapping_fn(agent_id, share_policy)

def policies_to_train(share_policy):
    return env.policies_to_train(share_policy)

env_cls = HumanoidFencingHierarchy

def config_override(spec):
    return env.config_override(spec)

class MyCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        episode.custom_metrics["player1_win"] = \
            episode.last_info_for("player1")["win"]
        episode.custom_metrics["player2_win"] = \
            episode.last_info_for("player2")["win"]

class EnvRenderer(env.EnvRenderer):
    def __init__(self, trainer, config, **kwargs):
        super().__init__(trainer, config, **kwargs)

def default_cam():
    return rm.camera.Camera(
        pos=np.array([8.0, 0.0, 8.0]),
        origin=np.array([0.0, 0.0, 0.0]), 
        vup=np.array([0.0, 0.0, 1.0]), 
        fov=30.0)

env.BASE_ENV_CLS = env_humanoid_fencing.Env
env.CALLBACK_FUNC = MyCallbacks
