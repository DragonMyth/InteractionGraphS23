import ray
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
from typing import Dict

import numpy as np
import render_module as rm

import rllib_env_hierarchy_base_moe as env
import env_humanoid_boxing

class HumanoidBoxingHierarchyMOE(env.HumanoidHierarchyBaseMOE):
    def __init__(self, env_config):
        super().__init__(env_config)
    
    def get_info(self):
        if self.base_env._game_type == "total_score":
            score = self.base_env._game_remaining_score
            init_score = self.base_env._game_init_score
            score1 = score[0]/init_score
            score2 = score[1]/init_score
        else:
            raise NotImplementedError 
        info = {
            self.player1: {
                # "info": info_env[0],
                "score": score1,
            },
            self.player2: {
                # "info": info_env[1],
                "score": score2,
            }
        }
        return info

def policy_mapping_fn(agent_id, share_policy):
    return env.policy_mapping_fn(agent_id, share_policy)

def policies_to_train(share_policy):
    return env.policies_to_train(share_policy)

env_cls = HumanoidBoxingHierarchyMOE

def config_override(spec):
    return env.config_override(spec)

class MyCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        episode.custom_metrics["player1_score"] = \
            episode.last_info_for("player1")["score"]
        episode.custom_metrics["player2_score"] = \
            episode.last_info_for("player2")["score"]

class EnvRenderer(env.EnvRenderer):
    def __init__(self, trainer, config, **kwargs):
        super().__init__(trainer, config, **kwargs)

def default_cam():
    return rm.camera.Camera(
        pos=np.array([12.0, 0.0, 12.0]),
        origin=np.array([0.0, 0.0, 0.0]), 
        vup=np.array([0.0, 0.0, 1.0]), 
        fov=30.0)

env.BASE_ENV_CLS = env_humanoid_boxing.Env
env.CALLBACK_FUNC = MyCallbacks
