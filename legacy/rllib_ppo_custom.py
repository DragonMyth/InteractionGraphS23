
import ray
import ray.rllib.agents.ppo as ppo

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.policy.tf_policy import LearningRateSchedule, EntropyCoeffSchedule

from ray.rllib.utils import try_import_tf

from ray.tune.registry import register_trainable

def optimizer(policy, config):
    assert "lr" in config.keys()
    assert "adam_epsilon" in config.keys()
    return tf.train.AdamOptimizer(
        learning_rate=config["lr"], 
        epsilon=config["adam_epsilon"])

Policy = PPOTFPolicy.with_updates(
    name="PPOTFCustomPolicy",
    optimizer_fn=optimizer)
        
DEFAULT_CONFIG = ppo.DEFAULT_CONFIG.copy()
DEFAULT_CONFIG['adam_epsilon'] = 1.0e-5

Trainer = PPOTrainer.with_updates(
    name="PPO_CUSTOM",
    default_config=DEFAULT_CONFIG,
    default_policy=Policy,
    get_policy_class=lambda config: Policy)

register_trainable("PPO_CUSTOM", Trainer)

tf = try_import_tf()
