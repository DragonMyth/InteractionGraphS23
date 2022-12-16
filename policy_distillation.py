import os
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.distributions.categorical import Categorical

import ray
from ray import tune

import pickle

import torch_models
import rllib_model_torch as policy_models

import gym
from gym.spaces import Box

import argparse

import json 

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--data_train", required=True, type=str)
    parser.add_argument("--data_test", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--lr_schedule", type=str, default="step")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--checkpoint_freq", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--cluster", action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--local_dir", type=str, default='~/ray_results')
    parser.add_argument("--mode", choices=["sim", "policy"])
    parser.add_argument("--sim_layer", type=str, default=None)
    parser.add_argument("--vae_kl_coeff", type=float, default=2.0)
    parser.add_argument("--z_model", type=str, default="vae")

    return parser

'''
Our data is a list of episode data like [ep1, ep2, ..., epN].
Each episode data is a list of transitions, where each transition 
is a dictionary {'state_task', 'state_body', 'action'}.
'''

def load_dataset_for_AR(
    file, 
    num_samples=None, 
    lookahead=1, 
    cond="abs", 
    use_a_gt=True):
    with open(file, "rb") as f:
        ''' Load saved data '''
        data = pickle.load(f)
        dim_state_body = data["dim_state_body"]
        dim_action = data["dim_action"]
        episodes = data["episodes"]
        X, Y = [], []

        assert lookahead >= 1

        for ep in episodes:
            num_tuples = len(ep["time"])
            assert num_tuples >= lookahead
            # print('num_tuples:', num_tuples)
            # print('lookahead:', lookahead)
            for i in range(num_tuples-lookahead):
                if num_samples is not None and len(X) > num_samples:
                    break
                x = []
                y = []
                for j in range(lookahead):
                    state_body_t1 = ep["state_body"][i+j]
                    state_body_t2 = ep["state_body"][i+j+1]
                    if use_a_gt:
                        action = ep["action_gt"][i+j]
                    else:
                        action = ep["action"][i+j]
                    # assert dim_state_body == len(state_body_t1)
                    # assert dim_state_body == len(state_body_t2)
                    # assert dim_action == len(action)
                    if cond == "abs":
                        x.append(np.hstack([state_body_t1, state_body_t2]))
                    elif cond == "rel":
                        x.append(np.hstack([state_body_t1, state_body_t2-state_body_t1]))
                    else:
                        raise NotImplementedError
                    y.append(action)
                X.append(np.vstack(x))
                Y.append(np.vstack(y))

        print("------------------Data Loaded------------------")
        print("File:", file)
        print("Num Episode:", len(episodes))
        print("Num Transition:", len(X))
        print("-----------------------------------------------")
        return torch_models.DatasetBase(
                np.array(X), np.array(Y), normalize_x=False, normalize_y=False)
    raise Exception('File error')

def load_dataset_for_pure_distillation(file, num_samples=None):
    with open(file, "rb") as f:
        ''' Load saved data '''
        data = pickle.load(f)
        dim_state = data["dim_state"]
        dim_state_body = data["dim_state_body"]
        dim_state_task = data["dim_state_task"]
        dim_action = data["dim_action"]
        episodes = data["episodes"]
        X, Y = [], []
        for ep in episodes:
            num_trans = len(ep["time"])
            for i in range(num_trans):
                if num_samples is not None and len(X) > num_samples:
                    break
                state_body = ep["state_body"][i]
                state_task = ep["state_task"][i]
                if "action_gt" in ep.keys():
                    action = ep["action_gt"][i]
                else:
                    action = ep["action"][i]
                assert dim_state_body == len(state_body)
                assert dim_state_task == len(state_task)
                assert dim_action == len(action)
                X.append(np.hstack([state_body, state_task]))
                Y.append(action)        
        print("------------------Data Loaded------------------")
        print("File:", file)
        print("Num Episode:", len(episodes))
        print("Num Transition:", len(X))
        print("-----------------------------------------------")
        return torch_models.DatasetBase(
                np.array(X), np.array(Y), normalize_x=False, normalize_y=False)
    raise Exception('File error')

def create_model(config):
    model_config = config["model"]
    obs_space = model_config["custom_model_config"]["observation_space"]
    action_space = model_config["custom_model_config"]["action_space"]
    return policy_models.TaskAgnosticPolicyType1(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=2*action_space.shape[0],
        model_config=model_config,
        name="custom_policy"
        )

# def inspect_dataset(file):
#     with open(file, "rb") as f:
#         ''' Load saved data '''
#         data = pickle.load(f)
#         dim_state = data["dim_state"]
#         dim_state_body = data["dim_state_body"]
#         dim_state_task = data["dim_state_task"]
#         dim_action = data["dim_action"]
#         return dim_state, dim_state_body, dim_state_task, dim_action
#     raise Exception('File error')

def inspect_dataset(file):
    with open(file, "rb") as f:
        ''' Load saved data '''
        data = pickle.load(f)
        # dim_state_body = data["dim_state_body"]
        # dim_action = data["dim_action"]
        ep0 = data["episodes"][0]
        dim_state_body = len(ep0["state_body"][0])
        dim_action = len(ep0["action"][0])
        return 2*dim_state_body, dim_state_body, dim_state_body, dim_action
    raise Exception('File error')

MODEL_CONFIG = {
    "custom_model": "task_agnostic_policy_type1",
    "custom_model_config": {
        "project_dir": None,
        
        "log_std_type": "constant",
        "sample_std": 0.1,

        "task_encoder_type": "mlp",
        "task_encoder_inputs": ["body", "task"],
        "task_encoder_load_weights": None,
        "task_encoder_learnable": True,
        "task_encoder_output_dim": 32,
        "task_encoder_autoreg": False,
        "task_encoder_autoreg_alpha": 0.95,
        "task_encoder_vae": False,
        "task_encoder_layers": [
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
            # {"type": "hardmax"},
        ],

        "task_decoder_enable": False,
        # "task_decoder_layers": [
        #     {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
        #     {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
        #     {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
        # ],
        # "task_decoder_load_weights": None,
        # "task_decoder_learnable": True,
        
        "body_encoder_enable": False,
        # "body_encoder_load_weights": None,
        # "body_encoder_output_dim": 32,
        # "body_encoder_learnable": True,

        "motor_decoder_type": "mlp",
        "motor_decoder_inputs": ["body", "task"],
        "motor_decoder_load_weights": None,
        "motor_decoder_learnable": True,
        "motor_decoder_task_bypass": False,
        "motor_decoder_layers": [
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
        ],
        "motor_decoder_gate_fn_inputs": ["body", "task"],
        "motor_decoder_gate_fn_layers": [
            {"type": "fc", "hidden_size": 64, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 64, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 4, "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
            {"type": "softmax"}
        ],

        "observation_space": None,
        "observation_space_body": None,
        "observation_space_task": None,
        "action_space": None,

        "future_pred_enable": False,
        "future_pred_layers": [
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
        ],

        "future_pred_learnable": True,
        "future_pred_load_weights": None,
    }
}

def gen_layers(width, depth, out_size="output", act_hidden="relu", act_out="linear", add_softmax=False):
    assert depth > 0 and width > 0
    layers = []
    for i in range(depth):
        layers.append(
            {"type": "fc", "hidden_size": width, "activation": act_hidden, "init_weight": {"name": "normc", "std": 1.0}}
        )
    layers.append(
        {"type": "fc", "hidden_size": out_size, "activation": act_out, "init_weight": {"name": "normc", "std": 0.01}}
    )
    if add_softmax:
        layers.append({"type": "softmax"})
    return layers

def get_trainer_config(args):
    dim_state, dim_state_body, dim_state_task, dim_action = \
        inspect_dataset(file=args.data_train)

    ob_scale = 1000.0
    ac_scale = 3.0
    obs_space = \
        Box(low=-ob_scale*np.ones(dim_state),
            high=ob_scale*np.ones(dim_state),
            dtype=np.float64)
    obs_space_body = \
        Box(low=-ob_scale*np.ones(dim_state_body),
            high=ob_scale*np.ones(dim_state_body),
            dtype=np.float64)
    obs_space_task = \
        Box(low=-ob_scale*np.ones(dim_state_task),
            high=ob_scale*np.ones(dim_state_task),
            dtype=np.float64)
    action_space = \
        Box(low=-ac_scale*np.ones(dim_action),
            high=ac_scale*np.ones(dim_action),
            dtype=np.float64)

    model_config = MODEL_CONFIG.copy()
    custom_model_config = model_config["custom_model_config"]
    custom_model_config["observation_space"] = obs_space
    custom_model_config["observation_space_body"] = obs_space_body
    custom_model_config["observation_space_task"] = obs_space_task
    custom_model_config["action_space"] = action_space

    if args.mode == "sim":
        custom_model_config["task_encoder_learnable"] = False
        custom_model_config["motor_decoder_learnable"] = False
        custom_model_config["future_pred_learnable"] = True
        custom_model_config["future_pred_load_weights"] = None
        use_a_gt = tune.grid_search([False])
        vae_kl_coeff = 0.0
        a_rec_coeff = 0.0
        FP_use = True
        FP_rec_coeff = tune.grid_search([1.0])
        FP_cyc_coeff = 0.0
        z_model = tune.grid_search([args.z_model])
        z_dim = 32
    elif args.mode == "policy":
        assert args.sim_layer is not None
        custom_model_config["task_encoder_learnable"] = True
        custom_model_config["motor_decoder_learnable"] = True
        custom_model_config["future_pred_learnable"] = False
        custom_model_config["future_pred_load_weights"] = args.sim_layer
        
        use_a_gt = tune.grid_search([False])
        
        # vae_kl_coeff = tune.grid_search([0.2, 2.0])
        vae_kl_coeff = tune.grid_search([2.0])
        
        a_rec_coeff = tune.grid_search([1.0])
        FP_use = True
        FP_rec_coeff = 0.0
        
        # FP_cyc_coeff = tune.grid_search([0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
        FP_cyc_coeff = tune.grid_search([3e-3])
        
        # z_model = tune.grid_search(["vae"])
        z_model = tune.grid_search([args.z_model])
        z_dim = tune.grid_search([1, 2, 4, 8, 16, 32, 64, 128])
    else:
        raise NotImplementedError

    trainer_config = {
        "model": model_config,
        "lr": args.lr,
        # "lr_schedule": "cosine",
        # "lr_schedule_params": {"T_max": 200},
        # "lr_schedule": "cosine_restart",
        # "lr_schedule_params": {"T_0": 100, "T_mult": 2}, # 100+200+400+800=1500
        # "lr_schedule": "step",
        # "lr_schedule_params": tune.grid_search([
        #     # {"step_size": 50, "gamma": 0.90}, 
        #     # {"step_size": 50, "gamma": 0.80},
        #     {"step_size": 50, "gamma": 0.70},
        #     # {"step_size": 50, "gamma": 0.60},
        #     # {"step_size": 50, "gamma": 0.50},
        #     # {"step_size": 50, "gamma": 0.40},
        #     # {"step_size": 25, "gamma": 0.99},
        #     # {"step_size": 25, "gamma": 0.95},
        #     # {"step_size": 25, "gamma": 0.90},
        #     # {"step_size": 25, "gamma": 0.85},
        #     # {"step_size": 25, "gamma": 0.80},
        #     # {"step_size": 25, "gamma": 0.75},
        #     ]),
        "lr_schedule_params": {"step_size": 50, "gamma": 0.70},
        # "lr_schedule_params": {"step_size": 20, "gamma": 0.70},
        "lr_schedule": args.lr_schedule,
        # "weight_decay": tune.grid_search([0.0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]),
        # "weight_decay": tune.grid_search([5e-7, 1e-6]),
        "weight_decay": 0.0,
        "dataset_train": args.data_train,
        "dataset_test": args.data_test,
        "use_gpu": False,
        "loss": "MSE",
        "loss_test": "MSE",
        "batch_size": args.batch_size,
        "suffle_data": True,
        # "z_dim": tune.grid_search([8, 16, 32, 64, 128]),
        # "z_dim": tune.grid_search([8]),
        "z_dim": z_dim,
        "z_model": z_model,
        
        ## VAE related parameters
        "vae_kl_coeff": vae_kl_coeff,
        
        "a_rec_coeff": a_rec_coeff,

        ## Task Decoder
        # "TD_use": tune.grid_search([False]),
        "TD_use": False,
        # "s_rec_coeff": tune.grid_search([1.0]),

        ## Motor Decoder
        
        "MD_type": tune.grid_search(["mlp"]),
        # "MD_type": tune.grid_search(["moe"]),
        # "MD_in": tune.grid_search([["body", "task"],["body"]]),
        # "MD_in": tune.grid_search([["body", "task"]]),
        "MD_in": ["body", "task"],
        # "MD_bypass": tune.grid_search([False]),
        "MD_bypass": False,
        # "MD_width": tune.grid_search([64, 128, 256, 512]),
        # "MD_depth": tune.grid_search([2, 3, 4]),
        # "MD_width": tune.grid_search([64, 128, 256, 512]),
        # "MD_width": tune.grid_search([512, 1024]),
        "MD_width": tune.grid_search([512]),
        # "MD_depth": tune.grid_search([3]),
        "MD_depth": tune.grid_search([3]),
        # "MD_width": 512,
        # "MD_depth": 2,

        # "entropy_coeff": tune.grid_search([0.0,-1e-7,-3e-7,-1e-6,-3e-6,-1e-5,-3e-5,-1e-4,-3e-4,-1e-3,-3e-3,-1e-2]),

        # "GT_in": tune.grid_search([["body", "task"]]),
        # "GT_in": tune.grid_search([["task"]]),
        # "GT_reg_coeff": tune.grid_search([1.0]),
        # "GT_width": tune.grid_search([128, 256]),
        # "GT_depth": tune.grid_search([2]),
        # "GT_out": tune.grid_search([4,6]),
        # "GT_out": tune.grid_search([8]),

        # "lookahead": tune.grid_search([1, 5, 10, 20, 30, 40, 50, 60])
        "lookahead": tune.grid_search([1]),
        # "lookahead": tune.grid_search([1,2,4,8]),
        # "use_a_gt": tune.grid_search([False]),
        # "use_a_gt": tune.grid_search([True]),
        "use_a_gt": use_a_gt,

        ## Task Encoder
        # "TE_in": tune.grid_search([["body", "task"]]),
        "TE_in": ["body", "task"],
        # "TE_cond": tune.grid_search(["abs"]),
        # "TE_in": tune.grid_search([["task"]]),
        "TE_cond": tune.grid_search(["abs"]),
        # "TE_width": tune.grid_search([256, 512]),
        # "TE_depth": tune.grid_search([2]),
        "TE_width": 256,
        "TE_depth": 2,

        # "act_fn": tune.grid_search(["relu"]),
        "act_fn": "relu",

        "FP_use": FP_use,
        "FP_width": tune.grid_search([1024]),
        "FP_depth": tune.grid_search([2]),
        "FP_rec_coeff": FP_rec_coeff,
        "FP_cyc_coeff": FP_cyc_coeff,
    }

    return trainer_config

def update_model_config(trainer_config):
    model_config = trainer_config["model"]["custom_model_config"]
    model_config["task_encoder_output_dim"] = trainer_config.get("z_dim")
    model_config["task_encoder_inputs"] = trainer_config.get("TE_in")
    model_config["task_encoder_layers"] = \
        gen_layers(
            width=trainer_config.get("TE_width"),
            depth=trainer_config.get("TE_depth"),
            act_hidden=trainer_config.get("act_fn"),
        )
    model_config["motor_decoder_type"] = trainer_config.get("MD_type")
    model_config["motor_decoder_inputs"] = trainer_config.get("MD_in")
    model_config["motor_decoder_task_bypass"] = trainer_config.get("MD_bypass")
    model_config["motor_decoder_gate_fn_inputs"] = trainer_config.get("GT_in", None)
    model_config["motor_decoder_layers"] = \
        gen_layers(
            width=trainer_config.get("MD_width"),
            depth=trainer_config.get("MD_depth"),
            act_hidden=trainer_config.get("act_fn"),
        )
    if trainer_config.get("MD_type") == 'moe':
        model_config["motor_decoder_gate_fn_layers"] = \
            gen_layers(
                width=trainer_config.get("GT_width"),
                depth=trainer_config.get("GT_depth"),
                out_size=trainer_config.get("GT_out"),
                act_hidden=trainer_config.get("act_fn"),
                add_softmax=True,
            )
    
    z_model = trainer_config.get("z_model")
    if z_model == "softmax":
        model_config["task_encoder_layers"].append({"type": "softmax"})
    elif z_model == "hardmax":
        model_config["task_encoder_layers"].append({"type": "hardmax"})
    elif z_model == "vae":
        model_config["task_encoder_vae"] = True
    elif z_model == "none":
        pass
    else:
        raise NotImplementedError

    model_config["task_decoder_enable"] = trainer_config.get("TD_use")
    model_config["future_pred_enable"] = trainer_config.get("FP_use")

    if model_config["future_pred_enable"]:
        model_config["future_pred_layers"] = \
            gen_layers(
                width=trainer_config.get("FP_width"),
                depth=trainer_config.get("FP_depth"),
                act_hidden=trainer_config.get("act_fn"),
            )

class TrainModel(torch_models.TrainModel):
    def setup(self, config):        
        update_model_config(config)
        
        self.z_model = config.get("z_model")
        self.vae_kl_coeff = config.get("vae_kl_coeff", 0.0)
        
        self.use_task_decoder = config.get("TD_use")
        self.s_rec_coeff = config.get("s_rec_coeff", 0.0)

        self.use_moe_motor_decoder = config.get("MD_type") == "moe"
        self.entropy_coeff = config.get("entropy_coeff", 0.0)
        self.gate_fn_reg_coeff = config.get("GT_reg_coeff", 0.0)

        self.use_future_pred = config.get("FP_use")
        self.FP_rec_coeff = config.get("FP_rec_coeff")
        self.FP_cyc_coeff = config.get("FP_cyc_coeff")
        
        self.a_rec_coeff = config.get("a_rec_coeff")
        self.lookahead = config.get("lookahead")
        self.TE_cond = config.get("TE_cond")
        self.use_a_gt = config.get("use_a_gt")

        ''' Save the used parameteres as a file '''
        def format(d, tab=0):
            s = ['{\n']
            if isinstance(d, dict):
                for k,v in d.items():
                    if isinstance(v, dict) or isinstance(v, list):
                        v = format(v, tab+1)
                    else:
                        v = repr(v)
                    s.append('%s%r: %s,\n' % ('  '*tab, k, v))
                s.append('%s}' % ('  '*tab))
            elif isinstance(d, list):
                for v in d:
                    if isinstance(v, dict) or isinstance(v, list):
                        v = format(v, tab+1)
                    else:
                        v = repr(v)
                    s.append('%s%s,\n' % ('  '*tab, v))
                s.append('%s}' % ('  '*tab))
            return ''.join(s)

        with open("params.txt", "w") as f:
            s = format(config, tab=1)
            f.write(s)
            f.close()

        super().setup(config)
    def load_dataset(self, file):
        return load_dataset_for_AR(
            file, lookahead=self.lookahead, cond=self.TE_cond, use_a_gt=self.use_a_gt)
    def create_model(self, config):
        return create_model(config)
    def compute_model(self, x):
        logits, state = self.model(
            input_dict={"obs": x, "obs_flat": x}, state=None, seq_lens=None)
        return logits[...,:logits.shape[1]//2]
    def compute_loss(self, y, x):
        y_recon = None
        
        total_loss = 0.0
        loss_a = loss_kl = loss_s = loss_entropy = loss_FP_rec = loss_FP_cyc = 0.0

        # State at time t
        s1 = x[..., 0, :self.model.dim_state_body]

        for t in range(self.lookahead):

            x_t_gt = x[..., t, :].squeeze()
            y_t_gt = y[..., t, :].squeeze()

            # State at time t
            s1_gt = x_t_gt[..., :self.model.dim_state_body]
            # State at time t+1
            s2_gt = x_t_gt[..., self.model.dim_state_body:]

            x_t = torch.cat([s1, s2_gt], axis=-1)
            y_t = self.compute_model(x_t)

            # Action Reconstruction Loss
            if self.a_rec_coeff > 0.0:
                loss_a += self.loss_fn(y_t_gt, y_t)
                if self.z_model == "vae" and self.vae_kl_coeff > 0.0:
                    # KL-loss
                    mu = self.model._cur_task_encoder_mu
                    logvar = self.model._cur_task_encoder_logvar
                    loss_kl += torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

            # Task Decoder Reconstruction Loss
            if self.use_task_decoder and self.s_rec_coeff > 0.0:
                x_recon = self.model._cur_task_decoder_out
                loss_s += self.loss_fn(x_recon, x)

            # MoE Motor Decoder Entropy Loss
            if self.use_moe_motor_decoder:
                if self.entropy_coeff != 0.0:
                    w = self.model._cur_motor_decoder_expert_weights
                    logvar = torch.log(torch.var(w, dim=0))
                    loss_entropy += torch.mean(logvar)

            # Sim Layer Loss
            if self.use_future_pred:
                if self.FP_rec_coeff > 0:
                    s2_pred, _ = self.model.forward_prediction(s1, y_t_gt)
                    loss_FP_rec += self.loss_fn(s2_gt, s2_pred)
                if self.FP_cyc_coeff > 0:
                    s2_pred = self.model._cur_future_pred
                    loss_FP_cyc += self.loss_fn(s2_gt, s2_pred)

            s1 = self.model._cur_future_pred
        
        if self.lookahead > 1:
            N = float(self.lookahead)
            loss_a /= N
            loss_kl /= N
            loss_s /= N
            loss_entropy /= N
            loss_FP_rec /= N
            loss_FP_cyc /= N

        total_loss = \
            self.a_rec_coeff * loss_a + \
            self.vae_kl_coeff * loss_kl + \
            self.s_rec_coeff * loss_s + \
            self.entropy_coeff * loss_entropy + \
            self.FP_rec_coeff * loss_FP_rec + \
            self.FP_cyc_coeff * loss_FP_cyc

        # print("iter:", self.iter)
        # print(">> loss_a:", loss_a)
        # print(">> loss_kl", loss_kl)
        # print(">> loss_s:", loss_s)
        # print(">> loss_FP_rec:", loss_FP_rec)
        # print(">> loss_FP_cyc:", loss_FP_cyc)
        return total_loss
    
    def compute_test_loss(self, y, x):
        y_recon = None
        
        total_loss = 0.0
        loss_a = loss_kl = loss_s = loss_entropy = loss_FP_rec = loss_FP_cyc = 0.0

        for t in range(self.lookahead):

            x_t = x[..., t, :].squeeze()
            y_t = y[..., t, :].squeeze()

            # State at time t
            s1 = x_t[..., :self.model.dim_state_body]
            # State at time t+1
            s2 = x_t[..., self.model.dim_state_body:]

            y_recon_t = self.compute_model(x_t)

            # Reconstruction Loss
            if self.a_rec_coeff > 0.0:
                loss_a += self.loss_fn(y_recon_t, y_t)
                if self.z_model == "vae" and self.vae_kl_coeff > 0.0:
                    # KL-loss
                    mu = self.model._cur_task_encoder_mu
                    logvar = self.model._cur_task_encoder_logvar
                    loss_kl += torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

            if self.use_future_pred and self.FP_learnable:
                if self.FP_rec_coeff > 0:
                    s2_recon, _ = self.model.forward_prediction(s1, y_t)
                    loss_FP_rec += self.loss_fn(s2_recon, s2)
                    
                if self.FP_cyc_coeff > 0:
                    s2_recon = self.model._cur_future_pred
                    loss_FP_cyc += self.loss_fn(s2_recon, s2)
        
        if self.lookahead > 1:
            N = float(self.lookahead)
            loss_a /= N
            loss_FP_rec /= N
            loss_FP_cyc /= N

        total_loss = \
            self.a_rec_coeff * loss_a + \
            self.FP_rec_coeff * loss_FP_rec + \
            self.FP_cyc_coeff * loss_FP_cyc

        return total_loss

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = super().save_checkpoint(checkpoint_dir)
        if self.model._task_encoder:
            checkpoint = os.path.join(
                checkpoint_dir, "task_encoder.pt")
            self.model.save_weights_task_encoder(checkpoint)
            print("Saved:", checkpoint)
        if self.model._task_decoder:
            checkpoint = os.path.join(
                checkpoint_dir, "task_decoder.pt")
            self.model.save_weights_task_decoder(checkpoint)
            print("Saved:", checkpoint)
        if self.model._motor_decoder:
            checkpoint = os.path.join(
                checkpoint_dir, "motor_decoder.pt")
            self.model.save_weights_motor_decoder(checkpoint)
            print("Saved:", checkpoint)
        if self.model._future_pred:
            checkpoint = os.path.join(
                checkpoint_dir, "sim_layer.pt")
            self.model.save_weights_future_pred(checkpoint)
            print("Saved:", checkpoint)
        return checkpoint_path

if __name__ == "__main__":

    args = arg_parser().parse_args()

    ''' Prepare data and initialize ray '''

    if args.cluster:
        print('>> Trying to initialize Ray')
        ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
        print('>> Ray was initialized')

    exit(0)

    trainer_config = get_trainer_config(args)

    if args.checkpoint is None:
        # print(args.checkpoint_freq)
        # exit(0)

        ''' Train a Model '''

        analysis = tune.run(
            TrainModel,
            # scheduler=sched,
            stop={
                # "mean_accuracy": 0.95,
                "training_iteration": args.max_iter,
            },
            resources_per_trial={
                "cpu": 1,
                "gpu": 1 if args.num_gpus > 0 else 0,
            },
            num_samples=1,
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_at_end=True,
            config=trainer_config,
            reuse_actors=True,
            local_dir=args.local_dir,
            resume=args.resume,
            name=args.name)

        logdir = analysis.get_best_logdir(
            metric='training_iteration', mode='max')
        checkpoint = analysis.get_best_checkpoint(
            logdir, metric='training_iteration', mode='max')
    else:
        checkpoint = args.checkpoint

    if args.output is not None:
        trainer = torch_models.TrainModel(trainer_config)
        trainer.restore(checkpoint)
        torch.save(
            trainer.model.state_dict(), 
            args.output,
        )
        print('Model Saved:', args.output)

    if args.cluster:
        ray.shutdown()