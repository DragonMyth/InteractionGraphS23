import argparse
import os

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from basecode.motion import kinematics_simple as kinematics
from basecode.utils import basics
from basecode.math import mmMath

import pickle
import gzip

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

EPSILON = np.finfo(np.float32).eps

import env_renderer as er
import render_module as rm

v_up = np.array([0.0, 1.0, 0.0])
v_face = np.array([0.0, 0.0, 1.0])
v_up_env = np.array([0.0, 0.0, 1.0])

class Dataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self.X_mean, self.X_std = np.mean(self.X, axis=0), np.std(self.X, axis=0)
        self.Y_mean, self.Y_std = np.mean(self.Y, axis=0), np.std(self.Y, axis=0)

    def __getitem__(self, index):
        # """Returns one data pair (source, target)."""
        # src_seq = (self.src_seqs[index] - self.mean)/(
        #     self.std + constants.EPSILON
        # )
        # tgt_seq = (self.tgt_seqs[index] - self.mean)/(
        #     self.std + constants.EPSILON
        # )
        # src_seq = torch.Tensor(src_seq).to(device=self.device).double()
        # tgt_seq = torch.Tensor(tgt_seq).to(device=self.device).double()

        x = self.preprocess_x(self.X[index])
        y = self.preprocess_y(self.Y[index])

        return x, y

    def __len__(self):
        return len(self.X)

    def preprocess_x(self, x, return_tensor=True):
        x_new = (x - self.X_mean) / (self.X_std + EPSILON)
        if return_tensor:
            x_new = torch.Tensor(x_new)
        return x_new

    def postprocess_x(self, x, return_tensor=True):
        x_new = self.X_mean + np.multiply(x, self.X_std)
        if return_tensor:
            x_new = torch.Tensor(x_new)
        return x_new

    def preprocess_y(self, y, return_tensor=True):
        y_new = torch.Tensor((y - self.Y_mean) / (self.Y_std + EPSILON))
        if return_tensor:
            y_new = torch.Tensor(y_new)
        return y_new

    def postprocess_y(self, y, return_tensor=True):
        y_new = self.Y_mean + np.multiply(y, self.Y_std)
        if return_tensor:
            y_new = torch.Tensor(y_new)
        return y_new

def load_all_motions_in_dir(dir):
    motion_files = basics.files_in_dir(dir, ext="bvh")
    motion_all = []
    for file in motion_files:
        motion = kinematics.Motion(file=file,
                                   scale=1.0,
                                   v_up_skel=v_up,
                                   v_face_skel=v_face,
                                   v_up_env=v_up_env)
        motion_all.append(motion)
        print('Loaded: %s'%file)
    return motion_all

def generate_dataset(motions, output_file="./dataset.pkl.gzip"):
    x, y = [], []
    skel = motions[0].skel
    for m in motions:
        for p in m.postures:
            pose_vec = []
            for j in skel.joints:
                if j==skel.root_joint: 
                    continue
                T = p.get_transform(j, local=True)
                axis_angle = mmMath.logSO3(mmMath.T2R(T))
                pose_vec.append(axis_angle)
            pose_vec = np.hstack(pose_vec)
            x.append(pose_vec)
            y.append(pose_vec)
    dataset = (np.array(x), np.array(y))
    with gzip.open(output_file, "wb") as f:
        pickle.dump(dataset, f)
        print(">> Dataset is stored at: %s"%(output_file))
    return dataset

def load_dataset(file, verbose=True):
    ''' Prepare dataset to learn '''
    if verbose:
        print(">> Loading dataset ...")
    if file.endswith('gzip'):
        with gzip.open(file, "rb") as f:
            X, Y = pickle.load(f)
    elif file.endswith('pkl'):
        with open(file, "rb") as f:
            X, Y = pickle.load(f)
    else:
        raise Exception('Unknown file format')
    return Dataset(X, Y)

def normc_initializer(std=1.0):
    def initializer(tensor):
        tensor.data.normal_(0, 1)
        tensor.data *= std / torch.sqrt(
            tensor.data.pow(2).sum(1, keepdim=True))
    return initializer

def const_initializer(val=0.0):
    def initializer(tensor):
        nn.init.constant_(tensor, val)
    return initializer

class SimpleFC(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 activation_fn=None,
                 initializer_weight=None,
                 initializer_bias=None,
                 ):
        super(SimpleFC, self).__init__()
        layers = []
        linear = nn.Linear(in_size, out_size)
        if initializer_weight:
            initializer_weight(linear.weight)
        if initializer_bias:
            initializer_bias(linear.bias)
        layers.append(linear)
        if activation_fn:
            layers.append(activation_fn())
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)

class Autoencoder(nn.Module):
    def __init__(self, dim_feature, dim_embedding):
        super(Autoencoder, self).__init__()

        assert dim_feature >= dim_embedding

        self.dim_feature = dim_feature
        self.dim_embedding = dim_embedding
        
        self.encoder = nn.Sequential(
            SimpleFC(dim_feature, 64, nn.Tanh, normc_initializer(1.0), const_initializer(0.0)),
            SimpleFC(64, 64, nn.Tanh, normc_initializer(1.0), const_initializer(0.0)),
            SimpleFC(64, dim_embedding, nn.Tanh, normc_initializer(1.0), const_initializer(0.0)),
            )

        self.decoder = nn.Sequential(             
            SimpleFC(dim_embedding, 64, nn.Tanh, normc_initializer(1.0), const_initializer(0.0)),
            SimpleFC(64, 64, nn.Tanh, normc_initializer(1.0), const_initializer(0.0)),
            SimpleFC(64, dim_feature, None, normc_initializer(1.0), const_initializer(0.0))
            )

    def forward(self, x):
        y = self.encoder(x)
        x = self.decoder(y)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, y):
        return self.decoder(y)

class TrainAutoEncoder(tune.Trainable):
    def _setup(self, config):
        ''' Load datasets for train and test and prepare the loaders'''
        file_dataset_train = config.get("dataset_train")
        file_dataset_test = config.get("dataset_test")
        self.dataset_train = load_dataset(file_dataset_train)
        self.dataset_test = load_dataset(file_dataset_test)
        self.train_loader = get_data_loader(self.dataset_train)
        self.test_loader = get_data_loader(self.dataset_test)

        dim_feature = self.dataset_train.X.shape[1]

        ''' Setup torch settings and AE model '''
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = Autoencoder(
            dim_feature,
            config.get("dim_embedding"),
            ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5))
        loss = config.get("loss", "MSE")
        if loss=="MSE":
            self.loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError

    def _train(self):
        ''' For train dataset '''
        mean_train_loss = 0.0
        self.model.train()
        for data in self.train_loader:
            x, y = data
            x = x.to(self.device)
            self.optimizer.zero_grad()
            x_reconstructed = self.model(x)
            loss = self.loss_fn(x_reconstructed, x)
            loss.backward()
            self.optimizer.step()
            mean_train_loss += loss.item()
        mean_train_loss /= len(self.train_loader)
        
        ''' For test dataset '''
        mean_test_loss = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                x, y = data
                x = x.to(self.device)
                x_reconstructed = self.model(x)
                loss = self.loss_fn(x_reconstructed, x)
                mean_test_loss += loss.item()
        mean_test_loss /= len(self.test_loader)

        return {"mean_train_loss": mean_train_loss, "mean_test_loss": mean_test_loss}

    def _save(self, checkpoint_dir):
        print(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

def get_data_loader(dataset):
    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    # with FileLock(os.path.expanduser("~/data.lock")):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True)
    return data_loader

def render_pose(pose, body_model, color, flag):
    skel = pose.skel
    if body_model=='stick_figure':
        for j in skel.joints:
            T = pose.get_transform(j, local=False)
            pos = mmMath.T2p(T)
            rm.gl_render.render_point(pos, radius=0.03, color=[0.8, 0.8, 0.0, 1.0])
            if flag['joint_xform']:
                rm.gl_render.render_transform(T, scale=0.1)
            if j.parent_joint is not None:
                pos_parent = mmMath.T2p(pose.get_transform(j.parent_joint, local=False))
                rm.gl_render.render_line(p1=pos_parent, p2=pos, color=color)
    elif body_model=='stick_figure2':
        for j in skel.joints:
            T = pose.get_transform(j, local=False)
            pos = mmMath.T2p(T)
            rm.gl_render.render_point(pos, radius=0.03, color=color)
            if flag['joint_xform']:
                rm.gl_render.render_transform(T, scale=0.1)
            if j.parent_joint is not None:
                # returns X that X dot vec1 = vec2 
                pos_parent = mmMath.T2p(pose.get_transform(j.parent_joint, local=False))
                p = 0.5 * (pos_parent + pos)
                l = np.linalg.norm(pos_parent-pos)
                r = 0.05
                R = mmMath.getSO3FromVectors(np.array([0, 0, 1]), pos_parent-pos)
                rm.gl_render.render_capsule(mmMath.Rp2T(R,p), l, r, color=color, slice=16)
                # gl_render.render_line(p1=pos_parent, p2=pos, color=color)

class EnvRenderer(er.EnvRenderer):
    def __init__(self, trainer, ref_pose, **kwargs):
        self.trainer = trainer
        self.ref_pose = ref_pose
        self.time_checker_auto_play = basics.TimeChecker()
        super().__init__(**kwargs)
    def sample_random_pose(self):
        y_random = torch.Tensor(np.random.uniform(-1.0, 1.0, self.trainer.model.dim_embedding))
        x = self.trainer.model.decode(y_random).detach().numpy()
        x = self.trainer.dataset_train.postprocess_x(x)

        # x_random = np.random.uniform(-1.0, 1.0, self.trainer.model.dim_feature)
        # x_reconstructed = self.trainer.model(x_random)        

        idx = 0
        for j in self.ref_pose.skel.joints:
            if j==self.ref_pose.skel.root_joint: continue
            self.ref_pose.set_transform(j, mmMath.R2T(mmMath.exp(x[idx:idx+3])), local=True)
            idx += 3
    def render_callback(self):
        flag = {
            "joint_xform": True,
        }
        rm.gl.glEnable(rm.gl.GL_LIGHTING)
        render_pose(self.ref_pose, 'stick_figure2', color=[0.5, 0.5, 0.5, 1.0], flag=flag)
    def idle_callback(self):
        time_elapsed = self.time_checker_auto_play.get_time(restart=False)
        # if self.rm.flag['auto_play'] and time_elapsed >= self.env.base_env._dt_con:
        #     self.time_checker_auto_play.begin()
        #     self.one_step()
    def keyboard_callback(self, key):
        if key == b'r':
            self.sample_random_pose()

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch Pose Embedding")
    parser.add_argument(
        "--ray_address", type=str, default=None, help="The Redis address of the cluster.")
    parser.add_argument(
        "--num_cpus", type=int, default=1, help="Number of CPUs")
    parser.add_argument(
        "--num_gpus", type=int, default=0, help="Number of GPUs")
    parser.add_argument(
        "--output", type=str, help="Dataset File")
    parser.add_argument(
        "--data_train", type=str, help="Dataset File for Train")
    parser.add_argument(
        "--data_test", type=str, help="Dataset File for Test")
    parser.add_argument(
        "--dir", type=str, help="Directory having data")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["gen_dataset", "train", "test"], help="Directory having data")
    parser.add_argument(
        "--checkpoint", type=str, help="Trainer Checkpoint")
    
    args = parser.parse_args()

    if args.data_test is None:
        args.data_test = args.data_train

    if args.mode=="gen_dataset":
        motions = load_all_motions_in_dir(args.dir)
        generate_dataset(motions, output_file=args.output)
    else:
        trainer_config = {
            "args": args,
            "lr": 0.001,
            "dim_embedding": 8,
            "dataset_train": os.path.abspath(args.data_train),
            "dataset_test": os.path.abspath(args.data_test),
            "use_gpu": args.num_gpus > 0,
        }
        if args.mode=="train":
            ''' Initialize ray '''
            ray.init(address=args.ray_address, num_cpus=args.num_cpus, object_store_memory=10**8)
            
            ''' Start to learn with Tune '''
            # sched = ASHAScheduler(metric="mean_accuracy")
            analysis = tune.run(
                TrainAutoEncoder,
                # scheduler=sched,
                stop={
                    # "mean_accuracy": 0.95,
                    "training_iteration": 1000,
                },
                resources_per_trial={
                    "cpu": args.num_cpus,
                    "gpu": args.num_gpus,
                },
                num_samples=1,
                checkpoint_at_end=True,
                checkpoint_freq=10,
                config=trainer_config)
            # print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))
        elif args.mode=="test":
            rm.initialize()
            ref_pose = kinematics.Motion(file="data/motion/amass/amass_hierarchy.bvh",
                                         scale=1.0, 
                                         load_motion=True,
                                         v_up_skel=v_up, 
                                         v_face_skel=v_face, 
                                         v_up_env=v_up_env, 
                                         ).postures[0]

            trainer = TrainAutoEncoder(trainer_config)
            trainer.restore(args.checkpoint)
            model = trainer.model

            cam = rm.camera.Camera(pos=np.array([0.0, 3.0, 2.0]),
                                   origin=np.array([0.0, 0.0, 0.0]), 
                                   vup=np.array([0.0, 0.0, 1.0]), 
                                   fov=45.0)
            renderer = EnvRenderer(trainer=trainer, cam=cam, ref_pose=ref_pose)
            renderer.run()
        else:
            raise NotImplementedError