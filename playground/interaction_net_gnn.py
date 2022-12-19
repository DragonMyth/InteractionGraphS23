from torch_geometric.nn import GCNConv, GATConv, EdgeConv, DynamicEdgeConv, radius_graph
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.nn import Sequential as GeoSequential
import logging
from black import out
import numpy as np
import os
from math import floor

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, SlimConv2d, normc_initializer
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
# from ray.rllib.utils.framework import get_activation_fn
from ray.rllib.utils import try_import_torch
from gym.spaces import Box
import time
from ray.rllib.models import ModelCatalog
torch, nn = try_import_torch()


def get_activation_fn(name=None):

    if name in ["linear", None]:
        return None
    if name in ["swish", "silu"]:
        from ray.rllib.utils.torch_ops import Swish
        return Swish
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "sigmoid":
        return nn.Sigmoid
    if name == "elu":
        return nn.ELU

    raise ValueError("Unknown activation ({})={}!".format(name))


def get_initializer(info):
    if info['name'] == "normc":
        return normc_initializer(info['std'])
    elif info['name'] == 'xavier_normal':
        def initializer(tensor):
            return nn.init.xavier_normal_(tensor, gain=info['gain'])
        return initializer
    elif info['name'] == 'xavier_uniform':
        def initializer(tensor):
            return nn.init.xavier_uniform_(tensor, gain=info['gain'])
        return initializer
    else:
        raise NotImplementedError


class Hardmax(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        idx = torch.argmax(x, dim=1)
        # print(x.shape)
        # print(idx, self.num_classes)
        y = F.one_hot(idx, num_classes=self.num_classes)
        # print(y)
        return y


class AppendLogStd(nn.Module):
    '''
    An appending layer for log_std.
    '''

    def __init__(self, type, init_val, dim):
        super().__init__()
        self.type = type

        if np.isscalar(init_val):
            init_val = init_val * np.ones(dim)
        elif isinstance(init_val, (np.ndarray, list)):
            assert len(init_val) == dim
        else:
            raise NotImplementedError

        self.init_val = init_val

        if self.type == "constant":
            self.log_std = torch.Tensor(init_val)
        elif self.type == "state_independent":
            self.log_std = torch.nn.Parameter(
                torch.Tensor(init_val))
            self.register_parameter("log_std", self.log_std)
        else:
            raise NotImplementedError

    def set_val(self, val):
        assert self.type == "constant", \
            "Change value is only allowed in constant logstd"
        assert np.isscalar(val), \
            "Only scalar is currently supported"

        self.log_std[:] = val

    def forward(self, x):
        assert x.shape[-1] == self.log_std.shape[-1]

        shape = list(x.shape)
        for i in range(0, len(shape)-1):
            shape[i] = 1
        log_std = torch.reshape(self.log_std, shape)
        shape = list(x.shape)
        shape[-1] = 1
        log_std = log_std.repeat(shape)

        out = torch.cat([x, log_std], axis=-1)
        return out


class FC(nn.Module):
    ''' 
    A network with fully connected layers.
    '''

    def __init__(self, size_in, size_out, layers, append_log_std=False,
                 log_std_type='constant', sample_std=1.0):
        super().__init__()
        nn_layers = []
        prev_layer_size = size_in
        for l in layers:
            layer_type = l['type']
            if layer_type == 'fc':
                assert isinstance(l['hidden_size'],
                                  int) or l['hidden_size'] == 'output'
                hidden_size = l['hidden_size'] if l['hidden_size'] != 'output' else size_out
                layer = SlimFC(
                    in_size=prev_layer_size,
                    out_size=hidden_size,
                    initializer=get_initializer(l['init_weight']),
                    activation_fn=get_activation_fn(l['activation'])
                )
                prev_layer_size = hidden_size
            elif layer_type in ['bn', 'batch_norm']:
                layer = nn.BatchNorm1d(prev_layer_size)
            elif layer_type in ['sm', 'softmax']:
                layer = nn.Softmax(dim=1)
            elif layer_type in ['hm', 'hardmax']:
                layer = Hardmax(num_classes=prev_layer_size)
            else:
                raise NotImplementedError(
                    "Unknown Layer Type:", layer_type)
            nn_layers.append(layer)

        if append_log_std:
            nn_layers.append(AppendLogStd(
                type=log_std_type,
                init_val=np.log(sample_std),
                dim=size_out))

        self._model = nn.Sequential(*nn_layers)

    def forward(self, x):
        return self._model(x)

    def save_weights(self, file):
        torch.save(self.state_dict(), file)

    def load_weights(self, file):
        self.load_state_dict(torch.load(file))
        self.eval()
class Conv2D(nn.Module):
    ''' 
    A network with conv_layers.
    '''
    def __init__(self, channel_in, channel_out, layers):
        super().__init__()
        nn_layers = []
        prev_channel_size = channel_in
        for l in layers:
            layer_type = l['type']
            if layer_type == 'conv':
                assert isinstance(l['channel_out'] , int) or l['channel_out'] =='output'
                channel_out_size = l['channel_out'] if l['channel_out'] != 'output' else channel_out
                kernel = l['kernel']
                stride = l['stride']
                padding= l['padding']
                # in_channels: int,
                # out_channels: int,
                # kernel: Union[int, Tuple[int, int]],
                # stride: Union[int, Tuple[int, int]],
                # padding: Union[int, Tuple[int, int]],
                # # Defaulting these to nn.[..] will break soft torch import.
                # initializer: Any = "default",
                # activation_fn: Any = "default",
                layer = SlimConv2d(
                    in_channels = prev_channel_size,
                    out_channels=channel_out_size,
                    kernel=kernel,
                    stride =stride,
                    padding = padding,
                    # in_size=prev_layer_size,
                    # out_size=hidden_size,
                    initializer=get_initializer(l['init_weight']),
                    activation_fn=get_activation_fn(l['activation'])
                )
                prev_channel_size = channel_out_size
            
            elif layer_type in ['bn', 'batch_norm']:
                layer = nn.BatchNorm2d(prev_channel_size)

            elif layer_type in ['mp', 'max_pooling']:
                kernel = l['kernel']
                stride = l['stride']
                padding= l['padding']
                layer = nn.MaxPool2d(kernel,stride,padding)
            else:
                raise NotImplementedError(
                    "Unknown Layer Type:", layer_type)
            nn_layers.append(layer)


        self._model = nn.Sequential(*nn_layers)
    def forward(self, x):
        ret =  self._model(x)
        ret = ret.reshape(ret.shape[0],-1)
        return ret
    def save_weights(self, file):
        torch.save(self.state_dict(), file)

    def load_weights(self, file):
        self.load_state_dict(torch.load(file))
        self.eval()
class GAT(nn.Module):
    ''' 
    A network with conv_layers.
    '''
    def __init__(self, channel_in, channel_out, out_dim, num_nodes_per_batch,gcn_layers,fc_layers):
        super().__init__()
        self._num_nodes_per_batch = num_nodes_per_batch
        gcn_nn_layers = []
        fc_nn_layers = []
        prev_channel_size = channel_in
        for l in gcn_layers:
            assert isinstance(l['channel_out'] , int) or l['channel_out'] =='output'
            channel_out_size = l['channel_out'] if l['channel_out'] != 'output' else channel_out

            layer = GATConv(prev_channel_size,channel_out_size,add_self_loops=False)
            
            gcn_nn_layers.append((layer,"x, edge_index, edge_attr -> x"))
            prev_channel_size = channel_out_size

        self._gcn_model = GeoSequential('x, edge_index, edge_attr',gcn_nn_layers)
        size_in = self._num_nodes_per_batch*prev_channel_size
        prev_layer_size = size_in
        for l in fc_layers:
            hidden_size = l['hidden_size'] if l['hidden_size'] != 'output' else out_dim
            layer = SlimFC(
                in_size=prev_layer_size,
                out_size=hidden_size,
                initializer=get_initializer(l['init_weight']),
                activation_fn=get_activation_fn(l['activation'])
            )
            prev_layer_size = hidden_size
            fc_nn_layers.append(layer)

        self._fc_model = nn.Sequential(*fc_nn_layers)

    def forward(self, x, edge_index):
        num_batch = x.shape[0]//self._num_nodes_per_batch
        gcn_out = self._gcn_model(x,edge_index)
        gcn_out = gcn_out.reshape(num_batch,-1)

        fc_out = self._fc_model(gcn_out)
        return fc_out

    def save_weights(self, file):
        torch.save(self.state_dict(), file)

    def load_weights(self, file):
        self.load_state_dict(torch.load(file))
        self.eval()
    pass
class GCN(nn.Module):
    ''' 
    A network with conv_layers.
    '''
    def __init__(self, channel_in, channel_out, out_dim, num_nodes_per_batch,gcn_layers,fc_layers):
        super().__init__()
        self._num_nodes_per_batch = num_nodes_per_batch
        gcn_nn_layers = []
        fc_nn_layers = []
        prev_channel_size = channel_in
        for l in gcn_layers:
            assert isinstance(l['channel_out'] , int) or l['channel_out'] =='output'
            channel_out_size = l['channel_out'] if l['channel_out'] != 'output' else channel_out

            layer = GCNConv(prev_channel_size,channel_out_size)
            
            activation = get_activation_fn(l['activation'])
            gcn_nn_layers.append((layer,"x, edge_index -> x"))
            gcn_nn_layers.append(activation())
            prev_channel_size = channel_out_size

        self._gcn_model = GeoSequential('x, edge_index',gcn_nn_layers)
        size_in = self._num_nodes_per_batch*prev_channel_size
        prev_layer_size = size_in
        for l in fc_layers:
            hidden_size = l['hidden_size'] if l['hidden_size'] != 'output' else out_dim
            layer = SlimFC(
                in_size=prev_layer_size,
                out_size=hidden_size,
                initializer=get_initializer(l['init_weight']),
                activation_fn=get_activation_fn(l['activation'])
            )
            prev_layer_size = hidden_size
            fc_nn_layers.append(layer)

        self._fc_model = nn.Sequential(*fc_nn_layers)

    def forward(self, x, edge_index):
        num_batch = x.shape[0]//self._num_nodes_per_batch
        gcn_out = self._gcn_model(x,edge_index)
        gcn_out = gcn_out.reshape(num_batch,-1)

        fc_out = self._fc_model(gcn_out)
        return fc_out

    def save_weights(self, file):
        torch.save(self.state_dict(), file)

    def load_weights(self, file):
        self.load_state_dict(torch.load(file))
        self.eval()
class InteractionPolicy(TorchModelV2, nn.Module):
    ''' 
    A policy that generates action and value with FCNN
    '''
    DEFAULT_CONFIG = {
        "log_std_type": "constant",
        "sample_std": 1.0,
        "interaction_out_channel":3,
        "interaction_fc_out_shape":128,
        "interaction_net_type":"conv",
        "interaction_layers":[
            {"type": "conv","channel_out":10,"kernel":5,"stride":2,"padding":2,"activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "mp", "kernel":3,"stride":1,"padding":0},
            {"type": "conv","channel_out":"output","kernel":3,"stride":1,"padding":1,"activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
        ],
        "interaction_fc_layers":[
            {"type": "fc", "hidden_size": "out", "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
        ],


        "policy_fn_layers": [
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
        ],
        
        "log_std_fn_hiddens": [64, 64],
        "log_std_fn_activations": ["relu", "relu", "linear"],
        "log_std_fn_init_weights": [1.0, 1.0, 0.01],
        "log_std_fn_base": 0.0,
        
        "value_fn_layers": [
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
        ],
        "interaction_obs_dim" : None,
        "interaction_obs_num" : None,
        "interaction_feature_dim": None,
    }
    """Generic fully connected network."""
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **model_kwargs):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        ''' Load and check configuarations '''

        custom_model_config = InteractionPolicy.DEFAULT_CONFIG.copy()
        custom_model_config_by_user = model_config.get("custom_model_config")
        if custom_model_config_by_user:
            custom_model_config.update(custom_model_config_by_user)

        '''
        constant
            log_std will not change during the training
        state_independent
            log_std will be learned during the training
            but it does not depend on the state of the agent
        state_dependent:
            log_std will be learned during the training
            and it depens on the state of the agent
        '''

        log_std_type = custom_model_config.get("log_std_type")
        assert log_std_type in \
            ["constant", "state_independent", "state_dependent"]

        sample_std = custom_model_config.get("sample_std")
        assert np.array(sample_std).all() > 0.0, \
            "The value shoulde be positive"

        assert num_outputs % 2 == 0, (
            "num_outputs must be divisible by two", num_outputs)
        num_outputs = num_outputs//2
        append_log_std = (log_std_type != "state_dependent")

        policy_fn_layers = custom_model_config.get("policy_fn_layers")
        
        value_fn_layers = custom_model_config.get("value_fn_layers")

        interaction_layers = custom_model_config.get("interaction_layers")
        interaction_out_channel = custom_model_config.get("interaction_out_channel")
        interaction_fc_out_shape = custom_model_config.get("interaction_fc_out_shape")
        interaction_fc_layers = custom_model_config.get("interaction_fc_layers")
        self.interaction_net_type = custom_model_config.get("interaction_net_type")
        self._interaction_obs_dim = custom_model_config.get("interaction_obs_dim")
        self._interaction_obs_num = custom_model_config.get("interaction_obs_num")
        self._interaction_feature_dim = custom_model_config.get("interaction_feature_dim")


        dim_state = int(np.product(obs_space.shape))
        self._dim_fc = dim_state - self._interaction_obs_dim*self._interaction_obs_num

        print("Interaction Network Dimension: \nFC Input: %d, \nInteraction Input Dim: %d \nInteraction Input Num %d"%(self._dim_fc,self._interaction_obs_dim,self._interaction_obs_num))
        if self.interaction_net_type=="conv":

            def compute_conv_out_flat_dim(dim_in,layers):
                new_dim = dim_in

                for l in layers:
                    if l['type']=='conv' or l['type']=='mp':
                        k = l['kernel']
                        s = l['stride']
                        p = l['padding']
                        new_dim = floor((new_dim-k+2*p)/s)+1
                    else:
                        continue
                return new_dim*new_dim*interaction_out_channel 
            
            def conv2d_fc_fn(params):
                
                out_shape = compute_conv_out_flat_dim(self._interaction_feature_dim[0],params['layers'])
                conv_fn = Conv2D(**params)

                l = interaction_fc_layers[0]
                fc_fn = SlimFC(
                    in_size=out_shape,
                    out_size=interaction_fc_out_shape,
                    initializer=get_initializer(l['init_weight']),
                    activation_fn=get_activation_fn(l['activation'])
                )
                
                out_fn = nn.Sequential(conv_fn,fc_fn)
                return out_fn

            conv_params = {
                "channel_in":3,
                "channel_out":interaction_out_channel,
                "layers":interaction_layers,
            }

            self._pos_interaction_net = conv2d_fc_fn(conv_params)
            self._vel_interaction_net = conv2d_fc_fn(conv_params)

            downstream_in_size =  self._dim_fc+self._interaction_obs_num*interaction_fc_out_shape*2
        elif self.interaction_net_type == "gcn":
            num_nodes_per_batch = self._interaction_feature_dim[0]

            gnn_params = {
                "channel_in":3,
                "channel_out":interaction_out_channel,
                "out_dim": interaction_fc_out_shape,
                "num_nodes_per_batch":num_nodes_per_batch,

                "gcn_layers" : interaction_layers,
                "fc_layers" : interaction_fc_layers            
            }

            self._pos_interaction_net = GCN(**gnn_params)
            self._vel_interaction_net = GCN(**gnn_params)
            downstream_in_size =  self._dim_fc+self._interaction_obs_num*interaction_fc_out_shape*2
            print("Downstream Size Total: %d \nFC dim: %d, \nInteraction Output: %d"%(downstream_in_size,self._dim_fc,self._interaction_obs_num*interaction_fc_out_shape*2))
        elif self.interaction_net_type == "gcn_fast":
            num_nodes_per_batch = self._interaction_feature_dim[0]

            gnn_params = {
                "channel_in":3,
                "channel_out":interaction_out_channel,
                "out_dim": interaction_fc_out_shape,
                "num_nodes_per_batch":num_nodes_per_batch,

                "gcn_layers" : interaction_layers,
                "fc_layers" : interaction_fc_layers            
            }

            self._pos_interaction_net = GCN(**gnn_params)
            self._vel_interaction_net = GCN(**gnn_params)
            downstream_in_size =  self._dim_fc+self._interaction_obs_num*interaction_fc_out_shape*2
            print("Downstream Size Total: %d \nFC dim: %d, \nInteraction Output: %d"%(downstream_in_size,self._dim_fc,self._interaction_obs_num*interaction_fc_out_shape*2))
        elif self.interaction_net_type == "fc":
            downstream_in_size =  dim_state

        ''' Construct the policy function '''
        param = {
            "size_in": downstream_in_size, 
            "size_out": num_outputs, 
            "layers": policy_fn_layers,
            "append_log_std": append_log_std,
            "log_std_type": log_std_type,
            "sample_std": sample_std
        }
        self._policy_fn = FC(**param)

        ''' Construct the value function '''

        param = {
            "size_in": downstream_in_size, 
            "size_out": 1, 
            "layers": value_fn_layers,
            "append_log_std": False
        }
        self._value_fn = FC(**param)

        ''' Keep the latest output of the value function '''

        self._cur_value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        
        obs = input_dict["obs_flat"].float()
        obs = obs.reshape(obs.shape[0], -1)
        begin_time = time.time()

        if self.interaction_net_type=="conv":
            outs = []
            for i in range(self._interaction_obs_num):
                
                seg_length = self._interaction_obs_dim
                seg = obs[:,self._dim_fc+i*seg_length:self._dim_fc+(i+1)*seg_length]
                interaction_point_dim = self._interaction_feature_dim[0]*self._interaction_feature_dim[1]
                seg = seg[:, interaction_point_dim:]
                seg = seg.reshape(seg.shape[0],self._interaction_feature_dim[1],self._interaction_feature_dim[0],self._interaction_feature_dim[0])
                seg_pos_info = seg[:,:3,:,:]
                seg_vel_info = seg[:,3:,:,:]

                pos_out = self._pos_interaction_net(seg_pos_info)
                vel_out = self._vel_interaction_net(seg_vel_info)

                outs.append(pos_out)
                outs.append(vel_out)
            outs.append(obs[:,:self._dim_fc])
            obs = torch.cat(outs, axis=-1)   
        elif self.interaction_net_type == "gcn_fast":
            outs = []
            max_edges = self._interaction_feature_dim[0] * self._interaction_feature_dim[0]
             
            for i in range(self._interaction_obs_num):

                feat_dim = self._interaction_feature_dim[1]//2
                seg_length = self._interaction_obs_dim
                seg = obs[:,self._dim_fc+i*seg_length:self._dim_fc+(i+1)*seg_length]
                interaction_point_dim = self._interaction_feature_dim[0]*self._interaction_feature_dim[1]

                num_edges = 36

                total_num_edges = self._interaction_feature_dim[0]*self._interaction_feature_dim[0]
                seg_interaction_points = seg[:,:interaction_point_dim]
                seg_interaction_points = seg_interaction_points.reshape(seg_interaction_points.shape[0],self._interaction_feature_dim[0],self._interaction_feature_dim[1])

                seg_interaction_edges_connectivity = seg[:,interaction_point_dim+1:interaction_point_dim+1+total_num_edges*2]
                seg_interaction_edges_connectivity = seg_interaction_edges_connectivity.reshape(seg_interaction_edges_connectivity.shape[0],2,total_num_edges)
                seg_interaction_edges_connectivity = seg_interaction_edges_connectivity[:,:,:max_edges]
                seg_interaction_edges_connectivity = seg_interaction_edges_connectivity.to(torch.long)
                
                seg_interaction_edges_features = seg[:,interaction_point_dim+1+total_num_edges*2:]
                seg_interaction_edges_features = seg_interaction_edges_features.reshape(seg_interaction_edges_features.shape[0],-1,self._interaction_feature_dim[1])
                seg_interaction_edges_features = seg_interaction_edges_features[:,:max_edges,:]

                start_time = time.time()
                batch_edge_index = Batch.from_data_list([Data(edge_attr=seg_interaction_edges_features[i,:num_edges,:],edge_index=seg_interaction_edges_connectivity[i,:,:num_edges],num_nodes=self._interaction_feature_dim[0]) for i in range(seg_interaction_edges_features.shape[0])])
                # batch_edge_index = Batch.from_data_list([Data(edge_index=seg_interaction_edges_connectivity[i,:,:num_edges],num_nodes=self._interaction_feature_dim[0]) for i in range(seg_interaction_edges_features.shape[0])])

                print("---GCN Building Blocked Diag Matrix: %.4f seconds ---" % (time.time() - start_time))
                start_time = time.time()


                seg_interaction_points = seg_interaction_points.reshape(seg_interaction_points.shape[0],-1,self._interaction_feature_dim[1])
                seg_interaction_points_pos = seg_interaction_points[:,:,:3]
                seg_interaction_points_pos = seg_interaction_points_pos.reshape(-1,feat_dim)

                seg_interaction_points_vel = seg_interaction_points[:,:,3:]
                seg_interaction_points_vel = seg_interaction_points_vel.reshape(-1,feat_dim)

                # edge_index = radius_graph(seg_interaction_points_pos,10000,batch=batch)

                pos_out = self._pos_interaction_net(seg_interaction_points_pos,batch_edge_index.edge_index)
                vel_out = self._vel_interaction_net(seg_interaction_points_vel,batch_edge_index.edge_index)
                print("---GCN Passing Data: %.4f seconds ---" % (time.time() - start_time))

                outs.append(pos_out)
                outs.append(vel_out)
            outs.append(obs[:,:self._dim_fc])
            obs = torch.cat(outs, axis=-1)
        elif self.interaction_net_type=="gcn":
            outs = []

            for i in range(self._interaction_obs_num):

                feat_dim = self._interaction_feature_dim[1]//2
                seg_length = self._interaction_obs_dim
                seg = obs[:,self._dim_fc+i*seg_length:self._dim_fc+(i+1)*seg_length]

                interaction_point_dim = self._interaction_feature_dim[0]*self._interaction_feature_dim[1]
                seg_interaction_points = seg[:,:interaction_point_dim]
                seg_interaction_edges = seg[:,interaction_point_dim:]
                seg_interaction_edges = seg_interaction_edges.reshape(seg_interaction_edges.shape[0],self._interaction_feature_dim[1],self._interaction_feature_dim[0],self._interaction_feature_dim[0])
                seg_interaction_edges_splits = seg_interaction_edges.split(1,dim=1)
                diag_block_edges = []

                start_time = time.time()
                for edges in seg_interaction_edges_splits:
                    squeezed = edges.squeeze(1)
                    block_mat = torch.block_diag(*squeezed)
                    block_mat = block_mat.unsqueeze(0)
                    diag_block_edges.append(block_mat)

                diag_block_edges = torch.cat(diag_block_edges,dim=0)
                # print(diag_block_edges.shape)
                edge_index, edge_attr = dense_to_sparse(diag_block_edges)
                # print(edge_index)

                print("---GCN Building Blocked Diag Matrix: %.4f seconds ---" % (time.time() - start_time))
                start_time = time.time()

                num_batch = seg.shape[0]
                batches = torch.arange(num_batch)
                batch = torch.repeat_interleave(batches,(seg_interaction_points.shape[1]//self._interaction_feature_dim[1]))


                seg_interaction_points = seg_interaction_points.reshape(seg_interaction_points.shape[0],-1,self._interaction_feature_dim[1])
                seg_interaction_points_pos = seg_interaction_points[:,:,:3]
                seg_interaction_points_pos = seg_interaction_points_pos.reshape(-1,feat_dim)

                seg_interaction_points_vel = seg_interaction_points[:,:,3:]
                seg_interaction_points_vel = seg_interaction_points_vel.reshape(-1,feat_dim)

                # edge_index = radius_graph(seg_interaction_points_pos,10000,batch=batch)

                pos_out = self._pos_interaction_net(seg_interaction_points_pos,edge_index)
                vel_out = self._vel_interaction_net(seg_interaction_points_vel,edge_index)
                print("---GCN Passing Data: %.4f seconds ---" % (time.time() - start_time))

                outs.append(pos_out)
                outs.append(vel_out)
            outs.append(obs[:,:self._dim_fc])
            obs = torch.cat(outs, axis=-1)
        print("---Upstream passing Data: %.4f seconds ---" % (time.time() - begin_time))

        start_time = time.time()
        logits = self._policy_fn(obs)
        self._cur_value = self._value_fn(obs).squeeze(1)
        print("---Downstream passing Data: %.4f seconds ---" % (time.time() - start_time))

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def save_policy_weights(self, file):
        torch.save(self._policy_fn.state_dict(), file)
        # print(self._policy_fn.state_dict())
        # print(self._value_fn.state_dict())

    def load_policy_weights(self, file):
        self._policy_fn.load_state_dict(torch.load(file))
        self._policy_fn.eval()


if __name__ == "__main__":

    model_gcn_config = {'_use_default_native_models': False, '_disable_preprocessor_api': False, 'fcnet_hiddens': [256, 256], 'fcnet_activation': 'tanh', 'conv_filters': None, 'conv_activation': 'relu', 'post_fcnet_hiddens': [], 'post_fcnet_activation': 'relu', 'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': False, 'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256, 'lstm_use_prev_action': False, 'lstm_use_prev_reward': False, '_time_major': False, 'use_attention': False, 'attention_num_transformer_units': 1, 'attention_dim': 64, 'attention_num_heads': 1, 'attention_head_dim': 32, 'attention_memory_inference': 50, 'attention_memory_training': 50, 'attention_position_wise_mlp_dim': 32, 'attention_init_gru_gate_bias': 2.0, 'attention_use_n_prev_actions': 0, 'attention_use_n_prev_rewards': 0, 'framestack': True, 'dim': 84, 'grayscale': False, 'zero_mean': True, 'custom_model': 'interaction_net', 'custom_model_config': {'interaction_net_type': 'gcn', 'interaction_fc_out_shape': 64, 'interaction_out_channel': 5, 'interaction_layers': [{'channel_out': 4, 'activation': 'relu', 'init_weight': {'name': 'normc', 'std': 1.0}}, {'channel_out': 'output', 'activation': 'relu', 'init_weight': {'name': 'normc', 'std': 1.0}}], 'interaction_fc_layers': [{'hidden_size': 128, 'activation': 'relu', 'init_weight': {'name': 'normc', 'std': 1.0}}, {'hidden_size': 'output', 'activation': 'relu', 'init_weight': {'name': 'normc', 'std': 1.0}}], 'log_std_type': 'constant', 'sample_std': 0.1, 'project_dir': '/private/home/yzhang3027/ScaDive/', 'interaction_obs_dim': 252, 'interaction_obs_num': 3, 'interaction_feature_dim': (6, 6)}, 'custom_action_dist': None, 'custom_preprocessor': None, 'lstm_use_prev_action_reward': -1}
    model_gcn_fast_config = {'_use_default_native_models': False, '_disable_preprocessor_api': False, 'fcnet_hiddens': [256, 256], 'fcnet_activation': 'tanh', 'conv_filters': None, 'conv_activation': 'relu', 'post_fcnet_hiddens': [], 'post_fcnet_activation': 'relu', 'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': False, 'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256, 'lstm_use_prev_action': False, 'lstm_use_prev_reward': False, '_time_major': False, 'use_attention': False, 'attention_num_transformer_units': 1, 'attention_dim': 64, 'attention_num_heads': 1, 'attention_head_dim': 32, 'attention_memory_inference': 50, 'attention_memory_training': 50, 'attention_position_wise_mlp_dim': 32, 'attention_init_gru_gate_bias': 2.0, 'attention_use_n_prev_actions': 0, 'attention_use_n_prev_rewards': 0, 'framestack': True, 'dim': 84, 'grayscale': False, 'zero_mean': True, 'custom_model': 'interaction_net', 'custom_model_config': {'interaction_net_type': 'gcn_fast', 'interaction_fc_out_shape': 64, 'interaction_out_channel': 5, 'interaction_layers': [{'channel_out': 4, 'activation': 'relu', 'init_weight': {'name': 'normc', 'std': 1.0}}, {'channel_out': 'output', 'activation': 'relu', 'init_weight': {'name': 'normc', 'std': 1.0}}], 'interaction_fc_layers': [{'hidden_size': 128, 'activation': 'relu', 'init_weight': {'name': 'normc', 'std': 1.0}}, {'hidden_size': 'output', 'activation': 'relu', 'init_weight': {'name': 'normc', 'std': 1.0}}], 'log_std_type': 'constant', 'sample_std': 0.1, 'project_dir': '/private/home/yzhang3027/ScaDive/', 'interaction_obs_dim': 325, 'interaction_obs_num': 3, 'interaction_feature_dim': (6, 6)}, 'custom_action_dist': None, 'custom_preprocessor': None, 'lstm_use_prev_action_reward': -1}

    model_cnn_config = {'_use_default_native_models': False, '_disable_preprocessor_api': False, 'fcnet_hiddens': [256, 256], 'fcnet_activation': 'tanh', 'conv_filters': None, 'conv_activation': 'relu', 'post_fcnet_hiddens': [], 'post_fcnet_activation': 'relu', 'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': False, 'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256, 'lstm_use_prev_action': False, 'lstm_use_prev_reward': False, '_time_major': False, 'use_attention': False, 'attention_num_transformer_units': 1, 'attention_dim': 64, 'attention_num_heads': 1, 'attention_head_dim': 32, 'attention_memory_inference': 50, 'attention_memory_training': 50, 'attention_position_wise_mlp_dim': 32, 'attention_init_gru_gate_bias': 2.0, 'attention_use_n_prev_actions': 0, 'attention_use_n_prev_rewards': 0, 'framestack': True, 'dim': 84, 'grayscale': False, 'zero_mean': True, 'custom_model': 'interaction_net', 'custom_model_config': {'interaction_net_type': 'conv', 'interaction_fc_out_shape': 64, 'interaction_out_channel': 5, 'interaction_layers': [{'type': 'conv', 'channel_out': 10, 'kernel': 5, 'stride': 1, 'padding': 2, 'activation': 'relu', 'init_weight': {'name': 'normc', 'std': 1.0}}, {'type': 'mp', 'kernel': 3, 'stride': 1, 'padding': 1}, {'type': 'conv', 'channel_out': 'output', 'kernel': 3, 'stride': 1, 'padding': 1, 'activation': 'relu', 'init_weight': {'name': 'normc', 'std': 1.0}}], 'interaction_fc_layers': [{'type': 'fc', 'hidden_size': 'output', 'activation': 'relu', 'init_weight': {'name': 'normc', 'std': 1.0}}], 'log_std_type': 'constant', 'sample_std': 0.1, 'project_dir': '/private/home/yzhang3027/ScaDive/', 'interaction_obs_dim': 252, 'interaction_obs_num': 3, 'interaction_feature_dim': (6, 6)}, 'custom_action_dist': None, 'custom_preprocessor': None, 'lstm_use_prev_action_reward': -1}
    model_fc_config = {'_use_default_native_models': False, '_disable_preprocessor_api': False, 'fcnet_hiddens': [256, 256], 'fcnet_activation': 'tanh', 'conv_filters': None, 'conv_activation': 'relu', 'post_fcnet_hiddens': [], 'post_fcnet_activation': 'relu', 'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': False, 'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256, 'lstm_use_prev_action': False, 'lstm_use_prev_reward': False, '_time_major': False, 'use_attention': False, 'attention_num_transformer_units': 1, 'attention_dim': 64, 'attention_num_heads': 1, 'attention_head_dim': 32, 'attention_memory_inference': 50, 'attention_memory_training': 50, 'attention_position_wise_mlp_dim': 32, 'attention_init_gru_gate_bias': 2.0, 'attention_use_n_prev_actions': 0, 'attention_use_n_prev_rewards': 0, 'framestack': True, 'dim': 84, 'grayscale': False, 'zero_mean': True, 'custom_model': 'interaction_net', 'custom_model_config': {'interaction_net_type': 'fc', 'log_std_type': 'constant', 'sample_std': 0.1, 'project_dir': '/private/home/yzhang3027/ScaDive/', 'interaction_obs_dim': 252, 'interaction_obs_num': 3, 'interaction_feature_dim': (6, 6)}, 'custom_action_dist': None, 'custom_preprocessor': None, 'lstm_use_prev_action_reward': -1}
    name = "default_model"
    obs_space = Box(-1000 * np.ones(756),
                1000 * np.ones(756),
                dtype=np.float64)

    obs_space_sparse = Box(-1000 * np.ones(975),
                1000 * np.ones(975),
                dtype=np.float64)
    action_space = Box(-3 * np.ones(51),
                3 * np.ones(51),
                dtype=np.float64)
    num_outputs = 102

    gcn_model = InteractionPolicy(obs_space,action_space,num_outputs,model_gcn_config,name)
    gcn_fast_model = InteractionPolicy(obs_space_sparse,action_space,num_outputs,model_gcn_fast_config,name)

    cnn_model = InteractionPolicy(obs_space,action_space,num_outputs,model_cnn_config,name)
    fc_model = InteractionPolicy(obs_space,action_space,num_outputs,model_fc_config,name)

    data1 = torch.zeros((1,756))
    data1_dict = { 
        'obs':data1 ,
        'obs_flat' :data1
        }



    data2 = torch.zeros((250,756))
    data2_dict = { 
        'obs':data2 ,
        'obs_flat' :data2
        }

    data3 = torch.zeros((500,756))
    data3_dict = { 
        'obs':data3 ,
        'obs_flat' :data3
        }  

    data3_fast = torch.zeros((500,975))
    
    data3_fast_dict = { 
        'obs':data3_fast ,
        'obs_flat' :data3_fast
        }  
    # start_time = time.time()
    # output = fc_model(data1_dict,[],None)
    # print("---FC Batch = 1 Input Execution: %.4f seconds ---" % (time.time() - start_time))
    # start_time = time.time()
    # output = fc_model(data2_dict,[],None)
    # print("---FC Batch = 250 Input Execution: %.4f seconds ---" % (time.time() - start_time))
    # start_time = time.time()
    # output = fc_model(data3_dict,[],None)
    # print("---FC Batch = 500 Input Execution: %.4f seconds ---" % (time.time() - start_time))

    
    # start_time = time.time()
    # output = cnn_model(data1_dict,[],None)
    # print("---CNN Batch = 1 Input Execution: %.4f seconds ---" % (time.time() - start_time))
    # start_time = time.time()
    # output = cnn_model(data2_dict,[],None)
    # print("---CNN Batch = 250 Input Execution: %.4f seconds ---" % (time.time() - start_time))
    # start_time = time.time()
    # output = cnn_model(data3_dict,[],None)
    # print("---CNN Batch = 500 Input Execution: %.4f seconds ---" % (time.time() - start_time))


    # start_time = time.time()
    # output = gcn_model(data1_dict,[],None)
    # print("---GCN Batch = 1 Input Execution: %.4f seconds ---" % (time.time() - start_time))
    # start_time = time.time()
    # output = gcn_model(data2_dict,[],None)
    # print("---GCN Batch = 250 Input Execution: %.4f seconds ---" % (time.time() - start_time))
    start_time = time.time()
    output = gcn_model(data3_dict,[],None)
    print("---GCN Batch = 500 Input Execution: %.4f seconds ---" % (time.time() - start_time))
    start_time = time.time()
    output = gcn_fast_model(data3_fast_dict,[],None)
    print("---GCN Batch = 500 Input Execution: %.4f seconds ---" % (time.time() - start_time))
