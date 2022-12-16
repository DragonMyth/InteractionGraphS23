import logging
from black import out
import numpy as np
import os

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, SlimConv2d,normc_initializer
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
# from ray.rllib.utils.framework import get_activation_fn
from ray.rllib.utils import try_import_torch
from ray.rllib.models import ModelCatalog
torch, nn = try_import_torch()
from torch_geometric.nn import Sequential as GeoSequential
from torch_geometric.nn import GCNConv,GATConv,EdgeConv,DynamicEdgeConv,radius_graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


import torch.nn.functional as F


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

class GAT(nn.Module):
    ''' 
    A network with conv_layers.
    '''
    def __init__(self, channel_in, channel_out, out_dim, num_nodes_per_batch,edge_dim,gcn_layers,fc_layers):
        super().__init__()
        self._num_nodes_per_batch = num_nodes_per_batch
        gcn_nn_layers = []
        fc_nn_layers = []
        prev_channel_size = channel_in
        for l in gcn_layers:
            assert isinstance(l['channel_out'] , int) or l['channel_out'] =='output'
            channel_out_size = l['channel_out'] if l['channel_out'] != 'output' else channel_out

            layer = GATConv(prev_channel_size,channel_out_size,add_self_loops=False,edge_dim=edge_dim)
            
            gcn_nn_layers.append((layer,"x, edge_index, edge_attr, size,return_attention_weights -> x, attention"))
            prev_channel_size = channel_out_size

        self._gcn_model = GeoSequential('x, edge_index, edge_attr, size,return_attention_weights',gcn_nn_layers)
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
        self._attention_weight= None
    def forward(self, x, edge_index,edge_attr):
        num_batch = x.shape[0]//self._num_nodes_per_batch
        gcn_out,attention = self._gcn_model(x,edge_index,edge_attr,size=None,return_attention_weights=True)
        gcn_out = gcn_out.reshape(num_batch,-1)

        fc_out = self._fc_model(gcn_out)
        self._attention_weight = attention
        return fc_out

    def save_weights(self, file):
        torch.save(self.state_dict(), file)

    def load_weights(self, file):
        self.load_state_dict(torch.load(file))
        self.eval()
    pass
if __name__ == "__main__":
    feat_dim = 6

    nodes = torch.rand(100,6)

    rows = np.random.choice(100,500)
    cols = np.random.choice(100,500)

    edge_index = torch.tensor([rows,cols])
    edge_attr = torch.rand(500,6)

    # print("Nodes: ", nodes)    graphData = Data(data,edge_index=edge_index,edge_attr=edge_attr)

    gcn_layers = [
        {"channel_out":4, "init_weight": {"name": "normc", "std": 1.0}},

        {"channel_out":"output", "init_weight": {"name": "normc", "std": 1.0}}
    ]

    fc_layers = [
        {"hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
        {"hidden_size": "output", "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
    ]
    model = GAT(feat_dim,4,64,100,6,gcn_layers,fc_layers)
    loss = model(nodes,edge_index,edge_attr)
    print(loss)

