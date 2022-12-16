import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import to_undirected


class MGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, use_skip=False):
        super(MGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0],hidden_channels[1])
        self.linear1 = nn.Linear (94, 100) # We have 94 local features
        self.linear2 = nn.Linear(110, 81) # 110 is 100 from the linear + 10 from the gcn
        self.linear3 = nn.Linear (81, 2)
        self.use_skip = use_skip
        if self.use_skip:
            self.weight = nn.init.xavier_normal_(Parameter(torch.Tensor(num_node_features, 2)))


    def forward(self, data):
        x_graph = data.x[0]
        x_linear = data.x[1]
        x = self.conv1(x_graph, data.edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)
        
        # Linearly transformed local feature data
        x_local = self.linear1 (x_linear)
        
        # Concatinating
        x_concat = torch.cat ((x, x_local), 1)
        
        # MLP
        x = x_concat.relu()
        x = self.linear2 (x)
        x = x.relu()
        x = self.linear3 (x)
            
        if self.use_skip:
            x = F.softmax(x+torch.matmul(data.x, self.weight), dim=-1)
        else:
#             x = F.softmax(x, dim=-1)
            x = F.log_softmax(x, dim=1)
        return x

    def embed(self, data):
        x = self.conv1(data.x[0], data.edge_index)
        return x