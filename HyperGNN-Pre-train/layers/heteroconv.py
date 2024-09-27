"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/09/27 16:38
  @Email: 2665109868@qq.com
  @function
"""
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv,  SAGEConv,GATConv

class Heteroconv(nn.Module):
    def __init__(self,hidden_dim):
        super(Heteroconv, self).__init__()
        self.heteroconv = HeteroConv({
            ('Drug', 'Drug-Protein', 'Protein'): GATConv((-1, -1), out_channels=hidden_dim, add_self_loops=False),
            ('Drug', 'DDI', 'Drug'): SAGEConv((-1, -1), out_channels=hidden_dim, add_self_loops=False),
            ('Protein', 'Protein-Pathway', 'Pathway'): GATConv((-1, -1), out_channels=hidden_dim, add_self_loops=False),
            ('Drug', 'Drug-Pathway', 'Pathway'): GATConv((-1, -1), out_channels=hidden_dim, add_self_loops=False),
            ('Protein', 'Protein-Disease', 'Disease'): GATConv((-1, -1), 128, add_self_loops=False),
            ('Protein', 'PPI', 'Protein'): GATConv((-1, -1), out_channels=hidden_dim, add_self_loops=False),
            ('Drug', 'Drug-Disease', 'Disease'): GATConv((-1, -1), out_channels=hidden_dim, add_self_loops=False)
        }, aggr='sum')

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, data):

        out = self.heteroconv(data.x_dict, data.edge_index_dict)


        return out
        # return out.type(torch.float)