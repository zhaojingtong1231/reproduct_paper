"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/09/27 16:39
  @Email: 2665109868@qq.com
  @function
"""
import torch.nn as nn
import torch

from torch_geometric.nn import HGTConv, Linear,HANConv



class HGT(torch.nn.Module):
    def __init__(self, data,hidden_channels, num_heads, num_layers):
        super().__init__()

        self.data = data
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads)
            self.convs.append(conv)


    def forward(self, x_dict, edge_index_dict,batch,lp=True):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)


        if lp:
            for node_type in self.data.node_types:
                x_dict[node_type] = x_dict[node_type].unsqueeze(0)
            return x_dict
        else:
            for node_type in self.data.node_types:
                x_dict[node_type] = x_dict[node_type].unsqueeze(0)
            return x_dict

class HAN(torch.nn.Module):
    def __init__(self, data,hidden_channels, num_heads, num_layers):
        super().__init__()

        self.data = data

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HANConv(hidden_channels, hidden_channels, data.metadata(),
                            num_heads)
            self.convs.append(conv)


    def forward(self, x_dict, edge_index_dict,batch,lp=True):


        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)


        if lp:
            for node_type in self.data.node_types:
                x_dict[node_type] = x_dict[node_type].unsqueeze(0)
            return x_dict
        else:
            for node_type in self.data.node_types:
                x_dict[node_type] = x_dict[node_type].unsqueeze(0)
            return x_dict