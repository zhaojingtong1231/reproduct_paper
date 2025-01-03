"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/09/27 16:39
  @Email: 2665109868@qq.com
  @function
"""
import torch.nn as nn
import torch
from layers import Heteroconv, AvgReadout
from torch_geometric.nn import HGTConv, Linear
class HeteroConvLayers(nn.Module):
    def __init__(self, data,hidden_dim,num_layers_num,dropout):
        super(HeteroConvLayers, self).__init__()

        self.act=torch.nn.ReLU()
        self.num_layers_num=num_layers_num
        self.data = data
        self.g_net, self.bns = self.create_net(hidden_dim,self.num_layers_num)

        self.dropout=torch.nn.Dropout(p=dropout)

    def create_net(self,hidden_dim,num_layers):

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):

            self.convs.append(Heteroconv(self.data,hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        return self.convs,self.bns


    def forward(self, data,LP=False):

        for i in range(self.num_layers_num):
            # print("i",i)

            graph_output = self.convs[i](data)
            # print("graphout1",graph_output)
            # print("graphout1",graph_output.shape)
            if LP:
                # print("graphout1",graph_output.shape)
                for node_type in data.x_dict:
                    graph_output[node_type] = self.bns[i](graph_output[node_type])

                    graph_output[node_type] = self.dropout(graph_output[node_type])
            # print("graphout2",graph_output)
            # print("graphout2",graph_output.shape)
        for node_type in data.x_dict:
            graph_output[node_type] = graph_output[node_type].unsqueeze(0)

        return graph_output


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
        # self.pred = Predictor(n_hid=hidden_channels,
        #                       w_rels=self.w_rels, G=data, rel2idx=self.rel2idx)


    def forward(self, x_dict, edge_index_dict,batch,edge_type,lp=True):
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