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

class HeteroConvLayers(nn.Module):
    def __init__(self, hidden_dim,num_layers_num,dropout):
        super(HeteroConvLayers, self).__init__()

        self.act=torch.nn.ReLU()
        self.num_layers_num=num_layers_num
        self.g_net, self.bns = self.create_net(hidden_dim,self.num_layers_num)

        self.dropout=torch.nn.Dropout(p=dropout)

    def create_net(self,hidden_dim,num_layers):

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):

            self.convs.append(Heteroconv(hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        return self.convs,self.bns


    def forward(self, data,LP):

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