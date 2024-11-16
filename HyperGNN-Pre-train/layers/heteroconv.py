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

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv

class Heteroconv(nn.Module):
    def __init__(self,  data,hidden_dim):
        super(Heteroconv, self).__init__()

        # 定义卷积层字典
        conv_dict = {}

        # 遍历异构数据的边类型并动态添加卷积层
        for edge_type in data.edge_types:

            # 默认使用 GATConv
            conv_layer = SAGEConv((-1, -1), out_channels=hidden_dim)

            # 将卷积层添加到字典中
            conv_dict[edge_type] = conv_layer

        # 初始化 HeteroConv
        self.heteroconv = HeteroConv(conv_dict, aggr='sum')


    def forward(self, data):
        # 执行前向传播
        out = self.heteroconv(data.x_dict, data.edge_index_dict)
        return out

        # return out.type(torch.float)

