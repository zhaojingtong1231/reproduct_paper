"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/11/04 14:10
  @Email: 2665109868@qq.com
  @function
"""
import torch
import torch.nn as nn
from .layers import  AvgReadout, HeteroDiscriminator

import torch
import torch.nn as nn
class DGI_heter(nn.Module):
    def __init__(self,hidden_dim,ntypes):
        super(DGI_heter, self).__init__()

        # 创建 HeteroConv，用于异构图
        self.ntypes = ntypes

        self.read = AvgReadout()  # 平均读取层

        self.sigm = nn.Sigmoid()  # 激活函数

        self.disc = HeteroDiscriminator(hidden_dim)  # 判别器
        self.prompt = {}
        for ntype in ntypes:
            self.prompt[ntype] = nn.Parameter(torch.FloatTensor(1, hidden_dim), requires_grad=True).cuda()
            torch.nn.init.xavier_uniform_(self.prompt[ntype])


    def forward(self, rgcn, G,blocks):

        # 使用 HeteroConv 进行消息传递
        h_1 = rgcn(blocks,dgicorrupt=False,lp=False)


        # # 将 prompt 与节点特征进行元素乘
        h_3 = {key: h + self.prompt[key] for key, h in h_1.items()}
        # #
        # # 计算 readout，使用掩码 msk
        c = {key: self.read(h, None) for key, h in h_1.items()}
        c = {key: self.sigm(c_val) for key, c_val in c.items()}


        # 第二个序列通过 GCN
        h_2 = rgcn(blocks,dgicorrupt=True,lp=False)

        # # 第二个序列与 prompt 相乘
        h_4 = {key: h+ self.prompt[key] for key, h in h_2.items()}

        ret = {key: self.disc(c[key], h_3[key], h_4[key], None , None) for key in c.keys()}


        return ret


class DGI_heterprompt(nn.Module):
    def __init__(self,hidden_dim,ntypes):
        super(DGI_heterprompt, self).__init__()

        # 创建 HeteroConv，用于异构图
        self.ntypes = ntypes

        self.read = AvgReadout()  # 平均读取层

        self.sigm = nn.Sigmoid()  # 激活函数

        self.disc = HeteroDiscriminator(hidden_dim)  # 判别器
        self.prompt = {}
        for ntype in ntypes:
            self.prompt[ntype] = nn.Parameter(torch.FloatTensor(1, hidden_dim), requires_grad=True).cuda()
            torch.nn.init.xavier_uniform_(self.prompt[ntype])


    def forward(self, rgcn, G,blocks):
        blocks1 = blocks
        blocks2 = blocks
        blocks1[0].srcdata['inp'] = {key: h * self.prompt[key] for key, h in blocks[0].srcdata['inp'].items()}

        # 使用 HeteroConv 进行消息传递
        h_1 = rgcn(blocks1,dgicorrupt=False,lp=False)


        # # 计算 readout，使用掩码 msk
        c = {key: self.read(h, None) for key, h in h_1.items()}
        c = {key: self.sigm(c_val) for key, c_val in c.items()}


        blocks2[0].srcdata['inp'] = {ntype: feats[torch.randperm(feats.shape[0])] for ntype, feats in blocks[0].srcdata['inp'].items()}
        blocks2[0].srcdata['inp'] = {key: h * self.prompt[key] for key, h in blocks2[0].srcdata['inp'].items()}

        # 第二个序列通过 GCN
        h_2 = rgcn(blocks2,dgicorrupt=False,lp=False)


        ret = {key: self.disc(c[key], h_1[key], h_2[key], None , None) for key in c.keys()}


        return ret


