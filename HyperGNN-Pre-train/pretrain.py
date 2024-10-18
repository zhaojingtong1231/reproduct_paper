"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/09/27 19:54
  @Email: 2665109868@qq.com
  @function
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import  GraphCL, Lp,DGIprompt,GraphCLprompt,Lpprompt,DGI_heter,GraphCL_heter
from layers import GCN, AvgReadout
import tqdm
import numpy as np

class Pretrain(nn.Module):
    def __init__(self, hidden_dim,device):
        super(Pretrain, self).__init__()
        self.dgi = DGI_heter(hidden_dim=hidden_dim)

        self.graphcledge = GraphCL_heter(hidden_dim=hidden_dim)




        self.read = AvgReadout()

        self.device = device
        self.loss = nn.BCEWithLogitsLoss()
        self.act = nn.ELU()

    def Heter_BCEWithLogitsLoss(self,logits, labels):
        total_loss = 0
        for node_type in logits.keys():
            # 获取当前节点类型的预测值（logits）和对应的标签
            logit = logits[node_type]  # shape: [num_nodes, out_features]
            label = labels[node_type]  # shape: [num_nodes, out_features]

            label = label.to(self.device)
            # logits.to(device)
            # labels.to(device)
            # 计算当前节点类型的损失
            node_loss = self.loss(logit, label)
            # 累加各节点类型的损失
            total_loss += node_loss
        return total_loss

    def forward(self, hetero_conv,data,neg_data, aug_features1edge, aug_features2edge,
                data1,aug1edge_index1,aug1edge_index2,msk, samp_bias1, samp_bias2,aug_type,labels):

        logits_dgi = self.dgi(hetero_conv,data,neg_data, aug_features1edge, aug_features2edge, msk, samp_bias1, samp_bias2)

        logits_graphcl = self.graphcledge(hetero_conv,data,neg_data, aug_features1edge, aug_features2edge,data1,aug1edge_index1,aug1edge_index2,msk, samp_bias1, samp_bias2,aug_type=aug_type)




        dgiloss = self.Heter_BCEWithLogitsLoss(logits_dgi,labels)
        graphclloss = self.Heter_BCEWithLogitsLoss(logits_graphcl,labels)



        ret = dgiloss + graphclloss



        return ret

    def embed(self, seq, adj, sparse, msk, LP):
        # print("seq",seq.shape)
        # print("adj",adj.shape)
        h_1 = self.gcn(seq, adj, sparse, LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

