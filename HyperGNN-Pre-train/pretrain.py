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
from models import  GraphCL, Lp_heter,DGIprompt,GraphCLprompt,Lpprompt,DGI_heter,GraphCL_heter
from layers import GCN, AvgReadout
import tqdm
import numpy as np
from sklearn.metrics import average_precision_score

from models.heteroconvlayer import HGT,HeteroConvLayers
from models.Predictor import Predictor
class Pretrain(nn.Module):
    def __init__(self,data, hidden_dim,batch_size,num_layers_num,dropout,device):
        super(Pretrain, self).__init__()

        # self.hetero_conv = HGTlayer(data=data,hidden_channels=128, num_heads=2, num_layers=1)
        self.hetero_conv = HeteroConvLayers(data=data, hidden_dim=hidden_dim, num_layers_num=num_layers_num, dropout=dropout)
        # self.hetero_conv = HGT(data=data,hidden_channels=128, num_heads=2, num_layers=1)


        self.dgi = DGI_heter(hidden_dim=hidden_dim)
        self.graphcledge = GraphCL_heter(hidden_dim=hidden_dim)
        self.lp = Lp_heter(data, hidden_dim,batch_size,device)


        self.device = device
        self.loss = nn.BCEWithLogitsLoss()
        self.act = nn.ELU()
        self.w_rels = nn.Parameter(torch.Tensor(len(data.edge_types), hidden_dim))

        rel2idx = dict(zip(data.edge_types, list(range(len(data.edge_types)))))
        self.W = self.w_rels
        self.batch_size = batch_size
        self.rel2idx = dict(zip(data.edge_types, list(range(len(data.edge_types)))))
        self.device = device

        self.pred = Predictor(n_hid=hidden_dim,
                              w_rels=self.w_rels, G=data, rel2idx=rel2idx)

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

    def forward(self, data,neg_data, aug_features1edge, aug_features2edge,
                data1,aug1edge_index1,aug1edge_index2,msk, samp_bias1, samp_bias2,aug_type,labels,edge_type,batch,lp):


        # logits_dgi = self.dgi(self.hetero_conv,data,neg_data, aug_features1edge, aug_features2edge, msk, samp_bias1, samp_bias2)

        # logits_graphcl = self.graphcledge(self.hetero_conv,data,neg_data, aug_features1edge, aug_features2edge,data1,aug1edge_index1,aug1edge_index2,msk, samp_bias1, samp_bias2,aug_type=aug_type)

        # dgi_loss = self.Heter_BCEWithLogitsLoss(logits_dgi,labels)

        # graphcl_loss = self.Heter_BCEWithLogitsLoss(logits_graphcl,labels)
        graphcl_loss = 0



        logits_lp = self.lp(self.hetero_conv, batch, pretrain_model=True)

        pretrain_model = True

        pre_score = self.pred(logits_lp,pretrain_model,batch,edge_type)
        lp_loss = F.binary_cross_entropy(torch.sigmoid(pre_score), batch[edge_type]['edge_label'].to(self.device))

        ret = lp_loss

        # ret = 0.3*dgiloss + 0.5*graphclloss +lploss
        # ret = dgiloss + graphclloss
        # ret = lploss
        return ret,pre_score
    def forword_minibatch(self,pos_g, neg_g,   edge_type,pretrain_model = True):
        logits_pos = self.lp(self.hetero_conv, pos_g, pretrain_model=True)
        logits_neg = self.lp(self.hetero_conv, neg_g, pretrain_model=True)


        pred_score_pos = self.pred(logits_pos,pretrain_model,pos_g,edge_type)

        pred_score_neg = self.pred(logits_neg,pretrain_model,neg_g,edge_type)

        return pred_score_pos, pred_score_neg




    def predict(self,G, g_neg, g_pos,edge_types):
        input_dict = {ntype: G[ntype].x for ntype in G.node_types}
        g_pos.x_dict = input_dict
        g_neg.x_dict = input_dict
        etypes_dd = [('drug', 'contraindication', 'disease'),
                     ('drug', 'indication', 'disease'),
                     ('drug', 'off-label use', 'disease'),
                     ('disease', 'rev_contraindication', 'drug'),
                     ('disease', 'rev_indication', 'drug'),
                     ('disease', 'rev_off-label use', 'drug')]
        pretrain_model=False
        all_auc = {}
        for edge_type in [edge_types]:

            logits_pos = self.lp(self.hetero_conv,  g_pos, pretrain_model=True)
            logits_neg = self.lp(self.hetero_conv,  g_neg, pretrain_model=True)

            pos_score = self.pred(logits_pos, pretrain_model, g_pos, edge_type)
            neg_score = self.pred(logits_neg, pretrain_model, g_neg, edge_type)

            pre_score = torch.cat((pos_score, neg_score), dim=0)
            label =  [1] * len(pos_score) + [0] * len(neg_score)

            auc = average_precision_score(label,torch.sigmoid(pre_score).cpu().detach().numpy())
            all_auc[edge_type]=auc
        return all_auc



