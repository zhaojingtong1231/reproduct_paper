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
from models import Lp_heter,DGIprompt,DGI_heter,GraphCL_heter
from layers import GCN, AvgReadout
import tqdm
import numpy as np
from sklearn.metrics import average_precision_score

from models.heteroconvlayer import HGT,HAN
from models.Predictor import Predictor
class Pretrain(nn.Module):
    def __init__(self,data, hidden_dim,batch_size,num_layers_num,dropout,device):
        super(Pretrain, self).__init__()

        # self.hetero_conv = HGTlayer(data=data,hidden_channels=128, num_heads=2, num_layers=1)
        # self.hetero_conv = HeteroConvLayers(data=data, hidden_dim=hidden_dim, num_layers_num=num_layers_num, dropout=dropout)

        # self.hetero_conv = HGT(data=data,hidden_channels=256, num_heads=2, num_layers=2)

        self.hetero_conv = HAN(data=data,hidden_channels=512, num_heads=4, num_layers=2)


        self.lp = Lp_heter(data, hidden_dim,batch_size,device)


        self.device = device

        self.act = nn.ELU()
        self.w_rels = nn.Parameter(torch.Tensor(len(data.edge_types), hidden_dim))

        rel2idx = dict(zip(data.edge_types, list(range(len(data.edge_types)))))
        self.W = self.w_rels
        self.batch_size = batch_size
        self.rel2idx = dict(zip(data.edge_types, list(range(len(data.edge_types)))))
        self.device = device

        self.pred = Predictor(n_hid=hidden_dim,
                              w_rels=self.w_rels, G=data, rel2idx=rel2idx)


    def forward(self, data,neg_data, aug_features1edge, aug_features2edge,
                data1,aug1edge_index1,aug1edge_index2,msk, samp_bias1, samp_bias2,aug_type,labels,edge_type,batch,lp):



        logits_lp = self.lp(self.hetero_conv, batch, pretrain_model=True)

        pretrain_model = True

        pre_score = self.pred(logits_lp,pretrain_model,batch,edge_type)
        lp_loss = F.binary_cross_entropy(torch.sigmoid(pre_score), batch[edge_type]['edge_label'].to(self.device))

        ret = lp_loss

        return ret,pre_score
    def forword_minibatch(self,pos_g, neg_g,   edge_type,pretrain_model = True):

        logits_pos = self.lp(self.hetero_conv, pos_g, pretrain_model=True)

        logits_neg = self.lp(self.hetero_conv, neg_g, pretrain_model=True)


        pred_score_pos = self.pred(logits_pos,pretrain_model,pos_g,edge_type)

        pred_score_neg = self.pred(logits_neg,pretrain_model,neg_g,edge_type)

        return pred_score_pos, pred_score_neg
    def predict(self,pos_g, neg_g,   edge_type,pretrain_model = True):

        logits_pos = self.lp(self.hetero_conv, pos_g, pretrain_model=True)

        logits_neg = self.lp(self.hetero_conv, neg_g, pretrain_model=True)


        pred_score_pos = self.pred.predict(logits_pos,pretrain_model,pos_g,edge_type)

        pred_score_neg = self.pred.predict(logits_neg,pretrain_model,neg_g,edge_type)

        return pred_score_pos, pred_score_neg







