"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2023/08/23 20:04
  @Email: 2665109868@qq.com
  @function
"""
from ProG.prompt import GNN, LightPrompt
from torch import nn, optim
import torch

# load pre-trained GNN
gnn = GNN(128, hid_dim=128, out_dim=128, gcn_layer_num=2, gnn_type="GCN")
pre_train_path = './pre_trained_gnn/{}.SimGRACE.{}.pth'.format("DDI", "GCN")
gnn.load_state_dict(torch.load(pre_train_path))
print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
for p in gnn.parameters():
    p.requires_grad = False

# prompt with hand-crafted answering template (no answering head tuning)
PG = LightPrompt(token_dim=128, token_num_per_group=100, group_num=6, inner_prune=0.01)

opi = optim.Adam(filter(lambda p: p.requires_grad, PG.parameters()),
                 lr=0.001, weight_decay=0.00001)

lossfn = nn.CrossEntropyLoss(reduction='mean')

