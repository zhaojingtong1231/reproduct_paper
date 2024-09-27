"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/09/19 20:45
  @Email: 2665109868@qq.com
  @function
"""
import torch

import numpy as np
import scipy.sparse as sp
import random
from preprompt import PrePrompt
import preprompt
from utils import process

import torch.nn as nn
import torch.nn.functional as F
import aug
import os
import argparse
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborSampler
import torch


save_path = '/data/zhaojingtong/pharmrgdata/hetero_graph.pt'
data = torch.load(save_path)


parser = argparse.ArgumentParser("My DGI")

parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--save_name', type=str, default='../modelset/cora.pkl', help='save ckpt name')

args = parser.parse_args()

print('-' * 100)
print(args)
print('-' * 100)

aug_type = args.aug_type
drop_percent = args.drop_percent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
seed = args.seed
random.seed(seed)
np.random.seed(seed)


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

batch_size = 1
nb_epochs = 1000
patience = 20
lr = 0.0001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 256
sparse = True
useMLP = False

nonlinearity = 'prelu'  # special name to separate parameters

#特征归一化
# data = aug.heterodata_preprocess_features(data)

aug_features1edge = data
aug_features2edge = data

# 边数据增强
# aug1edge_index1 = aug.aug_heterodata_random_edge_edge_index(data, drop_percent=0.2)
# aug1edge_index2 = aug.aug_heterodata_random_edge_edge_index(data, drop_percent=0.2)

#节点特征掩码
aug_feature1 = aug.aug_heterodata_random_mask(data,drop_percent=0.2)
aug_feature1 = aug.aug_heterodata_random_mask(data,drop_percent=0.2)


from models import DGI_heter, GraphCL, Lp,GcnLayers,DGIprompt,GraphCLprompt,Lpprompt
from models import HeteroConvLayers

msk  =None
samp_bias1 = None
samp_bias2 = None
hidden_dim = 128
num_layers_num = 1
dropout = 0.1
data = data.cuda()
aug_features1edge = aug_features1edge.cuda()
aug_features2edge = aug_features2edge.cuda()


hetero_conv = HeteroConvLayers( hidden_dim,num_layers_num,dropout)
hetero_conv = hetero_conv.cuda()

dgi = DGI_heter()
dgi = dgi.cuda()
logits_dgi = dgi(hetero_conv,data, aug_features1edge, aug_features2edge, msk, samp_bias1, samp_bias2)
# print(logits_dgi['Drug'].shape)


# LP = False
#
#
# lista4 = [0.0001]
#
# best_accs = 0
# list1 = [128]
# list2 = [0.0001]
# for lr in list2:
#     for hid_units in list1:
#         for a4 in lista4:
#             a1 = 0.9
#             a2 = 0.9
#             a3 = 0.1
#             model = PrePrompt(ft_size, hid_units, nonlinearity, negetive_sample, a1, a2, a3, a4, 1, 0.3)
#             optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
#             if torch.cuda.is_available():
#                 print('Using CUDA')
#                 model = model.cuda()
#                 # model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
#                 features = features.cuda()
#                 aug_features1edge = aug_features1edge.cuda()
#                 aug_features2edge = aug_features2edge.cuda()
#                 aug_features1mask = aug_features1mask.cuda()
#                 aug_features2mask = aug_features2mask.cuda()
#                 if sparse:
#                     sp_adj = sp_adj.cuda()
#                     sp_aug_adj1edge = sp_aug_adj1edge.cuda()
#                     sp_aug_adj2edge = sp_aug_adj2edge.cuda()
#                     sp_aug_adj1mask = sp_aug_adj1mask.cuda()
#                     sp_aug_adj2mask = sp_aug_adj2mask.cuda()
#                 else:
#                     adj = adj.cuda()
#                     aug_adj1edge = aug_adj1edge.cuda()
#                     aug_adj2edge = aug_adj2edge.cuda()
#                     aug_adj1mask = aug_adj1mask.cuda()
#                     aug_adj2mask = aug_adj2mask.cuda()
#                 labels = labels.cuda()
#                 idx_train = idx_train.cuda()
#                 idx_val = idx_val.cuda()
#                 idx_test = idx_test.cuda()
#             b_xent = nn.BCEWithLogitsLoss()
#             xent = nn.CrossEntropyLoss()
#             cnt_wait = 0
#             best = 1e9
#             best_t = 0
#             for epoch in range(nb_epochs):
#                 model.train()
#                 optimiser.zero_grad()
#                 idx = np.random.permutation(nb_nodes)
#                 shuf_fts = features[:, idx, :]
#                 lbl_1 = torch.ones(batch_size, nb_nodes)
#                 lbl_2 = torch.zeros(batch_size, nb_nodes)
#                 lbl = torch.cat((lbl_1, lbl_2), 1)
#                 if torch.cuda.is_available():
#                     shuf_fts = shuf_fts.cuda()
#                     lbl = lbl.cuda()
#                 loss = model(features, shuf_fts, aug_features1edge, aug_features2edge, aug_features1mask,
#                              aug_features2mask,
#                              sp_adj if sparse else adj,
#                              sp_aug_adj1edge if sparse else aug_adj1edge,
#                              sp_aug_adj2edge if sparse else aug_adj2edge,
#                              sp_aug_adj1mask if sparse else aug_adj1mask,
#                              sp_aug_adj2mask if sparse else aug_adj2mask,
#                              sparse, None, None, None, lbl=lbl)
#                 print('Loss:[{:.4f}]'.format(loss.item()))
#
#                 loss.backward()
#                 optimiser.step()

