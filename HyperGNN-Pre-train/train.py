"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/09/19 20:45
  @Email: 2665109868@qq.com
  @function
"""

import numpy as np
import random
from pretrain import Pretrain
import aug
import os
import argparse
import torch


gpu_id = 1  # 选择你想使用的 GPU ID，例如 0, 1, 2 等
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

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
hidden_dim = 512
sparse = True
useMLP = False

nonlinearity = 'prelu'  # special name to separate parameters

#特征归一化
# data = aug.heterodata_preprocess_features(data)

aug_features1edge = data
aug_features2edge = data

# 边数据增强
# aug1edge_index1 = data
aug1edge_index1 = aug.aug_heterodata_random_edge_edge_index(data, drop_percent=0.2)
aug1edge_index2 = aug.aug_heterodata_random_edge_edge_index(data, drop_percent=0.2)

#节点特征掩码
aug_feature1 = aug.aug_heterodata_random_mask(data,drop_percent=0.2)
aug_feature1 = aug.aug_heterodata_random_mask(data,drop_percent=0.2)

from models import HeteroConvLayers

neg_data,labels = aug.generate_hetero_shuf_features_and_labels(data)


msk  =None
samp_bias1 = None
samp_bias2 = None
hidden_dim = 128
num_layers_num = 1
dropout = 0.1
data = data.to(device)
neg_data = neg_data.to(device)

aug_features1edge = aug_features1edge.to(device)
aug_features2edge = aug_features2edge.to(device)

hetero_conv = HeteroConvLayers( hidden_dim,num_layers_num,dropout)
hetero_conv = hetero_conv.to(device)
model = Pretrain(hidden_dim=hidden_dim,device=device)
model= model.to(device)

optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
for epoch in range(1000):
    optimiser.zero_grad()
    loss = model(hetero_conv,data,neg_data, aug_features1edge, aug_features2edge,
                    data,aug1edge_index1,aug1edge_index2,msk, samp_bias1, samp_bias2,aug_type='edge',labels = labels)
    loss.backward()
    optimiser.step()

    print(loss.item())
# loss = nn.BCEWithLogitsLoss()
#
# def Heter_BCEWithLogitsLoss(logits,labels):
#     total_loss = 0
#     for node_type in logits.keys():
#         # 获取当前节点类型的预测值（logits）和对应的标签
#         logit = logits[node_type]  # shape: [num_nodes, out_features]
#         label = labels[node_type]  # shape: [num_nodes, out_features]
#
#         label = label.to(device)
#         # logits.to(device)
#         # labels.to(device)
#         # 计算当前节点类型的损失
#         node_loss = loss(logit, label)
#         # 累加各节点类型的损失
#         total_loss += node_loss
#     return total_loss
#
#
# hetero_conv = HeteroConvLayers( hidden_dim,num_layers_num,dropout)
# hetero_conv = hetero_conv.to(device)
#
# dgi = DGI_heter(hidden_dim=hidden_dim)
# dgi = dgi.to(device)
# logits_dgi = dgi(hetero_conv,data,neg_data, aug_features1edge, aug_features2edge, msk, samp_bias1, samp_bias2)
#
#
# graphcledge = GraphCL_heter(hidden_dim=hidden_dim)
# graphcledge = graphcledge.to(device)
# logits_graphcl = graphcledge(hetero_conv,data,neg_data, aug_features1edge, aug_features2edge,data,aug1edge_index1,aug1edge_index1,msk, samp_bias1, samp_bias2,aug_type='edge',labels= labels)
#
# Lp_heter =Lp_heter(hidden_dim=hidden_dim)
# Lp_heter = Lp_heter.to(device)
# logits_lp = Lp_heter(hetero_conv,data)
#
# dgiloss = Heter_BCEWithLogitsLoss(logits_dgi,labels)
# graphclloss = Heter_BCEWithLogitsLoss(logits_graphcl,labels)
# print(dgiloss)
# print(graphclloss)

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

