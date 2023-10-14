"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2023/08/20 22:11
  @Email: 2665109868@qq.com
  @function
"""
from ProG.utils import mkdir, load_data4pretrain,myload_data4pretrain
from ProG import PreTrain

mkdir('./pre_trained_gnn/')

pretext = 'SimGRACE'  # 'GraphCL', 'SimGRACE'
gnn_type = 'GCN'  # 'GAT', 'GCN'
# dataname, num_parts, batch_size = 'CiteSeer', 20, 10
dataname, num_parts, batch_size = 'DDI', 20, 10

print("load small_data...")
graph_list, input_dim, hid_dim = myload_data4pretrain(dataname, num_parts)

print("create PreTrain instance...")
pt = PreTrain(pretext, gnn_type, input_dim, hid_dim, gln=2)

print("pre-training...")
pt.train(dataname, graph_list, batch_size=batch_size,
         aug1='dropN', aug2="permE", aug_ratio=None,
         lr=0.001, decay=0.0001, epochs=100)