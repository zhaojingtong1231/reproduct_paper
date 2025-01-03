"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/09/19 20:45
  @Email: 2665109868@qq.com
  @function
"""
import csv
import numpy as np
import torch.nn.functional as F
import random
from sklearn.metrics import average_precision_score
from pretrain import Pretrain
import aug
from torch_geometric.loader import LinkNeighborLoader, HGTLoader
from torch_geometric.sampler import NeighborSampler,NegativeSampling
from torch_geometric.utils import degree
import os
from tqdm import tqdm
import argparse
import torch
from PrimeKG import PreData
from utils import process

gpu_id = 1  # 选择你想使用的 GPU ID，例如 0, 1, 2 等
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

# save_path = '/data/zhaojingtong/pharmrgdata/hetero_graph.pt'
# data = torch.load(save_path)
preData = PreData(data_folder_path='/data/zhaojingtong/PrimeKG/data_all/')
g, df, df_train, df_valid, df_test, disease_eval_idx, no_kg ,g_valid_pos , g_valid_neg= preData.prepare_split(split='random', seed=42, no_kg=False)
g = process.initialize_node_embedding(g,128)
data = g


parser = argparse.ArgumentParser("My DGI")

parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--save_name', type=str, default='../modelset/cora.pkl', help='save ckpt name')

args = parser.parse_args()


aug_type = args.aug_type
drop_percent = args.drop_percent
seed = args.seed
random.seed(seed)
np.random.seed(seed)


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

batch_size = 1024
epochs = 20
patience = 20
lr = 1e-3
l2_coef = 0.0001
drop_prob = 0.0
sparse = True
useMLP = False

nonlinearity = 'prelu'  # special name to separate parameters

#特征归一化
# data = aug.heterodata_preprocess_features(data)

aug_features1edge = data
aug_features2edge = data

# 边数据增强
aug1edge_index1 = data
aug1edge_index2 = data
# print('aug edge')
# aug1edge_index1 = aug.aug_heterodata_random_edge_edge_index(data, drop_percent=0.2)
# aug1edge_index2 = aug.aug_heterodata_random_edge_edge_index(data, drop_percent=0.2)

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
dropout = 0.2
data = data.to(device)
neg_data = neg_data.to(device)

aug_features1edge = aug_features1edge.to(device)
aug_features2edge = aug_features2edge.to(device)

# hetero_conv = HeteroConvLayers( data, hidden_dim,num_layers_num,dropout)
# hetero_conv = hetero_conv.to(device)
model = Pretrain(data=data,hidden_dim=hidden_dim,batch_size=batch_size,num_layers_num=num_layers_num,dropout=dropout,device=device)
model= model.to(device)
# from PrimeKG import HGT
# model = HGT(data=data,hidden_channels=128, num_heads=2, num_layers=2)
# model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
print('begin train ......')
edge_types = data.edge_types
g_valid_pos = g_valid_pos.to(device)
g_valid_neg = g_valid_neg.to(device)
# 开始训练并保存结果
for epoch in range(epochs):
    print('epoch:' + str(epoch))
    epoch_results = [epoch]  # 当前 epoch 的结果
    loss_all =0
    auc_all =0

    for edge_type in edge_types:
        scores_list = []
        label_list = []

        all_loss = 0
        all_auc = 0
        neg_sampling_config = NegativeSampling(
            mode='triplet',  # 使用 binary 模式
            dst_weight=torch.pow(degree(data[edge_type].edge_index[1]), 0.75).float().to(device)  # 转移到设备
        )

        loader = LinkNeighborLoader(
            data,
            num_neighbors=[60],  # 为每个关系采样邻居个数
            batch_size=batch_size,
            edge_label_index=(edge_type, data[edge_type].edge_index),
            neg_sampling=neg_sampling_config,
            neg_sampling_ratio=1.0,  # 正负样本比例
            edge_label=torch.ones(data[edge_type].edge_index.size(1), device=device)  # 转移到设备
        )

        for batch in tqdm(loader):

            loss ,pre_score = model(data, neg_data, aug_features1edge, aug_features2edge,
                                data, aug1edge_index1, aug1edge_index2, msk, samp_bias1, samp_bias2,
                                aug_type='edge', labels=labels, edge_type=edge_type, batch=batch,lp=True)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            auc = average_precision_score(batch[edge_type]['edge_label'].cpu().detach().numpy(),
                                          torch.sigmoid(pre_score).cpu().detach().numpy())
            all_loss += loss.item()
            all_auc += auc

            # print(loss.item(),auc)


    #     print(all_loss / len(loader), all_auc / len(loader))
    #     loss_all += all_loss / len(loader)
    #     auc_all += all_auc / len(loader)
    # print(loss_all/len(edge_types),auc_all/len(edge_types))
        auc = model.predict(data, g_valid_pos, g_valid_neg,edge_types)
        print('auc:', auc)




