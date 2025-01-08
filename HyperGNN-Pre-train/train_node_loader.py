"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/09/19 20:45
  @Email: 2665109868@qq.com
  @function
"""
import csv
from utils import process
import numpy as np
import torch.nn.functional as F
import random
from sklearn.metrics import average_precision_score
from pretrain import Pretrain
import aug
from torch_geometric.loader import LinkNeighborLoader, HGTLoader,LinkLoader
from torch_geometric.sampler import NeighborSampler,NegativeSampling
from torch_geometric.utils import degree
import os
from PrimeKG import FullGraphNegSampler
from tqdm import tqdm
import argparse
import torch
from PrimeKG import PreData
from utils import process

gpu_id = 0  # 选择你想使用的 GPU ID，例如 0, 1, 2 等
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

# save_path = '/data/zhaojingtong/pharmrgdata/hetero_graph.pt'
# data = torch.load(save_path)
preData = PreData(data_folder_path='/data/zhaojingtong/PrimeKG/data_all/')
g, df, df_train, df_valid, df_test, disease_eval_idx, no_kg ,g_valid_pos , g_valid_neg= preData.prepare_split(split='random', seed=42, no_kg=False)
g = process.initialize_node_embedding(g,512)

data = g


batch_size = 1024
epochs = 150
patience = 20
lr = 1e-3
l2_coef = 0.0001
drop_prob = 0.0
sparse = True


nonlinearity = 'prelu'  # special name to separate parameters


from models import HeteroConvLayers

neg_data,labels = aug.generate_hetero_shuf_features_and_labels(data)

hidden_dim = 512
num_layers_num = 1
dropout = 0.2
data = data.to(device)
neg_data = neg_data.to(device)




model = Pretrain(data=data,hidden_dim=hidden_dim,batch_size=batch_size,num_layers_num=num_layers_num,dropout=dropout,device=device)
model= model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
print('begin train ......')
edge_types = data.edge_types
g_valid_pos = g_valid_pos.to(device)
g_valid_neg = g_valid_neg.to(device)

dd_etypes = [('drug', 'contraindication', 'disease'),
                  ('drug', 'indication', 'disease'),
                  ('drug', 'drug_drug', 'drug'),
                  ('gene/protein', 'protein_protein', 'gene/protein'),
                  ('disease', 'disease_disease', 'disease'),
                  ('drug', 'drug_protein', 'gene/protein'),
          ('gene/protein', 'disease_protein', 'disease')]
edge_type = dd_etypes[0]

pos_loader = LinkNeighborLoader(
    data,
    num_neighbors=[60],  # 为每个关系采样邻居个数
    batch_size=batch_size,
    edge_label_index=(edge_type, data[edge_type].edge_index)

)
test_loader = LinkNeighborLoader(
    g_valid_pos,
    num_neighbors=[60],  # 为每个关系采样邻居个数
    batch_size=2048,
    edge_label_index=(edge_type, g_valid_pos[edge_type].edge_index)

)
# 开始训练并保存结果
for epoch in range(epochs):
    print('epoch:' + str(epoch))
    epoch_results = [epoch]  # 当前 epoch 的结果

    loss_all =0
    auc_all =0

    print(edge_type)
    scores_list = []
    label_list = []
    all_loss = 0
    all_auc = 0
    test_auc = 0
    for pos_g in pos_loader:
        ng = FullGraphNegSampler(pos_g, k=1, method='fix_dst')
        neg_g = ng(pos_g)
        pred_score_pos, pred_score_neg = model.forword_minibatch(pos_g, neg_g,edge_type, pretrain_model=True)

        scores = torch.cat((pred_score_pos, pred_score_neg),dim=0)
        labels = [1] * len(pred_score_pos) + [0] * len(pred_score_neg)

        loss = F.binary_cross_entropy(torch.sigmoid(scores), torch.Tensor(labels).float().to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        auc = average_precision_score(torch.Tensor(labels).cpu().detach().float(),
                                      torch.sigmoid(scores).cpu().detach().numpy())
        all_loss += loss.item()
        all_auc +=auc

    print(all_auc/len(pos_loader))
    with torch.no_grad():
        for test_g in test_loader:
            ng = FullGraphNegSampler(test_g, k=1, method='fix_dst')
            neg_g = ng(pos_g)
            pred_score_pos, pred_score_neg = model.forword_minibatch(pos_g, neg_g,edge_type, pretrain_model=True)

            scores = torch.cat((pred_score_pos, pred_score_neg),dim=0)
            labels = [1] * len(pred_score_pos) + [0] * len(pred_score_neg)

            auc_t = average_precision_score(torch.Tensor(labels).cpu().detach().float(),
                                          torch.sigmoid(scores).cpu().detach().numpy())
            test_auc +=auc_t
        print(test_auc / len(test_loader))



