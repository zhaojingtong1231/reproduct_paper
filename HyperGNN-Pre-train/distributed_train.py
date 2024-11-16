import os
import csv
import numpy as np
import torch.nn.functional as F
import random
from sklearn.metrics import average_precision_score
from pretrain import Pretrain
import aug
from torch_geometric.loader import LinkNeighborLoader, HGTLoader
from torch_geometric.sampler import NeighborSampler, NegativeSampling
from torch_geometric.utils import degree
from tqdm import tqdm
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PrimeKG import PreData
from utils import process

# 初始化GPU的配置
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # 加载数据
    preData = PreData(data_folder_path='/data/zhaojingtong/PrimeKG/data_all/')
    g, df, df_train, df_valid, df_test, disease_eval_idx, no_kg = preData.prepare_split(split='random', seed=42, no_kg=False)
    g = process.initialize_node_embedding(g, 128)
    data = g.to(device)

    # 设置随机种子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # 设置超参数
    batch_size = 1024
    epochs = 20
    lr = 1e-3
    l2_coef = 0.0001

    aug_features1edge = data
    aug_features2edge = data
    neg_data, labels = aug.generate_hetero_shuf_features_and_labels(data)

    aug_features1edge = aug_features1edge.to(device)
    aug_features2edge = aug_features2edge.to(device)
    neg_data = neg_data.to(device)

    # 初始化模型
    hidden_dim = 128
    num_layers_num = 1
    dropout = 0.2

    model = Pretrain(data=data, hidden_dim=hidden_dim, batch_size=batch_size, num_layers_num=num_layers_num, dropout=dropout, device=device)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    edge_types = data.edge_types

    # 开始训练
    print(f'GPU {rank} 开始训练...')
    for epoch in range(epochs):
        print(f'GPU {rank} epoch: {epoch}')
        loss_all = 0
        auc_all = 0

        for edge_type in edge_types:
            optimizer.zero_grad()
            all_loss = 0
            all_auc = 0

            # 设置负采样配置
            neg_sampling_config = NegativeSampling(
                mode='triplet',
                dst_weight=torch.pow(degree(data[edge_type].edge_index[1]), 0.75).float().to(device)
            )

            # 创建数据加载器
            loader = LinkNeighborLoader(
                data,
                num_neighbors=[60],
                batch_size=batch_size,
                edge_label_index=(edge_type, data[edge_type].edge_index),
                neg_sampling=neg_sampling_config,
                neg_sampling_ratio=1.0,
                edge_label=torch.ones(data[edge_type].edge_index.size(1), device=device)
            )

            # 计算无监督损失
            dgi_loss, graphcl_loss = model(
                data, neg_data, aug_features1edge, aug_features2edge,
                data, data, data, None, None, None,
                aug_type='edge', labels=labels, edge_type=edge_type, batch=0, lp=False
            )

            for batch in tqdm(loader):
                lploss, pre_score = model(
                    data, neg_data, aug_features1edge, aug_features2edge,
                    data, data, data, None, None, None,
                    aug_type='edge', labels=labels, edge_type=edge_type, batch=batch, lp=True
                )

                loss = dgi_loss + graphcl_loss + lploss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                auc = average_precision_score(batch[edge_type]['edge_label'].cpu().detach().numpy(),
                                              torch.sigmoid(pre_score).cpu().detach().numpy())
                all_loss += loss.item()
                all_auc += auc

            print(f'GPU {rank} Loss: {all_loss / len(loader)}, AUC: {all_auc / len(loader)}')
            loss_all += all_loss / len(loader)
            auc_all += all_auc / len(loader)

        print(f'GPU {rank} 平均Loss: {loss_all / len(edge_types)}, 平均AUC: {auc_all / len(edge_types)}')

    cleanup()

# 多进程启动
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Distributed DGI")
    parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
    parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_name', type=str, default='../modelset/cora.pkl', help='save ckpt name')

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print("Let's use", world_size, "GPUs!")
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
