"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2023/10/04 21:30
  @Email: 2665109868@qq.com
  @function
"""
"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2023/09/12 12:00
  @Email: 2665109868@qq.com
  @function
"""
# -*- coding: utf-8 -*-
import argparse
import csv
import numpy as np
import time
import torch
import torch.nn.functional as F
import dgl
import scipy.sparse as sp
from gat_RGCN import GAT_RGCN_2
import torch.nn as nn
from loss import multihead_contrastive_loss,sub_structure_constrastive_loss
import warnings
from sklearn.metrics import f1_score, accuracy_score, recall_score, \
    precision_score
from data_preprocess_small import load_small_data
from data_preprocess_big import load_big_data
import os
import random
from datetime import datetime
from torch.optim import lr_scheduler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluation_metrics(y_label_list, y_pred_list):
    acc = accuracy_score(y_label_list, y_pred_list)
    f1 = f1_score(y_label_list, y_pred_list, average='macro')
    recall = recall_score(y_label_list, y_pred_list, average='macro')
    precision = precision_score(y_label_list, y_pred_list, average='macro',zero_division=0.0)

    return acc, f1, recall, precision


def get_label_list(batch):
    labels = batch[2]
    label_list = [int(x) for x in labels]
    return label_list


def prediction(model, data_loader):
    label_list = []
    pred_list = []
    for batch in data_loader:
        batch_label_list = get_label_list(batch)

        _, mlp_out,_,_ = model(features, data_o, batch)
        softmax_output = torch.softmax(mlp_out, dim=1)
        label_list += batch_label_list
        pred_list += [int(x) for x in list(torch.argmax(mlp_out, dim=1))]
    return label_list, pred_list


def test(model, data_o, features, adj, train_loader, val_loader, test_loader):



    test_label_list, test_pred_list = prediction(model, test_loader)
    # compute evaluation
    test_acc, test_f1_score, test_recall, test_precision = evaluation_metrics(y_label_list=test_label_list,
                                                                              y_pred_list=test_pred_list)


    print("test accuaray:{:.4f}; test f1_score:{:.4f}; test recall:{:.4f}; test precision:{:.4f}".format(test_acc,
                                                                                                         test_f1_score,
                                                                                                         test_recall,
                                                                                                         test_precision))



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # config_file = sys.argv[1]

    parser = argparse.ArgumentParser(description='GAT_RGCN')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="loader small_data")
    parser.add_argument("--dataset", type=str, default='small',
                        help="small or big dataset")
    parser.add_argument("--epochs", type=int, default=15,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--loss-rate-mlp", type=float, default=0.8,
                        help=" mlp loss rate ")
    parser.add_argument("--loss-rate-multi", type=float, default=0.5,
                        help="multi loss rate")
    parser.add_argument("--loss-rate-substructure",type=float,default=0.15,
                        help='sub_structure loss rate')
    parser.add_argument("--tau", type=float, default=1,
                        help="temperature-scales")
    parser.add_argument('--warmup-epoch', type=int, default=5,
                        help=" ")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed")
    parser.add_argument("--in-drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0.5,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--zhongzi', type=int, default=0,
                        help=" ")
    parser.add_argument('--save-dir', type=str, default='./result',
                        help="save dir")
    args = parser.parse_args()
    set_random_seed(seed=1)
    # model_config
    num_layers = args.num_layers
    num_heads = args.num_heads
    num_hidden = args.num_hidden
    tau = args.tau
    in_drop = args.in_drop
    attn_drop = args.attn_drop
    negative_slope = args.negative_slope
    # train_config
    lr = args.lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    epochs = args.epochs
    warmup_epoch = args.warmup_epoch
    loss_rate_mlp = args.loss_rate_mlp
    loss_rate_multi = args.loss_rate_multi
    loss_rate_substructure = args.loss_rate_substructure
    gpu = args.gpu
    seed = args.seed
    zhongzi = args.zhongzi
    dataset_size = args.dataset
    # save
    save_dir = args.save_dir

    params = vars(args)

    set_random_seed(1, deterministic=False)
    # data_o原始图数据
    if dataset_size=='small':
        data_o, train_loader, val_loader, test_loader = load_small_data(zhongzi, args.batch_size, workers=32)
    else:
        data_o, train_loader, val_loader, test_loader = load_big_data(zhongzi, args.batch_size, workers=32)
    e_type = data_o.edge_type
    # 假设data包含了图数据，包括边索引
    edge_index = data_o.edge_index
    # 获取节点的数量（可以根据需要调整维度）
    num_nodes = data_o.num_nodes
    # 创建一个零填充的邻接矩阵
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    # 使用边索引来填充邻接矩阵
    adjacency_matrix[edge_index[0], edge_index[1]] = 1  # 假设是无权图，如果是带权图，可以设置相应的权重
    # adjacency_matrix 就是邻接矩阵
    torch.manual_seed(args.seed)

    adj2 = sp.csr_matrix(adjacency_matrix)
    features = sp.csc_matrix(data_o.x)
    Y = data_o.edge_type
    #?
    features[features > 0] = 1
    g = dgl.from_scipy(adj2)
    if args.gpu >= 0 and torch.cuda.is_available():
        cuda = True
        g = g.int().to(args.gpu)
    else:
        cuda = False
    features = torch.FloatTensor(features.todense())

    labels = np.array(Y)
    adj = torch.tensor(adj2.todense())
    all_time = time.time()
    num_feats = features.shape[1]
    if dataset_size=='small':
        n_classes = len(set(Y))
    else:
        n_classes = 86
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    heads = ([num_heads] * num_layers)
    model = GAT_RGCN_2(g,
                        num_layers,
                        num_feats,
                        num_hidden,
                        heads,
                        F.elu,
                        in_drop,
                        attn_drop,
                        negative_slope,
                        n_classes)
    # 3. 加载保存的模型参数
    model_state_dict = torch.load('./model.pth', map_location=torch.device('cuda:0'))

    # 4. 将加载的参数应用到模型
    model.load_state_dict(model_state_dict)
    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    # initialize graph
    dur = []
    loss_classification = nn.CrossEntropyLoss()
    if cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        data_o.cuda()
    test(model, data_o, features, adj, train_loader, val_loader, test_loader)
