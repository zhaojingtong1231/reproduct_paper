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

def save_evalution_metrics(save_path, row, epoch, train_result=False):
    if train_result == False:

        with open(save_path, 'a', newline='') as file:
            writer = csv.writer(file)
            header = ['epoch','acc', 'f1_score', 'recall', 'precision']
            if epoch == 0:
                writer.writerow(header)

            writer.writerow(row)
    else:
        with open(save_path, 'a', newline='') as file:
            writer = csv.writer(file)
            header = ['loss', 'acc', 'f1_score', 'recall', 'precision']
            if epoch == 0:
                writer.writerow(header)

            writer.writerow(row)

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


def get_time_str():
    # 获取当前日期和时间
    now = datetime.now()
    # 提取月、日、小时和分钟
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    # 将提取的信息格式化为用下划线隔开的字符串
    formatted_str = f"{month:02d}_{day:02d}_{hour:02d}_{minute:02d}"
    return formatted_str


def train(model, data_o, features, adj, train_loader, val_loader, test_loader, tau, save_path):
    train_save_path = os.path.join(save_path, f"train_result.csv")
    val_save_path = os.path.join(save_path, f"val_result.csv")
    test_save_path = os.path.join(save_path, f"test_result.csv")
    # model.cuda()
    # adj.cuda()
    # features.cuda()
    # data_o.cuda()
    best_models = []  # 保存效果最好的模型
    best_accs = []  # 保存最大的acc
    scheduler = lr_scheduler.StepLR(optimizer, step_size=warmup_epoch, gamma=0.1)
    test_acc_history = 0
    for epoch in range(epochs):
        if epoch >= 0:
            t0 = time.time()
        model.train()
        optimizer.zero_grad()
        train_label_list = []
        train_pred_list = []
        loss_train_history = []

        for batch in train_loader:
            optimizer.zero_grad()
            label_list = get_label_list(batch)
            heads, mlp_out,drug1_list,drug2_list = model(features, data_o, batch)


            mlp_loss = loss_classification(mlp_out,
                                           target=torch.from_numpy(np.array(label_list, dtype=np.longlong)).cuda())
                                           # target=torch.from_numpy(np.array(label_list, dtype=np.longlong)))

            # multihead_con_loss = multihead_contrastive_loss(heads, adj, tau=tau)

            # sub_loss = sub_structure_constrastive_loss(heads,adj,tau = tau)
            # sub_loss = 0

            # loss_train = loss_rate_multi * multihead_con_loss + loss_rate_mlp * mlp_loss + sub_loss * loss_rate_substructure
            loss_train = loss_rate_mlp * mlp_loss
            loss_train_history.append(loss_train.cpu().detach().numpy())
            loss_train.backward()
            optimizer.step()
            train_label_list += label_list
            train_pred_list += [int(x) for x in list(torch.argmax(mlp_out, dim=1))]
        # scheduler.step()
        # train_acc, train_f1_score, tain_recall, train_precision = evaluation_metrics(y_label_list=train_label_list, y_pred_list=train_pred_list)
        # val_label_list, val_pred_list = prediction(model, val_loader)
        # val_acc, val_f1_score, val_recall, val_precision = evaluation_metrics(y_label_list=val_label_list,y_pred_list=val_pred_list)

        with torch.no_grad():
            test_label_list, test_pred_list = prediction(model, test_loader)
        # compute evaluation
        test_acc, test_f1_score, test_recall, test_precision = evaluation_metrics(y_label_list=test_label_list,
                                                                                  y_pred_list=test_pred_list)
        # train_eva_data = [loss_train, train_acc, train_f1_score, tain_recall, train_precision]
        # train_eva_data = [tensor.item() for tensor in train_eva_data]
        # val_save_data=[val_acc, val_f1_score, val_recall, val_precision]
        test_save_data = [epoch,test_acc, test_f1_score, test_recall, test_precision]
        if epoch >= 0:
            dur.append(time.time() - t0)
        time1 = time.time() - t0
        print("Epoch {:04d} | Time(s) {:.4f} | TrainLoss {:.4f} ".format(epoch + 1, time1,
                                                                         sum(loss_train_history) / len(
                                                                             loss_train_history)))
        print(
            "test accuaray:{:.4f}; test f1_score:{:.4f}; test recall:{:.4f}; test precision:{:.4f}".format(test_acc,
                                                                                                           test_f1_score,
                                                                                                           test_recall,
                                                                                                           test_precision))

        # save_evalution_metrics(train_save_path, train_eva_data,epoch, train_result=True)
        # save_evalution_metrics(val_save_path, val_save_data, epoch,train_result=False)
        save_evalution_metrics(test_save_path, test_save_data, epoch, train_result=False)
        if test_acc > test_acc_history:


            test_acc_history = test_acc
            checkpoint_path = os.path.join(checkpoints_path, f'model{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            best_models.append(os.path.join(checkpoints_path, f'model{epoch}.pth'))
            best_accs.append(test_acc)
            if len(best_models)>5:
                model_to_delete = best_models.pop(0)
                os.remove(model_to_delete)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # config_file = sys.argv[1]

    parser = argparse.ArgumentParser(description='GAT_RGCN')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="loader small_data")
    parser.add_argument("--dataset", type=str, default='small',
                        help="small or big dataset")
    parser.add_argument("--epochs", type=int, default=1200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--loss-rate-mlp", type=float, default=1,
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
    parser.add_argument("--lr", type=float, default=0.002,
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

    time_str = get_time_str()
    save_path = os.path.join(save_dir,
                             f"heads{num_heads}_num_hidden{num_hidden}_lr{lr}_batch{batch_size}_epochs{epochs}_mlp_loss{loss_rate_mlp}_multi_loss{loss_rate_multi}_sub_loss{loss_rate_substructure}_dataset{dataset_size}_time{time_str}_zhongzi{zhongzi}")
    checkpoints_path = os.path.join(save_path, f"./checkpoints")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    config_save_path = os.path.join(save_path, f"parameters.json")
    params = vars(args)
    # 将参数保存到文本文件中
    with open(config_save_path, 'w') as file:
        for key, value in params.items():
            file.write(f'{key}: {value}\n')
    set_random_seed(1, deterministic=False)
    # data_o原始图数据
    if dataset_size=='small':
        data_o, train_loader, val_loader, test_loader = load_small_data(zhongzi, args.batch_size, workers=4)
    else:
        data_o, train_loader, val_loader, test_loader = load_big_data(zhongzi, args.batch_size, workers=4)
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

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    # initialize graph
    dur = []
    loss_classification = nn.CrossEntropyLoss()
    if cuda:
        model = model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        data_o = data_o.cuda()
    # print(model)
    # print(features)
    # print(adj)
    # print(data_o)
    train(model, data_o, features, adj, train_loader, val_loader, test_loader, tau, save_path=save_path)
