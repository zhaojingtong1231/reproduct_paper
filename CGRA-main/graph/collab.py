import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from core.encoders import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset
# from torch_geometric.small_data import DataLoader
from torch_geometric.loader import DataLoader
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *

from arguments import arg_parse
from torch_geometric.transforms import Constant
import pdb


class GcnInfomax(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(GcnInfomax, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
        # self.global_d = MIFCNet(self.embedding_dim, mi_units)

        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):

        # batch_size = small_data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)

        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        mode = 'fd'
        measure = 'JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)

        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return local_global_loss + PRIOR


class simclr(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, p1, p2, p3,
                 alpha=0.5, beta=1., gamma=.1
                 ):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers,
                               p1, p2, p3)

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):

        # batch_size = small_data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)

        y = self.proj_head(y)

        return y

    def loss_cal(self, x, x_aug, tau):

        # T = 0.2
        T = tau
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss


import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


args = arg_parse()
setup_seed(args.seed)


epochs = 100
log_interval = 10
#batch_size = 128
batch_size = 512
lr = args.lr
DS = args.DS
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'small_data', DS)
# kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

dataset = TUDataset(path, name=DS, aug='none').shuffle()
dataset_eval = TUDataset(path, name=DS, aug='none').shuffle()
print(len(dataset))
print(dataset.get_num_feature())
try:
    dataset_num_features = dataset.get_num_feature()
except:
    dataset_num_features = 1

dataloader = DataLoader(dataset, batch_size=batch_size)
dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


def TT(space):
    accuracies = {'val': [], 'test': []}
    model = simclr(args.hidden_dim, args.num_gc_layers, space['p1'],
                   space['p2'], space['p3']).to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=space['lr'])

    # print('================')
    # print('lr: {}'.format(lr))
    # print('num_features: {}'.format(dataset_num_features))
    # print('hidden_dim: {}'.format(args.hidden_dim))
    # print('num_gc_layers: {}'.format(args.num_gc_layers))
    # print('================')

    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader_eval)
    # print(emb.shape, y.shape)

    """
    acc_val, acc = evaluate_embedding(emb, y)
    accuracies['val'].append(acc_val)
    accuracies['test'].append(acc)
    """

    for epoch in range(1, epochs + 1):
        loss_all = 0
        model.train()
        for data in dataloader:

            # print('start')
            data= data[0]
            optimizer.zero_grad()

            node_num, _ = data.x.size()
            data = data.to(device)
            x = model(data.x, data.edge_index, data.batch, data.num_graphs)

            # if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
            #     # node_num_aug, _ = data_aug.x.size()
            #     edge_idx = data_aug.edge_index.numpy()
            #     _, edge_num = edge_idx.shape
            #     idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]
            #
            #     node_num_aug = len(idx_not_missing)
            #     data_aug.x = data_aug.x[idx_not_missing]
            #
            #     data_aug.batch = small_data.batch[idx_not_missing]
            #     idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
            #     edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
            #                 not edge_idx[0, n] == edge_idx[1, n]]
            #     data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)
            #
            # data_aug = data_aug.to(device)
            #
            # '''
            # print(small_data.edge_index)
            # print(small_data.edge_index.size())
            # print(data_aug.edge_index)
            # print(data_aug.edge_index.size())
            # print(small_data.x.size())
            # print(data_aug.x.size())
            # print(small_data.batch.size())
            # print(data_aug.batch.size())
            # pdb.set_trace()
            # '''
            #
            # x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)
            x2 = model(data.x, data.edge_index, data.batch, data.num_graphs)
            # print(x)
            # print(x_aug)
            loss = model.loss_cal(x, x2, tau=space['tau'])
            # print(loss.item())
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
            # print('batch')
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)
    print(accuracies)
    print(space)
    a = np.array(accuracies['test']).max()
            # print(accuracies['val'][-1], accuracies['test'][-1])
    # return {'loss': -round(a, 4), 'status': STATUS_OK}

    # tpe = ('local' if args.local else '') + ('prior' if args.prior else '')
    # with open('logs/log_' + args.DS + '_' + args.aug, 'a+') as f:
    #     s = json.dumps(accuracies)
    #     f.write('{},{},{},{},{},{},{}\n'.format(args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr, s))
    #     f.write('\n')


# trials = Trials()
# space = {
#      "lr": hp.choice('lr', [1e-5, 5e-5, 8e-5, 1e-4, 5e-4, 8e-4, 1e-3, 5e-3, 8e-3, 1e-2, 5e-2, 8e-2]),
#      "p1": hp.quniform('p1', 0.0, 0.9, 0.1),
#      "p2": hp.quniform('p2', 0.0, 0.9, 0.1),
#      "p3": hp.quniform('p3', 0.0, 0.9, 0.1),
#      "tau": hp.quniform('tau', 0.1, 1.0, 0.1)
# }
# best = fmin(TT, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
# print(best)
# Protein
#TT({'p3': 0.7, 'lr': 8e-4, 'p1': 0.2, 'p2': 0.1, 'tau': 1.0})
#TT({'lr': 8e-4, 'p1': 0.5, 'p2': 0.2, 'p3': 0.60, 'tau': 1.0})
#TT({'lr': 5e-5, 'p1': 0.3, 'p2': 0.0, 'p3': 0.0, 'tau': 0.5})

#NCI1
#TT({'p3': 0.4, 'lr': 1e-5, 'p1': 0.0, 'p2': 0.2, 'tau': 0.4})
#MUTAG
#TT({'lr': 8e-4, 'p1': 0.70, 'p2': 0.60, 'p3': 0.5, 'tau': 0.1})
TT({'lr': 0.08, 'p1': 0.5, 'p2': 0.3, 'p3': 0.4, 'tau': 0.5})