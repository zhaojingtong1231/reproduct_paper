import torch
import sys
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator, Discriminator2

import pdb

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.disc2 = Discriminator2(n_h)
    '''
        seq1：正例样本1的节点特征。
        seq2：正例样本2的节点特征。
        seq3：负例样本1的节点特征。
        seq4：负例样本2的节点特征。
        adj：原始图的邻接矩阵。
        aug_adj1：用于增强正例样本的邻接矩阵（可能是原始邻接矩阵的变种）。
        aug_adj2：用于增强负例样本的邻接矩阵（可能是原始邻接矩阵的变种）。
        sparse：稀疏矩阵。
        msk：掩码，用于选择特定的节点。
        samp_bias1 和 samp_bias2：采样偏置，用于采样样本。
        aug_type：指定增强类型的字符串，可能是 'edge'、'mask'、'node' 或 'subgraph'
    '''
    def forward(self, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2, aug_type):
        #原图
        h_0 = self.gcn(seq1, adj, sparse)
        if aug_type == 'edge':
            h_1 = self.gcn(seq1, aug_adj1, sparse)
            h_3 = self.gcn(seq1, aug_adj2, sparse)
        elif aug_type == 'mask':
            h_1 = self.gcn(seq3, adj, sparse)
            h_3 = self.gcn(seq4, adj, sparse)
        elif aug_type == 'node' or aug_type == 'subgraph':
            h_1 = self.gcn(seq3, aug_adj1, sparse)
            h_3 = self.gcn(seq4, aug_adj2, sparse)
        else:
            assert False
        c_1 = self.read(h_1, msk)
        c_1= self.sigm(c_1)

        c_3 = self.read(h_3, msk)
        c_3= self.sigm(c_3)

        h_2 = self.gcn(seq2, adj, sparse)

        ret1 = self.disc(c_1, h_0, h_2, samp_bias1, samp_bias2)
        ret2 = self.disc(c_3, h_0, h_2, samp_bias1, samp_bias2)

        ret = ret1 + ret2
        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

