import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator

import torch
import torch.nn as nn



class DGI_heter(nn.Module):
    def __init__(self):
        super(DGI_heter, self).__init__()

        # 创建 HeteroConv，用于异构图


        self.read = AvgReadout()  # 平均读取层

        self.sigm = nn.Sigmoid()  # 激活函数

        self.disc = Discriminator(128)  # 判别器

        # 定义一个可训练的向量 prompt
        self.prompt = nn.Parameter(torch.FloatTensor(1, 128), requires_grad=True)

        # 初始化参数
        # self.reset_parameters()

    def forward(self, hetero_conv, data,seq1, seq2, msk, samp_bias1, samp_bias2):
        # 使用 HeteroConv 进行消息传递
        h_1 = hetero_conv(seq1, seq1)

        # 将 prompt 与节点特征进行元素乘
        h_3 = {key: h * self.prompt for key, h in h_1.items()}
        print(h_3['Drug'].shape)
        #
        # 计算 readout，使用掩码 msk
        c = {key: self.read(h, msk) for key, h in h_1.items()}
        c = {key: self.sigm(c_val) for key, c_val in c.items()}

        # 第二个序列通过 GCN
        h_2 = hetero_conv(seq2, seq2)

        # 第二个序列与 prompt 相乘
        h_4 = {key: h * self.prompt for key, h in h_2.items()}

        # 判别器返回预测结果
        ret = {key: self.disc(c[key], h_3[key], h_4[key], samp_bias1, samp_bias2) for key in c.keys()}

        #
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)  # 初始化 prompt 向量


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = gcn(seq1, adj, sparse)

        # print("h_1",h_1.shape)

        h_3 = h_1 * self.prompt

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = gcn(seq2, adj, sparse)

        h_4 = h_2 * self.prompt

        ret = self.disc(c, h_3, h_4
                        , samp_bias1, samp_bias2)

        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)

    # # Detach the return variables
    # def embed(self, seq, adj, sparse, msk):
    #     h_1 = self.gcn(seq, adj, sparse)
    #     c = self.read(h_1, msk)
    #
    #     return h_1.detach(), c.detach()
    #


class DGIprompt(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGIprompt, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.prompt = nn.Parameter(torch.FloatTensor(1, n_in), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        seq1 = seq1 * self.prompt
        h_1 = gcn(seq1, adj, sparse)

        # print("h_1",h_1.shape)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        seq2 = seq2 * self.prompt
        h_2 = gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2
                        , samp_bias1, samp_bias2)

        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)
