import torch
import torch.nn as nn
import math


import torch
import torch.nn as nn
import math

# class HeteroDiscriminator(nn.Module):
#     def __init__(self, n_hidden, ntypes):
#         super(HeteroDiscriminator, self).__init__()
#         # 为每种节点类型创建一个判别器的权重矩阵
#         self.weight = nn.ParameterDict({
#             ntype: nn.Parameter(torch.Tensor(n_hidden, n_hidden)) for ntype in ntypes
#         })
#         self.reset_parameters()  # 初始化参数
#
#     def reset_parameters(self):
#         # 使用 Xavier 均匀初始化每个权重矩阵
#         for param in self.weight.values():
#             nn.init.xavier_uniform_(param.data)
#
#     def forward(self, features, readout):
        # 对每种节点类型应用判别器
        # scores = {}
        # for ntype, feat in features.items():
        #     scores[ntype] = torch.matmul(feat, torch.matmul(self.weight[ntype], readout[ntype]))
        # return scores

class HeteroDiscriminator(nn.Module):
    def __init__(self, n_h):
        super(HeteroDiscriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)
        label1 = torch.ones_like(sc_1)
        label2 = torch.zeros_like(sc_2)
        label = torch.cat((label1, label2), 1)
        return (logits,label)

