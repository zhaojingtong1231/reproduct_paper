"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/11/12 15:42
  @Email: 2665109868@qq.com
  @function
"""
import torch
import torch.nn as nn

class weighted_prompt(nn.Module):
    def __init__(self, weightednum):
        super(weighted_prompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        self.weight[0][0].data.fill_(0.1)
        self.weight[0][1].data.fill_(1)

    def forward(self, graph_embedding):

        for key in graph_embedding:
            graph_embedding[key] = torch.mm(self.weight, graph_embedding[key])
        return graph_embedding

class featureprompt(nn.Module):
    def __init__(self, G, prompt1, prompt2):
        super(featureprompt, self).__init__()
        self.prompt = {}
        for ntype in G.ntypes:
            self.prompt[ntype] = torch.cat((prompt1[ntype], prompt2[ntype]), 0)
        self.weightprompt = weighted_prompt(2)

    def forward(self, feature):
        weight = self.weightprompt(self.prompt)

        for key in feature:
            feature[key] = weight[key] * feature[key]

        return feature