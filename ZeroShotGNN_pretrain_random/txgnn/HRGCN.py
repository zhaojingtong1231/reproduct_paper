"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/11/04 12:19
  @Email: 2665109868@qq.com
  @function
"""
import dgl
from dgl.ops import edge_softmax
import math
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch.utils import data
import pandas as pd
import copy
import os
import random
import torch.nn as nn
import torch
import torch.nn.functional as F


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })
        self.in_size = in_size
        self.out_size = out_size

        self.gate_storage = {}
        self.gate_score_storage = {}
        self.gate_penalty_storage = {}

    def add_graphmask_parameter(self, gate, baseline, layer):
        self.gate = gate
        self.baseline = baseline
        self.layer = layer

    def forward(self, G, feat_dict):
        funcs = {}
        etypes_all = [i for i in G.canonical_etypes if G.edges(etype=i)[0].shape[0] != 0]

        for srctype, etype, dsttype in etypes_all:
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')

        return {ntype: G.dstdata['h'][ntype] for ntype in list(G.dstdata['h'].keys())}

class HRGCN(nn.Module):
    def __init__(self,in_size, hidden_size,out_size, G):
        super(HRGCN, self).__init__()

        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self,blocks,dgicorrupt=False,lp=True):
        input_dict = blocks[0].srcdata['inp']
        if dgicorrupt:
            input_dict = {ntype: feats[torch.randperm(feats.shape[0])] for ntype, feats in input_dict.items()}
        h_dict = self.layer1(blocks[0], input_dict)
        h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
        h = self.layer2(blocks[1], h_dict)

        if lp:
            return h
        else:
            for ntypes, v in h.items():
                h[ntypes] = h[ntypes].unsqueeze(0)
            return h





















