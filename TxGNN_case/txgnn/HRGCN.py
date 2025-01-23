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


    def gm_online(self, edges):
        etype = edges._etype[1]
        srctype = edges._etype[0]
        dsttype = edges._etype[2]

        if srctype == dsttype:
            gate, penalty, gate_score, penalty_not_sum = self.gate[etype][self.layer](
                [edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype]])
        else:
            if etype[:3] == 'rev':
                gate, penalty, gate_score, penalty_not_sum = self.gate[etype][self.layer](
                    [edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype[4:]]])
            else:
                gate, penalty, gate_score, penalty_not_sum = self.gate[etype][self.layer](
                    [edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % 'rev_' + etype]])

        # self.penalty += len(edges.src['Wh_%s' % etype])/self.num_of_edges * penalty
        # self.penalty += penalty
        self.penalty.append(penalty)

        self.num_masked += len(torch.where(gate.reshape(-1) != 1)[0])
        if self.no_base:
            message = gate.unsqueeze(-1) * edges.src['Wh_%s' % etype]
        else:
            message = gate.unsqueeze(-1) * edges.src['Wh_%s' % etype] + (1 - gate.unsqueeze(-1)) * self.baseline[etype][
                self.layer].unsqueeze(0)

        if self.return_gates:
            self.gate_storage[etype] = copy.deepcopy(gate.to('cpu').detach())
            self.gate_penalty_storage[etype] = copy.deepcopy(penalty_not_sum.to('cpu').detach())
            self.gate_score_storage[etype] = copy.deepcopy(gate_score.to('cpu').detach())
        return {'m': message}

    def message_func_no_replace(self, edges):
        etype = edges._etype[1]
        # self.msg_emb[etype] = edges.src['Wh_%s' % etype].to('cpu')
        return {'m': edges.src['Wh_%s' % etype]}

    def graphmask_forward(self, G, feat_dict, graphmask_mode, return_gates, no_base):
        self.no_base = no_base
        self.return_gates = return_gates
        self.penalty = []
        self.num_masked = 0
        self.num_of_edges = G.number_of_edges()

        funcs = {}
        etypes_all = G.canonical_etypes

        for srctype, etype, dsttype in etypes_all:
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh

        for srctype, etype, dsttype in etypes_all:

            if graphmask_mode:
                ## replace the message!
                funcs[etype] = (self.gm_online, fn.mean('m', 'h'))
            else:
                ## normal propagation!
                funcs[etype] = (self.message_func_no_replace, fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')

        if graphmask_mode:
            self.penalty = torch.stack(self.penalty).reshape(-1, )
            # penalty_mean = torch.mean(self.penalty)
            # penalty_relation_reg = torch.sum(torch.log(self.penalty) * self.penalty)
            # penalty = penalty_mean + 0.1 * penalty_relation_reg
            penalty = torch.mean(self.penalty)
        else:
            penalty = 0

        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}, penalty, self.num_masked

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






















