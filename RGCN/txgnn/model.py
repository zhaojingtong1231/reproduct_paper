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

import warnings
warnings.filterwarnings("ignore")
from .utils import sim_matrix, exponential, obtain_disease_profile, obtain_protein_random_walk_profile, convert2str
from .graphmask.multiple_inputs_layernorm_linear import MultipleInputsLayernormLinear
from .graphmask.squeezer import Squeezer
from .graphmask.sigmoid_penalty import SoftConcrete

class DistMultPredictor(nn.Module):
    def __init__(self, n_hid, w_rels, G, rel2idx, proto, proto_num, sim_measure, bert_measure, agg_measure, num_walks, walk_mode, path_length, split, data_folder, exp_lambda, device):
        super().__init__()
        
        self.proto = proto
        self.sim_measure = sim_measure
        self.bert_measure = bert_measure
        self.agg_measure = agg_measure
        self.num_walks = num_walks
        self.walk_mode = walk_mode
        self.path_length = path_length
        self.exp_lambda = exp_lambda
        self.device = device
        self.W = w_rels
        self.rel2idx = rel2idx
        self.etypes_dd = [('drug', 'contraindication', 'disease'),
                           ('drug', 'indication', 'disease'),
                           ('drug', 'off-label use', 'disease'),
                           ('disease', 'rev_contraindication', 'drug'),
                           ('disease', 'rev_indication', 'drug'),
                           ('disease', 'rev_off-label use', 'drug')]
        self.etypes_all = [('drug', 'contraindication', 'disease'),
                          ('drug', 'indication', 'disease'),
                          ('drug', 'off-label use', 'disease'),
                          ('disease', 'rev_contraindication', 'drug'),
                          ('disease', 'rev_indication', 'drug'),
                          ('disease', 'rev_off-label use', 'drug'),
                          ('gene/protein', 'protein_protein', 'gene/protein'),
                          ('disease', 'disease_disease', 'disease'),
                          ('drug', 'drug_protein', 'gene/protein'),
                  ('gene/protein', 'disease_protein', 'disease')]
        self.node_types_dd = ['disease', 'drug','gene/protein']


    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        rel_idx = self.rel2idx[edges._etype]
        h_r = self.W[rel_idx]
        score = torch.sum(h_u * h_r * h_v, dim=1)
        return {'score': score}

    def forward(self, graph, G, h, pretrain_mode, mode, block = None, only_relation = None):
        with graph.local_scope():
            scores = {}
            s_l = []

            if len(graph.canonical_etypes) == 1:
                etypes_train = graph.canonical_etypes
            else:
                etypes_train = self.etypes_dd

            if only_relation is not None:
                if only_relation == 'indication':
                    etypes_train = [('drug', 'indication', 'disease'),
                                    ('disease', 'rev_indication', 'drug')]
                elif only_relation == 'contraindication':
                    etypes_train = [('drug', 'contraindication', 'disease'),
                                   ('disease', 'rev_contraindication', 'drug')]
                elif only_relation == 'off-label':
                    etypes_train = [('drug', 'off-label use', 'disease'),
                                   ('disease', 'rev_off-label use', 'drug')]
                else:
                    return ValueError

            graph.ndata['h'] = h

            if pretrain_mode:
                # during pretraining....
                etypes_all = [i for i in graph.canonical_etypes if graph.edges(etype = i)[0].shape[0] != 0]
                for etype in etypes_all:
                    graph.apply_edges(self.apply_edges, etype=etype)
                    out = torch.sigmoid(graph.edges[etype].data['score'])
                    s_l.append(out)
                    scores[etype] = out
            else:
                # finetuning on drug disease only...

                for etype in self.etypes_all:

                    graph.apply_edges(self.apply_edges, etype=etype)
                    out = graph.edges[etype].data['score']
                    s_l.append(out)
                    scores[etype] = out

                
            if pretrain_mode:
                s_l = torch.cat(s_l)             
            else: 
                s_l = torch.cat(s_l).reshape(-1,).detach().cpu().numpy()
            return scores, s_l


from .HRGCN import HRGCN
from .DGI import DGI_heter
from .downprompt import featureprompt
class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size, proto, proto_num, sim_measure, bert_measure, agg_measure, num_walks, walk_mode, path_length, split, data_folder, exp_lambda, device):
        super(HeteroRGCN, self).__init__()


        self.w_rels = nn.Parameter(torch.Tensor(len(G.canonical_etypes), out_size))
        nn.init.xavier_uniform_(self.w_rels, gain=nn.init.calculate_gain('relu'))
        rel2idx = dict(zip(G.canonical_etypes, list(range(len(G.canonical_etypes)))))

        self.pred = DistMultPredictor(n_hid = hidden_size, w_rels = self.w_rels, G = G, rel2idx = rel2idx, proto = proto, proto_num = proto_num, sim_measure = sim_measure, bert_measure = bert_measure, agg_measure = agg_measure, num_walks = num_walks, walk_mode = walk_mode, path_length = path_length, split = split, data_folder = data_folder, exp_lambda = exp_lambda, device = device)
        
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.etypes = G.etypes
        self.device = device
        self.prompt ={}
        for ntype in G.ntypes:
            self.prompt[ntype] = nn.Parameter(torch.FloatTensor(1, hidden_size), requires_grad=True).to(device)
            torch.nn.init.xavier_uniform_(self.prompt[ntype])
        self.loss = nn.BCEWithLogitsLoss()
        self.hrgcn = HRGCN(in_size, hidden_size, out_size, G)
        self.dgi = DGI_heter(hidden_dim=hidden_size, ntypes=G.ntypes)

    def Heter_BCEWithLogitsLoss(self, logits):
        total_loss = 0
        for node_type in logits.keys():
            # 获取当前节点类型的预测值（logits）和对应的标签
            (logit,label)  = logits[node_type]  # shape: [num_nodes, out_features]

            # 计算当前节点类型的损失
            node_loss = self.loss(logit, label)
            # 累加各节点类型的损失
            total_loss  = total_loss + node_loss
        return total_loss

    def forward_minibatch(self, pos_G, neg_G, blocks, G, mode = 'train', pretrain_mode = False):

        h  = self.hrgcn(blocks)
        # h_1 = {key: h + self.prompt[key] for key, h in h.items()}
        h_1 = h
        # logits_label_dgi = self.dgi(self.hrgcn, G, blocks)
        # dgi_loss = self.Heter_BCEWithLogitsLoss(logits_label_dgi)

        scores, out_pos = self.pred(pos_G, G, h_1, pretrain_mode, mode=mode + '_pos', block=blocks[1])
        scores_neg, out_neg = self.pred(neg_G, G, h_1, pretrain_mode, mode=mode + '_neg', block=blocks[1])

        return scores, scores_neg, out_pos, out_neg

    def forward(self,G, neg_G, eval_pos_G = None, return_h = False, return_att = False, mode = 'train', pretrain_mode = False):
        with G.local_scope():
            input_dict = {ntype : G.nodes[ntype].data['inp'] for ntype in G.ntypes}
            h_dict = self.hrgcn.layer1(G, input_dict)
            h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
            h = self.hrgcn.layer2(G, h_dict)
            # h = {key: h + self.prompt[key] for key, h in h.items()}
            if return_h:
                return h
            # full batch
            if eval_pos_G is not None:
                # eval mode
                scores, out_pos = self.pred(eval_pos_G, G, h, pretrain_mode, mode = mode + '_pos')
                scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode = mode + '_neg')
                return scores, scores_neg, out_pos, out_neg
            else:
                scores, out_pos = self.pred(G, G, h, pretrain_mode, mode = mode + '_pos')
                scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode = mode + '_neg')
                return scores, scores_neg, out_pos, out_neg
    def forward_prompt(self, feature_prompt,G, neg_G, eval_pos_G = None, return_h = False, return_att = False, mode = 'train', pretrain_mode = False):
        with G.local_scope():
            input_dict = {ntype : G.nodes[ntype].data['inp'] for ntype in G.ntypes}

            h_dict = self.hrgcn.layer1(G, input_dict)
            h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
            h = self.hrgcn.layer2(G, h_dict)
            # h = {key: h * self.feature_prompt[key] for key, h in h.items()}
            h_prompt = feature_prompt(h)
            h = {key: h + h_prompt[key] for key, h in h.items()}
            if return_h:
                return h

            # full batch
            if eval_pos_G is not None:
                # eval mode
                scores, out_pos = self.pred(eval_pos_G, G, h, pretrain_mode, mode = mode + '_pos')
                scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode = mode + '_neg')
                return scores, scores_neg, out_pos, out_neg
            else:
                scores, out_pos = self.pred(G, G, h, pretrain_mode, mode = mode + '_pos')
                scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode = mode + '_neg')
                return scores, scores_neg, out_pos, out_neg
    
    def graphmask_forward(self, G, pos_graph, neg_graph, graphmask_mode = False, return_gates = False, only_relation = None, no_base = False):
                    
        
        with G.local_scope():
            input_dict = {ntype : G.nodes[ntype].data['inp'] for ntype in G.ntypes}
            h_dict_l1, penalty_l1, num_masked_l1 = self.layer1.graphmask_forward(G, input_dict, graphmask_mode, return_gates, no_base)
            h_dict = {k : F.leaky_relu(h) for k, h in h_dict_l1.items()}
            h, penalty_l2, num_masked_l2 = self.layer2.graphmask_forward(G, h_dict, graphmask_mode, return_gates, no_base)         
            
            scores_pos, out_pos = self.pred(pos_graph, G, h, False, mode = 'train_pos', only_relation = only_relation)
            scores_neg, out_neg = self.pred(neg_graph, G, h, False, mode = 'train_neg', only_relation = only_relation)
            return scores_pos, scores_neg, penalty_l1 + penalty_l2, [num_masked_l1, num_masked_l2]

    
    def enable_layer(self, layer, graphmask = True):
        print("Enabling layer "+str(layer))
        
        for name in self.etypes:
            if graphmask:
                for parameter in self.gates_all[name][layer].parameters():
                    parameter.requires_grad = True
                self.baselines_all[name][layer].requires_grad = True
            else:
                for parameter in self.gates_all[name].parameters():
                    parameter.requires_grad = True
    
    def count_layers(self):
        return 2
    
    def get_gates(self):
        return [self.layer1.gate_storage, self.layer2.gate_storage]
    
    def get_gates_scores(self):
        return [self.layer1.gate_score_storage, self.layer2.gate_score_storage]
    
    def get_gates_penalties(self):
        return [self.layer1.gate_penalty_storage, self.layer2.gate_penalty_storage]
    
    
    def add_graphmask_parameters(self, G, threshold = 0.5, remove_key_parts = False, use_top_k = False, k = 0.05, gate_hidden_size = 32):
        gates_all, baselines_all = {}, {}
        hidden_size = self.hidden_size
        out_size = self.out_size
        print('gate_hidden_size: ', gate_hidden_size)
        for name in G.etypes:
            ## for each relation type

            gates = []
            baselines = []

            vertex_embedding_dims = [hidden_size, out_size]
            message_dims = [hidden_size, out_size]
            h_dims = message_dims

            for v_dim, m_dim, h_dim in zip(vertex_embedding_dims, message_dims, h_dims):
                gate_input_shape = [m_dim, m_dim]
                
                ### different layers have different gates
                gate = torch.nn.Sequential(
                    MultipleInputsLayernormLinear(gate_input_shape, gate_hidden_size),
                    nn.ReLU(),
                    nn.Linear(gate_hidden_size, 1),
                    Squeezer(),
                    SoftConcrete(threshold, remove_key_parts, use_top_k, k)
                )

                gates.append(gate)

                baseline = torch.FloatTensor(m_dim)
                stdv = 1. / math.sqrt(m_dim)
                baseline.uniform_(-stdv, stdv)
                baseline = torch.nn.Parameter(baseline, requires_grad=True)

                baselines.append(baseline)

            gates = torch.nn.ModuleList(gates)
            gates_all[name] = gates

            baselines = torch.nn.ParameterList(baselines)
            baselines_all[name] = baselines

        self.gates_all = nn.ModuleDict(gates_all)
        self.baselines_all = nn.ModuleDict(baselines_all)

        # Initially we cannot update any parameters. They should be enabled layerwise
        for parameter in self.parameters():
            parameter.requires_grad = False
            
        self.layer1.add_graphmask_parameter(self.gates_all, self.baselines_all, 0)
        self.layer2.add_graphmask_parameter(self.gates_all, self.baselines_all, 1)