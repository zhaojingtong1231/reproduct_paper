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
        
        self.node_types_dd = ['disease', 'drug']
        # 定义一个用于处理每个关系的MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_hid * 2, n_hid),  # 输入维度为 h_u, h_v 和 h_r 连接后的维度
            nn.ReLU(),
            nn.Linear(n_hid, 1)  # 输出得分，维度为 1
        )

        if proto:
            self.W_gate = {}
            for i in self.node_types_dd:
                temp_w = nn.Linear(n_hid * 2, 1)
                nn.init.xavier_uniform_(temp_w.weight)
                self.W_gate[i] = temp_w.to(self.device)
            self.k = proto_num
            self.m = nn.Sigmoid()

            self.diseases_profile = {}
            self.sim_all_etypes = {}
            self.diseaseid2id_etypes = {}
            self.diseases_profile_etypes = {}
            
            disease_etypes_all = ['disease_disease', 'rev_disease_protein', 'disease_phenotype_positive', 'rev_exposure_disease']
            disease_nodes_all = ['disease', 'gene/protein', 'effect/phenotype', 'exposure']
            
            disease_etypes = ['disease_disease', 'rev_disease_protein']
            disease_nodes = ['disease', 'gene/protein']
                        
            
            for etype in self.etypes_dd:
                src, dst = etype[0], etype[2]
                if src == 'disease':
                    all_disease_ids = torch.where(G.out_degrees(etype=etype) != 0)[0]
                elif dst == 'disease':
                    all_disease_ids = torch.where(G.in_degrees(etype=etype) != 0)[0]
                    
                if sim_measure == 'all_nodes_profile':
                    diseases_profile = {i.item(): obtain_disease_profile(G, i, disease_etypes, disease_nodes) for i in all_disease_ids}
                elif sim_measure == 'all_nodes_profile_more':
                    diseases_profile = {i.item(): obtain_disease_profile(G, i, disease_etypes_all, disease_nodes_all) for i in all_disease_ids}
                elif sim_measure == 'protein_profile':
                    diseases_profile = {i.item(): obtain_disease_profile(G, i, ['rev_disease_protein'], ['gene/protein']) for i in all_disease_ids}
                elif sim_measure == 'protein_random_walk':
                    diseases_profile = {i.item(): obtain_protein_random_walk_profile(i, num_walks, path_length, G, disease_etypes, disease_nodes, walk_mode) for i in all_disease_ids}
                    
                diseaseid2id = dict(zip(all_disease_ids.detach().cpu().numpy(), range(len(all_disease_ids))))
                disease_profile_tensor = torch.stack([diseases_profile[i.item()] for i in all_disease_ids])
                sim_all = sim_matrix(disease_profile_tensor, disease_profile_tensor)
                
                self.sim_all_etypes[etype] = sim_all
                self.diseaseid2id_etypes[etype] = diseaseid2id
                self.diseases_profile_etypes[etype] = diseases_profile
                
    def apply_edges(self, edges):
        # 获取源节点 h_u 和目标节点 h_v 的特征
        h_u = edges.src['h']
        h_v = edges.dst['h']

        # 将 h_u, h_v, 和 h_r 拼接起来，传递给 MLP
        combined_input = torch.cat([h_u, h_v], dim=1)  # 拼接源节点特征，目标节点特征和关系嵌入

        # 使用 MLP 计算评分
        score = self.mlp(combined_input)

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
                
                for etype in etypes_train:

                    if self.proto:
                        src, dst = etype[0], etype[2]
                        src_rel_idx = torch.where(graph.out_degrees(etype=etype) != 0)
                        dst_rel_idx = torch.where(graph.in_degrees(etype=etype) != 0)
                        src_h = h[src][src_rel_idx]
                        dst_h = h[dst][dst_rel_idx]

                        src_rel_ids_keys = torch.where(G.out_degrees(etype=etype) != 0)
                        dst_rel_ids_keys = torch.where(G.in_degrees(etype=etype) != 0)
                        src_h_keys = h[src][src_rel_ids_keys]
                        dst_h_keys = h[dst][dst_rel_ids_keys]

                        h_disease = {}

                        if src == 'disease':
                            h_disease['disease_query'] = src_h
                            h_disease['disease_key'] = src_h_keys
                            h_disease['disease_query_id'] = src_rel_idx
                            h_disease['disease_key_id'] = src_rel_ids_keys
                        elif dst == 'disease':
                            h_disease['disease_query'] = dst_h
                            h_disease['disease_key'] = dst_h_keys
                            h_disease['disease_query_id'] = dst_rel_idx
                            h_disease['disease_key_id'] = dst_rel_ids_keys

                        if self.sim_measure in ['protein_profile', 'all_nodes_profile', 'protein_random_walk', 'bert', 'profile+bert', 'all_nodes_profile_more']:

                            try:
                                sim = self.sim_all_etypes[etype][np.array([self.diseaseid2id_etypes[etype][i.item()] for i in h_disease['disease_query_id'][0]])]
                            except:
                                
                                disease_etypes = ['disease_disease', 'rev_disease_protein']
                                disease_nodes = ['disease', 'gene/protein']
                                disease_etypes_all = ['disease_disease', 'rev_disease_protein', 'disease_phenotype_positive', 'rev_exposure_disease']
                                disease_nodes_all = ['disease', 'gene/protein', 'effect/phenotype', 'exposure']
                                ## new disease not seen in the training set
                                for i in h_disease['disease_query_id'][0]:
                                    if i.item() not in self.diseases_profile_etypes[etype]:
                                        if self.sim_measure == 'all_nodes_profile':
                                            self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i, disease_etypes, disease_nodes)
                                        elif self.sim_measure == 'all_nodes_profile_more':
                                            self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i, disease_etypes_all, disease_nodes_all)    
                                        elif self.sim_measure == 'protein_profile':
                                            self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i, ['rev_disease_protein'], ['gene/protein'])
                                        elif self.sim_measure == 'protein_random_walk':
                                            self.diseases_profile_etypes[etype][i.item()] = obtain_protein_random_walk_profile(i, self.num_walks, self.path_length, G, disease_etypes, disease_nodes, self.walk_mode)
                                            
                                profile_query = [self.diseases_profile_etypes[etype][i.item()] for i in h_disease['disease_query_id'][0]]
                                profile_query = torch.cat(profile_query).view(len(profile_query), -1)

                                profile_keys = [self.diseases_profile_etypes[etype][i.item()] for i in h_disease['disease_key_id'][0]]
                                profile_keys = torch.cat(profile_keys).view(len(profile_keys), -1)

                                sim = sim_matrix(profile_query, profile_keys)

                            if src_h.shape[0] == src_h_keys.shape[0]:
                                ## during training...
                                coef = torch.topk(sim, self.k + 1).values[:, 1:]
                                coef = F.normalize(coef, p=1, dim=1)
                                embed = h_disease['disease_key'][torch.topk(sim, self.k + 1).indices[:, 1:]]
                            else:
                                ## during evaluation...
                                coef = torch.topk(sim, self.k).values[:, :]
                                coef = F.normalize(coef, p=1, dim=1)
                                embed = h_disease['disease_key'][torch.topk(sim, self.k).indices[:, :]]
                            out = torch.mul(embed, coef.unsqueeze(dim = 2).to(self.device)).sum(dim = 1)

                        if self.sim_measure in ['protein_profile', 'all_nodes_profile', 'all_nodes_profile_more', 'protein_random_walk', 'bert', 'profile+bert']:
                            # for protein profile, we are only looking at diseases for now...
                            if self.agg_measure == 'learn':
                                coef_all = self.m(self.W_gate['disease'](torch.cat((h_disease['disease_query'], out), dim = 1)))
                                proto_emb = (1 - coef_all)*h_disease['disease_query'] + coef_all*out
                            elif self.agg_measure == 'heuristics-0.8':
                                proto_emb = 0.8*h_disease['disease_query'] + 0.2*out
                            elif self.agg_measure == 'avg':
                                proto_emb = 0.5*h_disease['disease_query'] + 0.5*out
                            elif self.agg_measure == 'rarity':
                                if src == 'disease':
                                    coef_all = exponential(G.out_degrees(etype=etype)[torch.where(graph.out_degrees(etype=etype) != 0)], self.exp_lambda).reshape(-1, 1)
                                elif dst == 'disease':
                                    coef_all = exponential(G.in_degrees(etype=etype)[torch.where(graph.in_degrees(etype=etype) != 0)], self.exp_lambda).reshape(-1, 1)
                                proto_emb = (1 - coef_all)*h_disease['disease_query'] + coef_all*out
                            elif self.agg_measure == '100proto':
                                proto_emb = out
                            h['disease'][h_disease['disease_query_id']] = proto_emb


                        graph.ndata['h'] = h

                    graph.apply_edges(self.apply_edges, etype=etype)    
                    out = graph.edges[etype].data['score']
                    s_l.append(out)
                    scores[etype] = out

                    if self.proto:
                        # recover back to the original embeddings for other relations
                        h[src][src_rel_idx] = src_h
                        h[dst][dst_rel_idx] = dst_h
                
                
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
        h_1 = {key: h * self.prompt[key] for key, h in h.items()}


        scores, out_pos = self.pred(pos_G, G, h_1, pretrain_mode, mode=mode + '_pos', block=blocks[1])
        scores_neg, out_neg = self.pred(neg_G, G, h_1, pretrain_mode, mode=mode + '_neg', block=blocks[1])

        return scores, scores_neg, out_pos, out_neg

    def forward(self,G, neg_G, eval_pos_G = None, return_h = False, return_att = False, mode = 'train', pretrain_mode = False):
        with G.local_scope():
            input_dict = {ntype : G.nodes[ntype].data['inp'] for ntype in G.ntypes}




            h_dict = self.hrgcn.layer1(G, input_dict)
            h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
            h = self.hrgcn.layer2(G, h_dict)
            h = {key: h * self.prompt[key] for key, h in h.items()}

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
            h = feature_prompt(h)
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