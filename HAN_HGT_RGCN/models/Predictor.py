"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/10/27 17:13
  @Email: 2665109868@qq.com
  @function
"""
import torch.nn as nn
import torch
class Predictor(nn.Module):
    def __init__(self, n_hid, w_rels,G,rel2idx):
        super().__init__()


        self.W = w_rels
        self.rel2idx = rel2idx

        self.etypes_dd = [('drug', 'contraindication', 'disease'),
                          ('drug', 'indication', 'disease'),
                          ('drug', 'off-label use', 'disease'),
                          ('disease', 'rev_contraindication', 'drug'),
                          ('disease', 'rev_indication', 'drug'),
                          ('disease', 'rev_off-label use', 'drug')]

        self.node_types_dd = ['disease', 'drug']

        def compute_edge_score(self, src_h, dst_h, rel_emb):
            # 计算边上的得分，类似于 apply_edges 的功能
            score = torch.sum(src_h * rel_emb * dst_h, dim=1)
            return score
    def compute_edge_score(self,src_h, dst_h, rel_emb):
        # 计算边上的得分，类似于 apply_edges 的功能
        score = torch.sum(src_h * rel_emb * dst_h, dim=1)
        return score
    def forward(self,h,pretrain_model,batch,edge_type,only_relation = None):


        if pretrain_model:
            src_h = h[edge_type[0]][batch[edge_type]['edge_label_index'][0]]
            dst_h = h[edge_type[2]][batch[edge_type]['edge_label_index'][1]]


            rel_idx = self.rel2idx[edge_type]
            rel_emb = self.W[rel_idx] # 关系嵌入也转移到设备

            # 计算得分
            score = self.compute_edge_score(src_h, dst_h, rel_emb)
        else:
            src_h = h[edge_type[0]][batch[edge_type].edge_index[0]]
            dst_h = h[edge_type[2]][batch[edge_type].edge_index[1]]

            rel_idx = self.rel2idx[edge_type]
            rel_emb = self.W[rel_idx]  # 关系嵌入也转移到设备

            # 计算得分
            score = self.compute_edge_score(src_h, dst_h, rel_emb)

        return score

