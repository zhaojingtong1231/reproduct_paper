"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/10/27 17:13
  @Email: 2665109868@qq.com
  @function
"""
import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, n_hid, w_rels, G, rel2idx):
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

    def compute_edge_score(self, src_h, dst_h):
        """
        使用点积计算边的得分
        """
        # 点积计算得分
        score = torch.sum(src_h * dst_h, dim=1)
        return score

    def forward(self, h, pretrain_model, batch, edge_type, only_relation=None):
        if pretrain_model:
            src_h = h[edge_type[0]][batch[edge_type]['edge_label_index'][0]]
            dst_h = h[edge_type[2]][batch[edge_type]['edge_label_index'][1]]
        else:
            src_h = h[edge_type[0]][batch[edge_type].edge_index[0]]
            dst_h = h[edge_type[2]][batch[edge_type].edge_index[1]]

        # 计算得分（基于点积）
        score = self.compute_edge_score(src_h, dst_h)

        return score



