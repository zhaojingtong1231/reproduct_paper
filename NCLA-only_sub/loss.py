# -*- coding: utf-8 -*-
import copy

import torch
import torch.nn.functional as F


def drug_sub_loss(view, adj):
    # # 获取矩阵的对角线大小
    # diag_size = min(adj.size(0), adj.size(1))
    # adj1=copy.deepcopy(adj)
    # # 使用for循环将对角线元素加1
    # for i in range(diag_size):
    #     adj1[i, i] += 1
    # adj1[adj1 > 0] = 1
    # # drug_and_neighbors = adj[drug_list]
    # num_neighbors = torch.sum(adj1,dim=1)
    ###可学习参数
    adj = adj + torch.eye(adj.size(0), device=adj.device)  # Adding identity to the adjacency matrix
    num_neighbors = torch.sum(adj, dim=1)
    sub_structure = (adj @ view) / num_neighbors.view(-1, 1).float()# (572,572)*(572,128) #avgreadout

    return sub_structure


def sub_loss(view1, view2, adj,tau):
    view1_sub_loss = drug_sub_loss(view1, adj)#(572,128)
    view2_sub_loss = drug_sub_loss(view2, adj)#(572,128)
    similarity = F.cosine_similarity(view1_sub_loss, view2_sub_loss, dim=1) #(572,1)

    # loss = similarity.sum() / similarity.shape[0]#[1,1]
    loss = similarity.mean()
    # loss = torch.exp(loss)
    return -torch.log(loss)
    # return 1-loss

def sub_contrastive_loss(view1, view2, adj,tau):
    l1 = sub_loss(view1, view2, adj,tau)
    return l1



def sub_structure_constrastive_loss(heads,adj,tau):
    loss = 0
    for i in range(1, len(heads)):
        loss = loss + sub_contrastive_loss(heads[0], heads[i],adj,tau)
    return loss / (len(heads) - 1)

def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, adj,
                     mean: bool = True, tau: float = 1.0, hidden_norm: bool = True):
    l1 = nei_con_loss(z1, z2, tau, adj, hidden_norm)
    l2 = nei_con_loss(z2, z1, tau, adj, hidden_norm)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()
    return ret
def multihead_contrastive_loss(heads, adj, tau: float = 1.0):
    loss = torch.tensor(0, dtype=float, requires_grad=True)
    for i in range(1, len(heads)):
        loss = loss +contrastive_loss(heads[0], heads[i], adj, tau)
    return loss / (len(heads) - 1)


def sim(z1: torch.Tensor, z2: torch.Tensor, hidden_norm: bool = True):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def nei_con_loss(z1: torch.Tensor, z2: torch.Tensor, tau, adj, hidden_norm: bool = True):
    '''neighbor contrastive loss'''
    adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
    adj[adj > 0] = 1
    nei_count = torch.sum(adj, 1) * 2 + 1  # intra-view nei+inter-view nei+self inter-view
    nei_count = torch.squeeze(nei_count.clone().detach().requires_grad_(True))

    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z1, z1, hidden_norm))
    inter_view_sim = f(sim(z1, z2, hidden_norm))

    loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)) / \
           (intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())
    loss = loss / nei_count  # divided by the number of positive pairs for each node

    return -torch.log(loss)
