import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import copy
from functools import partial
from dgl.nn.pytorch.conv import RelGraphConv
# from basemodel import GraphAdjModel
import numpy as np
import tqdm

class Lp_heter(nn.Module):
    def __init__(self,  hidden_dim):
        super(Lp_heter, self).__init__()
        self.sigm = nn.ELU()
        self.act=torch.nn.LeakyReLU()
        # self.dropout=torch.nn.Dropout(p=config["dropout"])
        self.prompt = nn.Parameter(torch.FloatTensor(1, hidden_dim), requires_grad=True)

        self.reset_parameters()



    def forward(self,hetero_conv,seq):
        h_1 = hetero_conv(seq,LP=False)


        ret = {key: h * self.prompt for key, h in h_1.items()}

        ret = {key: self.sigm(c_val.squeeze(dim=0)) for key, c_val in ret.items()}
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)
class Lp(nn.Module):
    def __init__(self, n_in, n_h):
        super(Lp, self).__init__()
        self.sigm = nn.ELU()
        self.act=torch.nn.LeakyReLU()
        # self.dropout=torch.nn.Dropout(p=config["dropout"])
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)

        self.reset_parameters()



    def forward(self,gcn,seq,adj,sparse):
        h_1 = gcn(seq,adj,sparse,True)
        # ret = h_1
        ret = h_1 * self.prompt
        # ret = h_1 
        # print("ret1",ret)
        ret = self.sigm(ret.squeeze(dim=0))
                # print("ret2",ret)
        # ret = ret.squeeze(dim=0)
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)

class Lpprompt(nn.Module):
    def __init__(self, n_in, n_h):
        super(Lpprompt, self).__init__()
        self.sigm = nn.ELU()
        self.act=torch.nn.LeakyReLU()
        # self.dropout=torch.nn.Dropout(p=config["dropout"])
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_in), requires_grad=True)

        self.reset_parameters()



    def forward(self,gcn,seq,adj,sparse):
        
        seq = seq * self.prompt
        h_1 = gcn(seq,adj,sparse,True)
        ret = h_1
        # ret = h_1 * self.prompt
        # ret = h_1 
        # print("ret1",ret)
        ret = self.sigm(ret.squeeze(dim=0))
                # print("ret2",ret)
        # ret = ret.squeeze(dim=0)
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)
