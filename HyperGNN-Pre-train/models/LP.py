import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import LinkNeighborLoader, HGTLoader
from torch_geometric.sampler import NeighborSampler,NegativeSampling
from torch_geometric.utils import degree
from tqdm import tqdm
from .Predictor import Predictor

class Lp_heter(nn.Module):
    def __init__(self, data,hidden_dim,batch_size,device):
        super(Lp_heter, self).__init__()
        self.w_rels = nn.Parameter(torch.Tensor(len(data.edge_types), hidden_dim))

        rel2idx = dict(zip(data.edge_types, list(range(len(data.edge_types)))))
        self.W = self.w_rels
        self.batch_size = batch_size
        self.rel2idx = dict(zip(data.edge_types, list(range(len(data.edge_types)))))
        self.device = device

        self.pred = Predictor(n_hid=hidden_dim,
                                   w_rels=self.w_rels,G = data, rel2idx = rel2idx)

    def forward(self,hetero_conv,batch,pretrain_model=False):

        # h_1 = self.hetero_conv1(batch.x_dict, batch.edge_index_dict,batch,edge_type)
        # h_1 = hetero_conv(batch.x_dict, batch.edge_index_dict)
        # h_1 = hetero_conv(batch)
        h_1 = hetero_conv(batch.x_dict, batch.edge_index_dict,batch)

        h1  = {key: F.leaky_relu(h[0])  for key, h in h_1.items()}

        # out = self.pred(data,h1,pretrain_model,batch,edge_type)

        return h1


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
