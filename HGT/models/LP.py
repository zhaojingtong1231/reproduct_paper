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


        h_1 = hetero_conv(batch.x_dict, batch.edge_index_dict,batch)

        h1  = {key: F.leaky_relu(h[0])  for key, h in h_1.items()}



        return h1

