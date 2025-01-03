import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = gcn(seq1, adj, sparse) #(1,2708,256)


        # print("h_1",h_1.shape)

        h_3 = h_1 * self.prompt#(1,2708,256)

        c = self.read(h_1, msk)#(1,256)
        c = self.sigm(c)#(1,256)

        h_2 = gcn(seq2, adj, sparse)

        h_4 = h_2 * self.prompt

        ret = self.disc(c, h_3, h_4
                        , samp_bias1, samp_bias2)#（1,2*node_number）
        print(ret)
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)

    # # Detach the return variables
    # def embed(self, seq, adj, sparse, msk):
    #     h_1 = self.gcn(seq, adj, sparse)
    #     c = self.read(h_1, msk)
    #
    #     return h_1.detach(), c.detach()
    #
class DGIprompt(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGIprompt, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.prompt = nn.Parameter(torch.FloatTensor(1, n_in), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        
        
        seq1 = seq1 * self.prompt
        h_1 = gcn(seq1, adj, sparse)


        # print("h_1",h_1.shape)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        seq2 = seq2 * self.prompt
        h_2 = gcn(seq2, adj, sparse)


        ret = self.disc(c, h_1, h_2
                        , samp_bias1, samp_bias2)

        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)
