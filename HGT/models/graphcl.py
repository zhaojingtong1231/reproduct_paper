import torch
import torch.nn as nn



class GraphCL_heter(nn.Module):
    def __init__(self,hidden_dim):
        super(GraphCL_heter, self).__init__()


        self.sigm = nn.Sigmoid()

        self.prompt = nn.Parameter(torch.FloatTensor(1,hidden_dim), requires_grad=True)

        self.reset_parameters()

    def forward(self, hetero_conv, seq1, seq2, seq3, seq4, data, aug1edge_index1, aug1edge_index2, msk, samp_bias1, samp_bias2,
                aug_type):

        h_0 = hetero_conv(seq1.x_dict, seq1.edge_index_dict, batch=None, edge_type=None, lp=False)

        h_00 = {key:h *self.prompt for key,h in h_0.items()}
        if aug_type == 'edge':

            h_1 = hetero_conv(aug1edge_index1.x_dict, aug1edge_index1.edge_index_dict, batch=None, edge_type=None, lp=False)
            h_3 = hetero_conv(aug1edge_index2.x_dict, aug1edge_index2.edge_index_dict, batch=None, edge_type=None,lp=False)

        elif aug_type == 'mask':

            h_1 = hetero_conv(seq3,LP=False)
            h_3 = hetero_conv(seq4,LP=False)

        elif aug_type == 'node' or aug_type == 'subgraph':

            h_1 = hetero_conv(aug1edge_index1,LP=False)
            h_3 = hetero_conv(aug1edge_index1,LP=False)

        else:
            assert False


        h_11 = {key: h * self.prompt for key, h in h_1.items()}

        h_33 = {key: h * self.prompt for key, h in h_3.items()}


        c_1 = {key: self.read(h, msk) for key, h in h_11.items()}
        c_1 = {key: self.sigm(c_val) for key, c_val in c_1.items()}


        c_3 = {key: self.read(h, msk) for key, h in h_33.items()}
        c_3 = {key: self.sigm(c_val) for key, c_val in c_3.items()}

        h_2 = hetero_conv(seq2,LP=False)

        h_22 = {key: h * self.prompt for key, h in h_2.items()}


        ret1 = {key: self.disc(c_1[key], h_00[key], h_22[key], samp_bias1, samp_bias2) for key in c_1.keys()}

        ret2 = {key: self.disc(c_3[key], h_00[key], h_22[key], samp_bias1, samp_bias2) for key in c_3.keys()}


        ret = {key: ret1[key] + ret2[key] for key in ret1.keys()}

        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)

