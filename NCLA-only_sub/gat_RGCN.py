import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
import copy


class GAT_RGCN(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 n_classes
                 ):
        super(GAT_RGCN, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        self.rgcn = RGCNConv(num_hidden, num_hidden, num_relations=n_classes)
        # 参数非共享
        self.rgcn_layers = nn.ModuleList(
            [RGCNConv(num_hidden, num_hidden, num_relations=n_classes) for _ in range(heads[0])])

        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))

        self.mlp = nn.ModuleList([nn.Linear(heads[0] * num_hidden * 2, 1024),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(1024, 512),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(512, 256),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(256, 128),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(128, n_classes)
                                  ])

        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, False, self.activation))

    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp[i](vectors)
        return vectors

    def forward(self, inputs, data_o, batch):
        heads = []
        h = inputs
        x_o, adj, e_type = data_o.x, data_o.edge_index, data_o.edge_type
        e_type = torch.tensor(e_type, dtype=torch.int64)

        # get hidden_representation
        for l in range(self.num_layers):
            temp = h.flatten(1)
            h = self.gat_layers[l](self.g, temp)
            # (num_node,heads=4,32)
        m = torch.zeros_like(h)
        n = torch.zeros_like(m)

        # get heads
        # 参数非共享
        for i in range(h.shape[1]):
            m[:, i] = self.rgcn_layers[i](x=h[:, i], edge_index=adj, edge_type=e_type)
            # n[:, i] = self.rgcn_layers[i](x=m[:, i],  edge_index=adj, edge_type=e_type)
            heads.append(m[:, i])

        # mlp
        embeds = torch.cat(heads, axis=1)
        drug1_idx = batch[0]
        drug2_idx = batch[1]
        labels = batch[2]
        drug1_list = [int(x) for x in drug1_idx]
        drug2_list = [int(x) for x in drug2_idx]

        drug1_embedding = embeds[drug1_list]
        drug2_embedding = embeds[drug2_list]
        input_features = torch.cat((drug1_embedding, drug2_embedding), dim=1)
        mlp_out = self.MLP(input_features, len(self.mlp))

        return heads, mlp_out


class GAT_RGCN_2(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 n_classes
                 ):
        super(GAT_RGCN_2, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        self.rgcn = RGCNConv(num_hidden, num_hidden, num_relations=n_classes)
        # 参数非共享
        self.rgcn_layers = nn.ModuleList(
            [RGCNConv(num_hidden, num_hidden, num_relations=n_classes) for _ in range(heads[0])])

        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))

        self.mlp = nn.ModuleList([nn.Linear(heads[0] * num_hidden * 2, 1024),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(1024, 512),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(512, 256),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(256, 128),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(128, n_classes)
                                  ])

        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, False, self.activation))

    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp[i](vectors)
        return vectors

    def forward(self, inputs, data_o, batch):
        heads = []
        h = inputs
        x_o, adj, e_type = data_o.x, data_o.edge_index, data_o.edge_type
        e_type = torch.tensor(e_type, dtype=torch.int64)

        # get hidden_representation
        for l in range(self.num_layers):
            temp = h.flatten(1)
            h = self.gat_layers[l](self.g, temp)
            # (num_node,heads,hidden_dim)
        m = torch.zeros_like(h)
        n = torch.zeros_like(m)
        o = torch.zeros_like(m)
        # get heads
        # 参数非共享
        for i in range(h.shape[1]):
            m[:, i] = F.relu(self.rgcn_layers[i](x=h[:, i], edge_index=adj, edge_type=e_type))

            # heads.append(m[:, i])
        for i in range(m.shape[1]):
            n[:, i] = F.relu(self.rgcn_layers[i](m[:, i], edge_index=adj, edge_type=e_type))
            heads.append(n[:, i])

        # mlp
        embeds = torch.cat(heads, axis=1)
        drug1_idx = batch[0]
        drug2_idx = batch[1]
        labels = batch[2]
        drug1_list = [int(x) for x in drug1_idx]
        drug2_list = [int(x) for x in drug2_idx]

        drug1_embedding = embeds[drug1_list]
        drug2_embedding = embeds[drug2_list]
        input_features = torch.cat((drug1_embedding, drug2_embedding), dim=1)
        mlp_out = self.MLP(input_features, len(self.mlp))
        #sub_structure
        return heads, mlp_out,drug1_list,drug2_list


class GAT_RGCN_2_four_feature(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 n_classes
                 ):
        super(GAT_RGCN_2_four_feature, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        self.rgcn = RGCNConv(num_hidden, num_hidden, num_relations=n_classes)
        # 参数非共享
        self.rgcn_layers = nn.ModuleList(
            [RGCNConv(num_hidden, num_hidden, num_relations=n_classes) for _ in range(heads[0])])

        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))

        self.mlp = nn.ModuleList([nn.Linear(heads[0] * num_hidden * 2+256, 2048),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(2048, 1024),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(1024, 512),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(512, 256),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(256, 128),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(128, n_classes)
                                  ])

        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, False, self.activation))

    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp[i](vectors)
        return vectors

    def forward(self, inputs, data_o, batch):
        heads = []
        h = inputs
        x_o, adj, e_type = data_o.x, data_o.edge_index, data_o.edge_type
        e_type = torch.tensor(e_type, dtype=torch.int64)

        # get hidden_representation
        for l in range(self.num_layers):
            temp = h.flatten(1)
            h = self.gat_layers[l](self.g, temp)
            # (num_node,heads=4,32)
        m = torch.zeros_like(h)
        n = torch.zeros_like(m)
        o = torch.zeros_like(m)

        # get heads
        # 参数非共享

        for i in range(h.shape[1]):
            m[:, i] = F.relu(self.rgcn_layers[i](x=h[:, i], edge_index=adj, edge_type=e_type))

            # heads.append(m[:, i])
        for i in range(m.shape[1]):
            n[:, i] = F.relu(self.rgcn_layers[i](m[:, i], edge_index=adj, edge_type=e_type))
            heads.append(n[:, i])

        # mlp
        embeds = torch.cat(heads, axis=1)
        drug1_idx = batch[0]
        drug2_idx = batch[1]

        drug1_list = [int(x) for x in drug1_idx]
        drug2_list = [int(x) for x in drug2_idx]

        drug1_embedding = embeds[drug1_list]
        drug2_embedding = embeds[drug2_list]
        # 输入x 特征
        drug1_o_embedding = x_o[drug1_list]
        drug2_o_embedding = x_o[drug2_list]
        # input_features = torch.cat((drug1_embedding, drug2_embedding), dim=1)
        input_features = torch.cat((drug1_embedding, drug1_o_embedding,drug2_embedding,drug2_o_embedding), dim=1)
        mlp_out = self.MLP(input_features, len(self.mlp))

        return heads, mlp_out,drug1_list,drug2_list
