from .model import *
from .utils import *
from .downprompt import featureprompt
from .TxData import TxData
from .TxEval import TxEval

from .graphmask.moving_average import MovingAverage
from .graphmask.lagrangian_optimization import LagrangianOptimization
import sys
import warnings
from dgl.nn.pytorch import SAGEConv, HeteroGraphConv, GCN2Conv, GraphConv, GATConv, GATv2Conv

warnings.filterwarnings("ignore")
from dgl import apply_each

torch.manual_seed(0)


# device = torch.device("cuda:0")
class SAGE(nn.Module):
    def __init__(self, in_size, hidden_size, rel_names):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroGraphConv({
            rel: SAGEConv(in_size, hidden_size, "pool")
            for rel in rel_names
        }))
        self.layers.append(HeteroGraphConv({
            rel: SAGEConv(hidden_size, hidden_size, "pool")
            for rel in rel_names
        }))
        self.hidden_size = hidden_size
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, blocks, x):

        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):

            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = {k: F.leaky_relu(h) for k, h in hidden_x.items()}
        return hidden_x

    def forward_all(self, G, x):

        hidden_x = x

        for layer_idx, (layer, block) in enumerate(zip(self.layers, [G, G])):
            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = {k: F.leaky_relu(h) for k, h in hidden_x.items()}
        return hidden_x

class RGCN(nn.Module):
    def __init__(self, in_size, hidden_size, rel_names):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroGraphConv({
            rel: GraphConv(in_size, hidden_size)
            for rel in rel_names
        }))
        self.layers.append(HeteroGraphConv({
            rel: GraphConv(hidden_size, hidden_size)
            for rel in rel_names
        }))
        self.hidden_size = hidden_size

    def forward(self, blocks, x):

        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):

            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = {k: F.leaky_relu(h) for k, h in hidden_x.items()}
        return hidden_x

    def forward_all(self, G, x):

        hidden_x = x

        for layer_idx, (layer, block) in enumerate(zip(self.layers, [G, G])):
            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = {k: F.leaky_relu(h) for k, h in hidden_x.items()}
        return hidden_x

class RGAT(nn.Module):
    def __init__(self, in_size, hidden_size, rel_names, n_heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroGraphConv({
            rel: GATConv(in_size, hidden_size // n_heads, n_heads)
            for rel in rel_names
        }))
        self.layers.append(HeteroGraphConv({
            rel: GATConv(hidden_size, hidden_size // n_heads, n_heads)
            for rel in rel_names
        }))

    def forward(self, blocks, x):

        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):

            hidden_x = layer(block, hidden_x)
            hidden_x = apply_each(hidden_x, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))

            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = {k: F.leaky_relu(h) for k, h in hidden_x.items()}
        return hidden_x

    def forward_all(self, G, x):

        hidden_x = x

        for layer_idx, (layer, block) in enumerate(zip(self.layers, [G, G])):
            hidden_x = layer(block, hidden_x)
            hidden_x = apply_each(hidden_x, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = {k: F.leaky_relu(h) for k, h in hidden_x.items()}
        return hidden_x

class DistMultPredictor(nn.Module):
    def __init__(self, n_hid, w_rels, G, rel2idx, proto, proto_num, sim_measure, bert_measure, agg_measure, num_walks,
                 exp_lambda, device):
        super().__init__()

        self.proto = proto
        self.sim_measure = sim_measure
        self.bert_measure = bert_measure
        self.agg_measure = agg_measure
        self.num_walks = num_walks
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

            disease_etypes_all = ['disease_disease', 'rev_disease_protein', 'disease_phenotype_positive',
                                  'rev_exposure_disease']
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
                    diseases_profile = {i.item(): obtain_disease_profile(G, i, disease_etypes, disease_nodes) for i in
                                        all_disease_ids}
                elif sim_measure == 'all_nodes_profile_more':
                    diseases_profile = {i.item(): obtain_disease_profile(G, i, disease_etypes_all, disease_nodes_all)
                                        for i in all_disease_ids}

                diseaseid2id = dict(zip(all_disease_ids.detach().cpu().numpy(), range(len(all_disease_ids))))
                disease_profile_tensor = torch.stack([diseases_profile[i.item()] for i in all_disease_ids])
                sim_all = sim_matrix(disease_profile_tensor, disease_profile_tensor)

                self.sim_all_etypes[etype] = sim_all
                self.diseaseid2id_etypes[etype] = diseaseid2id
                self.diseases_profile_etypes[etype] = diseases_profile

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        rel_idx = self.rel2idx[edges._etype]
        h_r = self.W[rel_idx]
        score = torch.sum(h_u * h_r * h_v, dim=1)
        return {'score': score}

    def forward(self, graph, G, h, pretrain_mode, mode, block=None, only_relation=None):
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
                etypes_all = [i for i in graph.canonical_etypes if graph.edges(etype=i)[0].shape[0] != 0]
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

                        if self.sim_measure in ['protein_profile', 'all_nodes_profile', 'protein_random_walk', 'bert',
                                                'profile+bert', 'all_nodes_profile_more']:

                            try:
                                sim = self.sim_all_etypes[etype][np.array(
                                    [self.diseaseid2id_etypes[etype][i.item()] for i in
                                     h_disease['disease_query_id'][0]])]
                            except:

                                disease_etypes = ['disease_disease', 'rev_disease_protein']
                                disease_nodes = ['disease', 'gene/protein']
                                disease_etypes_all = ['disease_disease', 'rev_disease_protein',
                                                      'disease_phenotype_positive', 'rev_exposure_disease']
                                disease_nodes_all = ['disease', 'gene/protein', 'effect/phenotype', 'exposure']
                                ## new disease not seen in the training set
                                for i in h_disease['disease_query_id'][0]:
                                    if i.item() not in self.diseases_profile_etypes[etype]:
                                        if self.sim_measure == 'all_nodes_profile':
                                            self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i,
                                                                                                                   disease_etypes,
                                                                                                                   disease_nodes)
                                        elif self.sim_measure == 'all_nodes_profile_more':
                                            self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i,
                                                                                                                   disease_etypes_all,
                                                                                                                   disease_nodes_all)
                                profile_query = [self.diseases_profile_etypes[etype][i.item()] for i in
                                                 h_disease['disease_query_id'][0]]
                                profile_query = torch.cat(profile_query).view(len(profile_query), -1)

                                profile_keys = [self.diseases_profile_etypes[etype][i.item()] for i in
                                                h_disease['disease_key_id'][0]]
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
                            out = torch.mul(embed, coef.unsqueeze(dim=2).to(self.device)).sum(dim=1)

                        if self.sim_measure in ['protein_profile', 'all_nodes_profile', 'all_nodes_profile_more',
                                                'protein_random_walk', 'bert', 'profile+bert']:
                            # for protein profile, we are only looking at diseases for now...
                            if self.agg_measure == 'learn':
                                coef_all = self.m(
                                    self.W_gate['disease'](torch.cat((h_disease['disease_query'], out), dim=1)))
                                proto_emb = (1 - coef_all) * h_disease['disease_query'] + coef_all * out
                            elif self.agg_measure == 'heuristics-0.8':
                                proto_emb = 0.8 * h_disease['disease_query'] + 0.2 * out
                            elif self.agg_measure == 'avg':
                                proto_emb = 0.5 * h_disease['disease_query'] + 0.5 * out
                            elif self.agg_measure == 'rarity':
                                if src == 'disease':
                                    coef_all = exponential(
                                        G.out_degrees(etype=etype)[torch.where(graph.out_degrees(etype=etype) != 0)],
                                        self.exp_lambda).reshape(-1, 1)
                                elif dst == 'disease':
                                    coef_all = exponential(
                                        G.in_degrees(etype=etype)[torch.where(graph.in_degrees(etype=etype) != 0)],
                                        self.exp_lambda).reshape(-1, 1)
                                proto_emb = (1 - coef_all) * h_disease['disease_query'] + coef_all * out
                            elif self.agg_measure == '100proto':
                                proto_emb = out
                            h['disease'][h_disease['disease_query_id']] = proto_emb
                        else:
                            if self.agg_measure == 'learn':
                                coef_src = self.m(self.W_gate[src](torch.cat((src_h, sim_emb_src), dim=1)))
                                coef_dst = self.m(self.W_gate[dst](torch.cat((dst_h, sim_emb_dst), dim=1)))
                            elif self.agg_measure == 'rarity':
                                # give high weights to proto embeddings for nodes that have low degrees
                                coef_src = exponential(
                                    G.out_degrees(etype=etype)[torch.where(graph.out_degrees(etype=etype) != 0)],
                                    self.exp_lambda).reshape(-1, 1)
                                coef_dst = exponential(
                                    G.in_degrees(etype=etype)[torch.where(graph.in_degrees(etype=etype) != 0)],
                                    self.exp_lambda).reshape(-1, 1)
                            elif self.agg_measure == 'heuristics-0.8':
                                coef_src = 0.2
                                coef_dst = 0.2
                            elif self.agg_measure == 'avg':
                                coef_src = 0.5
                                coef_dst = 0.5
                            elif self.agg_measure == '100proto':
                                coef_src = 1
                                coef_dst = 1

                            proto_emb_src = (1 - coef_src) * src_h + coef_src * sim_emb_src
                            proto_emb_dst = (1 - coef_dst) * dst_h + coef_dst * sim_emb_dst

                            h[src][src_rel_idx] = proto_emb_src
                            h[dst][dst_rel_idx] = proto_emb_dst

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
                s_l = torch.cat(s_l).reshape(-1, ).detach().cpu().numpy()
            return scores, s_l


class TxGNNPrompt:
    def __init__(self, data,
                 weight_bias_track=False,
                 proj_name='TxGNNPrompt',
                 exp_name='TxGNNPrompt',
                 device='cuda:0'):
        self.device = torch.device(device)
        self.weight_bias_track = weight_bias_track
        self.G = data.G
        self.df, self.df_train, self.df_valid, self.df_test = data.df, data.df_train, data.df_valid, data.df_test
        self.data_folder = data.data_folder
        self.disease_eval_idx = data.disease_eval_idx
        self.split = data.split
        self.no_kg = data.no_kg

        self.disease_rel_types = ['rev_contraindication', 'rev_indication', 'rev_off-label use']

        self.dd_etypes = [('drug', 'contraindication', 'disease'),
                          ('drug', 'indication', 'disease'),
                          ('drug', 'off-label use', 'disease'),
                          ('disease', 'rev_contraindication', 'drug'),
                          ('disease', 'rev_indication', 'drug'),
                          ('disease', 'rev_off-label use', 'drug')]

        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)
            self.wandb = wandb
        else:
            self.wandb = None
        self.config = None

    def model_initialize(self, n_hid=128,
                         n_inp=128,
                         n_out=128,
                         proto=True,
                         proto_num=5,
                         attention=False,
                         sim_measure='all_nodes_profile',
                         bert_measure='disease_name',
                         agg_measure='rarity',
                         exp_lambda=0.7,
                         num_walks=200,
                         walk_mode='bit',
                         path_length=2,
                         model='rgcn'):

        if self.no_kg and proto:
            print('Ablation study on No-KG. No proto learning is used...')
            proto = False

        self.G = self.G.to('cpu')
        self.G = initialize_node_embedding(self.G, n_inp)

        # self.g_valid_pos, self.g_valid_neg = evaluate_graph_construct(self.df_valid, self.G, 'fix_dst', 1, self.device)
        self.g_test_pos, self.g_test_neg = evaluate_graph_construct(self.df_test, self.G, 'fix_dst', 1, self.device)

        self.config = {'n_hid': n_hid,
                       'n_inp': n_inp,
                       'n_out': n_out,
                       'proto': proto,
                       'proto_num': proto_num,
                       'attention': attention,
                       'sim_measure': sim_measure,
                       'bert_measure': bert_measure,
                       'agg_measure': agg_measure,
                       'num_walks': num_walks,
                       'walk_mode': walk_mode,
                       'path_length': path_length
                       }
        self.w_rels = nn.Parameter(torch.Tensor(len(self.G.canonical_etypes), n_hid))
        nn.init.xavier_uniform_(self.w_rels, gain=nn.init.calculate_gain('relu'))
        rel2idx = dict(zip(self.G.canonical_etypes, list(range(len(self.G.canonical_etypes)))))

        if model == 'rgcn':
            self.model = RGCN(n_hid, hidden_size=n_hid, rel_names=self.G.canonical_etypes)
        elif model == 'rsage':
            self.model = SAGE(n_hid,hidden_size=n_hid,rel_names=self.G.canonical_etypes)
        elif model == 'rgat':
            self.model = RGAT(n_hid, hidden_size=n_hid, rel_names=self.G.canonical_etypes, n_heads=4)


        self.pred = DistMultPredictor(n_hid=n_hid, w_rels=self.w_rels, G=self.G, rel2idx=rel2idx, proto=proto,
                                      proto_num=proto_num, sim_measure=sim_measure, bert_measure=bert_measure,
                                      agg_measure=agg_measure, num_walks=num_walks,
                                      exp_lambda=exp_lambda, device=self.device)

    def pretrain(self, n_epoch=1, learning_rate=1e-3, batch_size=1024, train_print_per_n=20, save_model_path='./',
                 sweep_wandb=None):

        if self.no_kg:
            raise ValueError('During No-KG ablation, pretraining is infeasible because it is the same as finetuning...')
        print('Creating minibatch pretraining dataloader...')
        train_eid_dict = {etype: self.G.edges(form='eid', etype=etype) for etype in self.G.canonical_etypes}

        rel_unique = self.df.relation.unique()
        reverse_etypes = {}
        for rel in rel_unique:
            if 'rev_' in rel:
                reverse_etypes[rel] = rel[4:]
            elif 'rev_' + rel in rel_unique:
                reverse_etypes[rel] = 'rev_' + rel
            else:
                reverse_etypes[rel] = rel

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

        dataloader = dgl.dataloading.DistEdgeDataLoader(
            self.G, train_eid_dict, sampler,
            negative_sampler=Minibatch_NegSampler(self.G, 1, 'fix_dst'),
            shuffle=True,
            batch_size=batch_size,
            drop_last=False,
            num_workers=0)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        print('Start pre-training with #param: %d' % (get_n_params(self.model)))
        self.G = self.G.to(self.device)
        best_auprc = -1
        best_models = []
        best_auprc_list = []
        self.model = self.model.to(self.device)
        self.pred = self.pred.to(self.device)
        for epoch in range(n_epoch):

            for step, (nodes, pos_g, neg_g, blocks) in enumerate(dataloader):

                blocks = [i.to(self.device) for i in blocks]
                pos_g = pos_g.to(self.device)
                neg_g = neg_g.to(self.device)
                ret = self.model.forward(blocks, blocks[0].srcdata['inp'])
                scores, out_pos = self.pred(pos_g, self.G, ret, True, mode='_pos', block=blocks[1])
                scores_neg, out_neg = self.pred(neg_g, self.G, ret, True, mode='_neg', block=blocks[1])

                scores1 = torch.cat((out_pos, out_neg)).reshape(-1, )
                labels = [1] * len(out_pos) + [0] * len(out_neg)

                loss = F.binary_cross_entropy(scores1, torch.Tensor(labels).float().to(self.device))
                #
                # all_loss = 0.1*dgi_loss+loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step) % train_print_per_n == 0:
                    with open(save_model_path+"/output.txt", "a") as f:
                        # Redirect stdout to the file
                        sys.stdout = f
                        print(epoch, step, loss.item())

                        with torch.no_grad():
                            with self.G.local_scope():
                                input_dict = {ntype: self.G.nodes[ntype].data['inp'] for ntype in self.G.ntypes}

                                ret1 = self.model.forward_all(self.G, input_dict)

                                scores2, out_pos2 = self.pred(self.g_test_pos, self.G, ret1, False, mode='_pos')
                                scores_neg2, out_neg2 = self.pred(self.g_test_neg, self.G, ret1, False, mode='_neg')

                                dd_etypes = [('drug', 'contraindication', 'disease'),
                                             ('drug', 'indication', 'disease'),
                                             ('drug', 'off-label use', 'disease'),
                                             ('disease', 'rev_contraindication', 'drug'),
                                             ('disease', 'rev_indication', 'drug'),
                                             ('disease', 'rev_off-label use', 'drug')]

                                pos_score = torch.cat([scores2[i] for i in dd_etypes])
                                neg_score = torch.cat([scores_neg2[i] for i in dd_etypes])

                                scores = torch.sigmoid(torch.cat((pos_score, neg_score)).reshape(-1, ))
                                labels2 = [1] * len(pos_score) + [0] * len(neg_score)

                                loss2 = F.binary_cross_entropy(scores, torch.Tensor(labels2).float().to(self.device))

                                (auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc,
                                 macro_auprc), loss, pred_pos, pred_neg = get_all_metrics_fb(scores2, scores_neg2,
                                                                                             scores.reshape(
                                                                                                 -1, ).detach().cpu().numpy(),
                                                                                             labels2,
                                                                                             self.G,
                                                                                             True), loss2.item(), scores2, scores_neg2
                                print(
                                    'Testing Loss %.4f Testing Micro AUROC %.4f Testing Micro AUPRC %.4f Testing Macro AUROC %.4f Testing Macro AUPRC %.4f' % (
                                        loss,
                                        micro_auroc,
                                        micro_auprc,
                                        macro_auroc,
                                        macro_auprc
                                    ))
                                print(auroc_rel)
                                print(auprc_rel)
                    sys.stdout = sys.__stdout__
                    if macro_auprc > best_auprc:
                        best_auprc = macro_auprc
                        checkpoint_path = os.path.join(save_model_path, f'model{epoch}_{step}.pth')
                        torch.save(self.model.state_dict(), checkpoint_path)
                        best_models.append(os.path.join(save_model_path, f'model{epoch}_{step}.pth'))
                        best_auprc_list.append(macro_auprc)
                        if len(best_models) > 5:
                            model_to_delete = best_models.pop(0)
                            os.remove(model_to_delete)

    def finetune(self, n_epoch=500,
                 learning_rate=1e-3,
                 train_print_per_n=5,
                 valid_per_n=25,
                 model_path='./',
                 sweep_wandb=None,
                 save_name=None):

        best_val_acc = 0
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.G = self.G.to(self.device)
        self.pred = self.pred.to(self.device)
        neg_sampler = Full_Graph_NegSampler(self.G, 1, 'fix_dst', self.device)
        # torch.nn.init.xavier_uniform(self.w_rels) # reinitialize decoder

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.8)
        negative_graph = neg_sampler(self.G)
        for epoch in range(n_epoch):


            # negative_graph = negative_graph.to(self.device)
            input_dict = {ntype: self.G.nodes[ntype].data['inp'] for ntype in self.G.ntypes}

            ret1 = self.model.forward_all(self.G, input_dict)

            scores, out_pos = self.pred(self.G, self.G, ret1, False, mode='_pos')
            scores_neg, out_neg = self.pred(negative_graph, self.G, ret1, False, mode='_neg')

            dd_etypes = [('drug', 'contraindication', 'disease'),
                         ('drug', 'indication', 'disease'),
                         ('drug', 'off-label use', 'disease'),
                         ('disease', 'rev_contraindication', 'drug'),
                         ('disease', 'rev_indication', 'drug'),
                         ('disease', 'rev_off-label use', 'drug')]

            pos_score = torch.cat([scores[i] for i in dd_etypes])
            neg_score = torch.cat([scores_neg[i] for i in dd_etypes])

            scores = torch.sigmoid(torch.cat((pos_score, neg_score)).reshape(-1, ))
            labels = [1] * len(pos_score) + [0] * len(neg_score)

            loss = F.binary_cross_entropy(scores, torch.Tensor(labels).float().to(self.device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step(loss)

            if self.weight_bias_track:
                self.wandb.log({"Training Loss": loss})

            if epoch % train_print_per_n == 0:
                print(epoch)
                # training tracking...
                with torch.no_grad():
                    with self.G.local_scope():
                        input_dict = {ntype: self.G.nodes[ntype].data['inp'] for ntype in self.G.ntypes}

                        ret1 = self.model.forward_all(self.G, input_dict)

                        scores2, out_pos2 = self.pred(self.g_test_pos, self.G, ret1, False, mode='_pos')
                        scores_neg2, out_neg2 = self.pred(self.g_test_neg, self.G, ret1, False, mode='_neg')

                        dd_etypes = [('drug', 'contraindication', 'disease'),
                                     ('drug', 'indication', 'disease'),
                                     ('drug', 'off-label use', 'disease'),
                                     ('disease', 'rev_contraindication', 'drug'),
                                     ('disease', 'rev_indication', 'drug'),
                                     ('disease', 'rev_off-label use', 'drug')]

                        pos_score = torch.cat([scores2[i] for i in dd_etypes])
                        neg_score = torch.cat([scores_neg2[i] for i in dd_etypes])

                        scores = torch.sigmoid(torch.cat((pos_score, neg_score)).reshape(-1, ))
                        labels2 = [1] * len(pos_score) + [0] * len(neg_score)

                        loss2 = F.binary_cross_entropy(scores, torch.Tensor(labels2).float().to(self.device))

                        (auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc,
                         macro_auprc), loss, pred_pos, pred_neg = get_all_metrics_fb(scores2, scores_neg2,
                                                                                     scores.reshape(
                                                                                         -1, ).detach().cpu().numpy(),
                                                                                     labels2,
                                                                                     self.G,
                                                                                     True), loss2.item(), scores2, scores_neg2
                        print(
                            'Testing Loss %.4f Testing Micro AUROC %.4f Testing Micro AUPRC %.4f Testing Macro AUROC %.4f Testing Macro AUPRC %.4f' % (
                                loss,
                                micro_auroc,
                                micro_auprc,
                                macro_auroc,
                                macro_auprc
                            ))
                        print(auroc_rel)
                        print(auprc_rel)

