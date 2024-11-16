"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/10/18 18:23
  @Email: 2665109868@qq.com
  @function
"""
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import csv
import pandas as pd
import sys
import os
from utils import process
import warnings
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils.convert import to_dgl
from torch_geometric.loader import LinkNeighborLoader, HGTLoader,LinkLoader
from torch_geometric.sampler import NeighborSampler,NegativeSampling
import dgl
import torch
from torch_geometric.utils import degree
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")
from torch_geometric.nn import HGTConv, Linear
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling


def construct_negative_graph_each_etype(graph, k, etype, method, weights):
    src, dst = graph[etype].edge_index
    utype, _, vtype = etype

    if method == 'corrupt_dst':
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, graph[vtype].num_nodes, (len(src) * k,))
    elif method == 'corrupt_src':
        neg_dst = dst.repeat_interleave(k)
        neg_src = torch.randint(0, graph[utype].num_nodes, (len(dst) * k,))
    elif method == 'corrupt_both':
        neg_src = torch.randint(0, graph[utype].num_nodes, (len(dst) * k,))
        neg_dst = torch.randint(0, graph[vtype].num_nodes, (len(src) * k,))
    elif method in ['multinomial_src', 'inverse_src', 'fix_src']:
        neg_dst = dst.repeat_interleave(k)
        try:
            neg_src = weights[etype].multinomial(len(neg_dst), replacement=True)
        except:
            neg_src = torch.Tensor([])
    elif method in ['multinomial_dst', 'inverse_dst', 'fix_dst']:
        neg_src = src.repeat_interleave(k)
        try:
            neg_dst = weights[etype].multinomial(len(neg_src), replacement=True)
        except:
            neg_dst = torch.Tensor([])

    return {etype: (neg_src, neg_dst)}


class FullGraphNegSampler:
    def __init__(self, graph, k, method):
        self.weights = {}
        if method == 'multinomial_src':
            self.weights = {
                etype: torch.pow(degree(graph[etype].edge_index[1]), 0.75).float()
                for etype in graph.edge_types
            }
        elif method == 'multinomial_dst':
            self.weights = {
                etype: torch.pow(degree(graph[etype].edge_index[1]), 0.75).float()
                for etype in graph.edge_types
            }
        elif method == 'inverse_dst':
            self.weights = {
                etype: -torch.pow(degree(graph[etype].edge_index[1]), 0.75).float()
                for etype in graph.edge_types
            }
        elif method == 'inverse_src':
            self.weights = {
                etype: -torch.pow(degree(graph[etype].edge_index[0]), 0.75).float()
                for etype in graph.edge_types
            }
        elif method == 'fix_dst':
            self.weights = {
                etype: (degree(graph[etype].edge_index[1])>0).float()

                for etype in graph.edge_types
            }
        elif method == 'fix_src':
            self.weights = {
                etype: (degree(graph[etype].edge_index[0])>0).float()
                for etype in graph.edge_types
            }

        self.k = k
        self.method = method

    def __call__(self, graph):
        out = {}
        for etype in graph.edge_types:
            temp = construct_negative_graph_each_etype(graph, self.k, etype, self.method, self.weights)
            if len(temp[etype][0]) != 0:
                out.update(temp)
        from copy import deepcopy
        negative_graph = deepcopy(graph)
        for etype, (src, dst) in out.items():
            utype, rel, vtype = etype
            negative_graph[(utype, rel, vtype)].edge_index = torch.stack([src, dst], dim=0)


        return negative_graph


def evaluate_graph_construct(df_valid, data, neg_sampler, k):
    edge_dict = {}
    df_in = df_valid[['x_idx', 'relation', 'y_idx']]
    for etype in data.edge_types:
        utype, rel, vtype = etype
        df_temp = df_in[df_in.relation == rel]
        src = torch.tensor(df_temp.x_idx.values, dtype=torch.long)
        dst = torch.tensor(df_temp.y_idx.values, dtype=torch.long)
        edge_dict[etype] = torch.stack([src, dst], dim=0)

    # Construct the validation graph
    g_valid = HeteroData()
    for ntype in data.node_types:
        g_valid[ntype].num_nodes = data[ntype].num_nodes
    for etype, edge_index in edge_dict.items():
        g_valid[etype].edge_index = edge_index
    g_valid = process.initialize_node_embedding(g_valid, 128)
    # Negative sampling
    ng = FullGraphNegSampler(g_valid, k, neg_sampler)
    g_neg_valid = ng(g_valid)

    return g_valid, g_neg_valid



class PreData:

    def __init__(self, data_folder_path):
        if not os.path.exists(data_folder_path):
            os.mkdir(data_folder_path)

        self.data_folder = data_folder_path  # the data folder, contains the kg.csv

    def prepare_split(self, split='complex_disease',
                      disease_eval_idx=None,
                      seed=42,
                      no_kg=False,
                      test_size=0.05,
                      mask_ratio=0.1,
                      one_hop=False):

        if split not in ['random', 'complex_disease', 'complex_disease_cv', 'disease_eval', 'cell_proliferation',
                         'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland', 'autoimmune',
                         'metabolic_disorder', 'diabetes', 'neurodigenerative', 'full_graph', 'downstream_pred',
                         'few_edeges_to_kg', 'few_edeges_to_indications']:
            raise ValueError(
                "Please select one of the following supported splits: 'random', 'complex_disease', 'disease_eval', 'cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland'")

        if disease_eval_idx is not None:
            split = 'disease_eval'
            print('disease eval index is not none, use the individual disease split...')
        self.split = split

        if split in ['cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland', 'autoimmune',
                     'metabolic_disorder', 'diabetes', 'neurodigenerative']:

            if test_size != 0.05:
                folder_name = split + '_kg' + '_frac' + str(test_size)
            elif one_hop:
                folder_name = split + '_kg' + '_one_hop_ratio' + str(mask_ratio)
            else:
                folder_name = split + '_kg'

            if not os.path.exists(os.path.join(self.data_folder, folder_name)):
                os.mkdir(os.path.join(self.data_folder, folder_name))
            kg_path = os.path.join(self.data_folder, folder_name, 'kg_directed.csv')
        else:
            kg_path = os.path.join(self.data_folder, 'kg_directed.csv')

        if os.path.exists(kg_path):
            print('Found saved processed KG... Loading...')
            df = pd.read_csv(kg_path)
        else:
            if os.path.exists(os.path.join(self.data_folder, 'kg.csv')):
                print('First time usage... Mapping TxData raw KG to directed csv... it takes several minutes...')
                process.preprocess_kg(self.data_folder, split, test_size, one_hop, mask_ratio)
                df = pd.read_csv(kg_path)
            else:
                raise ValueError("KG file path does not exist...")

        if split == 'disease_eval':
            split_data_path = os.path.join(self.data_folder, self.split + '_' + str(disease_eval_idx))
        elif split == 'downstream_pred':
            split_data_path = os.path.join(self.data_folder, self.split + '_downstream_pred')
            disease_eval_idx = [11394., 6353., 12696., 14183., 12895., 9128., 12623., 15129.,
                                12897., 12860., 7611., 13113., 4029., 14906., 13438., 13177.,
                                13335., 12896., 12879., 12909., 4815., 12766., 12653.]
        elif no_kg:
            split_data_path = os.path.join(self.data_folder, self.split + '_no_kg_' + str(seed))
        elif test_size != 0.05:
            split_data_path = os.path.join(self.data_folder, self.split + '_' + str(seed)) + '_frac' + str(test_size)
        elif one_hop:
            split_data_path = os.path.join(self.data_folder, self.split + '_' + str(seed)) + '_one_hop_ratio' + str(
                mask_ratio)
        else:
            split_data_path = os.path.join(self.data_folder, self.split + '_' + str(seed))

        if no_kg:
            sub_kg = ['off-label use', 'indication', 'contraindication']
            df = df[df.relation.isin(sub_kg)].reset_index(drop=True)

        if not os.path.exists(os.path.join(split_data_path, 'train.csv')):
            if not os.path.exists(split_data_path):
                os.mkdir(split_data_path)
            print('Creating splits... it takes several minutes...')
            df_train, df_valid, df_test = process.create_split(df, split, disease_eval_idx, split_data_path, seed)
        else:
            print('Splits detected... Loading splits....')
            df_train = pd.read_csv(os.path.join(split_data_path, 'train.csv'))
            df_valid = pd.read_csv(os.path.join(split_data_path, 'valid.csv'))
            df_test = pd.read_csv(os.path.join(split_data_path, 'test.csv'))

        if split not in ['random', 'complex_disease', 'complex_disease_cv', 'disease_eval', 'full_graph',
                         'downstream_pred', 'few_edeges_to_indications', 'few_edeges_to_kg']:
            # in disease area split
            df_test = process.process_disease_area_split(self.data_folder, df, df_test, split)

        print('Creating PyG graph....')
        #create pyg graph
        g = process.create_pyg_graph(df_valid,df)
        g_valid_pos , g_valid_neg = evaluate_graph_construct(df_valid,g,'multinomial_dst', 1)



        print('Done!')
        return g,df, df_train, df_valid, df_test,disease_eval_idx,no_kg,g_valid_pos , g_valid_neg


    def retrieve_id_mapping(self):
        df = self.df
        df['x_id'] = df.x_id.apply(lambda x: process.convert2str(x))
        df['y_id'] = df.y_id.apply(lambda x: process.convert2str(x))

        idx2id_drug = dict(df[df.x_type == 'drug'][['x_idx', 'x_id']].drop_duplicates().values)
        idx2id_drug.update(dict(df[df.y_type == 'drug'][['y_idx', 'y_id']].drop_duplicates().values))

        idx2id_disease = dict(df[df.x_type == 'disease'][['x_idx', 'x_id']].drop_duplicates().values)
        idx2id_disease.update(dict(df[df.y_type == 'disease'][['y_idx', 'y_id']].drop_duplicates().values))

        df_ = pd.read_csv(os.path.join(self.data_folder, 'kg.csv'))
        df_['x_id'] = df_.x_id.apply(lambda x: process.convert2str(x))
        df_['y_id'] = df_.y_id.apply(lambda x: process.convert2str(x))

        id2name_disease = dict(df_[df_.x_type == 'disease'][['x_id', 'x_name']].drop_duplicates().values)
        id2name_disease.update(dict(df_[df_.y_type == 'disease'][['y_id', 'y_name']].drop_duplicates().values))

        id2name_drug = dict(df_[df_.x_type == 'drug'][['x_id', 'x_name']].drop_duplicates().values)
        id2name_drug.update(dict(df_[df_.y_type == 'drug'][['y_id', 'y_name']].drop_duplicates().values))

        return {'id2name_drug': id2name_drug,
                'id2name_disease': id2name_disease,
                'idx2id_disease': idx2id_disease,
                'idx2id_drug': idx2id_drug
                }


from tqdm import tqdm
if __name__ == '__main__':
    preData = PreData(data_folder_path='/data/zhaojingtong/PrimeKG/data_all/')
    g,df, df_train, df_valid, df_test,disease_eval_idx,no_kg, g_valid_pos , g_valid_neg= preData.prepare_split(split='random', seed=42, no_kg=False)
    g = process.initialize_node_embedding(g, 128)
    data = g
    gpu_id = 0  # 选择你想使用的 GPU ID，例如 0, 1, 2 等
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    for ntype in g.node_types:
        loader = HGTLoader(
            data,
            # Sample 512 nodes per type and per iteration for 4 iterations
            num_samples={key: [256] * 2 for key in data.node_types},
            # Use a batch size of 128 for sampling training nodes of type paper
            batch_size=1024,
            input_nodes=(ntype)
        )
        sampled_hetero_data = next(iter(loader))
        ng = FullGraphNegSampler(sampled_hetero_data, 1,'fix_dst')
        g_neg_valid = ng(sampled_hetero_data)
        print(sampled_hetero_data)






























