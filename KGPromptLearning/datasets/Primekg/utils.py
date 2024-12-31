import scipy.io
import urllib.request
import dgl
from dgl.ops import edge_softmax
import math
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch.utils import data
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import copy
import pickle
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from random import choice
from collections import Counter
import requests
from zipfile import ZipFile 

import warnings
warnings.filterwarnings("ignore")

#device = torch.device("cuda:0")

from data_splits.datasplit import DataSplitter


def preprocess_kg(path, split, test_size = 0.05, one_hop = False, mask_ratio = 0.1):
    if split in ['cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland','autoimmune', 'metabolic_disorder', 'diabetes', 'neurodigenerative']:
        
        print('Generating disease area using ontology... might take several minutes...')
        name2id = { 
                    'cell_proliferation': '14566',
                    'mental_health': '150',
                    'cardiovascular': '1287',
                    'anemia': '2355',
                    'adrenal_gland': '9553',
                    'autoimmune': '417',
                    'metabolic_disorder': '655',
                    'diabetes': '9351',
                    'neurodigenerative': '1289'
                  }

        ds = DataSplitter(kg_path = path)

        test_kg = ds.get_test_kg_for_disease(name2id[split], test_size = test_size, one_hop = one_hop, mask_ratio = mask_ratio)
        all_kg = ds.kg
        all_kg['split'] = 'train'
        test_kg['split'] = 'test'
        df = pd.concat([all_kg, test_kg]).drop_duplicates(subset = ['x_index', 'y_index'], keep = 'last').reset_index(drop = True)
        
        print('test size: ', test_size)
        if test_size != 0.05:
            folder_name = split + '_kg_frac' + str(test_size)
        elif one_hop:
            folder_name = split + '_kg' + '_one_hop_ratio' + str(mask_ratio)
        else:
            folder_name = split + '_kg'
        
        path = os.path.join(path, folder_name)
        
        if not os.path.exists(path):
            os.mkdir(path)
        print('save kg.csv to ', os.path.join(path, 'kg.csv'))
        df.to_csv(os.path.join(path, 'kg.csv'), index = False)
        df = df[['x_name','x_type', 'x_id', 'relation', 'y_name','y_type', 'y_id', 'split']]

    else:
        ## random, complex disease splits
        df = pd.read_csv(os.path.join(path, 'kg.csv'))
        df = df[['x_name','x_index','x_type', 'x_id', 'relation', 'y_name','y_index','y_type', 'y_id']]
    unique_relation = np.unique(df.relation.values)
    undirected_index = []
    
    print('Iterating over relations...')
    
    for i in tqdm(unique_relation):
        if ('_' in i) and (i.split('_')[0] == i.split('_')[1]):
            # homogeneous graph
            df_temp = df[df.relation == i]
            df_temp['check_string'] = df_temp.apply(lambda row: '_'.join(sorted([str(row['x_id']), str(row['y_id'])])), axis=1)
            undirected_index.append(df_temp.drop_duplicates('check_string').index.values.tolist())
        else:
            # undirected
            d_off = df[df.relation == i]
            undirected_index.append(d_off[d_off.x_type == d_off.x_type.iloc[0]].index.values.tolist())
    flat_list = [item for sublist in undirected_index for item in sublist]
    df = df[df.index.isin(flat_list)]
    unique_node_types = np.unique(np.append(np.unique(df.x_type.values), np.unique(df.y_type.values)))

    df['x_id'] = df.x_id.apply(lambda x: convert2str(x))
    df['y_id'] = df.y_id.apply(lambda x: convert2str(x))


    print('save kg_directed.csv...')


    df.to_csv(os.path.join(path, 'kg_directed.csv'), index = False)

def random_fold(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()
    # to avoid extreme minority types don't exist in valid/test
    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac = test_frac, replace = False, random_state = fold_seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = pd.concat([df_train, train])
        df_valid = pd.concat([df_valid, val])
        df_test = pd.concat([df_test, test])
        
    return {'train': df_train.reset_index(drop = True), 
            'valid': df_valid.reset_index(drop = True), 
            'test': df_test.reset_index(drop = True)}

def disease_eval_fold(df, fold_seed, disease_idx):
    if not isinstance(disease_idx, list):
        disease_idx = np.array([disease_idx])
    else:
        disease_idx = np.array(disease_idx)
        
    dd_rel_types = ['contraindication', 'indication', 'off-label use']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)]
    
    unique_diseases = df_dd.y_idx.unique()
   
    # remove the unique disease out of training
    train_diseases = np.setdiff1d(unique_diseases, disease_idx)
    df_dd_train_val = df_dd[df_dd.y_idx.isin(train_diseases)]                               
    df_dd_test = df_dd[df_dd.y_idx.isin(disease_idx)]
    
    # randomly get 5% disease-drug pairs for validation 
    df_dd_val = df_dd_train_val.sample(frac = 0.05, replace = False, random_state = fold_seed)
    df_dd_train = df_dd_train_val[~df_dd_train_val.index.isin(df_dd_val.index)]
                                       
    df_train = pd.concat([df_not_dd, df_dd_train])
    df_valid = df_dd_val
    df_test = df_dd_test                               
                                   
    #np.random.seed(fold_seed)
    #np.random.shuffle(unique_diseases)
    #train, valid = np.split(unique_diseases, int(0.95*len(unique_diseases)))
    return {'train': df_train.reset_index(drop = True), 
            'valid': df_valid.reset_index(drop = True), 
            'test': df_test.reset_index(drop = True)}                      

def complex_disease_fold(df, fold_seed, frac):
    dd_rel_types = ['contraindication', 'indication', 'off-label use']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)] 
    
    unique_diseases = df_dd.y_idx.unique()
    np.random.seed(fold_seed)
    np.random.shuffle(unique_diseases)
    train, valid, test = np.split(unique_diseases, [int(frac[0]*len(unique_diseases)), int((frac[0] + frac[1])*len(unique_diseases))])
    
    df_dd_train = df_dd[df_dd.y_idx.isin(train)]
    df_dd_valid = df_dd[df_dd.y_idx.isin(valid)]
    df_dd_test = df_dd[df_dd.y_idx.isin(test)]
    
    df = df_not_dd
    train_frac, val_frac, test_frac = frac
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()

    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac = test_frac, replace = False, random_state = fold_seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = pd.concat([df_train, train])
        df_valid = pd.concat([df_valid, val])
        df_test = pd.concat([df_test, test])
    
    df_train = pd.concat([df_train, df_dd_train])
    df_valid = pd.concat([df_valid, df_dd_valid])
    df_test = pd.concat([df_test, df_dd_test])
    
    return {'train': df_train.reset_index(drop = True), 
            'valid': df_valid.reset_index(drop = True), 
            'test': df_test.reset_index(drop = True)}
        
def few_edeges_to_kg_fold(df, fold_seed, frac):
    
    dd_rel_types = ['contraindication', 'indication', 'off-label use']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)]
    
    disease2num_neighbors_1 = dict(df_not_dd[df_not_dd.x_type == 'disease'].groupby('x_idx').y_id.agg(len))
    disease2num_neighbors_2 = dict(df_not_dd[df_not_dd.y_type == 'disease'].groupby('y_idx').x_id.agg(len))

    disease2num_neighbors = {}

    # Iterating through keys in both dictionaries
    for key in set(disease2num_neighbors_1).union(disease2num_neighbors_2):
        disease2num_neighbors[key] = disease2num_neighbors_1.get(key, 0) + disease2num_neighbors_2.get(key, 0)
    
    disease_with_less_than_3_connections_in_kg = np.array([i for i,j in disease2num_neighbors.items() if j <= 3])
    unique_diseases = df_dd.y_idx.unique()
    train_val_diseases = np.setdiff1d(unique_diseases, disease_with_less_than_3_connections_in_kg)
    test = np.intersect1d(unique_diseases, disease_with_less_than_3_connections_in_kg)
    print('Number of testing diseases: ', len(test))
    np.random.seed(fold_seed)
    np.random.shuffle(train_val_diseases)
    train, valid = np.split(train_val_diseases, [int(frac[0]*len(unique_diseases))])
    print('Number of train diseases: ', len(train))
    print('Number of valid diseases: ', len(valid))
    
    df_dd_train = df_dd[df_dd.y_idx.isin(train)]
    df_dd_valid = df_dd[df_dd.y_idx.isin(valid)]
    df_dd_test = df_dd[df_dd.y_idx.isin(test)]
    
    df = df_not_dd
    train_frac, val_frac, test_frac = frac
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()

    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac = test_frac, replace = False, random_state = fold_seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = pd.concat([df_train, train])
        df_valid = pd.concat([df_valid, val])
        df_test = pd.concat([df_test, test])
    
    df_train = pd.concat([df_train, df_dd_train])
    df_valid = pd.concat([df_valid, df_dd_valid])
    df_test = pd.concat([df_test, df_dd_test])
    
    return {'train': df_train.reset_index(drop = True), 
            'valid': df_valid.reset_index(drop = True), 
            'test': df_test.reset_index(drop = True)}
    
    
def few_edeges_to_indications_fold(df, fold_seed, frac):
    
    dd_rel_types = ['contraindication', 'indication', 'off-label use']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)]
    
    disease2num_indications = dict(df_dd[(df_dd.y_type == 'disease') & (df_dd.relation == 'indication')].groupby('x_idx').y_id.agg(len))
    
    disease_with_less_than_3_indications_in_kg = np.array([i for i,j in disease2num_indications.items() if j <= 3])
    unique_diseases = df_dd.y_idx.unique()
    train_val_diseases = np.setdiff1d(unique_diseases, disease_with_less_than_3_indications_in_kg)
    test = np.intersect1d(unique_diseases, disease_with_less_than_3_indications_in_kg)
    print('Number of testing diseases: ', len(test))
    np.random.seed(fold_seed)
    np.random.shuffle(train_val_diseases)
    train, valid = np.split(train_val_diseases, [int(frac[0]*len(unique_diseases))])
    print('Number of train diseases: ', len(train))
    print('Number of valid diseases: ', len(valid))
    
    df_dd_train = df_dd[df_dd.y_idx.isin(train)]
    df_dd_valid = df_dd[df_dd.y_idx.isin(valid)]
    df_dd_test = df_dd[df_dd.y_idx.isin(test)]
    
    df = df_not_dd
    train_frac, val_frac, test_frac = frac
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()

    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac = test_frac, replace = False, random_state = fold_seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = pd.concat([df_train, train])
        df_valid = pd.concat([df_valid, val])
        df_test = pd.concat([df_test, test])
    
    df_train = pd.concat([df_train, df_dd_train])
    df_valid = pd.concat([df_valid, df_dd_valid])
    df_test = pd.concat([df_test, df_dd_test])
    
    return {'train': df_train.reset_index(drop = True), 
            'valid': df_valid.reset_index(drop = True), 
            'test': df_test.reset_index(drop = True)}
    
def create_fold_cv(df, split_num, num_splits):
    dd_rel_types = ['contraindication', 'indication', 'off-label use']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)] 
    
    unique_diseases = df_dd.y_idx.unique()
    np.random.seed(42)
    np.random.shuffle(unique_diseases)
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=num_splits)
    
    split_num_idx = {}
    
    for i, (train_index, test_index) in enumerate(kf.split(unique_diseases)):
        train_index, valid_index, _ = np.split(train_index, [int(0.9*len(train_index)), int(len(train_index))])
        split_num_idx[i+1] = {'train': unique_diseases[train_index],
                              'valid': unique_diseases[valid_index],
                              'test': unique_diseases[test_index]
                             }
        
    train, valid, test = split_num_idx[split_num]['train'], split_num_idx[split_num]['valid'], split_num_idx[split_num]['test']
    df_dd_train = df_dd[df_dd.y_idx.isin(train)]
    df_dd_valid = df_dd[df_dd.y_idx.isin(valid)]
    df_dd_test = df_dd[df_dd.y_idx.isin(test)]
    
    df = df_not_dd
    train_frac, val_frac, test_frac = [0.83125, 0.11875, 0.05]
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()

    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac = test_frac, replace = False, random_state = split_num)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = pd.concat([df_train, train])
        df_valid = pd.concat([df_valid, val])
        df_test = pd.concat([df_test, test])
    
    df_train = pd.concat([df_train, df_dd_train])
    df_valid = pd.concat([df_valid, df_dd_valid])
    df_test = pd.concat([df_test, df_dd_test])
    
    return df_train.reset_index(drop = True), df_valid.reset_index(drop = True), df_test.reset_index(drop = True)
    
    
def create_fold(df, fold_seed = 100, frac = [0.7, 0.1, 0.2], method = 'random', disease_idx = 0.0):
    if method == 'random':
        out = random_fold(df, fold_seed, frac)
    elif method == 'complex_disease':
        out = complex_disease_fold(df, fold_seed, frac)
    elif method == 'few_edeges_to_kg':
        out = few_edeges_to_kg_fold(df, fold_seed, [0.7, 0.1, 0.2])
    elif method == 'few_edeges_to_indications':
        out = few_edeges_to_indications_fold(df, fold_seed, [0.7, 0.1, 0.2])
    elif method == 'downstream_pred':
        out = disease_eval_fold(df, fold_seed, disease_idx)        
    elif method == 'disease_eval':
        out = disease_eval_fold(df, fold_seed, disease_idx)
    elif method == 'full_graph':
        out = random_fold(df, fold_seed, [0.95, 0.05, 0.0])
        out['test'] = out['valid'] # this is to avoid error but we are not using testing set metric here
    else:
        # disease split
        train_val = df[df.split == 'train'].reset_index(drop = True)
        test = df[df.split == 'test'].reset_index(drop = True)
        out = random_fold(train_val, fold_seed, [0.875, 0.125, 0.0])
        out['test'] = test
    return out['train'], out['valid'], out['test']


def create_split(df, split, disease_eval_index, split_data_path, seed):
    print('split_data_path: ', split_data_path)
    if split == 'complex_disease_cv':
        if seed < 1 or seed > 20:
            raise ValueError('Complex disease cross validation 20 folds, select seed from 1-20.')
        df_train, df_valid, df_test = create_fold_cv(df, split_num = seed, num_splits = 20)
    else:
        df_train, df_valid, df_test = create_fold(df, fold_seed = seed, frac = [0.83125, 0.11875, 0.05], method = split, disease_idx = disease_eval_index)


    unique_relations = df['relation'].drop_duplicates().reset_index(drop=True)

    # 创建带编号的 DataFrame
    output_df = pd.DataFrame({
        'relation': unique_relations,
        'id': range(len(unique_relations))
    })
    # 保存为 CSV 文件
    output_df.to_csv(os.path.join(split_data_path, 'relation.txt'), sep="\t",index=False, header=False)

    x_df = df[['x_name', 'x_index']].rename(columns={'x_name': 'name', 'x_index': 'id'})
    y_df = df[['y_name', 'y_index']].rename(columns={'y_name': 'name', 'y_index': 'id'})
    combined_df = pd.concat([x_df, y_df])
    unique_df = combined_df.drop_duplicates().reset_index(drop=True)
    unique_df.to_csv(os.path.join(split_data_path, 'entities.txt'),sep="\t", index=False, header=False)
    df_train = df_train[['x_name','relation','y_name']]
    df_valid = df_valid[['x_name','relation','y_name']]
    df_test = df_test[['x_name','relation','y_name']]

    df_train.to_csv(os.path.join(split_data_path, 'train.txt'), sep="\t",index = False, header=False)
    df_valid.to_csv(os.path.join(split_data_path, 'valid.txt'), sep="\t",index = False, header=False)
    df_test.to_csv(os.path.join(split_data_path, 'test.txt'),sep="\t", index = False, header=False)
    
    return df_train, df_valid, df_test
    
def construct_negative_graph_each_etype(graph, k, etype, method, weights, device):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    
    if method == 'corrupt_dst':
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,))
    elif method == 'corrupt_src':
        neg_dst = dst.repeat_interleave(k)
        neg_src = torch.randint(0, graph.number_of_nodes(utype), (len(dst) * k,))
    elif method == 'corrupt_both':
        neg_src = torch.randint(0, graph.number_of_nodes(utype), (len(dst) * k,))
        neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,))
    elif (method == 'multinomial_src') or (method == 'inverse_src') or (method == 'fix_src'):
        neg_dst = dst.repeat_interleave(k)
        try:
            neg_src = weights[etype].multinomial(len(neg_dst), replacement=True)
        except:
            neg_src = torch.Tensor([])
    elif (method == 'multinomial_dst') or (method == 'inverse_dst') or (method == 'fix_dst'):
        neg_src = src.repeat_interleave(k)
        try:
            neg_dst = weights[etype].multinomial(len(neg_src), replacement=True)
        except:
            neg_dst = torch.Tensor([])
    return {etype: (neg_src.to(device), neg_dst.to(device))}

def construct_negative_graph(graph, k, device):
    out = {}   
    for etype in graph.canonical_etypes:
        out.update(construct_negative_graph_each_etype(graph, k, etype, device))
    return dgl.heterograph(out, num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})


def evaluate_mb(model, g_pos, g_neg, G, dd_etypes, device, return_embed = False, mode = 'valid'):
    model.eval()
    #model = model.to('cpu')
    pred_score_pos, pred_score_neg, pos_score, neg_score = model.forward_minibatch(g_pos.to(device), g_neg.to(device), [G.to(device), G.to(device)], G.to(device), mode = mode, pretrain_mode = False)
    
    pos_score = torch.cat([pred_score_pos[i] for i in dd_etypes])
    neg_score = torch.cat([pred_score_neg[i] for i in dd_etypes])
    
    scores = torch.sigmoid(torch.cat((pos_score, neg_score)).reshape(-1,))
    labels = [1] * len(pos_score) + [0] * len(neg_score)
    loss = F.binary_cross_entropy(scores, torch.Tensor(labels).float().to(device))
    
    model = model.to(device)
    if return_embed:
        return get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, G, True), loss.item(), pred_score_pos, pred_score_neg
    else:
        return get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, G, True), loss.item()

## disable all gradient
def disable_all_gradients(module):
    for param in module.parameters():
        param.requires_grad = False

def print_dict(x, dd_only = True):
    if dd_only:
        etypes = [('drug', 'contraindication', 'disease'), 
                  ('drug', 'indication', 'disease'), 
                  ('drug', 'off-label use', 'disease'),
                  ('disease', 'rev_contraindication', 'drug'), 
                  ('disease', 'rev_indication', 'drug'), 
                  ('disease', 'rev_off-label use', 'drug')]
        
        for i in etypes:
            print(str(i) + ': ' + str(x[i]))
    else:
        for i, j in x.items():
            print(str(i) + ': ' + str(j))
        
def to_wandb_table(auroc, auprc):
    return [[idx, i[1], j, auprc[i]] for idx, (i, j) in enumerate(auroc.items())]

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def process_df(df_train, edge_dict):
    df_train['relation_idx'] = [edge_dict[i] for i in df_train['relation']]
    df_train = df_train[['x_type', 'x_idx', 'relation_idx', 'y_type', 'y_idx', 'degree', 'label']].rename(columns = {'x_type': 'head_type', 
                                                                                    'x_idx': 'head', 
                                                                                    'relation_idx': 'relation',
                                                                                    'y_type': 'tail_type',
                                                                                    'y_idx': 'tail'})
    df_train['head'] = df_train['head'].astype(int)
    df_train['tail'] = df_train['tail'].astype(int)
    return df_train


def reverse_rel_generation(df, df_valid, unique_rel):
    
    for i in unique_rel.values:
        temp = df_valid[df_valid.relation == i[1]]
        temp = temp.rename(columns={"x_type": "y_type", 
                     "x_id": "y_id", 
                     "x_idx": "y_idx",
                     "y_type": "x_type", 
                     "y_id": "x_id", 
                     "y_idx": "x_idx"})

        if i[0] != i[2]:
            # bi identity
            temp["relation"] = 'rev_' + i[1]
        df_valid = pd.concat([df_valid,temp])
    return df_valid.reset_index(drop = True)


def get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, mode):
    
    results = {
              mode + " Micro AUROC": micro_auroc,
              mode + " Micro AUPRC": micro_auprc,
              mode + " Macro AUROC": macro_auroc,
              mode + " Macro AUPRC": macro_auprc
    }
    
    relations = [('drug', 'contraindication', 'disease'),
                 ('drug', 'indication', 'disease'),
                 ('drug', 'off-label use', 'disease'),
                 ('disease', 'rev_contraindication', 'drug'),
                 ('disease', 'rev_indication', 'drug'),
                 ('disease', 'rev_off-label use', 'drug')
                ]
    
    name_mapping = {('drug', 'contraindication', 'disease'): ' Contraindication ',
                    ('drug', 'indication', 'disease'): ' Indication ',
                    ('drug', 'off-label use', 'disease'): ' Off-Label ',
                    ('disease', 'rev_contraindication', 'drug'): ' Rev-Contraindication ',
                    ('disease', 'rev_indication', 'drug'): ' Rev-Indication ',
                    ('disease', 'rev_off-label use', 'drug'): ' Rev-Off-Label '
                   }
    
    for i in relations:
        if i in auroc_rel:
            results.update({mode + name_mapping[i] + "AUROC": auroc_rel[i]})
        if i in auprc_rel:
            results.update({mode + name_mapping[i] + "AUPRC": auprc_rel[i]})
    return results

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def obtain_protein_random_walk_profile(disease, num_walks, path_len, g, disease_etypes, disease_nodes, walk_mode):
    random_walks = []
    num_nodes = len(g.nodes('gene/protein'))
    for _ in range(num_walks):
        successor = g.successors(disease, etype = 'rev_disease_protein')
        if len(successor) > 0:
            current = choice(successor)
        else:
            continue
        path = [current.item()]
        for path_idx in range(path_len):
            successor = g.successors(current, etype = 'protein_protein')
            if len(successor) > 0:
                current = choice(successor)
                path.append(current.item())
            else:
                break

        random_walks = random_walks + path
        
    if walk_mode == 'bit':
        visted_nodes = np.unique(np.array(random_walks))
        node_profile = torch.zeros((num_nodes,))
        node_profile[visted_nodes] = 1.
    elif walk_mode == 'prob':
        visted_nodes = Counter(random_walks)
        node_profile = torch.zeros((num_nodes,))
        for x, y in visted_nodes.items():
            node_profile[x] = y/len(random_walks)
    return node_profile

def obtain_disease_profile(G, disease, disease_etypes, disease_nodes):
    profiles_for_each_disease_types = []
    for idx, disease_etype in enumerate(disease_etypes):
        nodes = G.successors(disease, etype=disease_etype)
        num_nodes = len(G.nodes(disease_nodes[idx]))
        node_profile = torch.zeros((num_nodes,))
        node_profile[nodes] = 1.
        profiles_for_each_disease_types.append(node_profile)
    return torch.cat(profiles_for_each_disease_types)

def exponential(x, lamb):
    return lamb * torch.exp(-lamb * x) + 0.2

def convert2str(x):
    try:
        if '_' in str(x): 
            pass
        else:
            x = float(x)
    except:
        pass

    return str(x)

def map_node_id_2_idx(x, id2idx):
        id_ = convert2str(x)
        if id_ in id2idx:
            return id2idx[id_]
        else:
            return 'null'
        
def process_disease_area_split(data_folder, df, df_test, split):
    disease_file_path = os.path.join(data_folder, 'disease_files')
    disease_list = pd.read_csv(os.path.join(disease_file_path, split + '.csv'))
    
    id2idx = dict(df[df.x_type == 'disease'][['x_id', 'x_idx']].drop_duplicates().values)
    id2idx.update(dict(df[df.y_type == 'disease'][['y_id', 'y_idx']].drop_duplicates().values))

    temp_dict = {}

    # for merged disease ids
    for i,j in id2idx.items():
        try:
            if '_' in i:
                for x in i.split('_'):
                    temp_dict[str(float(x))] = j
        except:
            temp_dict[str(float(i))] = j

    id2idx.update(temp_dict)

    disease_list['node_idx'] = disease_list.node_id.apply(lambda x: map_node_id_2_idx(x, id2idx))

    disease_rel_types = ['rev_contraindication', 'rev_indication', 'rev_off-label use']
    temp = df_test[df_test.relation.isin(disease_rel_types)]
    df_test = df_test.drop(temp[~temp.x_idx.isin(disease_list.node_idx.unique())].index)
    
    return df_test



