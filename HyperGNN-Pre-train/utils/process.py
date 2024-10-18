import os

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import torch
import torch.nn as nn
import tqdm
import pandas as pd
from data_splits.datasplit import DataSplitter


def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret

# Process a (subset of) a TU dataset into standard form
def process_tu(data, nb_nodes):
    nb_graphs = len(data)
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros(nb_graphs)
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))
       
    for g in range(nb_graphs):
        sizes[g] = data[g].x.shape[0]
        features[g, :sizes[g]] = data[g].x
        labels[g] = data[g].y[0]
        masks[g, :sizes[g]] = 1.0
        e_ind = data[g].edge_index
        coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
        adjacency[g] = coo.todense()

    return features, adjacency, labels, sizes, masks

def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))
    
    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    current_path = os.path.dirname(__file__)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_train, idx_val, idx_test

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_kg(path, split, test_size=0.05, one_hop=False, mask_ratio=0.1):
    if split in ['cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland', 'autoimmune',
                 'metabolic_disorder', 'diabetes', 'neurodigenerative']:

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

        ds = DataSplitter(kg_path=path)

        test_kg = ds.get_test_kg_for_disease(name2id[split], test_size=test_size, one_hop=one_hop,
                                             mask_ratio=mask_ratio)
        all_kg = ds.kg
        all_kg['split'] = 'train'
        test_kg['split'] = 'test'
        df = pd.concat([all_kg, test_kg]).drop_duplicates(subset=['x_index', 'y_index'], keep='last').reset_index(
            drop=True)

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
        df.to_csv(os.path.join(path, 'kg.csv'), index=False)
        df = df[['x_type', 'x_id', 'relation', 'y_type', 'y_id', 'split']]

    else:
        ## random, complex disease splits
        df = pd.read_csv(os.path.join(path, 'kg.csv'))
        df = df[['x_type', 'x_id', 'relation', 'y_type', 'y_id']]
    unique_relation = np.unique(df.relation.values)
    undirected_index = []

    print('Iterating over relations...')

    for i in tqdm(unique_relation):
        if ('_' in i) and (i.split('_')[0] == i.split('_')[1]):
            # homogeneous graph
            df_temp = df[df.relation == i]
            df_temp['check_string'] = df_temp.apply(lambda row: '_'.join(sorted([str(row['x_id']), str(row['y_id'])])),
                                                    axis=1)
            undirected_index.append(df_temp.drop_duplicates('check_string').index.values.tolist())
        else:
            # undirected
            d_off = df[df.relation == i]
            undirected_index.append(d_off[d_off.x_type == d_off.x_type.iloc[0]].index.values.tolist())
    flat_list = [item for sublist in undirected_index for item in sublist]
    df = df[df.index.isin(flat_list)]
    unique_node_types = np.unique(np.append(np.unique(df.x_type.values), np.unique(df.y_type.values)))

    df['x_idx'] = np.nan
    df['y_idx'] = np.nan
    df['x_id'] = df.x_id.apply(lambda x: convert2str(x))
    df['y_id'] = df.y_id.apply(lambda x: convert2str(x))

    idx_map = {}
    print('Iterating over node types...')
    for i in tqdm(unique_node_types):
        names = np.unique(np.append(df[df.x_type == i]['x_id'].values, df[df.y_type == i]['y_id'].values))
        names2idx = dict(zip(names, list(range(len(names)))))
        df.loc[df.x_type == i, 'x_idx'] = df[df.x_type == i]['x_id'].apply(lambda x: names2idx[x])
        df.loc[df.y_type == i, 'y_idx'] = df[df.y_type == i]['y_id'].apply(lambda x: names2idx[x])
        idx_map[i] = names2idx

    print('save kg_directed.csv...')
    df.to_csv(os.path.join(path, 'kg_directed.csv'), index=False)


# process data


def preprocess_kg(path, split, test_size=0.05, one_hop=False, mask_ratio=0.1):
    if split in ['cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland', 'autoimmune',
                 'metabolic_disorder', 'diabetes', 'neurodigenerative']:

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

        ds = DataSplitter(kg_path=path)

        test_kg = ds.get_test_kg_for_disease(name2id[split], test_size=test_size, one_hop=one_hop,
                                             mask_ratio=mask_ratio)
        all_kg = ds.kg
        all_kg['split'] = 'train'
        test_kg['split'] = 'test'
        df = pd.concat([all_kg, test_kg]).drop_duplicates(subset=['x_index', 'y_index'], keep='last').reset_index(
            drop=True)

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
        df.to_csv(os.path.join(path, 'kg.csv'), index=False)
        df = df[['x_type', 'x_id', 'relation', 'y_type', 'y_id', 'split']]

    else:
        ## random, complex disease splits
        df = pd.read_csv(os.path.join(path, 'kg.csv'))
        df = df[['x_type', 'x_id', 'relation', 'y_type', 'y_id']]
    unique_relation = np.unique(df.relation.values)
    undirected_index = []

    print('Iterating over relations...')

    for i in tqdm(unique_relation):
        if ('_' in i) and (i.split('_')[0] == i.split('_')[1]):
            # homogeneous graph
            df_temp = df[df.relation == i]
            df_temp['check_string'] = df_temp.apply(lambda row: '_'.join(sorted([str(row['x_id']), str(row['y_id'])])),
                                                    axis=1)
            undirected_index.append(df_temp.drop_duplicates('check_string').index.values.tolist())
        else:
            # undirected
            d_off = df[df.relation == i]
            undirected_index.append(d_off[d_off.x_type == d_off.x_type.iloc[0]].index.values.tolist())
    flat_list = [item for sublist in undirected_index for item in sublist]
    df = df[df.index.isin(flat_list)]
    unique_node_types = np.unique(np.append(np.unique(df.x_type.values), np.unique(df.y_type.values)))

    df['x_idx'] = np.nan
    df['y_idx'] = np.nan
    df['x_id'] = df.x_id.apply(lambda x: convert2str(x))
    df['y_id'] = df.y_id.apply(lambda x: convert2str(x))

    idx_map = {}
    print('Iterating over node types...')
    for i in tqdm(unique_node_types):
        names = np.unique(np.append(df[df.x_type == i]['x_id'].values, df[df.y_type == i]['y_id'].values))
        names2idx = dict(zip(names, list(range(len(names)))))
        df.loc[df.x_type == i, 'x_idx'] = df[df.x_type == i]['x_id'].apply(lambda x: names2idx[x])
        df.loc[df.y_type == i, 'y_idx'] = df[df.y_type == i]['y_id'].apply(lambda x: names2idx[x])
        idx_map[i] = names2idx

    print('save kg_directed.csv...')
    df.to_csv(os.path.join(path, 'kg_directed.csv'), index=False)

def convert2str(x):
    try:
        if '_' in str(x):
            pass
        else:
            x = float(x)
    except:
        pass

    return str(x)
def create_split(df, split, disease_eval_index, split_data_path, seed):
    print('split_data_path: ', split_data_path)
    if split == 'complex_disease_cv':
        if seed < 1 or seed > 20:
            raise ValueError('Complex disease cross validation 20 folds, select seed from 1-20.')
        df_train, df_valid, df_test = create_fold_cv(df, split_num=seed, num_splits=20)
    else:
        df_train, df_valid, df_test = create_fold(df, fold_seed=seed, frac=[0.83125, 0.11875, 0.05], method=split,
                                                  disease_idx=disease_eval_index)

    unique_rel = df[['x_type', 'relation', 'y_type']].drop_duplicates()
    df_train = reverse_rel_generation(df, df_train, unique_rel)
    df_valid = reverse_rel_generation(df, df_valid, unique_rel)
    df_test = reverse_rel_generation(df, df_test, unique_rel)
    df_train.to_csv(os.path.join(split_data_path, 'train.csv'), index=False)
    df_valid.to_csv(os.path.join(split_data_path, 'valid.csv'), index=False)
    df_test.to_csv(os.path.join(split_data_path, 'test.csv'), index=False)

    return df_train, df_valid, df_test


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
        df_valid = df_valid.append(temp)
    return df_valid.reset_index(drop=True)

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
        train_index, valid_index, _ = np.split(train_index, [int(0.9 * len(train_index)), int(len(train_index))])
        split_num_idx[i + 1] = {'train': unique_diseases[train_index],
                                'valid': unique_diseases[valid_index],
                                'test': unique_diseases[test_index]
                                }

    train, valid, test = split_num_idx[split_num]['train'], split_num_idx[split_num]['valid'], split_num_idx[split_num][
        'test']
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
        test = df_temp.sample(frac=test_frac, replace=False, random_state=split_num)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac=val_frac / (1 - test_frac), replace=False, random_state=1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = df_train.append(train)
        df_valid = df_valid.append(val)
        df_test = df_test.append(test)

    df_train = pd.concat([df_train, df_dd_train])
    df_valid = pd.concat([df_valid, df_dd_valid])
    df_test = pd.concat([df_test, df_dd_test])

    return df_train.reset_index(drop=True), df_valid.reset_index(drop=True), df_test.reset_index(drop=True)

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


def random_fold(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()
    # to avoid extreme minority types don't exist in valid/test
    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac=test_frac, replace=False, random_state=fold_seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac=val_frac / (1 - test_frac), replace=False, random_state=1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = df_train.append(train)
        df_valid = df_valid.append(val)
        df_test = df_test.append(test)

    return {'train': df_train.reset_index(drop=True),
            'valid': df_valid.reset_index(drop=True),
            'test': df_test.reset_index(drop=True)}


def complex_disease_fold(df, fold_seed, frac):
    dd_rel_types = ['contraindication', 'indication', 'off-label use']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)]

    unique_diseases = df_dd.y_idx.unique()
    np.random.seed(fold_seed)
    np.random.shuffle(unique_diseases)
    train, valid, test = np.split(unique_diseases, [int(frac[0] * len(unique_diseases)),
                                                    int((frac[0] + frac[1]) * len(unique_diseases))])

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
        test = df_temp.sample(frac=test_frac, replace=False, random_state=fold_seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac=val_frac / (1 - test_frac), replace=False, random_state=1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = df_train.append(train)
        df_valid = df_valid.append(val)
        df_test = df_test.append(test)

    df_train = pd.concat([df_train, df_dd_train])
    df_valid = pd.concat([df_valid, df_dd_valid])
    df_test = pd.concat([df_test, df_dd_test])

    return {'train': df_train.reset_index(drop=True),
            'valid': df_valid.reset_index(drop=True),
            'test': df_test.reset_index(drop=True)}


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

    disease_with_less_than_3_connections_in_kg = np.array([i for i, j in disease2num_neighbors.items() if j <= 3])
    unique_diseases = df_dd.y_idx.unique()
    train_val_diseases = np.setdiff1d(unique_diseases, disease_with_less_than_3_connections_in_kg)
    test = np.intersect1d(unique_diseases, disease_with_less_than_3_connections_in_kg)
    print('Number of testing diseases: ', len(test))
    np.random.seed(fold_seed)
    np.random.shuffle(train_val_diseases)
    train, valid = np.split(train_val_diseases, [int(frac[0] * len(unique_diseases))])
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
        test = df_temp.sample(frac=test_frac, replace=False, random_state=fold_seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac=val_frac / (1 - test_frac), replace=False, random_state=1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = df_train.append(train)
        df_valid = df_valid.append(val)
        df_test = df_test.append(test)

    df_train = pd.concat([df_train, df_dd_train])
    df_valid = pd.concat([df_valid, df_dd_valid])
    df_test = pd.concat([df_test, df_dd_test])

    return {'train': df_train.reset_index(drop=True),
            'valid': df_valid.reset_index(drop=True),
            'test': df_test.reset_index(drop=True)}


def few_edeges_to_indications_fold(df, fold_seed, frac):
    dd_rel_types = ['contraindication', 'indication', 'off-label use']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)]

    disease2num_indications = dict(
        df_dd[(df_dd.y_type == 'disease') & (df_dd.relation == 'indication')].groupby('x_idx').y_id.agg(len))

    disease_with_less_than_3_indications_in_kg = np.array([i for i, j in disease2num_indications.items() if j <= 3])
    unique_diseases = df_dd.y_idx.unique()
    train_val_diseases = np.setdiff1d(unique_diseases, disease_with_less_than_3_indications_in_kg)
    test = np.intersect1d(unique_diseases, disease_with_less_than_3_indications_in_kg)
    print('Number of testing diseases: ', len(test))
    np.random.seed(fold_seed)
    np.random.shuffle(train_val_diseases)
    train, valid = np.split(train_val_diseases, [int(frac[0] * len(unique_diseases))])
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
        test = df_temp.sample(frac=test_frac, replace=False, random_state=fold_seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac=val_frac / (1 - test_frac), replace=False, random_state=1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = df_train.append(train)
        df_valid = df_valid.append(val)
        df_test = df_test.append(test)

    df_train = pd.concat([df_train, df_dd_train])
    df_valid = pd.concat([df_valid, df_dd_valid])
    df_test = pd.concat([df_test, df_dd_test])

    return {'train': df_train.reset_index(drop=True),
            'valid': df_valid.reset_index(drop=True),
            'test': df_test.reset_index(drop=True)}


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
    df_dd_val = df_dd_train_val.sample(frac=0.05, replace=False, random_state=fold_seed)
    df_dd_train = df_dd_train_val[~df_dd_train_val.index.isin(df_dd_val.index)]

    df_train = pd.concat([df_not_dd, df_dd_train])
    df_valid = df_dd_val
    df_test = df_dd_test

    # np.random.seed(fold_seed)
    # np.random.shuffle(unique_diseases)
    # train, valid = np.split(unique_diseases, int(0.95*len(unique_diseases)))
    return {'train': df_train.reset_index(drop=True),
            'valid': df_valid.reset_index(drop=True),
            'test': df_test.reset_index(drop=True)}