import networkx as nx

from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData
import torch


# from utils import *
# from .utils import normalize
import pandas as pd
import csv
import random
from tqdm import tqdm
import copy
import scipy.sparse as sp

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
import numpy as np
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=True)


class Data_class(Dataset):

    def __init__(self, triple):
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.relationtype=triple[:,2]
        #self.label = triple[:, 3]

    def __len__(self):
        return len(self.relationtype)

    def __getitem__(self, index):


        return  (self.entity1[index], self.entity2[index], self.relationtype[index])


def load_big_data(zhongzi,batch,workers):
    """Read small_data from path, convert small_data into loader, return features and symmetric adjacency"""
    # read small_data

    drug_list = []
    with open('big_data/drug_smiles.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            drug_list.append(row[0])

    # print(len(drug_list))

    zhongzi=zhongzi

    def loadtrainvaltest():
        #train dataset
        train=pd.read_csv('big_data/'+str(zhongzi)+'/ddi_training1.csv')
        train_pos=[(h, t, r) for h, t, r in zip(train['d1'], train['d2'], train['type'])]
        np.random.shuffle(train_pos)
        train_pos = np.array(train_pos)
        for i in range(train_pos.shape[0]):
            train_pos[i][0] = int(drug_list.index(train_pos[i][0]))
            train_pos[i][1] = int(drug_list.index(train_pos[i][1]))
            train_pos[i][2] = int(train_pos[i][2])
        label_list=[]
        for i in range(train_pos.shape[0]):
            label=np.zeros((86))
            label[int(train_pos[i][2])]=1
            label_list.append(label)
        label_list=np.array(label_list)
        train_data= np.concatenate([train_pos, label_list],axis=1)

        #val dataset
        val = pd.read_csv('big_data/'+str(zhongzi)+'/ddi_validation1.csv')
        val_pos = [(h, t, r) for h, t, r in zip(val['d1'], val['d2'], val['type'])]
        np.random.shuffle(val_pos)
        val_pos= np.array(val_pos)
        for i in range(len(val_pos)):
            val_pos[i][0] = int(drug_list.index(val_pos[i][0]))
            val_pos[i][1] = int(drug_list.index(val_pos[i][1]))
            val_pos[i][2] = int(val_pos[i][2])
        label_list = []
        for i in range(val_pos.shape[0]):
            label = np.zeros((86))
            label[int(val_pos[i][2])] = 1
            label_list.append(label)
        label_list = np.array(label_list)
        val_data = np.concatenate([val_pos, label_list], axis=1)

        #test dataset
        test = pd.read_csv('big_data/'+str(zhongzi)+'/ddi_test1.csv')
        test_pos = [(h, t, r) for h, t, r in zip(test['d1'],test['d2'], test['type'])]
        np.random.shuffle(test_pos)
        test_pos= np.array(test_pos)
        # print(test_pos[0])
        for i in range(len(test_pos)):
            test_pos[i][0] = int(drug_list.index(test_pos[i][0]))
            test_pos[i][1] = int(drug_list.index(test_pos[i][1]))
            test_pos[i][2] = int(test_pos[i][2])
        label_list = []
        for i in range(len(test_pos)):
            label = np.zeros(86)
            label[int(test_pos[i][2])] = 1
            label_list.append(label)
        label_list = np.array(label_list)
        test_data = np.concatenate([test_pos, label_list], axis=1)
        # print(train_data.shape)
        # print(val_data.shape)
        # print(test_data.shape)
        return train_data,val_data,test_data

    train_data,val_data,test_data=loadtrainvaltest()
    params = {'batch_size': batch, 'shuffle': False, 'num_workers': workers, 'drop_last': False}

    training_set = Data_class(train_data)
    train_loader = DataLoader(training_set, **params)

    validation_set = Data_class(val_data)

    val_loader = DataLoader(validation_set, **params)


    test_set = Data_class(test_data)

    test_loader = DataLoader(test_set, **params)

    # print('Extracting features...')

    features = np.load('trimnet_big/drug_emb_trimnet'+str(zhongzi)+'.npy')
    # features = np.load('/project/reproduct_paper/NCLA-main/MRCGNN_code/drug_emb_trimnet'+str(zhongzi)+'.npy')
    ids = np.load('trimnet_big/drug_idsbig.npy')
    ids=ids.tolist()
    features1=[]
    for i in range(len(drug_list)):
        features1.append(features[ids.index(drug_list[i])])
    print(len(features1))
    print()
    features=np.array(features1)
    features_o = normalize(features)

    dimensions = features_o.shape[1]

    # adversarial nodes
    id = np.arange(features_o.shape[0])

    x_o = torch.tensor(features_o, dtype=torch.float)
    positive1=copy.deepcopy(train_data)

    edge_index_o = []
    label_list = []
    label_list11 = []
    for i in range(positive1.shape[0]):

    #for h, t, r ,label in positive1:
        a = []
        a.append(int(positive1[i][0]))
        a.append(int(positive1[i][1]))
        edge_index_o.append(a)
        label_list.append(int(positive1[i][2]))
        a = []
        a.append(int(positive1[i][1]))
        a.append(int(positive1[i][0]))
        edge_index_o.append(a)
        label_list.append(int(positive1[i][2]))
        b = []
        b.append(int(positive1[i][2]))
        b.append(int(positive1[i][2]))
        label_list11.append(b)

    edge_index_o = torch.tensor(edge_index_o, dtype=torch.long)

    data_o = Data(x=x_o, edge_index=edge_index_o.t().contiguous(), edge_type=label_list)

    # print('Loading finished!')
    return data_o, train_loader, val_loader, test_loader
