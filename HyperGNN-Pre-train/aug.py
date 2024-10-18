import torch
import copy
import random
import pdb
import scipy.sparse as sp
import numpy as np

def main():
    pass


import torch
import random
import copy
import numpy as np
import scipy.sparse as sp

def generate_hetero_shuf_features_and_labels(data):
    neg_data = copy.deepcopy(data)
    # 存放不同节点类型的正负样本特征及标签
    pos_features = {}
    neg_features = {}
    labels = {}

    # 遍历 HeteroData 中的所有节点类型
    for node_type in data.node_types:
        # 提取节点特征
        features = data[node_type].x  # 节点特征张量，形状 [num_nodes, feature_dim]
        nb_nodes = features.size(0)          # 节点数量

        # 生成随机打乱的负样本特征
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[idx, :]  # 随机打乱后的特征

        # 生成正负样本特征字典
        neg_features[node_type] = shuf_fts

        # 创建正负样本标签
        lbl_1 = torch.ones(1,nb_nodes)  # 正样本标签，全 1
        lbl_2 = torch.zeros(1,nb_nodes) # 负样本标签，全 0

        # 合并正负样本标签
        labels[node_type] = torch.cat((lbl_1, lbl_2), 1)  # 拼接标签
    neg_data.x_dict = neg_features

    return  neg_data, labels
def heterodata_preprocess_features(data):
    hererodata_feature_dict = {}
    input_feature_dict = data.x_dict
    new_data = copy.deepcopy(data)
    for node_type, input_feature in input_feature_dict.items():
        rowsum = np.array(input_feature.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        fearture = r_mat_inv.dot(input_feature)

        # 将掩码后的特征矩阵存入结果字典
        hererodata_feature_dict[node_type] = torch.FloatTensor(fearture)

    new_data.x_dict = hererodata_feature_dict
    return new_data

def aug_random_edge_edge_index(edge_index, drop_percent=0.2):
    edge_num = edge_index.shape[1]
    percent = drop_percent / 2
    add_drop_num = int(edge_num * percent)

    src_nodes = edge_index[0].unique().tolist()  # 获取唯一的源节点列表
    dst_nodes = edge_index[1].unique().tolist()  # 获取唯一的目标节点列表
    edge_list = edge_index.t().tolist()  # 转换为二维列表

    drop_idx = random.sample(range(edge_num), add_drop_num)  # 随机选择要删除的边的索引
    drop_idx = sorted(drop_idx, reverse=True)

    # 删除选中的边
    # edge_list = [edge_list[i] for i in range(edge_num) if i not in drop_idx]
    for i in drop_idx:
        edge_list.pop(i)

    # 转换现有的边为集合，便于快速查找
    existing_edges = set(map(tuple, edge_list))

    # 优化：从源节点和目标节点中随机采样未存在的边
    add_list = []
    attempts = 0
    max_attempts = 10 * add_drop_num  # 为防止死循环，设置最大尝试次数

    while len(add_list) < add_drop_num and attempts < max_attempts:
        src = random.choice(src_nodes)  # 随机选择一个源节点
        dst = random.choice(dst_nodes)  # 随机选择一个目标节点
        new_edge = (src, dst)
        if new_edge not in existing_edges:
            add_list.append(new_edge)
            existing_edges.add(new_edge)  # 更新现有边集合
        attempts += 1

    # 如果采样边数不足，可以通过再次采样或终止采样
    if len(add_list) < add_drop_num:
        print(f"Warning: Only {len(add_list)} new edges were added out of {add_drop_num}.")

    # 增加新边
    edge_list.extend(add_list)
    augmented_edge_index = torch.tensor(edge_list).t()

    return augmented_edge_index


def aug_heterodata_random_edge_edge_index(hetero_data, drop_percent=0.2):
    for key in hetero_data.edge_types:
        print('边数据增强')
        print(key)
        edge_index = hetero_data[key]['edge_index']
        augmented_edge_index = aug_random_edge_edge_index(edge_index, drop_percent)
        hetero_data[key]['edge_index'] = augmented_edge_index  # 更新增强后的边索引

    return hetero_data


import torch
import copy
import random


def aug_heterodata_random_mask(data, drop_percent=0.2):
    aug_feature_dict = {}
    input_feature_dict = data.x_dict
    new_data = copy.deepcopy(data)
    for node_type, input_feature in input_feature_dict.items():
        node_num = input_feature.shape[1]  # 获取当前节点类型的节点数
        mask_num = int(node_num * drop_percent)
        node_idx = [i for i in range(node_num)]
        mask_idx = random.sample(node_idx, mask_num)  # 随机选择掩码节点
        input_feature = input_feature.unsqueeze(0)
        # 深拷贝当前节点类型的特征
        aug_feature = copy.deepcopy(input_feature)

        # 生成全零向量，与节点特征的维度一致
        zeros = torch.zeros_like(aug_feature[0][0])

        # 进行掩码操作
        for j in mask_idx:
            aug_feature[0][j] = zeros

        # 将掩码后的特征矩阵存入结果字典
        aug_feature_dict[node_type] = aug_feature[0]
    new_data.x_dict = aug_feature_dict
    return new_data


def aug_random_mask_hetero(input_features_dict, drop_percent=0.2):
    # input_features_dict: 是一个字典，key 是节点类型，value 是节点特征
    aug_features_dict = {}
    for node_type, input_feature in input_features_dict.items():
        node_num = input_feature.shape[0]
        mask_num = int(node_num * drop_percent)
        mask_idx = random.sample(range(node_num), mask_num)
        aug_feature = copy.deepcopy(input_feature)
        aug_feature[mask_idx] = torch.zeros_like(aug_feature[0])  # 将这些节点的特征置为 0
        aug_features_dict[node_type] = aug_feature
    return aug_features_dict


def aug_random_edge_hetero(edge_index_dict, drop_percent=0.2):
    # edge_index_dict: 是一个字典，key 是边的类型 (src_node_type, edge_type, tgt_node_type)
    # value 是对应的边索引矩阵
    aug_edge_index_dict = {}
    for edge_type, edge_index in edge_index_dict.items():
        src, tgt = edge_index
        edge_num = src.shape[0]
        drop_num = int(edge_num * drop_percent / 2)

        # 删除边
        drop_idx = random.sample(range(edge_num), drop_num)
        aug_src = np.delete(src.cpu().numpy(), drop_idx)
        aug_tgt = np.delete(tgt.cpu().numpy(), drop_idx)

        # 添加随机边
        num_nodes_src = max(src) + 1  # 假设节点是从 0 编号的
        num_nodes_tgt = max(tgt) + 1
        new_edges_src = np.random.randint(0, num_nodes_src, drop_num)
        new_edges_tgt = np.random.randint(0, num_nodes_tgt, drop_num)

        aug_src = np.concatenate((aug_src, new_edges_src))
        aug_tgt = np.concatenate((aug_tgt, new_edges_tgt))

        aug_edge_index_dict[edge_type] = torch.tensor([aug_src, aug_tgt], dtype=torch.long)

    return aug_edge_index_dict


def aug_drop_node_hetero(input_features_dict, edge_index_dict, drop_percent=0.2):
    # 删除某种类型的节点
    aug_features_dict = {}
    aug_edge_index_dict = {}

    for node_type, input_feature in input_features_dict.items():
        node_num = input_feature.shape[0]
        drop_num = int(node_num * drop_percent)
        drop_idx = sorted(random.sample(range(node_num), drop_num))

        aug_features_dict[node_type] = delete_row_col(input_feature, drop_idx, only_row=True)

    # 更新边，删除与丢弃节点相关的边
    for edge_type, edge_index in edge_index_dict.items():
        src, tgt = edge_index
        src_node_type, _, tgt_node_type = edge_type
        if src_node_type in aug_features_dict and tgt_node_type in aug_features_dict:
            keep_src = [i for i in range(len(src)) if src[i] not in drop_idx]
            keep_tgt = [i for i in range(len(tgt)) if tgt[i] not in drop_idx]
            keep_idx = list(set(keep_src).intersection(set(keep_tgt)))
            aug_edge_index_dict[edge_type] = edge_index[:, keep_idx]

    return aug_features_dict, aug_edge_index_dict


def aug_subgraph_hetero(input_features_dict, edge_index_dict, drop_percent=0.2):
    aug_features_dict = {}
    aug_edge_index_dict = {}

    for node_type, input_feature in input_features_dict.items():
        node_num = input_feature.shape[0]
        s_node_num = int(node_num * (1 - drop_percent))
        center_node_id = random.randint(0, node_num - 1)

        sub_node_id_list = [center_node_id]
        all_neighbor_list = []

        for i in range(s_node_num - 1):
            for edge_type, edge_index in edge_index_dict.items():
                if edge_type[0] == node_type:
                    all_neighbor_list += torch.nonzero(edge_index[0] == sub_node_id_list[i], as_tuple=False).squeeze(
                        1).tolist()
                elif edge_type[2] == node_type:
                    all_neighbor_list += torch.nonzero(edge_index[1] == sub_node_id_list[i], as_tuple=False).squeeze(
                        1).tolist()

            all_neighbor_list = list(set(all_neighbor_list))
            new_neighbor_list = [n for n in all_neighbor_list if n not in sub_node_id_list]
            if new_neighbor_list:
                new_node = random.sample(new_neighbor_list, 1)[0]
                sub_node_id_list.append(new_node)
            else:
                break

        drop_node_list = sorted([i for i in range(node_num) if i not in sub_node_id_list])
        aug_features_dict[node_type] = delete_row_col(input_feature, drop_node_list, only_row=True)

    # 更新边
    for edge_type, edge_index in edge_index_dict.items():
        src, tgt = edge_index
        src_node_type, _, tgt_node_type = edge_type
        if src_node_type in aug_features_dict and tgt_node_type in aug_features_dict:
            keep_idx = [i for i in range(len(src)) if src[i] in sub_node_id_list and tgt[i] in sub_node_id_list]
            aug_edge_index_dict[edge_type] = edge_index[:, keep_idx]

    return aug_features_dict, aug_edge_index_dict


def delete_row_col(input_matrix, drop_list, only_row=False):

    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out




if __name__ == "__main__":
    main()
    
