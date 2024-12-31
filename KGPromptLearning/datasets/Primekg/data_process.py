"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/11/26 21:41
  @Email: 2665109868@qq.com
  @function
"""
from utils_all import *
import os, shutil

import pandas as pd

from utils import preprocess_kg, create_split, process_disease_area_split

import warnings

warnings.filterwarnings("ignore")


class PrimekgData:

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
                preprocess_kg(self.data_folder, split, test_size, one_hop, mask_ratio)
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

        if not os.path.exists(os.path.join(split_data_path, 'train.txt')):
            if not os.path.exists(split_data_path):
                os.mkdir(split_data_path)
            print('Creating splits... it takes several minutes...')

            df_train, df_valid, df_test = create_split(df, split, disease_eval_idx, split_data_path, seed)

        if split not in ['random', 'complex_disease', 'complex_disease_cv', 'disease_eval', 'full_graph',
                         'downstream_pred', 'few_edeges_to_indications', 'few_edeges_to_kg']:
            # in disease area split
            df_test = process_disease_area_split(self.data_folder, df, df_test, split)

        print('Done!')



def data_progress(dataset, triple_format='htr'):
    # 1. read data
    file_list = ['train', 'valid', 'test']
    triples_dict = {file: [] for file in file_list}
    for file in file_list:
        triples_dict[file] = read_triple(os.path.join('/data/zhaojingtong/PrimeKG/', dataset, file + '.txt'), triple_format=triple_format)
    all_triples = triples_dict['train'] + triples_dict['valid'] + triples_dict['test']
    all_triples = list(set(all_triples))
    entities, relations = get_entities_relations_from_triples(all_triples)
    entity2id, id2entity = set2dict(entities)
    relation2id, id2relation = set2dict(relations)
    for file in file_list:
        triples_dict[file] = triple2ids(triples_dict[file], entity2id, relation2id)

    valid_filter_triples = triples_dict['train'] + triples_dict['valid']
    test_filter_triples = triples_dict['train'] + triples_dict['valid'] + triples_dict['test']

    # 2. build cases
    kg = KG(triples_dict['train'], len(entity2id), len(relation2id))
    cases = kg.build_cases_for_large_graph(case_num=15, enclosing=False, hop=3)

    # 3. sample training_data
    train_data = kg.sample_train_data_by_relation(num=3000)

    # 4. write data
    for file in file_list:
        if not os.path.exists(os.path.join(output_dir, dataset, file)):
            os.makedirs(os.path.join(output_dir, dataset, file))
        else:
            shutil.rmtree(os.path.join(output_dir, dataset, file))
            os.makedirs(os.path.join(output_dir, dataset, file))

        # background.txt = train.txt
        if file in ['train', 'valid']:
            write_triple(os.path.join(output_dir, dataset, file, 'background.txt'), triples_dict['train'])
        else:
            write_triple(os.path.join(output_dir, dataset, file, 'background.txt'), triples_dict['train'] + triples_dict['valid'])
        # entity2id.txt = entity2id
        write_dict(os.path.join(output_dir, dataset, file, 'entity2id.txt'), entity2id)
        # relation2id.txt = relation2id
        write_dict(os.path.join(output_dir, dataset, file, 'relation2id.txt'), relation2id)
        if file == 'train':
            write_triple(os.path.join(output_dir, dataset, file, 'facts.txt'), train_data)
        else:
            write_triple(os.path.join(output_dir, dataset, file, 'facts.txt'), triples_dict[file])
        if file in ['train', 'valid']:
            write_triple(os.path.join(output_dir, dataset, file, 'filter.txt'), valid_filter_triples)
        else:
            write_triple(os.path.join(output_dir, dataset, file, 'filter.txt'), test_filter_triples)
        # store cases
        if not os.path.exists(os.path.join(output_dir, dataset, file, 'cases')):
            os.makedirs(os.path.join(output_dir, dataset, file, 'cases'))
        for relation in relation2id.keys():
            if not os.path.exists(os.path.join(output_dir, dataset, file, 'cases', relation)):
                os.makedirs(os.path.join(output_dir, dataset, file, 'cases', relation))
        for relation in relation2id.keys():
            if cases[relation2id[relation]] is None or len(cases[relation2id[relation]]) == 0:
                continue
            for i in range(len(cases[relation2id[relation]])):
                write_cases(os.path.join(output_dir, dataset, file, 'cases', relation, str(i)), cases[relation2id[relation]][i])


split = 'random'
seed = 42

Preimekg = PrimekgData(data_folder_path = '/data/zhaojingtong/PrimeKG/KGPrompt/')
Preimekg.prepare_split(split = split, seed = seed, no_kg = False)



output_dir = '/data/zhaojingtong/PrimeKG/KGPrompt/'+split+'_'+str(seed)+'/processed_data/'

dataset_dir = '/data/zhaojingtong/PrimeKG/KGPrompt/random_42'
dataset = 'KGPrompt/'+split+'_'+str(seed)+'/'
data_progress(dataset, triple_format='hrt')