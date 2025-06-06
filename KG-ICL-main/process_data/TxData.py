import os
import pandas as pd


from utils import preprocess_kg, create_split, process_disease_area_split, create_dgl_graph, evaluate_graph_construct, convert2str, data_download_wrapper

import warnings
warnings.filterwarnings("ignore")


class TxData:
    
    def __init__(self, data_folder_path):
        if not os.path.exists(data_folder_path):
            os.mkdir(data_folder_path)
            
        self.data_folder = data_folder_path # the data folder, contains the kg.csv
        data_download_wrapper('https://dataverse.harvard.edu/api/access/datafile/7144484', os.path.join(self.data_folder, 'kg.csv'))
        data_download_wrapper('https://dataverse.harvard.edu/api/access/datafile/7144482', os.path.join(self.data_folder, 'node.csv'))
        data_download_wrapper('https://dataverse.harvard.edu/api/access/datafile/7144483', os.path.join(self.data_folder, 'edges.csv'))
        
        
    def prepare_split(self, split = 'complex_disease',
                     disease_eval_idx = None,
                     seed = 42,
                     no_kg = False,
                     test_size = 0.05,
                     mask_ratio = 0.1, 
                     one_hop = False):
        
        if split not in ['random', 'complex_disease', 'complex_disease_cv', 'disease_eval', 'cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland','autoimmune', 'metabolic_disorder', 'diabetes', 'neurodigenerative', 'full_graph', 'downstream_pred', 'few_edeges_to_kg', 'few_edeges_to_indications']:
            raise ValueError("Please select one of the following supported splits: 'random', 'complex_disease', 'disease_eval', 'cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland'")
            
        if disease_eval_idx is not None:
            split = 'disease_eval'
            print('disease eval index is not none, use the individual disease split...')
        self.split = split
        
        if split in ['cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland', 'autoimmune', 'metabolic_disorder', 'diabetes', 'neurodigenerative']:
            
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
                preprocess_kg(self.data_folder, split, test_size,one_hop, mask_ratio)
                df = pd.read_csv(kg_path)
            else:
                raise ValueError("KG file path does not exist...")
        
        if split == 'disease_eval':
            split_data_path = os.path.join(self.data_folder, self.split + '_' + str(disease_eval_idx))
        elif split == 'downstream_pred':
            split_data_path = os.path.join(self.data_folder, self.split + '_downstream_pred')
            disease_eval_idx = [11394.,  6353., 12696., 14183., 12895.,  9128., 12623., 15129.,
                                   12897., 12860.,  7611., 13113.,  4029., 14906., 13438., 13177.,
                                   13335., 12896., 12879., 12909.,  4815., 12766., 12653.]
        elif no_kg:
            split_data_path = os.path.join(self.data_folder, self.split + '_no_kg_' + str(seed))
        elif test_size != 0.05:
            split_data_path = os.path.join(self.data_folder, self.split + '_' + str(seed)) + '_frac' + str(test_size)
        elif one_hop:
            split_data_path = os.path.join(self.data_folder, self.split + '_' + str(seed)) + '_one_hop_ratio' + str(mask_ratio)
        else:
            split_data_path = os.path.join(self.data_folder, self.split + '_' + str(seed))
        
        if no_kg:
            sub_kg = ['off-label use', 'indication', 'contraindication']
            df = df[df.relation.isin(sub_kg)].reset_index(drop = True)        
        
        if not os.path.exists(os.path.join(split_data_path, 'train.csv')):
            if not os.path.exists(split_data_path):
                os.mkdir(split_data_path)           
            print('Creating splits... it takes several minutes...')
            df_train, df_valid, df_test = create_split(df, split, disease_eval_idx, split_data_path, seed)
        else:
            print('Splits detected... Loading splits....')
            df_train = pd.read_csv(os.path.join(split_data_path, 'train.csv'))
            df_valid = pd.read_csv(os.path.join(split_data_path, 'valid.csv'))
            df_test = pd.read_csv(os.path.join(split_data_path, 'test.csv'))
        
        if split not in ['random', 'complex_disease', 'complex_disease_cv', 'disease_eval', 'full_graph', 'downstream_pred', 'few_edeges_to_indications', 'few_edeges_to_kg']:
            # in disease area split
            df_test = process_disease_area_split(self.data_folder, df, df_test, split)
        

        self.df, self.df_train, self.df_valid, self.df_test = df, df_train, df_valid, df_test
        self.disease_eval_idx = disease_eval_idx
        self.no_kg = no_kg
        self.seed = seed
        print('Done!')
        
        
    def retrieve_id_mapping(self):
        df = self.df
        df['x_id'] = df.x_id.apply(lambda x: convert2str(x))
        df['y_id'] = df.y_id.apply(lambda x: convert2str(x))

        idx2id_drug = dict(df[df.x_type == 'drug'][['x_idx', 'x_id']].drop_duplicates().values)
        idx2id_drug.update(dict(df[df.y_type == 'drug'][['y_idx', 'y_id']].drop_duplicates().values))

        idx2id_disease = dict(df[df.x_type == 'disease'][['x_idx', 'x_id']].drop_duplicates().values)
        idx2id_disease.update(dict(df[df.y_type == 'disease'][['y_idx', 'y_id']].drop_duplicates().values))

        df_ = pd.read_csv(os.path.join(self.data_folder, 'kg.csv'))
        df_['x_id'] = df_.x_id.apply(lambda x: convert2str(x))
        df_['y_id'] = df_.y_id.apply(lambda x: convert2str(x))

        id2name_disease = dict(df_[df_.x_type == 'disease'][['x_id', 'x_name']].drop_duplicates().values)
        id2name_disease.update(dict(df_[df_.y_type == 'disease'][['y_id', 'y_name']].drop_duplicates().values))

        id2name_drug = dict(df_[df_.x_type == 'drug'][['x_id', 'x_name']].drop_duplicates().values)
        id2name_drug.update(dict(df_[df_.y_type == 'drug'][['y_id', 'y_name']].drop_duplicates().values))
        
        return {'id2name_drug': id2name_drug,
                'id2name_disease': id2name_disease,
                'idx2id_disease': idx2id_disease,
                'idx2id_drug': idx2id_drug
               }