"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/07/29 15:54
  @Email: 2665109868@qq.com
  @function
"""
import gzip
import pandas as pd
import shutil
source_id = 2
target_ids = [31, 14,  3, 17, 47, 37]
id_name_map = {31: 'bindingdb', 14: 'faers', 3: 'pdb', 17: 'pharmgkb', 47: 'rxnorm', 37: 'brenda'}

filename_template = "drugbank_to_{db_name}.txt.gz"
# 生成URL列表
filenames = [filename_template.format(db_name=id_name_map[target_id]) for target_id in target_ids]

filenames.append('chembl_to_drugbank.txt.gz')

names1 = ['drugbank','target_db1']
names2 = ['drugbank','target_db2']
for filename in filenames:
    print(filename)
    try:
        if filename == 'chembl_to_drugbank.txt.gz':
            with gzip.open('./drug_mapping_data/drugbank_to_chembl.txt.gz', 'rt') as f:

                mapping_data = pd.read_csv(f, delimiter='\t', header=None,names=names1)
            with gzip.open('./drug_mapping_supplement_data/' + filename, 'rt') as f:
                supplement_data = pd.read_csv(f, delimiter='\t', header=1, names=names2)
                supplement_data.iloc[:, [0, 1]] = supplement_data.iloc[:, [1, 0]].values
        else:
            with gzip.open('./drug_mapping_data/' + filename, 'rt') as f:

                mapping_data = pd.read_csv(f, delimiter='\t', header=None, names=names1)

            with gzip.open('./drug_mapping_supplement_data/' + filename, 'rt') as f:
                supplement_data = pd.read_csv(f, delimiter='\t', header=1, names=names2)

    except FileNotFoundError:
        print(filename + ' not found.')
        continue
    print(mapping_data[mapping_data['target_db1'] != '-'].shape)
    # 使用gzip模块解压缩并读取文件

    merged_data = pd.merge(mapping_data, supplement_data, on='drugbank', how='left')
    merged_data.fillna(-1, inplace=True)
    if filename == 'drugbank_to_bindingdb.txt.gz':
        merged_data['target_db2'] = merged_data['target_db2'].astype(int)
    merged_data['target_db1'] = merged_data.apply(lambda row: row['target_db2'] if row['target_db1'] == '-' and row['target_db2']!=-1 else row['target_db1'], axis=1)
    end_merged_data  = merged_data[['drugbank','target_db1']]
    end_merged_data = end_merged_data.drop_duplicates(['drugbank'])

    output_filename = f'./drug_merge_mapping_data/{filename}'
    if filename == 'chembl_to_drugbank.txt.gz':
        output_filename = f'./drug_merge_mapping_data/drugbank_to_chembl.txt.gz'
        with gzip.open(output_filename, 'wt', encoding='utf-8') as f:
            end_merged_data.to_csv(f, sep='\t', index=False, header=False)
    else:
        with gzip.open(output_filename, 'wt', encoding='utf-8') as f:
            end_merged_data.to_csv(f, sep='\t', index=False, header=False)
    # print(end_merged_data[end_merged_data['target_db1'] != '-'].shape)