import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_all import *
import shutil


def data_progress(dataset, triple_format='htr'):
    # 1. read data
    file_list = ['train', 'valid', 'test']
    triples_dict = {file: [] for file in file_list}
    for file in file_list:
        triples_dict[file] = read_triple(os.path.join('fully-inductive', dataset, file + '.txt'), triple_format=triple_format)
    triples_dict['inference'] = read_triple(os.path.join('fully-inductive', dataset, 'msg' + '.txt'), triple_format=triple_format)
    all_triples_train = triples_dict['train']
    all_triples_test = triples_dict['valid'] + triples_dict['test'] + triples_dict['inference']
    all_triples_train = list(set(all_triples_train))
    all_triples_test = list(set(all_triples_test))
    entities_train, relations_train = get_entities_relations_from_triples(all_triples_train)
    entities_test, relations_test = get_entities_relations_from_triples(all_triples_test)
    entity2id_train, id2entity_train = set2dict(entities_train)
    relation2id_train, id2relation_train = set2dict(relations_train)
    relation2id_test, id2relation_test = set2dict(relations_test)
    entity2id_test, id2entity_test = set2dict(entities_test)
    for file in file_list:
        if file == 'train':
            triples_dict[file] = triple2ids(triples_dict[file], entity2id_train, relation2id_train)
        else:
            triples_dict[file] = triple2ids(triples_dict[file], entity2id_test, relation2id_test)
    triples_dict['inference'] = triple2ids(triples_dict['inference'], entity2id_test, relation2id_test)

    valid_filter_triples = triples_dict['valid'] + triples_dict['inference']
    test_filter_triples = triples_dict['inference'] + triples_dict['valid'] + triples_dict['test']

    # 2. build cases
    kg_train = KG(triples_dict['train'], len(entity2id_train), len(relation2id_train))
    kg_test = KG(triples_dict['inference']+triples_dict['valid'], len(entity2id_test), len(relation2id_test))
    cases_train = kg_train.build_cases_for_large_graph(case_num=25, enclosing=False, hop=3)
    cases_test = kg_test.build_cases_for_large_graph(case_num=25, enclosing=False, hop=3)

    # 3. sample training_data
    train_data = kg_train.sample_train_data_by_relation(num=200)

    # 4. write data
    for file in file_list:
        if not os.path.exists(os.path.join(output_dir, dataset, file)):
            os.makedirs(os.path.join(output_dir, dataset, file))
        else:
            shutil.rmtree(os.path.join(output_dir, dataset, file))
            os.makedirs(os.path.join(output_dir, dataset, file))
        if file == 'train':
            write_triple(os.path.join(output_dir, dataset, file, 'background.txt'), triples_dict['train'])
            write_dict(os.path.join(output_dir, dataset, file, 'entity2id.txt'), entity2id_train)
            write_dict(os.path.join(output_dir, dataset, file, 'relation2id.txt'), relation2id_train)
        else:
            if file == 'valid':
                write_triple(os.path.join(output_dir, dataset, file, 'background.txt'), triples_dict['inference'])
            else:
                write_triple(os.path.join(output_dir, dataset, file, 'background.txt'), triples_dict['inference'] + triples_dict['valid'])
            write_dict(os.path.join(output_dir, dataset, file, 'entity2id.txt'), entity2id_test)
            write_dict(os.path.join(output_dir, dataset, file, 'relation2id.txt'), relation2id_test)

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
        if file == 'train':
            for relation in relation2id_train.keys():
                if not os.path.exists(os.path.join(output_dir, dataset, file, 'cases', relation)):
                    os.makedirs(os.path.join(output_dir, dataset, file, 'cases', relation))
            for relation in relation2id_train.keys():
                if cases_train[relation2id_train[relation]] is None or len(cases_train[relation2id_train[relation]]) == 0:
                    continue
                for i in range(len(cases_train[relation2id_train[relation]])):
                    write_cases(os.path.join(output_dir, dataset, file, 'cases', relation, str(i)), cases_train[relation2id_train[relation]][i])
        else:
            for relation in relation2id_test.keys():
                if not os.path.exists(os.path.join(output_dir, dataset, file, 'cases', relation)):
                    os.makedirs(os.path.join(output_dir, dataset, file, 'cases', relation))
            for relation in relation2id_test.keys():
                if cases_test[relation2id_test[relation]] is None or len(cases_test[relation2id_test[relation]]) == 0:
                    continue
                for i in range(len(cases_test[relation2id_test[relation]])):
                    write_cases(os.path.join(output_dir, dataset, file, 'cases', relation, str(i)), cases_test[relation2id_test[relation]][i])

dataset_list = ['FB-25', 'FB-50', 'FB-75', 'FB-100', 'NL-0', 'NL-25', 'NL-50', 'NL-75', 'NL-100', 'WK-25', 'WK-50', 'WK-75', 'WK-100']

output_dir = '../processed_data/'
for dataset in dataset_list:
    print(dataset)
    data_progress(dataset, triple_format='hrt')


