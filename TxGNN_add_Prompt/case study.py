"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2025/01/17 0:02
  @Email: 2665109868@qq.com
  @function
"""
from txgnn import TxEval
import torch
from txgnn import TxData, TxGNNPrompt, TxEval
TxData = TxData(data_folder_path='/data/zhaojingtong/PrimeKG/data_all')
TxData.prepare_split(split='random', seed=12, no_kg=False)
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

TxGNN = TxGNNPrompt(data=TxData,
                    weight_bias_track=False,
                    proj_name='TxGNNPrompt',
                    exp_name='TxGNNPrompt',
                    device=device)

TxGNN.model_initialize(n_hid=512,
                       n_inp=512,
                       n_out=512,
                       proto=True,
                       proto_num=3,
                       sim_measure='all_nodes_profile',
                       bert_measure='disease_name',
                       agg_measure='rarity',
                       num_walks=200,
                       walk_mode='bit',
                       path_length=2)

TxGNN.load_pretrained('/data/zhaojingtong/PrimeKG/our/random/lr0.001_batch2048_epochs10_hidden512_splitrandom_time12_24_10_46_seed22/fintune_model.pth')
TxEval = TxEval(model = TxGNN)

result = TxEval.eval_disease_centric(disease_idxs = 'test_set',
                                     show_plot = False,
                                     verbose = True,
                                     save_result = True,
                                     return_raw = False)
print(result)