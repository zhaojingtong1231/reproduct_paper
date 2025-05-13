"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2025/04/06 20:21
  @Email: 2665109868@qq.com
  @function
"""


from txgnn import TxData, TxGNN, TxEval


# Download/load knowledge graph dataset
TxData = TxData(data_folder_path = './data')
TxData.prepare_split(split = 'complex_disease', seed = 42)
TxGNN = TxGNN(data = TxData,
              weight_bias_track = False,
              proj_name = 'TxGNN', # wandb project name
              exp_name = 'TxGNN', # wandb experiment name
              device = 'cuda:1' # define your cuda device
              )


TxGNN.load_pretrained('/data/zhaojingtong/reproduct_paper/TxGNN-main/data/model.pt')

TxGNN.train_graphmask(relation = 'indication',
                      learning_rate = 3e-4,
                      allowance = 0.005,
                      epochs_per_layer = 3,
                      penalty_scaling = 1,
                      valid_per_n = 20)