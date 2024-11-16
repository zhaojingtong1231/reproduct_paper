"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/10/25 18:40
  @Email: 2665109868@qq.com
  @function
"""
from txgnn import TxData, TxGNNPrompt, TxEval
import torch
gpu_id = 1 # 选择你想使用的 GPU ID，例如 0, 1, 2 等
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
TxData = TxData(data_folder_path = '/data/zhaojingtong/PrimeKG/data_all/')
TxData.prepare_split(split = 'random', seed = 42, no_kg = False)

TxGNN = TxGNNPrompt(data = TxData,
              weight_bias_track = False,
              proj_name = 'TxGNNPrompt',
              exp_name = 'TxGNNPrompt',
              device=device

              )

# to load a pretrained model:
# TxGNN.load_pretrained('./model_ckpt')

TxGNN.model_initialize(n_hid = 512,
                      n_inp = 512,
                      n_out = 512,
                      proto = True,
                      proto_num = 3,
                      attention = False,
                      sim_measure = 'all_nodes_profile',
                      bert_measure = 'disease_name',
                      agg_measure = 'rarity',
                      num_walks = 200,
                      walk_mode = 'bit',
                      path_length = 2)
## here we did not run this, since the output is too long to fit into the notebook
TxGNN.pretrain(n_epoch = 10,
               learning_rate = 1e-3,
               batch_size = 1024,
               train_print_per_n = 50)
TxGNN.finetune(n_epoch = 1000,
               learning_rate = 5e-4,
               train_print_per_n = 5,
               valid_per_n = 20)