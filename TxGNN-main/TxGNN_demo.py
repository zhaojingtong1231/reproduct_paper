"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/11/17 17:18
  @Email: 2665109868@qq.com
  @function
"""
from datetime import datetime
def get_time_str():
    # 获取当前日期和时间
    now = datetime.now()
    # 提取月、日、小时和分钟
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    # 将提取的信息格式化为用下划线隔开的字符串
    formatted_str = f"{month:02d}_{day:02d}_{hour:02d}_{minute:02d}"
    return formatted_str
from txgnn import TxData, TxGNN, TxEval
seed = 22
TxData = TxData(data_folder_path = '/data/zhaojingtong/PrimeKG/data_all')
TxData.prepare_split(split = 'random', seed = seed, no_kg = False)

TxGNN = TxGNN(data = TxData,
              weight_bias_track = False,
              proj_name = 'TxGNN',
              exp_name = 'TxGNN'
              )
import os
# to load a pretrained model:
# TxGNN.load_pretrained('./model_ckpt')
lr = 0.001
time_str = get_time_str()
model_save_path = '/data/zhaojingtong/PrimeKG/TxGNN'
model_save_path = os.path.join(model_save_path,
                             f"time{time_str}seed{seed}")
os.makedirs(model_save_path, exist_ok=True)
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
TxGNN.pretrain(n_epoch = 2,
               learning_rate = 1e-3,
               batch_size = 1024,
               train_print_per_n = 1000,
               save_model_path = model_save_path)
## here as a demo, the n_epoch is set to 30. Change it to n_epoch = 500 when you use it
TxGNN.finetune(n_epoch = 500,
               learning_rate = 5e-4,
               train_print_per_n = 100,
               valid_per_n = 100,)