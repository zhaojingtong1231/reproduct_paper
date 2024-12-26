"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/11/17 17:18
  @Email: 2665109868@qq.com
  @function
"""
import argparse
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


parser = argparse.ArgumentParser(description='TxGNN_prompt')
parser.add_argument(
    "--split",
    type=str,
    default="random",
    choices=['random', 'complex_disease', 'disease_eval', 'cell_proliferation', 'mental_health', 'cardiovascular',
             'anemia', 'adrenal_gland', 'autoimmune', 'metabolic_disorder', 'diabetes', 'neurodigenerative',
             'full_graph', 'downstream_pred', 'few_edeges_to_kg', 'few_edeges_to_indications'],  # 指定合法的候选项
    help="Choose the data split type"
)
parser.add_argument("--seed", type=int, default=12,
                    help="random seed")
args = parser.parse_args()
split = args.split
seed = args.seed
TxData = TxData(data_folder_path = '/data/zhaojingtong/PrimeKG/data_all')
TxData.prepare_split(split = split, seed = seed, no_kg = False)

TxGNN = TxGNN(data = TxData,
              weight_bias_track = False,
              proj_name = 'TxGNN',
              exp_name = 'TxGNN'
              )
import os

lr = 0.001
time_str = get_time_str()
model_save_path = '/data/zhaojingtong/PrimeKG/TxGNN'
model_save_path = os.path.join(model_save_path,
                               f"lr{lr}_batch{1024}_epochs{2}_hidden{512}_split{split}_time{time_str}_seed{seed}")
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

