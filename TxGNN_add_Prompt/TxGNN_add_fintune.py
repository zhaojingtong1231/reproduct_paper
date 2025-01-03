"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/10/25 18:40
  @Email: 2665109868@qq.com
  @function
"""

import torch
from txgnn import TxData, TxGNNPrompt
import gc  # 引入垃圾回收模块
import os
import re
import glob
import argparse

parser = argparse.ArgumentParser(description='TxGNN_prompt')
parser.add_argument(
    "--model",
    type=str
)
args = parser.parse_args()
model_path = args.model
# 设置根目录路径
base_dir = "/data/zhaojingtong/PrimeKG/our"
TxData = TxData(data_folder_path='/data/zhaojingtong/PrimeKG/data_all')

print(model_path)
save_result_path = os.path.dirname(model_path)
match = re.search(r"split(.*?)_t", model_path)
split = match.group(1)
if split == 'random':
    seed = int(save_result_path[-2:])
else:
    seed = int(save_result_path[-1:])

TxData.prepare_split(split=split, seed=seed, no_kg=False)
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
# 创建 TxGNN 实例
TxGNN_instance = TxGNNPrompt(data=TxData,
                    weight_bias_track=False,
                    proj_name='TxGNNPrompt',
                    exp_name='TxGNNPrompt',
                    device=device)

TxGNN_instance.model_initialize(
    n_hid=512,
    n_inp=512,
    n_out=512,
    proto=True,
    proto_num=3,
    sim_measure='all_nodes_profile',
    bert_measure='disease_name',
    agg_measure='rarity',
    num_walks=200,
    walk_mode='bit',
    path_length=2
)

TxGNN_instance.finetune(
    n_epoch=1,
    learning_rate=5e-4,
    train_print_per_n=1500,
    valid_per_n=1500,
    model_path=model_path,
    save_result_path=save_result_path
)


