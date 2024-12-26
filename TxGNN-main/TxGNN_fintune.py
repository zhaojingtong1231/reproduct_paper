"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/11/18 20:55
  @Email: 2665109868@qq.com
  @function
"""
from txgnn import TxData, TxGNN
import gc  # 引入垃圾回收模块
import os
import re
import glob
# 设置根目录路径
base_dir = "/data/zhaojingtong/PrimeKG/TxGNN"
TxData = TxData(data_folder_path='/data/zhaojingtong/PrimeKG/data_all')

# 遍历 TxGNN 目录下所有子目录中的 model.pth 文件
for model_path in glob.glob(os.path.join(base_dir, "**/model.pth"), recursive=True):
    print(f"Processing: {model_path}")
    model_path = model_path
    save_result_path = os.path.dirname(model_path)
    match = re.search(r"split(.*?)_t", model_path)
    split = match.group(1)
    if split == 'random':
        seed = int(save_result_path[-2:])
    else:
        seed = int(save_result_path[-1:])

    TxData.prepare_split(split=split, seed=seed, no_kg=False)

    # 创建 TxGNN 实例
    TxGNN_instance = TxGNN(
        data=TxData,
        weight_bias_track=False,
        proj_name='TxGNN',
        exp_name='TxGNN'
    )
    TxGNN_instance.model_initialize(
        n_hid=512,
        n_inp=512,
        n_out=512,
        proto=True,
        proto_num=3,
        attention=False,
        sim_measure='all_nodes_profile',
        bert_measure='disease_name',
        agg_measure='rarity',
        num_walks=200,
        walk_mode='bit',
        path_length=2
    )

    TxGNN_instance.finetune(
        n_epoch=1000,
        learning_rate=5e-4,
        train_print_per_n=500,
        valid_per_n=500,
        model_path=model_path,
        save_result_path=save_result_path
    )

    # 删除实例并清理内存
    del TxGNN_instance
    gc.collect()  # 垃圾回收
