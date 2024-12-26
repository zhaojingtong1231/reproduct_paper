"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/10/25 18:40
  @Email: 2665109868@qq.com
  @function
"""
from txgnn import TxData, TxGNNPrompt, TxEval
import torch
import argparse
import os
from txgnn.utils import get_time_str
import re
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TxGNN_prompt')
    parser.add_argument("--gpu", type=int, default=1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--num-hidden", type=int, default=512,
                        help="number of hidden units")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of pretrain.sh epochs")
    parser.add_argument("--seed", type=int, default=12,
                        help="random seed")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="loader small_data")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument(
        "--split",
        type=str,
        default="random",
        choices=['random', 'complex_disease', 'disease_eval', 'cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland','autoimmune', 'metabolic_disorder', 'diabetes', 'neurodigenerative', 'full_graph', 'downstream_pred', 'few_edeges_to_kg', 'few_edeges_to_indications'],  # 指定合法的候选项
        help="Choose the data split type"
    )
    model_path = '/data/zhaojingtong/PrimeKG/our/random/lr0.001_batch1024_epochs30_hidden512_splitrandom_time12_22_16_56_seed12/model4_0.pth'
    save_result_path = os.path.dirname(model_path)
    seed = int(save_result_path[-2:])

    match = re.search(r'split([a-zA-Z]+)_', save_result_path)
    split = match.group(1)

    args = parser.parse_args()
    gpu_id = args.gpu # 选择你想使用的 GPU ID，例如 0, 1, 2 等
    hidden_dim = args.num_hidden
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    time_str = get_time_str()


    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    TxData = TxData(data_folder_path = '/data/zhaojingtong/PrimeKG/data_all')
    TxData.prepare_split(split = split, seed = seed, no_kg = False)

    TxGNN = TxGNNPrompt(data = TxData,
                  weight_bias_track = False,
                  proj_name = 'TxGNNPrompt',
                  exp_name = 'TxGNNPrompt',
                  device=device)

    TxGNN.model_initialize(n_hid = hidden_dim,
                          n_inp = hidden_dim,
                          n_out = hidden_dim,
                          proto = True,
                          proto_num = 3,
                          sim_measure = 'protein_profile',
                          bert_measure = 'disease_name',
                          agg_measure = 'rarity',
                          num_walks = 200,
                          walk_mode = 'bit',
                          path_length = 2)

    TxGNN.finetune(n_epoch = 1000,
                   learning_rate = 5e-4,
                   train_print_per_n = 100,
                   valid_per_n = 100,
                   model_save_path = model_path,save_result_path =save_result_path)