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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TxGNN_prompt')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--num-hidden", type=int, default=512,
                        help="number of hidden units")
    parser.add_argument("--epochs", type=int, default=12,
                        help="number of pretrain.sh epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="loader small_data")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")

    args = parser.parse_args()
    gpu_id = args.gpu # 选择你想使用的 GPU ID，例如 0, 1, 2 等
    hidden_dim = args.num_hidden
    seed = args.seed
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    time_str = get_time_str()
    model_save_path = '/data/zhaojingtong/PrimeKG/model'
    model_save_path = os.path.join(model_save_path,
                             f"lr{lr}_batch{batch_size}_epochs{epochs}_hidden{hidden_dim}_time{time_str}_seed{seed}")
    pretrain_model_path = '/data/zhaojingtong/PrimeKG/model/model39_3000.pth'

    # os.makedirs(model_save_path, exist_ok=True)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    TxData = TxData(data_folder_path = '/data/zhaojingtong/PrimeKG/data_all')
    TxData.prepare_split(split = 'random', seed = seed, no_kg = False)

    TxGNN = TxGNNPrompt(data = TxData,
                  weight_bias_track = False,
                  proj_name = 'TxGNNPrompt',
                  exp_name = 'TxGNNPrompt',
                  device=device)

    # to load a pretrained model:
    # TxGNN.load_pretrained('./model_ckpt')

    TxGNN.model_initialize(n_hid = hidden_dim,
                          n_inp = hidden_dim,
                          n_out = hidden_dim,
                          proto = True,
                          proto_num = 3,
                          attention = False,
                          sim_measure = 'all_nodes_profile_more',
                          bert_measure = 'disease_name',
                          agg_measure = 'rarity',
                          num_walks = 200,
                          walk_mode = 'bit',
                          path_length = 2)
    # here we did not run this, since the output is too long to fit into the notebook
    TxGNN.pretrain(n_epoch = epochs,
                   learning_rate = lr,
                   batch_size = batch_size,
                   train_print_per_n = 1000,
                   save_model_path = model_save_path)

    # TxGNN.finetune(n_epoch = 5000,
    #                learning_rate = 1e-3,
    #                train_print_per_n = 100,
    #                valid_per_n = 100,
    #                model_save_path = pretrain_model_path)