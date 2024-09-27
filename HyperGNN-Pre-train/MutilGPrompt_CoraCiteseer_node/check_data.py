"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/09/12 20:03
  @Email: 2665109868@qq.com
  @function
"""
import torch
idx_train = torch.load("./data/fewshot_cora/2-shot_cora/0/idx.pt").type(torch.long)
pass