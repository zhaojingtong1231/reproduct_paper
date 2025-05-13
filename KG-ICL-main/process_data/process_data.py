"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2025/05/12 17:02
  @Email: 2665109868@qq.com
  @function
"""

import TxData
from TxData import *
import torch
import argparse
import os


if __name__ == '__main__':
    seed = 101
    txData = TxData(data_folder_path = '/data/zhaojingtong/PrimeKG/data_all')
    txData.prepare_split(split = 'random', seed = seed, no_kg = False)
