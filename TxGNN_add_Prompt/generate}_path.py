"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2025/01/01 21:00
  @Email: 2665109868@qq.com
  @function
"""
import torch
from txgnn import TxData, TxGNNPrompt
import gc  # 引入垃圾回收模块
import os
import re
import glob
# 设置根目录路径
base_dir = "/data/zhaojingtong/PrimeKG/our"
TxData = TxData(data_folder_path='/data/zhaojingtong/PrimeKG/data_all')
# 存储模型路径的列表
model_paths = []
# 遍历 TxGNN 目录下所有子目录中的 model.pth 文件
for subfolder1 in glob.glob(os.path.join(base_dir, "*")):
    # 遍历每个子文件夹
    for subfolder in glob.glob(os.path.join(subfolder1, "*")):
        if os.path.isdir(subfolder):
            # 初始化最大数字和对应的文件路径
            max_num = -1
            best_model_path = None

            # 遍历当前子文件夹下的所有模型文件
            for model_path in glob.glob(os.path.join(subfolder, "**/model*.pth"), recursive=True):
                # 提取文件名中的数字部分
                filename = os.path.basename(model_path)

                # 使用正则表达式提取文件名中的所有数字
                try:
                    num_parts = [int(part) for part in filter(str.isdigit, filename)]
                    if len(num_parts) == 2:  # 假设文件名格式为 modelX_Y.pth
                        # 先比较数字的高位（5），然后比较低位（0）
                        num = num_parts[0] * 1000 + num_parts[1]  # 可以根据实际需要调整数字权重
                    else:
                        num = num_parts[0]
                except ValueError:
                    continue  # 如果提取失败，跳过此文件

                # 更新最大数字和路径
                if num > max_num:
                    max_num = num
                    best_model_path = model_path
            model_paths.append(best_model_path)
# 将所有路径保存到 txt 文件
with open("model_paths.txt", "w") as f:
    for path in model_paths:
        f.write(path + "\n")

print("模型路径已保存到 model_paths.txt")