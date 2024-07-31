"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/07/29 15:54
  @Email: 2665109868@qq.com
  @function
"""


# 使用gzip模块解压缩并读取文件
with gzip.open(file_path, 'rt') as f:
    # 假设文件是以制表符分隔的，可以根据实际情况修改delimiter参数
    df = pd.read_csv(f, delimiter='\t', header=None)