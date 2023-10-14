"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2023/09/02 21:05
  @Email: 2665109868@qq.com
  @function
"""
import os
import logging
import datetime

class TrainLogger:
    def __init__(self, params):
        self.params = params
        log_dir = params.get('log_dir', 'logs')  # 指定日志目录，默认为'logs'
        log_file = params.get('log_file', 'train.log')  # 指定日志文件，默认为'train.log'

        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 设置日志文件的完整路径
        log_file_path = os.path.join(log_dir, log_file)

        # 配置日志
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def info(self, message):
        logging.info(message)

    def get_model_dir(self):
        return os.path.join(self.params.get('save_dir', 'save'), self.params.get('model', 'model'))

# 其他TrainLogger类的方法可以根据需要添加，例如debug、warning、error等不同日志级别的记录方法。


