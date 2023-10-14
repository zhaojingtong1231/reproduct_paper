"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2023/09/02 21:00
  @Email: 2665109868@qq.com
  @function
"""

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
