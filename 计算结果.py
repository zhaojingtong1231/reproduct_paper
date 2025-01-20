"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2025/01/16 19:11
  @Email: 2665109868@qq.com
  @function
"""
from math import sqrt
import numpy as np
# 输入五个数
numbers = [0.8715408632262942, 0.8724166420236428, 0.8660784853256162, 0.8658621633838982,0.8728748552479244]  # 替换为你的数据

# 计算平均值
# mean = sum(numbers) / len(numbers)
#
# mean = round(mean, 3)
# 计算方差
# std_deviation = sqrt(sum((x - mean) ** 2 for x in numbers) / (len(numbers)))
# std_deviation = round(std_deviation, 3)

numbers = np.array(numbers)  # 替换为你的数据

mean = np.mean(numbers)
# 计算标准差
std_deviation = np.std(numbers)

mean = round(mean, 3)

std_deviation = round(std_deviation, 3)

# 输出结果
print(f"平均值: {mean}")
print(f"方差: {std_deviation}")
print(f"平均值: {mean}+{std_deviation}")