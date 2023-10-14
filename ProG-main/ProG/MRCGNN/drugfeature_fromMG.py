
import os
import torch
import random

import copy


import numpy as np
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=False)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
for k in range(5):


    os.chdir('/project/reproduct_paper/MRCGNN-main/MRCGNN_codes/trimnet_small')
    os.system('python train_big_data.py --zhongzi '+str(k))
