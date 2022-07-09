# -*- coding: utf-8 -*-
import torch
import random
from Parameter import Parameter as Para


def random_seed():
    torch.cuda.device(0)
    torch.backends.cudnn.deterministic = True
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)


def parameter(p):
    if p == 0:
        Para.NET_METHOD = 0             # BP
        Para.OPTIMIZER_FUNCTION = 1     # RMSprop
        Para.LOSS_FUNCTION = 1          # SmoothL1Loss
        Para.LR = 0.01
        Para.SORT = 1
        Para.EPOCH = 10
        Para.BATCH_SIZE = 32
        Para.SAVE_NAME = 'BP'

    elif p == 1:
        Para.NET_METHOD = 1  # DNN
        Para.OPTIMIZER_FUNCTION = 2  # Adam
        Para.LOSS_FUNCTION = 1  # SmoothL1Loss
        Para.LR = 0.001
        Para.EPOCH = 25
        Para.BATCH_SIZE = 32   #2^n
        Para.SAVE_NAME = 'DNN'
        Para.SORT = 1

    elif p == 2:
        Para.NET_METHOD = 2  # CNN
        Para.OPTIMIZER_FUNCTION = 2  # Adam
        Para.LOSS_FUNCTION = 0  # MSELoss
        Para.LR = 0.01
        Para.EPOCH = 15
        Para.BATCH_SIZE = 32
        Para.SAVE_NAME = 'CNN'
        Para.SORT = 1

    elif p == 3:
        Para.NET_METHOD = 3  # LR
        Para.OPTIMIZER_FUNCTION = 2  # Adam
        Para.LOSS_FUNCTION = 1  # MSELoss
        Para.LR = 0.01
        Para.EPOCH = 50
        Para.BATCH_SIZE = 32  #调整
        Para.SAVE_NAME = 'LR'
        Para.SORT = 0

    elif p == 4:
        Para.NET_METHOD = 4  # RF
        Para.OPTIMIZER_FUNCTION = 2  # Adam
        Para.LOSS_FUNCTION = 1  # Sigmoid
        Para.LR = 0.01
        Para.EPOCH = 50
        Para.BATCH_SIZE = 32
        Para.SAVE_NAME = 'RF'
        Para.SORT = 0