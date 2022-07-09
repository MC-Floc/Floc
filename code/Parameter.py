# -*- coding: utf-8 -*-

# 参数
class Parameter:
    # __init__.py
    ERROR_TRACE_METHOD = 2      # 0:RPS ; 1:SPS ; 2:RSPS
    CORRECT_TRACE_METHOD = 0    # 0:随机 ;
    TRACE_REPEAT = 1    # 0: 不重构路径集 ; 1: 重构路径集
    MATRIX_CONSTRUCT_METHOD = 0  # 0: 0/1 ; 1: 累加     矩阵构造方法
    SAVE_NAME = ''

    # TraceSearch.py
    CT_TRACE_METHOD = 0  # TODO
    ET_TRACE_METHOD = 2  # TODO
    CT_NUM = 4000   # 总数 = vs_num * ct_num
    ET_NUM = 2000   # 每个vs对应的路径数量
    C_ALL = 500
    E_ALL = 500
    REMOVE_DUPLICATE = 1    # 路径节点是否去重: 0: No ; 1: Yes
    SORT = 0               # 训练集是否排序： 0: No ; 1: Yes
    ERROR_SET = 1           #是否筛选错误路径集合： 0: No ; 1: Yes

    # NetConstruct.py
    NET_METHOD = 4 # 0: NN ; 1: DNN
    OPTIMIZER_FUNCTION = 2  # 0: SGD ; 1: RMSprop ; 2: Adam
    LOSS_FUNCTION = 1   # 0: MSELoss ; 1: SmoothL1Loss
    BATCH_SIZE = 32  # 16
    NUM_WORKERS = 10
    LR = 0.01
    EPOCH = 50

    Result = "None" #文件输出位置


