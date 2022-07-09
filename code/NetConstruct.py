# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import random
import NetClass
from Parameter import Parameter as Para

# 基于覆盖的基本方法 0/1
def matrix_construct_0(correct_traces, error_traces, length):
    trace_matrix = []
    # 构造训练集输入
#    print(length)
    for trace in correct_traces:
        vertex = [0] * length                  # 构造length个0：[0, 0, 0]
        for tr in trace:
            vertex[tr] = 1
        trace_matrix.append(vertex)
    # print(trace_matrix) #输出list,表示路径集
    # print(correct_traces)  # 输出list

    for trace in error_traces:
        vertex = [0] * length
        for tr in trace:
            vertex[tr] = 1
        trace_matrix.append(vertex)
    # print(trace_matrix) #输出list

    x = torch.from_numpy(np.array(trace_matrix, dtype=float))
    y = torch.cat((torch.zeros(len(correct_traces), 1), torch.ones(len(error_traces), 1)), 0)  # 正确路径标量值为0，错误路径标量值为1
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    test = torch.eye(length)  # 单位矩阵
    # print(x.size())
    # print(y.size())
    # print(len(test))
    # x为运行矩阵，若运行时经过x，则对应位置为1，y为结果矩阵，若结果正确则为0，结果错误则为1
    # 最终：运行过且结果错误的值为1，可以被找到
    return x, y, test


# 增加权重的对比方法 累加
def matrix_construct_1(correct_traces, error_traces, length):
    trace_matrix = []
    # num_all = {}
    # for i in range(0, length):
     #    num_all[i] = 0

    for trace in correct_traces:
        vertex = [0] * length
        for tr in trace:
            vertex[tr] += 1
    #         num_all[tr] += 1
        trace_matrix.append(vertex)
    # print(correct_traces)  # 输出list
    # print(trace_matrix)

    for trace in error_traces:
        vertex = [0] * length
        for tr in trace:
            vertex[tr] += 1
    #        num_all[tr] += 1
        trace_matrix.append(vertex)

    x = torch.from_numpy(np.array(trace_matrix, dtype=float))
    # x = x.cuda()
    y = torch.cat((torch.zeros(len(correct_traces), 1), torch.ones(len(error_traces), 1)), 0)
    # y = y.cuda()
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    # x = torch.tensor(x, dtype=torch.float32)
    # y = torch.tensor(y, dtype=torch.float32)
    test = torch.eye(length)                        # 生成单位矩阵
    # for i in range(0, length):
    #     test[i, i] = num_all[i] / Para.E_ALL
    return x, y, test

# 增加权重的对比方法 累加   排序
def matrix_construct_2(correct_traces, error_traces, length):
    trace_matrix = []
    # num_all = {}
    # for i in range(0, length):
    #    num_all[i] = 0

    for trace in correct_traces:
        trace_matrix.append(trace)
    # print(correct_traces)  # 输出list
    # print(trace_matrix)

    for trace in error_traces:
        trace_matrix.append(trace)
    x = torch.from_numpy(np.array(trace_matrix, dtype=float))
    # x = x.cuda()
    y = torch.cat((torch.zeros(len(correct_traces), 1), torch.ones(len(error_traces), 1)), 0)
    # y = y.cuda()
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    # x = torch.tensor(x, dtype=torch.float32)
    # y = torch.tensor(y, dtype=torch.float32)
    test = torch.eye(length)                        # 生成单位矩阵
    # for i in range(0, length):
    #     test[i, i] = num_all[i] / Para.E_ALL
    return x, y, test

def trace_net(x, y, test, length, i, j):
    fmodel_pre = 'Model/'
    if not os.path.exists(fmodel_pre):
        os.makedirs(fmodel_pre)
    if Para.NET_METHOD == 0:
        result = trace_net_nn(x, y, test, length)
    if Para.NET_METHOD == 1:
        result = trace_net_dnn(x, y, test, length, i, j)
    if Para.NET_METHOD == 2:
        result = trace_net_cnn(x, y, test, length, i, j) 
    if Para.NET_METHOD == 3:
        result = trace_net_LR(x, y, test, length)
    if Para.NET_METHOD == 4:
        result = trace_net_RF(x, y, test, length)
    return result


def data_loade(x, y):
    torch_dataset = Data.TensorDataset(x, y)
    return Data.DataLoader(
        dataset=torch_dataset,  # torch Tensor Dataset format
        batch_size=Para.BATCH_SIZE,  # mini batch size
        shuffle=True,
        num_workers=Para.NUM_WORKERS,
    )


def optimizer_function(net):  # 优化函数
    if Para.OPTIMIZER_FUNCTION == 0:
        return torch.optim.SGD(net.parameters(), lr=Para.LR)
    elif Para.OPTIMIZER_FUNCTION == 1:
        return torch.optim.RMSprop(net.parameters(), lr=Para.LR)
    elif Para.OPTIMIZER_FUNCTION == 2:
        return torch.optim.Adam(net.parameters(), lr=Para.LR, betas=(0.9, 0.99))


def loss_function():  # 损失函数
    if Para.LOSS_FUNCTION == 0:
        return torch.nn.MSELoss()
    elif Para.LOSS_FUNCTION == 1:
        return torch.nn.SmoothL1Loss()
    elif Para.LOSS_FUNCTION == 2:
        return torch.nn.Sigmoid()


def net_work(loader, net, test):
    print(torch.cuda.is_available())
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    test = test.to(device)
    net = net.to(device)
    optimizer = optimizer_function(net)
    loss_func = loss_function()

    running_loss = 0.0
    epoch = Para.EPOCH

    print(f"epoch: {epoch}")
    for i in range(epoch):
        for step, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()  # clear gradients for next train

            prediction = net(batch_x)  # input x and predict based on x
            loss = loss_func(prediction, batch_y)  # must be (1. nn output, 2. target)
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            running_loss += loss.item()
        print('[epoch %d] loss: %.3f' % (i + 1, running_loss))
        if running_loss < 0:
            break
        running_loss = 0.0

    test_list = list(net(test))
    result = []
    for temp in test_list:
        temp = temp.detach().cpu().numpy()
        result.append(temp)
    return result


def trace_net_nn(x, y, test, length):
    loader = data_loade(x, y)
    net = NetClass.Net(n_feature=length, n_hidden=length, n_output=1)  # define the network
    return net_work(loader, net, test)


def trace_net_dnn(x, y, test, length, i, j):
    loader = data_loade(x, y)
    # print(x.size(), y.size())
    # hidden_num = int(length / 3)
    # hidden_num = int(length / 2)
    net = NetClass.DNN(n_input=length, n_hidden1=length, n_hidden2=length,
                       n_hidden3=length, n_output=1)
    return net_work(loader, net, test)


def trace_net_cnn(x, y, test, length, i, j):
    #网络结构确定之后可以使用缓存
    # if os.path.exists("Model/cnn" + "_" + str(i) + "_" + str(j) + ".ptl"):
    #     loader = data_loade(x, y)
    #     net = torch.load("Model/cnn" + "_" + str(i) + "_" + str(j) + ".ptl")
    #     print("loading......")
    # else:
    #     print("calculating......")
    loader = data_loade(x, y)
    net = NetClass.CNN2(n_input=length)
    #torch.save(net, "Model/cnn" + "_" + str(i) + "_" + str(j) + ".ptl")
    # print(net)  #输出CNN网络结构
    return net_work(loader, net, test)


def trace_net_LR(x, y, test, length):
    lr = LogisticRegression(max_iter=3000)
    # x = x.cpu().detach().numpy()  # 累加
    # y = y.cpu().detach().numpy()
    y = y.numpy().ravel()    #0/1
    lr = lr.fit(x, y)
    #print(np.array(x).shape)
    #print(y)
    result = lr.predict_proba(test)
    #print(result)
    result = [i[1] for i in result]

    # result = list(result[1])
    # result = [np.array(i).reshape((1,)) for i in result]
    return result


def trace_net_RF(x, y, test, fmodel):
    forest = RandomForestClassifier()
    #x = x.cpu().detach().numpy()  # 累加
    #y = y.cpu().detach().numpy()
    # y = y.numpy().ravel()    #0/1
    forest = forest.fit(x, np.ravel(y))
    result = forest.predict_proba(test)
    result = [i[1] for i in result]   #运行外部数据时临时注释
    #print(f"result:\n {result}")
    #print(f"x.shape={x.shape}, \n{x}")
    #print(f"y.shape:{y.shape}, \n{y}")
    # forest = forest.fit(x, np.array(y).ravel())
    # result = list(result)
    # result = [np.array(i).reshape((1,)) for i in result]
    return result
