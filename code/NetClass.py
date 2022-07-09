# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

# 使用类建立网络


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 调用父类的构造函数
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        # activation function for hidden layer（隐藏层激活函数）
        x = torch.relu(self.hidden(x))
        x = self.predict(x)  # linear output（线性输出）
        return x


class DNN(torch.nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_hidden3, n_output):
        super(DNN, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.output = torch.nn.Linear(n_hidden3, n_output)
        # self.dropOut = torch.nn.Dropout(0.2)

    def forward(self, x):
        x = torch.sigmoid(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        x = torch.sigmoid(self.hidden3(x))
        x = self.output(x)
        return x

class LR(torch.nn.Module):
    def __init__(self,n_input):
        super(LR, self).__init__()
        self.lr = torch.nn.Linear(19,1)
        self.sm = torch.nn.Sigmoid()
    def forward(self, x):
        x=self.lr(x)
        x=self.sm(x)
        return x

class CNN(torch.nn.Module):
    def __init__(self, n_input):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=3, padding=1, stride=1)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(16, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=3, padding=1, stride=1)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=3, padding=1, stride=1)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * n_input, n_input),
            torch.nn.ReLU(),
            torch.nn.Linear(n_input, n_input),
            torch.nn.ReLU(),
            torch.nn.Linear(n_input, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x


class CNN1(torch.nn.Module):
    def __init__(self, n_input):
        super(CNN1, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=3, padding=1, stride=1)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=3, padding=1, stride=1)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * n_input, n_input),
            torch.nn.ReLU(),
            torch.nn.Linear(n_input, n_input),
            torch.nn.ReLU(),
            torch.nn.Linear(n_input, 1)
        )
    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x

class CNN2(torch.nn.Module):
    def __init__(self, n_input):
        super(CNN2, self).__init__()
        self.linear = torch.nn.Linear(n_input, 26)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=2),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 64, kernel_size=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 1))

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = x.view(x.size(0), 1, -1)
        x = self.layer1(x)
        x = self.layer2(x)
        # print(x.size())
        x = x.view(x.size(0), -1)  # view函数将张量x变形成一维的向量形式
        x = self.fc(x)
        return x

# 保存模型
# torch.save(CNN, 'Model/cnn.ptl')
# torch.load('Model/cnn.ptl')

# class CNN(torch.nn.Module):
#
#     def __init__(self, block, layers, w, h, num_classes=2):
#         self.in_planes = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, 1)
#         self.fc = nn.Linear(512 * block.expansion * w * h, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         down_sample = None
#         if stride != 1 or self.in_planes != planes * block.expansion:
#             down_sample = nn.Sequential(
#                 nn.Conv2d(self.in_planes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#         layers = list([])
#         layers.append(block(self.in_planes, planes, stride, down_sample))
#         self.in_planes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.in_planes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         # 输入图像的shape[3, w, h]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         # 此时的shape[64, w/4 , h/4]
#         x = self.layer1(x)
#         # 此时的shape[256, w/4 , h/4]
#         x = self.layer2(x)
#         # 此时的shape[512, w/8 , h/8]
#         x = self.layer3(x)
#         # 此时的shape[1024, w/16 , h/16]
#         x = self.layer4(x)
#         # 此时的shape[2048, w/32 , h/32]
#         x = self.avgpool(x)
#         # 此时的shape[2048, w/32 - 7 + 1 , h/32 - 7 + 1]
#         x = x.view(x.size(0), -1)
#         # 此时的shape[2048 * (w/32 - 7 + 1) * (h/32 - 7 + 1)]
#         x = self.fc(x)
#         return x


# class CNN(torch.nn.Module):
#     # 定义Net的初始化函数，这个函数定义了该神经网络的基本结构
#     # 一个输入层，两个卷积层，两个池化层，两个relu，3个全连接层
#     def __init__(self):
#         super(CNN, self).__init__()  # 复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
#         self.conv1 = torch.nn.Conv2d(1, 6, 5)  # 定义conv1函数的是图像卷积函数：输入为1个频道,输出为 6张特征图, 卷积核为5x5正方形
#         self.conv2 = torch.nn.Conv2d(6, 16, 5)  # 定义conv2函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
#         #self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)  # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
#         self.fc1 = torch.nn.Linear(1, 1024)  # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
#         self.fc2 = torch.nn.Linear(1024, 1024)
#         self.fc3 = torch.nn.Linear(1024, 1)
#
#         #self.fc2 = torch.nn.Linear(120, 84)  # 定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上。
#         #self.fc3 = torch.nn.Linear(84, 10)  # 定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上。
#
#     # 定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
#     def forward(self, x):
#         print(x.size())  #x[0]-x[31]
#         x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=True)
#         x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
#         #print("first")
#         print(x.size())  #torch.Size([1, 1, 32, 19])
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # 输入x经过卷积conv1之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
#         print(x.size()) #torch.Size([1, 6, 14, 7])
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 输入x经过卷积conv2之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
#         print("pool")
#         print(x.size()) #池化完成--torch.Size([1, 80])
#         x = x.view(-1, self.num_flat_features(x))  # view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
#        # print("tensor_one_dimension")
#        # x=torch.argmax(x,dim=0)
#         #print(x.size())
#         #print(x)
#         x = F.relu(self.fc1(x))  # 输入x经过全连接1，再经过ReLU激活函数，然后更新x
#         #print(x.size())
#        # print(x)
#         x = F.relu(self.fc2(x))  # 输入x经过全连接2，再经过ReLU激活函数，然后更新x
#         #x = self.fc3(x)  # 输入x经过全连接3，然后更新x
#         return x
#
#     # 使用num_flat_features函数计算张量x的总特征量（把每个数字都看出是一个特征，即特征总量），比如x是4*2*2的张量，那么它的特征总量就是16。
#     def num_flat_features(self, x):
#         size = x.size()[0:]  # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features


# class RNN(torch.nn.Module):
#     def __init__(self, length):
#         super(RNN, self).__init__()
#
#         self.rnn = torch.nn.RNN(
#             input_size=1,
#             hidden_size=32,     # rnn hidden unit
#             num_layers=1,       # 有几层 RNN layers
#             batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
#         )
#         self.out = torch.nn.Linear(32, 1)
#
#     def forward(self, x, h_state):
#         r_out, h_state = self.rnn(x, h_state)
#         r_out = r_out.view(-1, 32)
#         outs = self.out(r_out)
#         return outs.view(-1, 32, 10), h_state
