# -*- coding: UTF-8 -*-
# @Time        :   9:25
# @Author      :  Huangxiao
# @application :  
# @File        :  lenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
class LeNet(nn.Module):
    def __init__(self,in_channel = 3,out_channel = 10):
        '''
        :param in_channel(Int): 输入图像维度
        :param out_channel(Int): 输出类别数量
        '''
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channel,6,5)   # [bs,1,32,32] -> [bs，6，28，28]
        # params number: (5*5+1)*6 每个卷积含有5*5尺寸的参数，1为偏置参数bias

        self.pool1 = nn.AvgPool2d(2,2)    #[bs,6,28,28] -> [bs,6,14,14]

        self.conv2 = nn.Conv2d(6,16,5)   # [bs,6,14,14] ->[bs,16,10,10]
        #params number：(5*5+1) * 16

        self.pool2 = nn.AvgPool2d(2,2)    # [bs,16,10,10] -> [bs,16,5,5]

        self.fc1 = nn.Linear(16*5*5,120)  # [bs,16,5,5] -> [bs,120]
        #params number:400 * 120
        self.fc2 = nn.Linear(120,84)    # [bs,120] -> [bs,84]
        #params number:120*84
        self.fc3 = nn.Linear(84,out_channel)  # [bs,84] -> [bs,10]
        #params number: 84*10

    def forward(self,x):
        x = torch.sigmoid(self.conv1(x))
        x = self.pool1(x)

        x = torch.sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16*5*5)

        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        x = torch.sigmoid(self.fc3(x))
        return x

if __name__=='__main__':
    tensor = torch.randn([10,3,32,32])
    print(tensor.shape)
    model = LeNet(3,10)
    print(model)
    output = model(tensor)
    print(output.shape)