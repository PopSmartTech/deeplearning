# -*- coding: UTF-8 -*-
# @Time        :   1:57
# @Author      :  Huangxiao
# @application :  
# @File        :  test.py
import torch

def getAcc(outputs,label):
    predict_y = torch.max(outputs, dim=1)[1]
    accuracy = (predict_y == label).sum().item() / label.size(0)
    return accuracy