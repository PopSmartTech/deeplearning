# -*- coding: UTF-8 -*-
# @Time        :   17:43
# @Author      :  Huangxiao
# @application :  
# @File        :  train_lenet.py
import torch
import torchvision
import torch.nn as nn
import torch.optim as optimizer
import torchvision.transforms as transformer
from tensorboardX import SummaryWriter

from model import LeNet
from utils.logger import Logger
from utils.evaluator import getAcc

logPath = './log/letNet'


def main():
    device = 'cuda'
    lr = 1e-3
    epochs = 300
    writter = SummaryWriter(log_dir=logPath)
    log = Logger(logPath + '/log.log')
    #定义数据预处理方式，只是将原始图片padding2个像素，28->32,再转换为Tensor格式
    transformer_train = transformer.Compose([
        transformer.Pad(2, 2),
        transformer.ToTensor(),
    ])
    #定义数据集，使用MNIST手写数据集来训练模型
    train_set = torchvision.datasets.MNIST(root='./data',train = True,
                                             download= True,
                                             transform= transformer_train
                                             )
    #定义数据加载器，使用多线程/当线程来提前加载数据
    train_load = torch.utils.data.DataLoader(train_set,batch_size = 36,
                                             shuffle = True,num_workers = 2
                                             )
    val_set = torchvision.datasets.MNIST(root='./data',train = False,
                                         download= False,transform=transformer_train
                                         )
    val_load = torch.utils.data.DataLoader(val_set,batch_size = 36,
                                             shuffle = False,num_workers = 2
                                             )
    if torch.cuda.is_available() == False :
        device = 'cpu'
    log.info('------- data init Loader ------- ')
    model = LeNet(in_channel = 1,out_channel = 10).to(device)
    lossFun = nn.CrossEntropyLoss()
    optim = optimizer.Adam(model.parameters(),lr = lr)
    best_loss = 1000
    log.info('------- Model init ------- ')
    log.info(model)

    for epoch in range(epochs):
        model.train()
        # 模型训练
        for step,data in enumerate(train_load):
            x,y = data
            optim.zero_grad()
            output = model(x.to(device))
            loss = lossFun(output,y.to(device))
            acc = getAcc(output,y.to(device))
            log.info(('Epoch :%d,\titer:%d,\tloss:%.6f,\tAcc:%0.6f') % (epoch, step, loss.item(),acc))
            writter.add_scalar('train_loss',loss.item(),epoch * (len(train_load)) + step)
            loss.backward()
            optim.step()
        loss_val = 0.0

        #查看当前模型在验证集上的效果
        model.eval()
        for index,val_data in enumerate(val_load):
            x,y = val_data
            output = model(x.to(device))
            loss = lossFun(output, y.to(device)).item()
            acc = getAcc(output, y.to(device))
            log.info(('(Valid)Epoch :%d,\titer:%d,\tloss:%.6f,\tAcc:%0.6f') % (epoch, index, loss,acc))
            writter.add_scalar('val_loss', loss, epoch * (len(val_load)) + index)
            loss_val += loss
        loss_avg = loss_val / len(val_load)

        if loss_avg < best_loss:
            torch.save(model,logPath + '/lenet.pth',_use_new_zipfile_serialization=False)
            log.info('model saved !!! ')
            best_loss = loss_avg

if __name__=='__main__':
    main()