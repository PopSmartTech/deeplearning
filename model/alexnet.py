# -*- coding: UTF-8 -*-
# @Time        :   10:21
# @Author      :  Huangxiao
# @application :  
# @File        :  alexnet.py
import torch
import torch.nn as nn
from torchvision.models import AlexNet
from torchvision.models.utils import load_state_dict_from_url

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class alexnet(AlexNet):
    def __init__(self,num_class = 1000):
        # 继承自torchvision中AlexNe的网络结构，方便后续预训练权重的加载
        super(alexnet, self).__init__()

        #重新定义classifier，删除最后的分类层
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        #重新定义分类曾，支持预训练权重的任意类别的分类任务
        self.outFc = nn.Linear(4096,num_class)

    def forward(self,x):
        x = self.features(x)
        x = self.x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.outFc(x)
        return x

    def load_state_dict(self, state_dict, **kwargs):
        # 不加载原始模型参数的最后一层，从头开始训练最后一层参数
        state_dict.pop("classifier.6.bias")
        state_dict.pop("classifier.6.weight")
        super().load_state_dict(state_dict, **kwargs)

        # pass

def returnAlexNet(pretrained=False, progress=True, **kwargs):
    model = alexnet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)
    return model

if __name__ == '__main__':
    model = returnAlexNet(True,num_class = 3)
    print(model)
