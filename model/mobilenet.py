# -*- coding: UTF-8 -*-
# @Time        :   1:57
# @Author      :  Huangxiao
# @application :  
# @File        :  test.py
import torch
import torch.nn  as  nn

def DepthwiseSeparableConv(in_channel,out_channel,kernel_size = 3,stride = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel,out_channels=in_channel,
                  groups=in_channel,
                  kernel_size=kernel_size,
                  stride=stride,padding=1,
                  ),
        nn.BatchNorm2d(in_channel),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1,
                  stride=1,
                  ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True)
    )

class MobileNet(nn.Module):

    def __init__(self,in_channel = 3,class_nums = 1000):
        super(MobileNet, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        convs = []
        convs += [
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256,256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2)
        ]
        for i in range(5):
            convs.append(
                DepthwiseSeparableConv(512, 512, stride=1)
            )
        convs += [
            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024, stride=1)
        ]

        self.dws = nn.Sequential(*convs)
        self.avg_pool = nn.AvgPool2d(kernel_size=7,stride = 1)
        self.linear = nn.Linear(in_features=1024,out_features=class_nums)
        self.drop = nn.Dropout(p = 0.2)
        self.softmax = nn.Softmax(dim = 1)
        self.init_params()


    def init_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.project(x)
        x = self.dws(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        x = self.drop(x)
        x = self.linear(x)
        out = self.softmax(x)
        return out

if __name__ == "__main__":
    model = MobileNet(3,10).to('cuda')
    x = torch.rand((1,3,224,224)).to('cuda')
    model(x)