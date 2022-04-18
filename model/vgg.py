# -*- coding: UTF-8 -*-
# @Time        :   14:36
# @Author      :  HX
# @application :  
# @File        :  vgg.py
import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self,depthVersion: str = 'A',
                 in_channel:int = 3,
                 out_classes:int = 1000,
                 init_weight: bool = True
                 ):
        super(VGG, self).__init__()
        self.in_channel = in_channel
        self.model_base_config = {
            'A': [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'C': [64, 64, 'M', 128, 128, 'M', 256, 256, '1-256','M', 512, 512, '1-512','M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }
        self.features = self.make_layer(self.model_base_config[depthVersion])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, out_classes),
            # nn.Softmax(out_classes)
        )
        if init_weight:
            self.init_weight()

    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layer(self,config_list):
        channel = self.in_channel
        layers = []
        for item in config_list:
            if item == 'M':
                layers +=[nn.MaxPool2d(kernel_size=2,stride=2)]
            elif item in ['1-256', '1-512']:
                out_channel = int(item.replace('1-',''))
                layers += [nn.Conv2d(in_channels=channel, out_channels=out_channel,
                                     kernel_size=1,stride=1,padding=0),
                           nn.ReLU(inplace=True)
                           ]
                channel = out_channel
            else :
                layers += [nn.Conv2d(in_channels=channel, out_channels=item,
                                     kernel_size=3,stride=1,padding=1),
                           nn.ReLU(inplace = True)]
                channel = item
        return nn.Sequential(*layers)

if __name__ == "__main__":
    from torch.autograd import Variable
    model = VGG('D',3,100).cuda()
    torch.save(model,'vgg_16.pth',_use_new_zipfile_serialization=False)
    input_ = Variable(torch.randn(1, 3, 224, 224).float()).cuda()
    model.eval()
    torch.onnx._export(model, input_, "VGGNet.onnx", verbose=False, opset_version=11, input_names=['image'],
                       output_names=["output"])