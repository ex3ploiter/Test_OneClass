from torchattacks import FGSM, PGD



from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import copy
 
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
  def __init__(self, in_channels=1):
    super(FeatureExtractor, self).__init__()

 

    self.model = models.resnet18(pretrained=False)
    # self.model = models.resnet152(pretrained=False)

    self.model.fc = nn.Flatten()
    self.head = nn.Linear(512, 2)
    # self.head = nn.Linear(2048, 2)

  def forward(self, x):
 
    x = self.model(x)
    x = self.head(x)
    return x


class FeatureExtractorVIT(nn.Module):
  def __init__(self, in_channels=1):
    super(FeatureExtractorVIT, self).__init__()
    
 

    # self.model = models.resnet18(pretrained=True) norm
    self.model = models.vit_b_16(pretrained=True)
    num_inputs = self.model.heads.head.in_features

    self.model.heads.head = nn.Flatten()
    self.head = nn.Linear(num_inputs, 2)


  def forward(self, x):
     
    x = self.model(x)
    x = self.head(x)
    return x



_mean = (0.485, 0.456, 0.406)
_std = (0.229, 0.224, 0.225)

mu = torch.tensor(_mean).view(3,1,1).cuda()
std = torch.tensor(_std).view(3,1,1).cuda()


class PretrainedModel(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.norm = lambda x: ( x - mu ) / std

        self.backbone =models.resnet18(pretrained=False)
        checkpoint = torch.load("/kaggle/working/resnet18_linf_eps8.0.ckpt")
        state_dict_path = 'model'
        sd = checkpoint[state_dict_path]
        sd = {k[len('module.'):]:v for k,v in sd.items()}
        sd_t = {k[len('attacker.model.'):]:v for k,v in sd.items() if k.split('.')[0]=='attacker' and k.split('.')[1]!='normalize'}
        self.backbone.load_state_dict(sd_t)

        
        self.backbone.fc = torch.nn.Identity()
 
        self.backbone.fc__ = torch.nn.Linear(512, 2)
    def forward(self, x):
        x = self.norm(x)
        z1 = self.backbone(x)
        z2 = self.backbone.fc__(z1)
 
        return z2


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride:int=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride:int=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes:int=2):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])