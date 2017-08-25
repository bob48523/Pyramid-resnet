'''Pyramid ResNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from torch.autograd import Variable

FLOPS = 0

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class PreActBottleneck_p(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    widen_factor = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck_p, self).__init__()
    
        self.inplanes= in_planes
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
    
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.widen_factor, kernel_size=1, bias=False)   
    
        self.bn4 = nn.BatchNorm2d(planes*self.widen_factor)

        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2)             
            )
        
    def forward(self, x):
        global FLOPS
        out = self.bn1(x)
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
    
        out = self.conv1(out)  
        FLOPS += self.conv1.in_channels*self.conv1.out_channels*self.conv1.kernel_size[0]*self.conv1.kernel_size[1]*out.size(2)*out.size(3)
    
        out = self.conv2(F.relu(self.bn2(out)))
        FLOPS += self.conv2.in_channels*self.conv2.out_channels*self.conv2.kernel_size[0]*self.conv2.kernel_size[1]*out.size(2)*out.size(3)
        #FLOPS += temp // self.conv2.groups        

        out = self.conv3(F.relu(self.bn3(out)))
        FLOPS += self.conv3.in_channels*self.conv3.out_channels*self.conv3.kernel_size[0]*self.conv3.kernel_size[1]*out.size(2)*out.size(3) 
    
        out = self.bn4(out)
        if shortcut.size(1) == out.size(1): 
            out = shortcut+out
        else:
            out = torch.cat((shortcut[:,:self.inplanes,:,:]+out[:,:self.inplanes,:,:], out[:,self.inplanes:,:,:]), 1)
        return out


class ResNet(nn.Module):
    def __init__(self, block, depth, num_classes=10):
        super(ResNet, self).__init__()
        alpha = 48
        self.in_channels = 64
        self.temp_channels = 16
        n = (depth - 2)//9
        self.addchannel = alpha / (3*n)

        self.conv1 = conv3x3(3,self.in_channels)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        
        self.layer1 = self._make_layer(block, n, stride=1)
        self.layer2 = self._make_layer(block, n, stride=2)
        self.layer3 = self._make_layer(block, n, stride=2)
        self.linear = nn.Linear(self.in_channels, num_classes)
        #print(self.linear.in_features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            self.temp_channels += self.addchannel 
            layers.append(block(self.in_channels, int(self.temp_channels), stride))
            self.in_channels = 4*int(self.temp_channels)
        return nn.Sequential(*layers)

    def forward(self, x):
        global FLOPS
        FLOPS = 0
        out = F.relu(self.bn1(self.conv1(x)))
        FLOPS+=self.conv1.in_channels*self.conv1.out_channels*self.conv1.kernel_size[0]*self.conv1.kernel_size[1]*out.size(2)*out.size(3)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        FLOPS+=self.linear.in_features*self.linear.out_features
        return out


def PyramidResNet():
    return ResNet(PreActBottleneck_p, 101)

def CalcuFlops():
    net = ResNet(PreActBottleneck_p, 110)
    x = torch.randn(3,3,32,32)
    y = net(Variable(x))
    print ('flops:%d'%FLOPS)

# test()
