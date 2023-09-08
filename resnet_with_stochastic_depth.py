import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, Type, List

from stochastic_depth import StochasticDepth

def conv3x3(in_planes: int, 
            out_planes: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
           ) -> nn.Conv2d:
    # setting padding and dilation to the same value, when using a 3x3 kernel size, makes output the same size as input
    # (2*padding + height - 1 - 2*dialtion) // stride + 1
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int,
            out_planes: int, 
            stride: int = 1,
           ) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    """
    Basic Resnet Block is made of [Conv3x3, BN, ReLU, Conv3x3, BN, skip_connection, ReLU]
    """
    expansion: int = 1
        
    def __init__(self, 
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.stochastic_depth = StochasticDepth(0.2)
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # stochastic depth
        out = self.stochastic_depth(out)
        
        # perform projection to the input to match the shape of output
        if self.downsample is not None:
            identity = self.downsample(x)
        # residual connection
        out += identity
        out = self.relu(out)
        
        return out
    
class Bottleneck(nn.Module):
    """
    Bottleneck Block is made of [Conv1x1, BN, ReLU, Conv3x3, BN, ReLU, Conv1x1, BN, skip_connection, ReLU]
    """
    expansion: int = 4
        
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int =1,
                 base_width: int = 64,
                 dilation: int = 1,
                ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.downsample = downsample
        self.stride = stride

        self.stochastic_depth = StochasticDepth(0.2)
        
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(x))
        # stochastic depth 
        out = self.stochastic_depth(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out
        
        
class ResNet(nn.Module):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 101,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                ) -> None:
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # initialize gamma as 1 and beta as 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        # change the num_plane in the first block
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation)
            )
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x
    
def resnet18():
    return ResNet(block=BasicBlock, 
                  layers=[2, 2, 2, 2])

def resnet34():
    return ResNet(block=BasicBlock, 
                  layers=[2, 4, 6, 3])

def resnet50():
    """
    Replace each basic block in resnet34 with bottleneck block
    """
    return ResNet(block=Bottleneck, 
                 layers=[2, 4, 6, 3])

def resnet101():
    return ResNet(block=Bottleneck, 
                  layers=[3, 4, 23, 3])

def resnet152():
    return ResNet(block=Bottleneck, 
                 layers=[3, 8, 36, 3])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        