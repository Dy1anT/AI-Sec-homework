import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
 
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
 
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  num_blocks[0], stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512, num_classes)
 
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
 
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
 
 
def ResNet18():
    return ResNet(ResidualBlock, [2,2,2,2])

class Vgg16_Net(nn.Module):
    def __init__(self):
        super(Vgg16_Net, self).__init__()
        #2个卷积层和1个最大池化层
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride=1, padding=1),             # (32-3+2)/1+1 = 32  32*32*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64,64, kernel_size = 3, stride=1, padding=1),             # (32-3+2)/1+1 = 32  32*32*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, 2)                                                  # (32-2)/2+1 = 16    16*16*64
            
            )
        #2个卷积层和1个最大池化层
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride=1, padding=1),           # (16-3+2)/1+1 = 16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size = 3, stride=1, padding=1),          # (16-3+2)/1+1 = 16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, 2)                                                  # (16-2)/2+1 = 8    8*8*128
            )
        #3个卷积层和1个最大池化层
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, stride=1, padding=1),          # (8-3+2)/1+1 = 8  8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size = 3, stride=1, padding=1),          # (8-3+2)/1+1 = 8  8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size = 3, stride=1, padding=1),          # (8-3+2)/1+1 = 8  8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, 2),                                                 # (8-2)/2+1 = 4    4*4*256
            )
        #3个卷积层和1个最大池化层
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, stride=1, padding=1),          # (4-3+2)/1+1 = 4  4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size = 3, stride=1, padding=1),          # (4-3+2)/1+1 = 4  4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size = 3, stride=1, padding=1),          # (4-3+2)/1+1 = 4  4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, 2)                                                  # (4-2)/2+1 = 2    2*2*512
            )
        #3个卷积层和1个最大池化层
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, stride=1, padding=1),          # (2-3+2)/1+1 = 2  2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size = 3, stride=1, padding=1),          # (2-3+2)/1+1 = 2  2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size = 3, stride=1, padding=1),          # (2-3+2)/1+1 = 2  2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, 2)                                                  # (2-2)/2+1 = 1    1*1*512
            )
        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
            )
        self.fc = nn.Sequential(    
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
    
            nn.Linear(256, 10)
            )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x