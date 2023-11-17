import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, input_channels, out_classes, size=1):
        super(LeNet, self).__init__()
        self.structures = {1: [6,16,120,84],
                          2: [4,10,50,32],
                          3: [3,5,42,32],
                          4: [2,8,36,32],
                          5: [2,3,24,16],
                          6: [1,16,24,16],
                          7: [1,12,16,16],
                          8: [1,8,16,16],
                          9: [1,6,12,16],
                          10: [1,3,8,16]}
        self.size = size
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.set_size(size)
        
    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(x)
        x = self.maxpool0(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(-1, self.structure[2])
        x = self.linear0(x)
        x = F.relu(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = F.softmax(x, dim=1)
        return x
    
    def get_structure(self):
        return self.structure

    def set_size(self, size:int=1):
        self.size = size
        self.structure = self.structures[self.size]
        self.conv0 = nn.Conv2d(self.input_channels, self.structure[0], kernel_size=5, stride=1, padding=0)
        self.maxpool0 = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(self.structure[0], self.structure[1], kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(self.structure[1], self.structure[2], kernel_size=5, stride=1, padding=0)
        self.linear0 = nn.Linear(self.structure[2], self.structure[3])
        self.linear1 = nn.Linear(self.structure[3], self.out_classes)
        self.layers = [self.conv0, self.conv1, self.conv2, self.linear0, self.linear1]
    
    def set_structure(self, structure:list=None):
        if structure is not None:
            self.size = None
            self.structure = structure
            self.conv0 = nn.Conv2d(self.input_channels, self.structure[0], kernel_size=5, stride=1, padding=0)
            self.maxpool0 = nn.MaxPool2d(2)
            self.conv1 = nn.Conv2d(self.structure[0], self.structure[1], kernel_size=5, stride=1, padding=0)
            self.maxpool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(self.structure[1], self.structure[2], kernel_size=5, stride=1, padding=0)
            self.linear0 = nn.Linear(self.structure[2], self.structure[3])
            self.linear1 = nn.Linear(self.structure[3], self.out_classes)
            self.layers = [self.conv0, self.conv1, self.conv2, self.linear0, self.linear1]

    
    def get_macs(self, input_shape):
        return profile(self, inputs=(torch.empty(1, *input_shape),), verbose=False)[0]
    
    def get_weight_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

