import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, input_channels, out_classes, size=1):
        super(LeNet, self).__init__()
        self.structures = {1: [6,16,120,84],
                          2: [4,10,48,32],
                          3: [3,5,42,32],
                          4: [2,10,24,32],
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
    
    def __set_stucture(self):
        self.conv0 = nn.Conv2d(self.input_channels, self.structure[0], kernel_size=5, stride=1, padding=0)
        self.maxpool0 = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(self.structure[0], self.structure[1], kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(self.structure[1], self.structure[2], kernel_size=5, stride=1, padding=0)
        self.linear0 = nn.Linear(self.structure[2], self.structure[3])
        self.linear1 = nn.Linear(self.structure[3], self.out_classes)
        self.layers = [self.conv0, self.conv1, self.conv2, self.linear0, self.linear1]
        self.representation = nn.Sequential(self.conv0, nn.ReLU(), self.maxpool0, self.conv1, nn.ReLU(), self.maxpool1, self.conv2, nn.ReLU())
        self.classifier = nn.Sequential(self.linear0, nn.ReLU(), self.linear1, nn.Softmax(dim=1))
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_structure(self):
        return self.structure

    def set_size(self, size:int=1):
        self.size = size
        self.structure = self.structures[self.size]
        self.__set_stucture()
    
    def set_structure(self, structure:list=None):
        if structure is not None:
            self.size = None
            self.structure = structure
            self.__set_stucture()
    
    def get_macs(self, input_shape):
        return profile(self, inputs=(torch.empty(1, *input_shape),), verbose=False)[0]
    
    def get_weight_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_representation(self):
        return self.representation
    
    def get_classifier(self):
        return self.classifier