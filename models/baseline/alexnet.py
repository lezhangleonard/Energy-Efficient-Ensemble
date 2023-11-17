import torch 
import torch.nn as nn
from thop import profile
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, input_channels, out_classes, size=1):
        super(AlexNet, self).__init__()
        self.structures = {1: [64,192,384,256,256,2048],
                          2: [52,128,320,128,128,1024],
                          3: [49,84,268,108,80,1024],
                          4: [45,68,220,90,64,512],
                          5: [42,54,206,74,52,256],
                          6: [38,46,196,64,40,256],
                          7: [34,43,172,52,35,128],
                          8: [32,39,156,48,30,128],
                          9: [30,36,140,44,28,64],
                          10: [28,34,136,39,26,64]}
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
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = x.view(-1, self.structure[4])
        x = self.fc0(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x
    
    def get_structure(self):
        return self.structure
    
    def set_size(self, size:int=1):
        self.size = size
        self.structure = self.structures[self.size]

        self.conv0 = nn.Conv2d(self.input_channels, self.structure[0], kernel_size=5, stride=1, padding=0)
        self.maxpool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = nn.Conv2d(self.structure[0], self.structure[1], kernel_size=5, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(self.structure[1], self.structure[2], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.structure[2], self.structure[3], kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(self.structure[3], self.structure[4], kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)

        self.fc0 = nn.Linear(self.structure[4], self.structure[5])
        self.fc1 = nn.Linear(self.structure[5], self.out_classes)   

    def set_structure(self, structure:list=None):
        if structure is not None:
            self.size = None
            self.structure = structure
            self.conv0 = nn.Conv2d(self.input_channels, self.structure[0], kernel_size=5, stride=1, padding=0)
            self.maxpool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.conv1 = nn.Conv2d(self.structure[0], self.structure[1], kernel_size=5, stride=2, padding=1)
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.conv2 = nn.Conv2d(self.structure[1], self.structure[2], kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(self.structure[2], self.structure[3], kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(self.structure[3], self.structure[4], kernel_size=3, stride=1, padding=1)
            self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)

            self.fc0 = nn.Linear(self.structure[4], self.structure[5])
            self.fc1 = nn.Linear(self.structure[5], self.out_classes) 

    def get_macs(self, input_shape):
        return profile(self, inputs=(torch.empty(1, *input_shape),), verbose=False)[0]
    
    def get_weight_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
