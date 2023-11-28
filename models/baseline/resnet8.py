import torch.nn as nn
import torch.nn.functional as F

class ResNet8(nn.Module):
    def __init__(self, input_channels, out_classes, size=1):
        super(ResNet8, self).__init__()
        self.structures = {1: [16,16,32,64],
                          2: [10,10,26,42]}
        self.size = size
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.set_size(size)

    def __set_structure(self):
        self.conv0 = nn.Conv2d(self.input_channels, self.structure[0], kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(self.structure[0])

        # first residual convolutional layer
        self.res0 = ResidualLayer(self.structure[0], self.structure[1])

        # second residual convolutional layer
        self.res1 = ResidualLayer(self.structure[1], self.structure[2])

        # third residual convolutional layer
        self.res2 = ResidualLayer(self.structure[2], self.structure[3])

        self.pooling = nn.MaxPool2d(kernel_size=4, stride=1, padding=0) 
        self.classifier = nn.Linear(self.structure[3], self.out_classes)

        self.layers = [self.conv0, self.res0, self.res1, self.res2, self.classifier]

    def set_size(self, size):
        self.size = size
        self.structure = self.structures[self.size]
        self.__set_structure()
    
    def set_structure(self, structure:list=None):
        if structure is not None:
            self.size = None
            self.structure = structure
            self.__set_structure()
       

    def forward(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = F.relu(out)
        out = self.res0(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = F.softmax(out, dim=1)
        return out


class ResidualLayer(nn.Module):
    def __init__(self, input_channels, out_classes) -> None:
        super(ResidualLayer, self).__init__()
        
        self.mainpath = nn.Sequential(
            nn.Conv2d(input_channels, out_classes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_classes),
            nn.ReLU(),
            nn.Conv2d(out_classes, out_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_classes)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(input_channels, out_classes, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_classes)
        )
    
    def forward(self, x):
        out = self.mainpath(x)
        residual = self.shortcut(x)
        out += residual
        out = F.relu(out)
        return out
