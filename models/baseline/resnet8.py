import torch 
import torch.nn as nn
from thop import profile
import torch.nn.functional as F

class ResNet8(nn.Module):
    def __init__(self, input_channels, out_classes, quantization=False, act_int=0, act_dec=0, e2cnn_size=1 ):
        super(ResNet8, self).__init__()
        if e2cnn_size == 1:
            structure = [16, 16, 32, 64]
        elif e2cnn_size == 2:
            structure = [10, 10, 26, 42]

        self.conv = nn.Conv2d(input_channels, structure[0], k_size=3, stride=1, padding=0, with_bn=False, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec)

        self.residual = nn.Sequential( 
            ResidualLayer(structure[0], structure[1], skip_proj=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            ResidualLayer(structure[1], structure[2], skip_proj=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec), 
            ResidualLayer(structure[2], structure[3], skip_proj=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec) 
            )

        self.pooling = nn.MaxPool2d(kernel_size=8, stride=1, padding=0) 

        self.classifier = nn.Linear(structure[3], out_classes, with_relu=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec)
            

       

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = self.residual(x)
        # print(x.shape)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # print(x.shape)
        return x
