from torch import nn

class MetaLearner(nn.Module):
    def __init__(self, input_channels, out_classes) -> None:
        super(MetaLearner, self).__init__()
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.model = nn.Sequential(
            nn.Dropout(0.9),
            nn.Linear(input_channels, out_classes),
        )
    
    def forward(self, x):
        x = self.model(x)
        return x