import torch
from torch import nn

class Ensemble(nn.Module):
    def __init__(self, fusion = None):
        super(Ensemble, self).__init__()
        self.learners = nn.ModuleList()
        self.alphas = nn.ParameterList()
        self.fusion = fusion
    
    def forward(self, x):
        outputs = []
        for weak_learner in self.learners:
            outputs.append(weak_learner(x))
        if self.fusion is None:
            return outputs[0]
        return self.fusion(outputs, self.alphas)
    
    def add_weak_learner(self, weak_learner, alpha):
        self.learners.append(weak_learner)
        self.alphas.append(alpha)
    
    def get_weak_learners(self):
        return self.learners
    
    def get_alphas(self):
        return self.alphas
    
        
