import torch
from torch import nn

class GrowNet(torch.nn.Module):
    def __init__(self, fusion=None) -> None:
        super(GrowNet, self).__init__()
        self.learners = torch.nn.ModuleList()
        self.fusion = fusion
    
    def forward(self, x):
        inputs = [x, None]
        outputs = []
        for learner in self.learners:
            output = learner(inputs)
            residual = learner.get_residual(inputs)
            inputs = [x, residual]
            outputs.append(output)
        return self.fusion(outputs, None)

    def get_last_residual(self, x):
        if len(self.learners) == 0:
            return None
        return self.get_residuals(x)[-1]

    def get_residuals(self, x):
        if len(self.learners) == 0:
            return None
        inputs = [x, None]
        residuals = []
        for learner in self.learners:
            residual = learner.get_residual(inputs)
            residuals.append(residual)
            inputs = [x, residual]
        return residuals

    def get_last_residual_size(self):
        if len(self.learners) == 0:
            return 0
        return self.learners[-1].classifier[-2].in_features

    def add_learner(self, learner):
        self.learners.append(learner)


class WeakLearner(nn.Module):
    def __init__(self, base_model=None, residual_size=0) -> None:
        super(WeakLearner, self).__init__()
        self.base_model = base_model
        self.combine = nn.Linear(residual_size + base_model.get_classifier()[0].in_features, base_model.get_classifier()[0].out_features)
        self.representation = base_model.get_representation()
        self.classifier = nn.Sequential(self.combine, base_model.get_classifier()[1:])
        nn.init.xavier_uniform_(self.combine.weight)
        self.residual_size = residual_size
        self.alphas = nn.ParameterList()

    def forward(self, x):
        representation = self.representation(x[0])
        representation = representation.view(representation.size(0), -1)
        if self.residual_size > 0:
            residual = x[1]
            residual = residual.view(residual.size(0), -1)
            output = torch.cat((representation, residual), dim=1)
        else:
            output = representation
        return self.classifier(output)

    def get_residual(self, x):
        representation = self.representation(x[0])
        representation = representation.view(representation.size(0), -1)
        if self.residual_size > 0:
            residual = x[1]
            residual = residual.view(residual.size(0), -1)
            output = torch.cat((representation, residual), dim=1)
        else:
            output = representation
        return self.classifier[:-2](output)
        