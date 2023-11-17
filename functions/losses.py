import torch

class CategoricalCrossEntropyLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-9):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, pred_dist, target):
        # inputs: pred_dist (batch_size, num_classes), target (batch_size)
        # outputs: loss (batch_size)
        pred_dist = torch.clamp(pred_dist, min=self.epsilon, max=1-self.epsilon)
        log_pred_dist = torch.log(pred_dist)
        target = target.unsqueeze(1)
        loss = -torch.sum(log_pred_dist * target, dim=1)
        return loss
    