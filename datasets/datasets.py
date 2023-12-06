import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class WeightedDataset(Dataset):
    def __init__(self, data, targets, alpha, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.weights = torch.ones(len(self.data))
        self.alpha = alpha

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        w = self.weights[index]
        if self.transform:
            x = self.transform(x)
        return x, y, w

    def __len__(self):
        return len(self.data)

    def update_weights(self, indices, y_hat, y, eps=1e-9):
        y = y.to(self.weights.device)
        y_hat = y_hat.to(self.weights.device)
        y_onehot = torch.zeros_like(y_hat)
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        y_onehot = y_onehot.to(self.weights.device)
        y_hat = y_hat.unsqueeze(-1)
        y_onehot = y_onehot.unsqueeze(1)
        with torch.no_grad():
            x = torch.log(torch.clamp(y_hat, min=eps))
            x = torch.bmm(y_onehot, x)
            x = -self.alpha * (self.__len__() - 1) / self.__len__() * x
            x = torch.exp(torch.clamp(x, min=eps))
            x = x.squeeze()
            self.weights[indices] *= x
            self.weights = F.softmax(self.weights, dim=0)

def extract_data_targets(subset, dataset):
    return [dataset.data[idx] for idx in subset.indices], [dataset.targets[idx] for idx in subset.indices]