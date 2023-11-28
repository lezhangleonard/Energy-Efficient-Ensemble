import torch
from torch.utils.data import Dataset

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
        y_hat = y_hat.to(self.weights.device)
        y = y.to(self.weights.device)
        with torch.no_grad():
            x = torch.log(torch.clamp(y_hat, min=eps))
            x = torch.mm(torch.transpose(y, 0, 1), x)
            x = -self.alpha * (self.__len__ - 1) / self.__len__ * x
            x = torch.exp(torch.clamp(x, min=eps))
            x = x.view(-1)
            self.weights[indices] *= x
            self.weights /= self.weights.sum()

def extract_data_targets(subset, dataset):
    return [dataset.data[idx] for idx in subset.indices], [dataset.targets[idx] for idx in subset.indices]