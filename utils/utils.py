import torch
import datasets
import copy
from torch.nn.utils import prune
from datasets.datasets import *

def evaluate(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: str):
    model = model.to(device)
    criterion = criterion.to(device)
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for sample, target in data_loader:
            sample, target = sample.to(device), target.to(device)
            output = model(sample)
            if criterion.__class__.__name__ == 'JointLoss':
                loss = criterion(sample, output, target)
            else:
                loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy += 100. * correct / len(data_loader.dataset)
            sample.detach(), target.detach(), output.detach(), pred.detach()
            del sample, target, output, pred
    return accuracy, loss

def evaluate_weak_learner(model: torch.nn.Module, ensemble: torch.nn.Module, k: int, data_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: str):
    model = model.to(device)
    ensemble = ensemble.to(device)
    criterion = criterion.to(device)
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for sample, target in data_loader:
            sample, target = sample.to(device), target.to(device)
            if ensemble.get_residuals(sample) is None:
                residual = None
            else:
                residual = ensemble.get_residuals(sample)[k - 1]
            output = model([sample, residual])
            if criterion.__class__.__name__ == 'JointLoss':
                loss = criterion(sample, output, target)
            else:
                loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy += 100. * correct / len(data_loader.dataset)
            sample.detach(), target.detach(), output.detach(), pred.detach()
            if residual is not None:
                residual.detach()
            del sample, target, output, pred, residual
    return accuracy, loss

def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, 
          epochs: int, criterion, optimizer, scheduler, device: str, weighted_dataset=False, weighted_train=False) -> (float, torch.nn.Module):
    if device == 'cuda':
        torch.cuda.empty_cache()
    min_lr = 1e-5
    best_accuracy = 0
    best_model = None
    model = model.to(device)
    criterion = criterion.to(device)
    model.train()
    for epoch in torch.arange(epochs):
        if weighted_dataset:
            for batch_idx, (sample, target, weight) in enumerate(train_loader):
                sample, target, weight = sample.to(device), target.to(device), weight.to(device)
                optimizer.zero_grad()
                output = model(sample)
                losses = criterion(output, target)
                weighted_losses = losses * weight
                loss = weighted_losses.mean()
                loss.backward()
                optimizer.step()
                if weighted_train:
                    update_dataset_weights(train_loader, batch_idx, output, target)
                sample.detach(), target.detach(), output.detach(), weight.detach()
                del sample, target, output, weight
        else:
            for sample, target in train_loader:
                sample, target = sample.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(sample)
                if criterion.__class__.__name__ == 'JointLoss':
                    loss = criterion(sample, output, target)
                else:
                    loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                sample.detach(), target.detach(), output.detach()
                del sample, target, output
        val_accuracy, val_losses = evaluate(model, val_loader, criterion, device)
        val_loss = val_losses.mean()
        scheduler.step(metrics=val_loss)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = copy.deepcopy(model)
        print("Epoch {0}: Accuracy={1:.1f}%, Loss={2:.6f}".format(epoch, val_accuracy, val_loss))
        if scheduler._last_lr[-1] < min_lr:
            break
    del criterion, model
    return best_accuracy, best_model

def train_weak_learner(model: torch.nn.Module, ensemble: torch.nn.Module, k: int, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, 
          epochs: int, criterion, optimizer, scheduler, device: str, weighted_dataset=False, weighted_train=False) -> (float, torch.nn.Module):
    if device == 'cuda':
        torch.cuda.empty_cache()
    min_lr = 1e-5
    best_accuracy = 0
    best_model = None
    model = model.to(device)
    ensemble = ensemble.to(device)
    criterion = criterion.to(device)
    model.train()
    for epoch in torch.arange(epochs):
        if weighted_dataset:
            for batch_idx, (sample, target, weight) in enumerate(train_loader):
                sample, target, weight = sample.to(device), target.to(device), weight.to(device)
                residual = ensemble.get_last_residual(sample)
                optimizer.zero_grad()
                output = model([sample, residual])
                losses = criterion(output, target)
                weighted_losses = losses * weight
                loss = weighted_losses.mean()
                loss.backward()
                optimizer.step()
                if weighted_train:
                    update_dataset_weights(train_loader, batch_idx, output, target)
                sample.detach(), target.detach(), output.detach(), weight.detach()
                if residual is not None:
                    residual.detach()       
                del sample, target, output, weight, residual
        else:
            for sample, target in train_loader:
                sample, target = sample.to(device), target.to(device)
                residual = ensemble.get_last_residual(sample)
                optimizer.zero_grad()
                output = model([sample, residual])
                if criterion.__class__.__name__ == 'JointLoss':
                    loss = criterion(sample, output, target)
                else:
                    loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                sample.detach(), target.detach(), output.detach()
                if residual is not None:
                    residual.detach()
                del sample, target, output, residual
        val_accuracy, val_losses = evaluate_weak_learner(model, ensemble, k, val_loader, criterion, device)
        val_loss = val_losses.mean()
        scheduler.step(metrics=val_loss)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = copy.deepcopy(model)
        print("Epoch {0}: Accuracy={1:.1f}%, Loss={2:.6f}".format(epoch, val_accuracy, val_loss))
        if scheduler._last_lr[-1] < min_lr:
            break
    del criterion, model
    return best_accuracy, best_model



def initialize(model: torch.nn.Module, weights: str) -> torch.nn.Module:
    if weights is None:
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0)
    else:
        model.load_state_dict(weights)
    return model

def get_device() -> str:
    torch.backends.cudnn.benchmark = True
    print("cudnn backends:", torch.backends.cudnn.version())
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device: {}".format(device))
    return device

def get_dataset(ds: str='mnist', batch_size: int=128, train_ratio: float=0.8, weighted: bool=False) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, int):
    if ds == 'mnist':
        input_channels = 1
        out_classes = 10
        input_shape = (1, 28, 28)
    elif ds == 'cifar10':
        input_channels = 3
        out_classes = 10
        input_shape = (3, 32, 32)
    elif ds == 'cifar100':
        input_channels = 3
        out_classes = 100
        input_shape = (3, 32, 32)
    
    dataset = datasets.__dict__[ds]
    test_loader = dataset.load_test_data(batch_size=batch_size)
    train_loader, valid_loader = dataset.load_train_val_data(batch_size=batch_size, train_val_split=train_ratio, weighted=weighted)

    return train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes

def prune_model(model: torch.nn.Module, pruned_model: torch.nn.Module, structure: list, prune_structure: list) -> torch.nn.Module:
    previous_num_filters = None
    previous_filter_indices = None
    for i in torch.arange(len(model.layers) - 1):
        layer = model.layers[i]
        if isinstance(layer, torch.nn.Conv2d):
            num_filters_to_keep = prune_structure[i]
            prune.ln_structured(layer, name='weight', amount=layer.out_channels - num_filters_to_keep, n=2, dim=0)
            filter_indices = torch.nonzero(layer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
            prune.remove(layer, 'weight')
            if previous_filter_indices is not None:
                current_weight = layer.weight.data[filter_indices][:, previous_filter_indices].clone()
            else:
                current_weight = layer.weight.data[filter_indices].clone()
            pruned_model_layer = pruned_model.layers[i]
            pruned_model_layer.weight.data = current_weight
            pruned_model_layer.weight.requires_grad = True
            pruned_model_layer.bias.data = layer.bias.data[filter_indices].clone()
            pruned_model_layer.bias.requires_grad = True
            previous_filter_indices = filter_indices
        if isinstance(layer, torch.nn.Linear):
            # set uniform random weights
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0)
            
            # num_neurons_to_keep = prune_structure[i]
            # prune.ln_structured(layer, name='weight', amount=layer.out_features - num_neurons_to_keep, n=2, dim=0)
            # neuron_indices = torch.nonzero(layer.weight_mask.sum(dim=1), as_tuple=False).view(-1)
            # prune.remove(layer, 'weight')
            # if previous_filter_indices is not None and layer.in_features == structure[i-1]:
            #     current_weight = layer.weight.data[neuron_indices][:, previous_filter_indices].clone()
            # else:
            #     current_weight = layer.weight.data[neuron_indices].clone()
            # pruned_model_layer = pruned_model.layers[i]
            # pruned_model_layer.weight.data = current_weight
            # pruned_model_layer.weight.requires_grad = True
            # pruned_model_layer.bias.data = layer.bias.data[neuron_indices].clone()
            # pruned_model_layer.bias.requires_grad = True
    return pruned_model

def update_dataset_weights(dataloader, batch_index, y_hat, y, eps=1e-9):
    if dataloader.dataset is WeightedDataset:
        indices = torch.arange(batch_index*dataloader.batch_size, (batch_index+1)*dataloader.batch_size)
        dataloader.dataset.update_weights(indices, y_hat, y, eps=eps)