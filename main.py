import torch
from utils.utils import *
from utils.prune import *
from models.baseline.alexnet import AlexNet
from models.baseline.lenet import LeNet
from models.baseline.resnet8 import ResNet8
from models.baseline.mobilenet import MobileNetV2
from models.meta.ensemble import Ensemble
from functions.fusions import *
from functions.losses import *
from models.meta.metalearner import MetaLearner

dataset = 'cifar10'
model_name = 'lenet'
batch_size = 32
train_ratio = 0.8
learning_rate = 0.001
device = get_device()

models = {'alexnet': AlexNet, 'lenet': LeNet, 'resnet8': ResNet8, 'mobilenetv2': MobileNetV2}
selectedModel = models[model_name]

# get dataloaders
train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes = get_dataset(ds=dataset, batch_size=batch_size, train_ratio=train_ratio, weighted=False)

# define baseline model
############################################
# model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=1)
# model = initialize(model, weights=None)
############################################

# get MACs/memory profile
############################################
# structures = model.structures
# for i in structures:
#     print("Number of learners: {0}".format(i))
#     model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=i)
#     # model = initialize(model, weights=None)
#     # get macs
#     macs = model.get_macs(input_shape)
#     print("{0} macs: {1:.2f}K".format(model_name, macs*i/1e3))
#     weight_size = model.get_weight_size()
#     print("{0} weight size: {1:.2f}K".format(model_name, weight_size*i/1e3))
############################################

# get MACs/memory profile for MobileNetV2
############################################
# interverted_residual_settings = model.interverted_residual_settings
# for i in interverted_residual_settings:
#     print("Number of learners: {0}".format(i))
#     model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=i)
#     # get macs
#     macs = model.get_macs(input_shape)
#     print("{0} macs: {1:.2f}K".format(model_name, macs*i/1e3))
#     weight_size = model.get_weight_size()
#     print("{0} weight size: {1:.2f}K".format(model_name, weight_size*i/1e3))
############################################

# train baseline model
############################################
# epochs = 64
# criterion = torch.nn.CrossEntropyLoss()
# best_model = None
# best_accuracy = 0
# train_iter = 16
# for i in range(train_iter):
#     model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=1)
#     model = initialize(model, weights=None)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
#     accuracy, model = train(model, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device)
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_model = copy.deepcopy(model)
# print("{0} accuracy: {1:.1f}%".format(model_name, accuracy))
# torch.save(best_model.state_dict(), './checkpoint/{}_{}.pt'.format(model_name, dataset))
############################################

# prune model
############################################
prune_size = 2          # number of learners in ensemble
prune_iter = 4          # number of iterations for each learner to find the best learner
prune_step = 4          # number of steps to gradually prune each learner
# create an ensemble
ensemble = Ensemble(fusion=unweighted_fusion)
criterion = torch.nn.CrossEntropyLoss()

for i in torch.arange(prune_size):
    # baseline model
    epochs = 64
    best_model = None
    best_accuracy = 0
    train_iter = 4
    for j in range(train_iter):
        print("Training model {0} iteration {1}".format(i, j))
        model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=1)
        model = initialize(model, weights=None)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
        accuracy, model = train(model, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)
    print("{0} accuracy: {1:.1f}%".format(model_name, accuracy))
    torch.save(best_model.state_dict(), './checkpoint/{}_{}_{}.pt'.format(model_name, dataset, i))

    # prune model
    best_accuracy = 0
    best_model = None
    for j in torch.arange(prune_iter):
        model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=1)
        model.load_state_dict(torch.load('./checkpoint/{}_{}_{}.pt'.format(model_name, dataset, i)))
        print("Pruning learner {0} iteration {1}".format(i, j))
        if isinstance(model, MobileNetV2):
            structure = []
            for t, c, n, s in model.interverted_residual_settings[1]:
                structure.append(c)
            target_structure = []
            for t, c, n, s in model.interverted_residual_settings[prune_size]:
                target_structure.append(c)
        else:
            structure = model.structures[1]
            target_structure = model.structures[prune_size]
        prev_structure = structure
        prev_model = model
        for k in range(1, prune_step+1):
            if k == prune_step: epochs = 64
            else: epochs = 8
            print('Pruning learner {0} iteration {1} Iterative pruning {2}'.format(i, j, k))
            current_structure = [int((target_structure[i]-structure[i])*k/prune_step+structure[i]) for i in torch.arange(len(structure))]
            print('current structure: {0}'.format(current_structure))
            pruned_model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=1)
            pruned_model.set_structure(structure=current_structure)
            optimizer = torch.optim.Adam(pruned_model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)
            pruned_model = prune_model(prev_model, pruned_model, prev_structure, current_structure)
            pruned_accuracy, pruned_model = train(pruned_model, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device)
            test_accuracy, test_loss = evaluate(pruned_model, test_loader, criterion, device)
            prev_model = copy.deepcopy(pruned_model)
            prev_structure = current_structure
        print("{0} pruned accuracy: {1:.1f}%".format(model_name, test_accuracy))
        # get best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = copy.deepcopy(pruned_model)

    ensemble.add_weak_learner(best_model, alpha=1.0)
# save ensemble
torch.save(ensemble, './checkpoint/ensemble_{}_{}_{}'.format(model_name, dataset, prune_size))
############################################

# Adaboost ensemble
# prune_size = 4          # number of learners in ensemble
# prune_iter = 1          # number of iterations for each learner to find the best learner
# prune_step = 4          # number of steps to gradually prune each learner
# train_loader, valid_loader, test_loader, input_shape, input_channels, out_classes = get_dataset(ds=dataset, batch_size=batch_size, train_ratio=train_ratio, weighted=True)
# # create an ensemble
# ensemble = Ensemble(fusion=unweighted_fusion)
# criterion = torch.nn.CrossEntropyLoss(reduction='none')

# for i in torch.arange(prune_size):
#     best_accuracy = 0
#     best_model = None
#     for j in torch.arange(prune_iter):
#         model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=1)
#         model.load_state_dict(torch.load('./checkpoint/{}_{}.pt'.format(model_name, dataset)))
#         print("Pruning learner {0} iteration {1}".format(i, j))
#         structure = model.structures[1]
#         target_structure = model.structures[prune_size]
#         prev_structure = structure
#         prev_model = model
#         for k in range(1, prune_step+1):
#             if k == prune_step: epochs = 64
#             else: epochs = 16
#             print('Pruning learner {0} iteration {1} Iterative pruning {2}'.format(i, j, k))
#             current_structure = [int((target_structure[i]-structure[i])*k/prune_step+structure[i]) for i in torch.arange(len(structure))]
#             print('current structure: {0}'.format(current_structure))
#             pruned_model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=1)
#             pruned_model.set_structure(structure=current_structure)
#             optimizer = torch.optim.Adam(pruned_model.parameters(), lr=1e-3, weight_decay=1e-4)
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)
#             pruned_model = prune_model(prev_model, pruned_model, prev_structure, current_structure)
#             pruned_accuracy, pruned_model = train(pruned_model, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device, weighted_dataset=True, weighted_train=True)
#             test_accuracy, test_loss = evaluate(pruned_model, test_loader, criterion, device)
#             prev_model = copy.deepcopy(pruned_model)
#             prev_structure = current_structure
#         print("{0} pruned accuracy: {1:.1f}%".format(model_name, test_accuracy))
#         # get best model
#         if test_accuracy > best_accuracy:
#             best_accuracy = test_accuracy
#             best_model = copy.deepcopy(pruned_model)
#     update_dataset_weights(best_model, train_loader, criterion, device)
#     ensemble.add_weak_learner(best_model, alpha=1.0)

# # save ensemble
# torch.save(ensemble, './checkpoint/ensemble_{}_{}_{}_adaboost'.format(model_name, dataset, prune_size))

# # evaluate ensemble
# ############################################
# load ensemble
ensemble = torch.load('./checkpoint/ensemble_{}_{}_{}'.format(model_name, dataset, prune_size))
for learner in ensemble.learners:
    learner_accuracy, learner_loss = evaluate(learner, test_loader, criterion, device)
    print("Learner accuracy: {0:.1f}%".format(learner_accuracy))
ensemble_accuracy, ensemble_loss = evaluate(ensemble, test_loader, criterion, device)
print("Ensemble accuracy: {0:.1f}%".format(ensemble_accuracy))
# ############################################

meta_learner = MetaLearner(input_channels=out_classes*len(ensemble.learners), out_classes=out_classes)
meta_learner = initialize(meta_learner, weights=None)
epochs = 64
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(meta_learner.parameters(), lr=1e-3, weight_decay=5e-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
accuracy, meta_learner = train_metalearner(meta_learner, train_loader, valid_loader, ensemble.learners, epochs, criterion, optimizer, scheduler, device)
meta_learner_accuracy, meta_learner_loss = evaluate_meta_learner(meta_learner, test_loader, ensemble.learners, criterion, device)
print("Meta learner accuracy: {0:.1f}%".format(meta_learner_accuracy))
############################################

# K = 2
# size = 2
# epochs = 4
# iter_step = 1
# ensemble = GrowNet(fusion=weighted_fusion)
# for k in range(K):
#     best_accuracy = 0
#     best_model = None
#     alpha = torch.nn.Parameter(torch.tensor([1/K])).requires_grad_(True)
#     best_alpha = None
#     for i in range(iter_step):
#         base_model = selectedModel(input_channels=input_channels, out_classes=out_classes, size=size)
#         weak_learner = WeakLearner(base_model=base_model, residual_size=ensemble.get_last_residual_size())
#         # train weak learner
#         print("Training weak learner {0} iteration {1}".format(k, i))
#         criterion = torch.nn.CrossEntropyLoss()
#         optimizer = torch.optim.Adam([{'params': weak_learner.parameters()}, {'params': alpha}], lr=1e-3, weight_decay=1e-4)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
#         accuracy, weak_learner = train_weak_learner(weak_learner, ensemble, alpha, k, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, device)
#         print("Weak learner accuracy: {0:.1f}%".format(accuracy))
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_model = copy.deepcopy(weak_learner)
#             best_alpha = copy.deepcopy(alpha)
#     # add weak learner to ensemble
#     ensemble.add_learner(best_model, best_alpha.item())

# evaluate ensemble
# ############################################
# for k in range(K):
#     learner = ensemble.learners[k]
#     learner_accuracy, learner_loss = evaluate_weak_learner(learner, ensemble, k, test_loader, criterion, device)
#     print("Learner accuracy: {0:.1f}%".format(learner_accuracy))
#     print("Learner weight: {0:.2f}".format(ensemble.alpha[k].item()))
# ensemble_accuracy, ensemble_loss = evaluate(ensemble, test_loader, criterion, device)
# print("Ensemble accuracy: {0:.1f}%".format(ensemble_accuracy))
############################################


