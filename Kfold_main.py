import torch
from torchsummary import summary
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from loader import get_train_dataloader, get_test_dataloader, parse_data
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.data as data_utils
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import KFold, StratifiedKFold

from models.SE import SE, SE1
from models.SK import SK
from settings import *
import seaborn as sns
import pandas as pd

from argparse import ArgumentParser
from evaluate import evaluate_v1


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def kfold_train():

    model_catalog = {
        'SE': SE(),
        'SK': SK()
    }

    dataset_catalog = {
        'normal': DATASET_PATH.joinpath('train'),
        'adasyn': DATASET_PATH.joinpath('train')
    }
    optimizer = {
        'Adam': optim.Adam,
        'RMSprop': optim.RMSprop
    }

    train_df = pd.read_csv(dataset_catalog[DATASET_NAME])
    x_train, y_train = parse_data(train_df)

    dataset_catalog_test = {
        'normal': DATASET_PATH.joinpath('test'),
        'adasyn': DATASET_PATH.joinpath('test')
    }
    test_df = pd.read_csv(dataset_catalog_test[DATASET_NAME])
    x_test, y_test = parse_data(train_df)

    #train_loader, val_loader, data_loader = get_train_dataloader(dataset_catalog[DATASET_NAME])
    #dataset = ConcatDataset([train_loader, val_loader])

    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Operation On:', device)

    train_loss_per_epoch = []
    val_loss_per_epoch = []
    max_val = float('inf')
    max_val_acc = float('-inf')

    k_folds = 5
    kf = KFold(n_splits=k_folds)
    skf = StratifiedKFold(n_splits=k_folds)

    results = {}

    for fold, (train_ids, test_ids) in enumerate(kf.split(x_train, y_train)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            data_utils.TensorDataset(torch.tensor(x_train.reshape((-1, 1, 11, 11))), torch.tensor(y_train)),
            batch_size=10, sampler=train_subsampler)

        testloader = torch.utils.data.DataLoader(
            data_utils.TensorDataset(torch.tensor(x_train.reshape((-1, 1, 11, 11))), torch.tensor(y_train)),
            batch_size=10, sampler=test_subsampler)

        model = model_catalog[MODEL_NAME]
        model.apply(reset_weights)
        model.cuda()

        opt = optimizer[CONFIGS['model']['optim']](model.parameters(), lr=LR)
        scheduler_opt = {
            'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=FACTOR,
                                                                            patience=PATIENCE, threshold=1e-3,
                                                                            min_lr=MIN_LR, verbose=True),
            'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=FACTOR)
        }
        scheduler = scheduler_opt[SCHEDULER]

        for epoch in range(0, EPOCHS):
            # Print epoch
            print(f'Starting epoch {epoch + 1}')

            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(testloader):

                # Get inputs
                inputs, targets = data

                inputs, targets = inputs.float().cuda(), targets.cuda()

                # Zero the gradients
                opt.zero_grad()

                # Perform forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, torch.argmax(targets, dim=1))

                # Perform backward pass
                loss.backward()

                # Perform optimization
                opt.step()

                # Print statistics
                current_loss += loss.item()
                if i % 500 == 499:
                    print('Loss after mini-batch %5d: %.3f' %
                          (i + 1, current_loss / 500))
                    current_loss = 0.0

            # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')

        # Saving the model
        save_path = f'./model-fold-{fold}.pth'
        torch.save(model.state_dict(), save_path)

        # Evaluationfor this fold
        correct, total = 0, 0
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader):
                # Get inputs
                inputs, targets = data

                inputs, targets = inputs.float().cuda(), targets.cuda()

                # Generate outputs
                outputs = model(inputs)

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('--------------------------------')
            results[fold] = 100.0 * (correct / total)

        # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum / len(results.items())} %')


if __name__ == '__main__':
    # config_catalog = []
    # for config in os.listdir(CONFIGS_PATH):
    # config_catalog.append(config)

    # print(config_catalog)
    # parser = ArgumentParser()
    # parser.add_argument('--configs', help='choose config file', type=str, choices=config_catalog)
    # CONFIG_NAME = parser.parse_args()
    # print(CONFIG_NAME)

    if TRAINABLE == 1:
        train_model = kfold_train()

