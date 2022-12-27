import torch
from torchsummary import summary
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

import models.BAM
from loader import get_train_dataloader, get_test_dataloader
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score

from models.SE import SE, SEForward
from models.SK import SK
from models.CBAM import CBAM
from models.Residual import RB, ResNet, ResidualBlock
from models.CNN import CNN

from settings import *
import seaborn as sns
import pandas as pd

from argparse import ArgumentParser
from evaluate import evaluate_v1


def train():
    # Model Configuration
    model_catalog = {
        'SE': SE(),
        'SK': SK(),
        'CBAM': CBAM(),
        'RB': ResNet(ResidualBlock, [2, 2, 2]),
        'CNN': CNN()

    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_catalog[MODEL_NAME]
    model.to(device)
    best_model = model
    print('Operation On:', device)
    summary(model=model, input_size=(1, 11, 11))

    # LOAD TRAINING DATA
    train_loader = get_train_dataloader(DATASET_PATH.joinpath('train'), mode=DATASET_TYPE)  # , mode='np'

    # Optimizer and Scheduler Configuration
    optimizer = {
        'Adam': optim.Adam,
        'RMSprop': optim.RMSprop
    }
    opt = optimizer[CONFIGS['model']['optim']](model.parameters(), lr=LR)
    scheduler_opt = {
        'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=FACTOR,
                                                                        patience=PATIENCE, threshold=1e-3,
                                                                        min_lr=MIN_LR, verbose=True),
        'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=FACTOR)
    }
    scheduler = scheduler_opt[SCHEDULER]

    # Define Loss Function
    criterion = nn.CrossEntropyLoss()

    train_loss_per_epoch = []
    min_train_loss = float('+inf')

    print('START TRAINING ...')
    for epoch in range(EPOCHS):

        train_loss = 0.0

        train_true_labels = []
        train_pred_labels = []
        for idx, (x_train, x_labels) in enumerate(train_loader):
            x_train, x_labels = x_train.float().to(device), x_labels.to(device)

            opt.zero_grad()
            output = model(x_train)

            loss = criterion(output, torch.argmax(x_labels, dim=1))
            loss.backward()
            opt.step()
            train_loss += loss.item()

            train_true_labels.append(np.argmax(x_labels.cpu().detach().numpy(), axis=1))
            train_pred_labels.append(np.argmax(output.cpu().detach().numpy(), axis=1))

        train_loss = train_loss / len(train_loader)
        train_loss_per_epoch.append(train_loss)

        train_true_labels = np.concatenate(train_true_labels)
        train_pred_labels = np.concatenate(train_pred_labels)
        train_acc = accuracy_score(train_true_labels, train_pred_labels)

        if SCHEDULER == 'ReduceLROnPlateau':
            scheduler.step(train_loss_per_epoch[-1])

        print('Epoch: {} \tTraining Loss: {:.5f}\tTraining Accuracy: {:.5f}\t'.format(epoch + 1, train_loss, train_acc))

        if round(train_loss_per_epoch[-1], 3) < round(min_train_loss, 3):
            min_loss_prev = min_train_loss
            min_train_loss = train_loss_per_epoch[-1]

            print('[*] Loss Reduced from {:.5f} to {:.5f}'.format(min_loss_prev, min_train_loss))
            best_model = model

    return best_model
    # torch.save(best_model, SAVE_PATH.joinpath(f'{MODEL_NAME}_{DATASET_NAME}.pth'))
    # assert (best_model_flag == True), torch.save(model, SAVE_PATH.joinpath(f'{classifier_description}.pth'))


def test(model=None):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and True) else "cpu")
    if TRAINABLE == 0:
        model_name = f'{MODEL_NAME}_{DATASET_NAME}'
        model_path = SAVE_PATH.joinpath(f'{model_name}.pth')
        trained_model = torch.load(str(model_path))
        trained_model.to(device=device)
        trained_model.trainable = False
        print(f'MODEL {model_name} LOADED')
    else:
        trained_model = model

    test_loader = get_test_dataloader(DATASET_PATH.joinpath('test'), mode=DATASET_TYPE)

    true_labels = []
    for _, y_test in test_loader:
        true_labels.append(torch.argmax(y_test, dim=1).numpy())
    true_labels = np.concatenate(true_labels)

    predictions = []
    for x_test, y_test in test_loader:
        x_test = x_test.float()
        x_test = x_test.to(device)

        output = trained_model(x_test)
        output = output.cpu().detach().numpy()
        predictions.append(np.argmax(output, axis=1))

    predictions = np.concatenate(predictions)

    classes = ["Dos", "Probe", "R2L", "U2R", "normal"]
    cfm = confusion_matrix(true_labels, predictions, labels=list(range(5)))
    cm_array_df = pd.DataFrame(cfm, index=classes, columns=classes)
    h_plot = sns.heatmap(cm_array_df, annot=True, fmt='g')

    acc = accuracy_score(true_labels, predictions)
    kappa = cohen_kappa_score(true_labels, predictions)
    metrics = evaluate_v1(true_labels, predictions)
    # print(f'\tAccuracy: {acc}\tKappa: {kappa}')
    print(metrics)

    result = {
        "accuracy": acc,
        "config": CONFIGS['model']
    }

    if os.path.isfile(f'{SAVE_PATH}/{MODEL_NAME}_{DATASET_NAME}.json'):
        # read old .json file check acc update if needed
        old_result = open(f'{SAVE_PATH}/{MODEL_NAME}_{DATASET_NAME}.json')
        old_result = json.load(old_result)
        if result['accuracy'] > old_result['accuracy']:
            torch.save(trained_model, SAVE_PATH.joinpath(f'{MODEL_NAME}_{DATASET_NAME}.pth'))

            figure = h_plot.get_figure()
            figure.savefig(SAVE_PATH.joinpath(f"cfm_{MODEL_NAME}_{DATASET_NAME}.jpg"))
            figure.clf()

            with open(f'{SAVE_PATH}/{MODEL_NAME}_{DATASET_NAME}.json', "w") as result_json:
                json.dump(result, result_json)
    else:
        torch.save(trained_model, SAVE_PATH.joinpath(f'{MODEL_NAME}_{DATASET_NAME}.pth'))

        figure = h_plot.get_figure()
        figure.savefig(SAVE_PATH.joinpath(f"cfm_{MODEL_NAME}_{DATASET_NAME}.jpg"))
        figure.clf()

        with open(f'{SAVE_PATH}/{MODEL_NAME}_{DATASET_NAME}.json', "w") as result_json:
            json.dump(result, result_json)


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
        train_model = train()
        test(train_model)
    else:
        test()
