import json
import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from tqdm import tqdm

from utils import parse_data
from Dataset2Image.main import deepinsight

from configs.setting import setting
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from utils import metrics_evaluate

from models.SE import SE
from models.SP import SP
from models.SK import SK
from models.CBAM import CBAM
from models.Residual import RB, ResNet, ResidualBlock
from models.CNN import CNN, CNN_MAGNETO
from models.spse import SPSE


def setup(args: Namespace):
    i = 1
    flag = True
    SAVE_PATH_ = ''
    TRAINED_MODEL_PATH_ = ''
    CHECKPOINT_PATH_ = ''
    config = {}
    BASE_DIR = Path(__file__).resolve().parent
    while flag:
        if args.model_dir is None:

            config, config_file = setting()
            TEMP_FILENAME = f"{config['DATASET_NAME']}-{config['CLASSIFICATION_MODE']}-{config['MODEL_NAME']}-{i}"
            TEMP_PATH = BASE_DIR.joinpath(f"session/{TEMP_FILENAME}")

            if os.path.isdir(TEMP_PATH):
                i += 1
            else:
                flag = False

                os.mkdir(BASE_DIR.joinpath(f"session/{TEMP_FILENAME}"))
                os.mkdir(BASE_DIR.joinpath(f"session/{TEMP_FILENAME}/trained_models"))
                SAVE_PATH_ = BASE_DIR.joinpath(f"session/{TEMP_FILENAME}")
                TRAINED_MODEL_PATH_ = BASE_DIR.joinpath(f"session/{TEMP_FILENAME}/trained_models")

                os.mkdir(BASE_DIR.joinpath(f'{SAVE_PATH_}/model_checkpoint'))
                CHECKPOINT_PATH_ = SAVE_PATH_.joinpath(f"model_checkpoint/{TEMP_FILENAME}.pt")

                with open(f'{SAVE_PATH_}/MODEL_CONFIG.json', 'w') as f:
                    json.dump(config_file, f)

                print(f'MODEL SESSION: {SAVE_PATH_}')
        else:
            flag = False
            SAVE_PATH_ = args.model_dir
            model_dir_name = str(SAVE_PATH_).split(os.sep)[-1]  # .pt file is saving with name of folder name in session
            CHECKPOINT_PATH_ = SAVE_PATH_.joinpath(f'model_checkpoint/{model_dir_name}.pt')
            TRAINED_MODEL_PATH_ = SAVE_PATH_.joinpath('trained_models')

            CONFIGS = open(SAVE_PATH_.joinpath('MODEL_CONFIG.json'))
            CONFIGS = json.load(CONFIGS)
            print('JSON CONFIG FILE LOADED')
            config, _ = setting(CONFIGS)

    train_df = pd.read_csv(Path(config['DATASET_PATH']).joinpath('train_' + config['CLASSIFICATION_MODE'] + '.csv'))
    test_df = pd.read_csv(Path(config['DATASET_PATH']).joinpath('test_' + config['CLASSIFICATION_MODE'] + '.csv'))

    if config['DEEPINSIGHT']['deepinsight']:
        X_train, X_val = deepinsight(config['DEEPINSIGHT'], config)
        _, y_train = parse_data(train_df, dataset_name=config['DATASET_NAME'], mode=config['DATASET_TYPE'],
                                classification_mode=config['CLASSIFICATION_MODE'])
        _, y_val = parse_data(test_df, dataset_name=config['DATASET_NAME'], mode=config['DATASET_TYPE'],
                              classification_mode=config['CLASSIFICATION_MODE'])
    else:
        X_train, y_train = parse_data(train_df, dataset_name=config['DATASET_NAME'], mode=config['DATASET_TYPE'],
                                      classification_mode=config['CLASSIFICATION_MODE'])
        X_val, y_val = parse_data(test_df, dataset_name=config['DATASET_NAME'], mode=config['DATASET_TYPE'],
                                  classification_mode=config['CLASSIFICATION_MODE'])

    # X_train, image_transformer, norm_scaler = deepinsight(X_train, y_train)
    # X_val = image_transformer.transform(norm_scaler.transform(X_val))[:, :, :, 0]

    # X_train = np.load('/home/faraz/PycharmProjects/IDS/Dataset2Image/train_14x14_MI.npy')
    # X_val = np.load('/home/faraz/PycharmProjects/IDS/Dataset2Image/test_14x14_MI.npy')

    print(f'train shape: x=>{X_train.shape}, y=>{y_train.shape}')
    print(f'train shape: x=>{X_val.shape}, y=>{y_val.shape}')

    image_A_size = config['DEEPINSIGHT']['Max_A_Size']
    image_B_size = config['DEEPINSIGHT']['Max_B_Size']

    train_ld = DataLoader(
        data_utils.TensorDataset(torch.tensor(X_train.reshape((-1, 1, image_A_size, image_B_size))), torch.tensor(y_train)),
        batch_size=config['BATCH_SIZE'],
        num_workers=config['NUM_WORKER'],
        shuffle=True)

    val_ld = DataLoader(
        data_utils.TensorDataset(torch.tensor(X_val.reshape((-1, 1, image_A_size, image_B_size))), torch.tensor(y_val)),
        batch_size=1,
        num_workers=config['NUM_WORKER'])

    # MODEL CONFIGURATION
    model_catalog = {
        'SP': SP(classification_mode=config['CLASSIFICATION_MODE']),
        'SE': SE(classification_mode=config['CLASSIFICATION_MODE']),
        'SPSE': SPSE(),
        'SK': SK(),
        'CBAM': CBAM(),
        'RB': ResNet(ResidualBlock, [2, 2, 2]),
        'CNN': CNN(classification_mode=config['CLASSIFICATION_MODE']),
        'CNN_MAGNETO': CNN_MAGNETO(classification_mode=config['CLASSIFICATION_MODE'])
    }

    # OPTIMIZER CONFIGURATION
    opt = {
        'Adam': optim.Adam,
        'AdamW': optim.AdamW,
        'RMSprop': optim.RMSprop,
        'SGD': optim.SGD
    }

    device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_ = 0
    best_val_criteria_ = 0
    if args.model_dir is None:
        net = model_catalog[config['MODEL_NAME']]
        net.to(device_)
        print('Operation On:', device_)
        # summary(model=net, input_size=(3, config['IMAGE_SIZE'], config['IMAGE_SIZE']))
        with open(f"{SAVE_PATH_}/{config['MODEL_NAME']}_summary.txt", 'a') as f:
            print(net, file=f)

        optimizer_ = opt[config['OPTIMIZER']](net.parameters(), lr=config['LR'], weight_decay=config['WEIGHT_DECAY'])
    else:
        net = model_catalog[config['MODEL_NAME']]
        net_checkpoint = torch.load(Path(CHECKPOINT_PATH_))
        net.load_state_dict(net_checkpoint['model_state_dict'])
        net.to(device_)
        print('Operation On:', device_)

        optimizer_ = opt[config['OPTIMIZER']](net.parameters(), lr=config['LR'], weight_decay=config['WEIGHT_DECAY'])
        optimizer_.load_state_dict(net_checkpoint['optimizer_state_dict'])

        epoch_ = net_checkpoint['epoch']
        best_val_criteria_ = net_checkpoint['best_val_criteria']

    # SCHEDULER CONFIGURATION
    # weight decay help: if overfitting then use higher value. if the model remains under-fit then use lower values.
    scheduler_opt = {
        'ReduceLROnPlateau': lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer_, mode='max', factor=config['FACTOR'],
            patience=config['PATIENCE'], threshold=1e-3,
            min_lr=config['MIN_LR'], verbose=True),

        'ExponentialLR': lr_scheduler.ExponentialLR(optimizer=optimizer_, gamma=config['GAMMA'], verbose=True),

        'lr_scheduler': lr_scheduler.MultiStepLR(optimizer=optimizer_, milestones=[20, 40, 50], gamma=config['GAMMA'],
                                                 verbose=True)
    }
    scheduler_ = scheduler_opt[config['SCHEDULER']]

    if config['CLASSIFICATION_MODE'] == 'binary':
        criterion_ = nn.BCELoss()
    elif config['CLASSIFICATION_MODE'] == 'multi':
        criterion_ = nn.CrossEntropyLoss()

    return net, train_ld, val_ld, optimizer_, scheduler_, criterion_, device_, SAVE_PATH_, \
        TRAINED_MODEL_PATH_, CHECKPOINT_PATH_, epoch_, best_val_criteria_, config


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_dir', help='path to model in session folder in order to resume training using '
                                            'checkpoint files', type=Path, required=False)

    # CREATE SESSION AND CONFIGURE FOR TRAINING
    model, train_loader, val_loader, optimizer, scheduler, criterion, device, SAVE_PATH, TRAINED_MODEL_PATH, \
        CHECKPOINT_PATH, pre_epoch, best_val_criteria, config = setup(args=parser.parse_args())

    # Set up logging and tensorboard
    logging.basicConfig(filename=f'{SAVE_PATH}/training.log', level=logging.INFO)
    writer = SummaryWriter(log_dir=f'{SAVE_PATH}')

    best_valid_acc = best_val_criteria
    epoch_since_improvement = 0
    train_loss_per_epoch = []

    # TRAINING LOOP
    print('START TRAINING ...')
    for epoch in range(config['EPOCHS']):
        # TRAIN THE MODEL
        epoch_iterator_train = tqdm(train_loader)
        train_loss = 0.0
        train_acc = 0
        train_true_labels = []
        train_pred_labels = []
        for step, (x_train, y_train) in enumerate(epoch_iterator_train):
            model.train()
            x_train, y_train = x_train.to(device).float(), y_train.to(device)

            optimizer.zero_grad()
            pred = model(x_train)

            if config['CLASSIFICATION_MODE'] == 'multi':
                loss = criterion(pred, torch.argmax(y_train, dim=1))
                train_true_labels.append(np.argmax(y_train.cpu().detach().numpy(), axis=1))
                train_pred_labels.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
            elif config['CLASSIFICATION_MODE'] == 'binary':
                loss = criterion(pred.squeeze(), y_train.squeeze().float())
                train_acc += (pred >= 0.5).float().eq(y_train).sum().item()

            if config['REGULARIZATION'] == 'L1':  # L2 regularization
                loss += 0.01 * torch.norm(model.fc.weight, 1)  # higher multiply factor apply more regularization
            elif config['REGULARIZATION'] == 'L2':
                loss += 0.01 * torch.norm(model.fc.weight, 2)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            epoch_iterator_train.set_postfix(
                batch_loss=(loss.item()), loss=(train_loss / (step + 1))
            )

        train_loss /= len(train_loader)
        train_loss_per_epoch.append(train_loss)

        if config['CLASSIFICATION_MODE'] == 'multi':
            train_true_labels = np.concatenate(train_true_labels)
            train_pred_labels = np.concatenate(train_pred_labels)
            train_acc = accuracy_score(train_true_labels, train_pred_labels)
        elif config['CLASSIFICATION_MODE'] == 'binary':
            total_samples = len(train_loader.dataset)
            train_acc = train_acc / total_samples

        # VALIDATION THE MODEL
        val_loss = 0
        val_acc = 0
        val_true_labels = []
        val_pred_labels = []
        epoch_iterator_val = tqdm(val_loader)
        with torch.no_grad():
            for step, (x_val, y_val) in enumerate(epoch_iterator_val):
                model.eval()
                x_val, y_val = x_val.to(device).float(), y_val.to(device)
                y_pred = model(x_val)

                if config['CLASSIFICATION_MODE'] == 'multi':
                    loss = criterion(y_pred, torch.argmax(y_val, dim=1))
                    val_true_labels.append(np.argmax(y_val.cpu().detach().numpy(), axis=1))
                    val_pred_labels.append(np.argmax(y_pred.cpu().detach().numpy(), axis=1))
                elif config['CLASSIFICATION_MODE'] == 'binary':
                    loss = criterion(y_pred.squeeze(), y_val.squeeze().float())
                    val_acc += (y_pred >= 0.5).float().eq(y_val).sum().item()

                val_loss += loss.item()
                epoch_iterator_val.set_postfix(
                    batch_loss=(loss.item()), loss=(val_loss / (step + 1))
                )

        val_loss /= len(val_loader)

        if config['CLASSIFICATION_MODE'] == 'multi':
            val_acc = accuracy_score(val_true_labels, val_pred_labels)
            valid_balanced_acc = balanced_accuracy_score(val_true_labels, val_pred_labels)
        elif config['CLASSIFICATION_MODE'] == 'binary':
            total_samples_val = len(val_loader.dataset)
            val_acc = val_acc / total_samples_val

        # Print epoch results
        log = f"Epoch {epoch + pre_epoch + 1}/{config['EPOCHS']}:\n" \
              f"LR: {scheduler.optimizer.param_groups[0]['lr']}\n" \
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}\n" \
              f"Valid Loss: {val_loss:.4f}, Valid Accuracy: {val_acc:.4f}, " \
            # f"Validation Balanced Accuracy: {valid_balanced_acc:.4f}"
        print(log)

        # Save Best Trained Model
        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            epoch_since_improvement = 0
            torch.save(model, TRAINED_MODEL_PATH.joinpath(f'EPOCH-{epoch + pre_epoch + 1}-VAL-ACC-{val_acc:.4f}.pth'))
            print('BEST MODEL SAVED')
            print(f'VALIDATION ACCURACY IMPROVED TO {val_acc:.4f}.')

        else:
            epoch_since_improvement += 1
            # Check if we should stop training early
            if epoch_since_improvement >= config['EARLY_STOP']:
                early_stop = config['EARLY_STOP']
                print(f'VALIDATION ACCURACY DID NOT IMPROVE FOR {early_stop} EPOCHS. TRAINING STOPPED.')
                break
            print(
                f'VALIDATION ACCURACY DID NOT IMPROVE. EPOCHS SINCE LAST LAST IMPROVEMENT: {epoch_since_improvement}.')

        # Update Learning Rate Scheduler
        if config['SCHEDULER'] in ['ExponentialLR', 'lr_scheduler']:
            print(config['SCHEDULER'])
            scheduler.step()
        else:
            scheduler.step(train_loss_per_epoch[-1])

        # LOGGING
        logging.info(log)
        writer.add_scalar('/Loss_train', train_loss, epoch + pre_epoch)
        writer.add_scalar('/Loss_validation', val_loss, epoch + pre_epoch)

        try:
            torch.save({
                'epoch': epoch + pre_epoch + 1,
                'best_val_criteria': best_valid_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, CHECKPOINT_PATH)
        except Exception as e:
            print('MODEL DID NOT SAVE!')
