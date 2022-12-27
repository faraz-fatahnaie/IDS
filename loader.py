import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from settings import *


def parse_data(df, mode='np'):
    classes = ['Dos', 'Probe', 'R2L', 'U2R', 'normal']
    glob_cl = set(range(len(df.columns)))
    cl_idx = set([df.columns.get_loc(cl) for cl in classes])
    target_feature_idx = list(glob_cl.difference(cl_idx))
    cl_idx = list(cl_idx)
    dt = df.iloc[:, target_feature_idx]
    lb = df.iloc[:, cl_idx]
    assert len(dt) == len(lb), 'Something Wrong!!\nnumber of data is not equal to labels'
    if mode == 'np':
        return dt.to_numpy(), lb.to_numpy()
    elif mode == 'df':
        return dt, lb


def get_train_dataloader(train_path, mode=None):
    batch_size = BATCH_SIZE
    n_worker = N_WORKER

    train_df = pd.read_csv(train_path)
    x_train, y_train = parse_data(train_df)
    if mode == 'np':
        x_train = np.load(str(train_path)+'.npy')
        print('train set loaded in numpy mode')

    print(f'train shape: x=>{x_train.shape}, y=>{y_train.shape}')

    train_loader = DataLoader(
        data_utils.TensorDataset(torch.tensor(x_train.reshape((-1, 1, 11, 11))), torch.tensor(y_train)),
        batch_size=batch_size,
        num_workers=n_worker,
        shuffle=True)

    return train_loader


def get_test_dataloader(test_path, mode=None):
    batch_size = BATCH_SIZE
    n_worker = N_WORKER

    test_df = pd.read_csv(test_path)
    x_test, y_test = parse_data(test_df)
    if mode == 'np':
        x_test = np.load(str(test_path)+'.npy')
        print('test set loaded in numpy mode')
    print(f'test shape: x=>{x_test.shape}, y=>{y_test.shape}')

    test_loader = DataLoader(
        data_utils.TensorDataset(torch.tensor(x_test.reshape((-1, 1, 11, 11))), torch.tensor(y_test)),
        batch_size=batch_size,
        num_workers=n_worker,
        shuffle=False)

    return test_loader