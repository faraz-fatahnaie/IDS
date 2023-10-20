import sklearn.metrics
import torch

from sklearn import metrics
import numpy as np
from numpy import ndarray
from pathlib import Path
import os
import pandas as pd
from pandas import DataFrame

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from functools import partial

# from pyDeepInsight.pyDeepInsight import ImageTransformer
# from pyDeepInsight.pyDeepInsight.utils import Norm2Scaler
# from sklearn.manifold import TSNE
from torch.autograd import Variable


def parse_data(df, dataset_name: str, classification_mode: str, mode: str = 'np'):
    classes = []
    if classification_mode == 'binary':
        classes = df.columns[-1:]
    elif classification_mode == 'multi':
        if dataset_name in ['NSL_KDD', 'KDD_CUP99']:
            classes = df.columns[-5:]
        elif dataset_name == 'UNSW_NB15':
            classes = df.columns[-10:]
        elif dataset_name == 'CICIDS':
            classes = df.columns[-15:]

    assert classes is not None, 'Something Wrong!!\nno class columns could be extracted from dataframe'
    glob_cl = set(range(len(df.columns)))
    cl_idx = set([df.columns.get_loc(c) for c in list(classes)])
    target_feature_idx = list(glob_cl.difference(cl_idx))
    cl_idx = list(cl_idx)
    dt = df.iloc[:, target_feature_idx]
    lb = df.iloc[:, cl_idx]
    assert len(dt) == len(lb), 'Something Wrong!!\nnumber of data is not equal to labels'
    if mode == 'np':
        return dt.to_numpy(), lb.to_numpy()
    elif mode == 'df':
        return dt, lb


def save_dataframe(dataframe: DataFrame, save_path: Path, dataframe_type: str = 'train',
                   classification_mode: str = 'binary') -> None:
    file_name = dataframe_type
    if classification_mode == 'binary':
        file_name = file_name + '_binary'
    elif classification_mode == 'multi':
        file_name = file_name + '_multi'
    train_file = os.path.join(save_path, file_name + '.csv')
    dataframe.to_csv(train_file, index=False)
    print('Saved:', train_file)


def sort_columns(train_df: DataFrame, test_df: DataFrame) -> (DataFrame, DataFrame):
    train_cols = train_df.columns
    test_sortedBasedOnTrain = pd.DataFrame(columns=train_cols)
    for col in test_sortedBasedOnTrain:
        test_sortedBasedOnTrain[col] = test_df[col]

    return train_df, test_sortedBasedOnTrain


def shuffle_dataframe(dataframe: DataFrame):
    return dataframe.sample(frac=1).reset_index(drop=True)


def metrics_evaluate(true_label: ndarray, pred_label: ndarray) -> dict:
    confusion_matrix = metrics.confusion_matrix(true_label, pred_label)
    metric_param = {'accuracy': metrics.accuracy_score(true_label, pred_label),
                    'f1_score': (metrics.f1_score(true_label, pred_label)),
                    'recall': metrics.recall_score(true_label, pred_label, average=None),
                    'confusion_matrix': confusion_matrix.tolist()}
    false_positive = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    false_negative = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    true_positive = np.diag(confusion_matrix)
    true_negative = confusion_matrix.sum() - (false_positive + false_negative + true_positive)

    metric_param['false_alarm_rate'] = (false_positive / (false_positive + true_negative)).tolist()
    metric_param['detection_rate'] = (true_positive / (true_positive + false_negative)).tolist()

    return metric_param


def cuda_test():
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")


class EQLv2Loss(nn.Module):
    def __init__(self, use_sigmoid=False, reduction='mean', class_weight=None,
                 loss_weight=1.0, num_classes=2, gamma=12, mu=0.8, alpha=4.0,
                 vis_grad=False):
        super(EQLv2Loss, self).__init__()
        self.pred_class_logits = None
        self.gt_classes = None
        self.n_i = None
        self.n_c = None
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        # cfg for eqlv2
        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self._pos_grad = None
        self._neg_grad = None
        self.pos_neg = None

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))

        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)

    def forward(self, y_true, y_pred):
        target, cls_score = y_true.detach().cpu(), y_pred.detach().cpu()
        # n_i = batch_size / n_c = number of classes
        self.n_i, self.n_c = cls_score.size(0), cls_score.size(1)
        self.gt_classes = target
        self.pred_class_logits = cls_score
        pos_w, neg_w = self.get_weight(cls_score)

        weight = pos_w * target + neg_w * (1 - target)
        # cls_loss = F.binary_cross_entropy_with_logits(cls_score, target, weight=weight, reduction='none')

        # Manually compute binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                  reduction='none')

        # if self.use_sigmoid:
        #     loss = -pos_w * (target * torch.log(torch.sigmoid(cls_score) + 1e-7)) - \
        #            neg_w * ((1 - target) * torch.log(1 - torch.sigmoid(cls_score) + 1e-7))
        # else:
        #     loss = -pos_w * (target * cls_score) - neg_w * ((1 - target) * cls_score)

        cls_loss = torch.sum(loss * weight) / self.n_c

        self.collect_grad(cls_score, target, weight)
        ret = Variable(self.loss_weight * cls_loss, requires_grad=True)

        return ret

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, cls_score):
        cls_score = torch.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)
        pos_grad = torch.sum(grad * target * weight)
        neg_grad = torch.sum(grad * (1 - target) * weight)

        self._pos_grad += pos_grad
        self._neg_grad += neg_grad
        self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)

    def get_weight(self, cls_score):
        if self._pos_grad is None:
            self._pos_grad = torch.zeros(self.num_classes, device=cls_score.device)
            self._neg_grad = torch.zeros(self.num_classes, device=cls_score.device)
            neg_w = torch.ones(self.n_c, device=cls_score.device)
            pos_w = torch.ones(self.n_c, device=cls_score.device)
        else:
            neg_w = self.map_func(self.pos_neg)
            pos_w = 1 + self.alpha * (1 - neg_w)
            neg_w = neg_w.view(1, -1)
            pos_w = pos_w.view(1, -1)

        return pos_w, neg_w

# def deepinsight_original(x: ndarray, y: ndarray, pixel_size: tuple = (11, 11)):
#     distance_metric = 'cosine'
#     pixel_size = pixel_size
#
#     reducer = TSNE(
#         n_components=2,
#         metric=distance_metric,
#         init='random',
#         learning_rate='auto',
#         n_jobs=-1
#     )
#
#     it = ImageTransformer(
#         feature_extractor=reducer,
#         pixels=pixel_size)
#
#     ln = Norm2Scaler()
#     x_norm = ln.fit_transform(x)
#     it.fit(x, y=y, plot=False)
#     x_transformed = it.transform(x)
#
#     return x_transformed[:, :, :, 0], it, ln
