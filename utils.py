import torch

from sklearn import metrics
import numpy as np
from numpy import ndarray


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


def metrics_evaluate(true_label: ndarray, pred_label: ndarray) -> dict:
    metric_param = {
        'accuracy': metrics.accuracy_score(true_label, pred_label),
        'recall': metrics.recall_score(true_label, pred_label, average=None),
        # 'precision': metrics.precision_score(true_label, pred_label, average=None),
    }
    conf_mat = metrics.confusion_matrix(true_label, pred_label)
    false_positive = conf_mat.sum(axis=0) - np.diag(conf_mat)
    false_negative = conf_mat.sum(axis=1) - np.diag(conf_mat)
    true_positive = np.diag(conf_mat)
    true_negative = conf_mat.sum() - (false_positive + false_negative + true_positive)

    metric_param['false_alarm_rate'] = false_positive / (false_positive + true_negative)
    metric_param['detection_rate'] = true_positive / (true_positive + false_negative)

    return metric_param


def cuda_test():
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
