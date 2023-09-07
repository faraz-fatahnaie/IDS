import torch

from sklearn import metrics
import numpy as np
from numpy import ndarray

from pyDeepInsight.pyDeepInsight import ImageTransformer
from pyDeepInsight.pyDeepInsight.utils import Norm2Scaler
from sklearn.manifold import TSNE


def parse_data(df, dataset_name: str, classification_mode: str, mode: str = 'np'):
    classes = []
    if classification_mode == 'binary':
        classes = df.columns[-1:]
    elif classification_mode == 'multi':
        if dataset_name in ['NSL_KDD', 'KDD_CUP99']:
            classes = df.columns[-5:]
        elif dataset_name == 'UNSW_NB15':
            classes = df.columns[-10:]

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


def deepinsight_original(x: ndarray, y: ndarray, pixel_size: tuple = (11, 11)):
    distance_metric = 'cosine'
    pixel_size = pixel_size

    reducer = TSNE(
        n_components=2,
        metric=distance_metric,
        init='random',
        learning_rate='auto',
        n_jobs=-1
    )

    it = ImageTransformer(
        feature_extractor=reducer,
        pixels=pixel_size)

    ln = Norm2Scaler()
    x_norm = ln.fit_transform(x)
    it.fit(x, y=y, plot=False)
    x_transformed = it.transform(x)

    return x_transformed[:, :, :, 0], it, ln


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
