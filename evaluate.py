from sklearn import metrics
import numpy as np
from numpy import ndarray


def evaluate_v1(true_label: ndarray, pred_label: ndarray) -> dict:
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
