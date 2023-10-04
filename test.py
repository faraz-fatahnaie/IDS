import os
import json
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from argparse import Namespace, ArgumentParser
from pathlib import Path
from configs.setting import setting
from utils import parse_data, metrics_evaluate
from Dataset2Image.main import deepinsight


def evaluate(args: Namespace):
    model_path: Path = args.model
    root_path = Path(model_path).resolve().parent.parent
    _, submission_name = os.path.split(model_path)
    submission_name = str(submission_name).replace('.pth', '')
    # submission_name = str(model_path).split("/")[-1].split(".")[:-1]
    # submission_name = ".".join(submission_name)

    # LOAD CONFIG FILE
    CONFIGS = open(Path(root_path).joinpath('MODEL_CONFIG.json'))
    CONFIGS = json.load(CONFIGS)
    print('JSON CONFIG FILE LOADED')
    config, _ = setting(CONFIGS)

    model = torch.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_df = pd.read_csv(Path(config['DATASET_PATH']).joinpath('test_' + config['CLASSIFICATION_MODE'] + '.csv'))
    if config['DEEPINSIGHT']['deepinsight']:
        _, X_test = deepinsight(config['DEEPINSIGHT'], config)
        _, y_test = parse_data(test_df, dataset_name=config['DATASET_NAME'], mode=config['DATASET_TYPE'],
                               classification_mode=config['CLASSIFICATION_MODE'])
    else:
        X_test, y_test = parse_data(test_df, dataset_name=config['DATASET_NAME'], mode=config['DATASET_TYPE'],
                                    classification_mode=config['CLASSIFICATION_MODE'])

    image_A_size = config['DEEPINSIGHT']['Max_A_Size']
    image_B_size = config['DEEPINSIGHT']['Max_B_Size']
    test_ld = DataLoader(
        data_utils.TensorDataset(torch.tensor(X_test.reshape((-1, 1, image_A_size, image_B_size))), torch.tensor(y_test)),
        batch_size=1,
        num_workers=config['NUM_WORKER'])

    labels = []
    preds = []
    probs = []
    accuracy = 0.0
    balanced_acc = 0.0
    # miss_classified_samples = []
    epoch_iterator_test = tqdm(test_ld)
    with torch.no_grad():
        for step, (X, y) in enumerate(epoch_iterator_test):
            model.eval()
            X, y = X.to(device).float(), y.to(device).long()
            y_pred = model(X)
            if config['CLASSIFICATION_MODE'] == 'multi':
                labels.append(np.argmax(y.cpu().detach().numpy(), axis=1))
                preds.append(np.argmax(y_pred.cpu().detach().numpy(), axis=1))
            elif config['CLASSIFICATION_MODE'] == 'binary':
                labels.append(y.squeeze().cpu().detach().numpy())
                probs.append(y_pred.squeeze().cpu().detach().numpy())
                preds.append((y_pred.squeeze() >= 0.5).float().cpu().detach().numpy())
                accuracy += (y_pred >= 0.5).float().eq(y).sum().item()

    if config['CLASSIFICATION_MODE'] == 'multi':
        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
        accuracy = accuracy_score(labels, preds)
        balanced_acc = balanced_accuracy_score(labels, preds)
        print('BALANCE ACCURACY: ', round(balanced_acc, 4))
    elif config['CLASSIFICATION_MODE'] == 'binary':
        total_samples = len(test_ld.dataset)
        accuracy = accuracy / total_samples

    if not os.path.exists(root_path.joinpath('submission')):
        os.mkdir(root_path.joinpath('submission'))

    metrics = metrics_evaluate(labels, preds)
    with open(str(Path(root_path).joinpath('submission', f'{submission_name}-METRICS' + ".json")), "w") as fp:
        json.dump(metrics, fp)
    print(metrics)
    print('ACCURACY: ', round(accuracy, 4))

    submission = pd.DataFrame({'prediction': preds, 'label': labels, 'probability': probs})
    submission_name = f'{submission_name}-TEST-ACC-{round(accuracy, 4)}'
    submission_file_path = Path(root_path).joinpath('submission', submission_name + ".csv")
    submission.to_csv(submission_file_path, index=False)

    # with open(str(Path(root_path).joinpath('submission', 'MISS-CLASSIFIED-' + submission_file_name)), "wb") as fp:
    #     pickle.dump(miss_classified_samples, fp)


if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate Trained Model on Test Set')
    parser.add_argument('--model', help="model weights", type=Path)

    evaluate(args=parser.parse_args())
