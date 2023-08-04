from pathlib import Path
import json


def setting(config_file=None):
    config = dict()
    BASE_DIR = Path(__file__).resolve().parent.parent

    if config_file is None:
        config_name = 'CONFIG'  # config file name in config dir
        config_dir = BASE_DIR.joinpath('configs')
        config_file = open(f'{config_dir}/{config_name}.json')
        config_file = json.load(config_file)

    # DATASET_DIR = BASE_DIR.joinpath('dataset')
    config['DATASET_NAME'] = config_file['dataset']['name']
    config['DATASET_TYPE'] = config_file['dataset']['type']
    config['DEEPINSIGHT'] = config_file['dataset']['deepinsight']
    config['CLASSIFICATION_MODE'] = config_file['dataset']['classification_mode']
    # DATASET_PATH = DATASET_DIR.joinpath(DATASET_NAME)
    if config['DATASET_NAME'] == 'NSL_KDD':
        config['DATASET_PATH'] = '/home/faraz/PycharmProjects/IDS/dataset/NSL_KDD/file/preprocessed'
    elif config['DATASET_NAME'] == 'UNSW_NB15':
        config['DATASET_PATH'] = '/home/faraz/PycharmProjects/IDS/dataset/UNSW_NB15/file/preprocessed'

    config['NUM_WORKER'] = config_file['dataset']['n_worker']
    config['MODEL_NAME'] = config_file['model']['name']

    config['LOSS_FUNCTION'] = config_file['model']['loss']
    config['EPOCHS'] = config_file['model']['epoch']
    config['BATCH_SIZE'] = config_file['model']['batch_size']
    config['OPTIMIZER'] = config_file['model']['optim']
    config['LR'] = config_file['model']['lr']
    config['WEIGHT_DECAY'] = config_file['model']['weight_decay']
    config['SCHEDULER'] = config_file['model']['scheduler']
    config['MIN_LR'] = config_file['model']['min_lr']
    config['PATIENCE'] = config_file['model']['patience']
    config['EARLY_STOP'] = config_file['model']['early_stop']
    config['FACTOR'] = config_file['model']['factor']
    config['GAMMA'] = config_file['model']['gamma']
    config['REGULARIZATION'] = config_file['model']['regularization']

    return config, config_file
