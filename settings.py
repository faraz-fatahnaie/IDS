import os
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

CONFIG_NAME = 'CONFIG' # it should load from input
CONFIGS_PATH = BASE_DIR.joinpath('configs')
CONFIGS = open(f'{CONFIGS_PATH}/{CONFIG_NAME}.json')
CONFIGS = json.load(CONFIGS)

DATASET_DIR = BASE_DIR.joinpath('dataset')
DATASET_NAME = CONFIGS['dataset']['name']
DATASET_PATH = DATASET_DIR.joinpath(DATASET_NAME)
N_WORKER = CONFIGS['dataset']['n_worker']
DATASET_TYPE = CONFIGS['dataset']['type']

MODEL_NAME = CONFIGS['model']['name']
MODEL_DIR = BASE_DIR.joinpath('models')
MODEL_PATH = MODEL_DIR.joinpath(MODEL_NAME)

TRAINABLE = CONFIGS['model']['trainable']
LR = CONFIGS['model']['lr']
EPOCHS = CONFIGS['model']['epoch']
BATCH_SIZE = CONFIGS['model']['batch_size']
MIN_LR = CONFIGS['model']['min_lr']
PATIENCE = CONFIGS['model']['patience']
FACTOR = CONFIGS['model']['factor']
SCHEDULER = CONFIGS['model']['scheduler']

SAVE_PATH = BASE_DIR.joinpath(f'best/{MODEL_NAME}_{DATASET_NAME}')
if not os.path.isdir(f'{SAVE_PATH}'):
    os.makedirs(f'{SAVE_PATH}')







