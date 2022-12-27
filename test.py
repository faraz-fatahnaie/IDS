from argparse import ArgumentParser
from settings import *

#if __name__ == '__main__':
config_catalog= []
for config in os.listdir(CONFIGS_PATH):
    config_catalog.append(config)

#print(config_catalog)
parser = ArgumentParser()
parser.add_argument('--configs', help='choose config file', type=str, choices=config_catalog)
CONFIG_NAME = parser.parse_args()
print(CONFIG_NAME)