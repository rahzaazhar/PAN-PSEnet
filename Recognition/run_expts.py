import os
import argparse
import numpy as np
import torch
from train_pmnist import run_learn_to_grow
import L2G_config as M

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name',help='name of Config to be used')
    arg = parser.parse_args()
    opts = getattr(M, arg.config_name)
    for name,config in opts.items():
      print('Loading.....',name+'_Config')
      run_learn_to_grow(config)
