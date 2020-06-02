import os, sys
import argparse
import numpy as np
import torch
from train_pmnist import run_learn_to_grow
import L2G_config as M

if __name__ == '__main__':
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name',help='name of Config to be used')
    arg = parser.parse_args()
    opts = getattr(M, arg.config_name)
    if isinstance(opts, dict):
        for name,config in opts.items():
          print('Loading.....',name+'_Config')
          run_learn_to_grow(config)
    else:
        task_ids, avg_acc, avg_diff, num_paras = run_learn_to_grow(opts)
        # each element in the vector res is the average acc/diff after the nth task
        print('run_expts:', task_ids, avg_acc, avg_diff, num_paras)