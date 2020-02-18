import os
import sys
import time
import random
import string
import argparse
from collections import deque
from dataclasses import dataclass, replace

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
from trainv2 import train
from train_utils import setup_model, setup_optimizer
from utils import LanguageData, get_vocab
import prune 
from prune import Pruning
import Config as M

from sparse_sharing.train_mtl import load_masks, MTL_Masker


def subnets_train(train_config,args):
	model = setup_model(train_config)
	if args.mask_dir is None:
        masks = None
    else:
        masks = load_masks(args.mask_dir)
    masker = MTL_Masker(model, masks)
    train(opt=train_config,model=model,optimizer=optim,criterion=criterion,masker=masker)
	#either do gradient step after every batch or accumulate gradient and then do a gradient step


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_config_name',help='name of Config to be used')
	#parser.add_argument('--Mask_trainer_config', help='name of Congig')
	parser.add_argument('--mask_dir', help='path to all the masks')

	""" Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(train_config.manualSeed)
    np.random.seed(train_config.manualSeed)
    torch.manual_seed(train_config.manualSeed)
    torch.cuda.manual_seed(train_config.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    train_config.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if train_config.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        train_config.workers = train_config.workers * train_config.num_gpu
        opt.batch_size = train_config.batch_size * train_config.num_gpus




