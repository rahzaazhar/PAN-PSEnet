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
from fastNLP import logger

from utils import LanguageData, get_vocab, tensorlog
from train_utils import setup_model, setup_optimizer, setup_loss
from sparse_sharing.train_mtl import load_masks, MTL_Masker
from trainv2 import validation, languagelog
import Config as M
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_iter(s):
    return float(s.split('/')[-1].split('_')[-1].split('.th')[0])


def find_best_mask(opt,masks_dir):
    model = setup_model(opt)
    loss = setup_loss(opt)
    data = list_models(masks_dir)
    data = list(sorted(data,key=lambda s: get_iter(s[1]),reverse=True)) 
    os.makedirs(train_config.save_path, exist_ok=True)
    tflogger = tensorlog(opt)
    metrics = {}
    lang_data_dict = {}
    for lang,iterr,m,t_id in zip(opt.langs,opt.pli,opt.mode,opt.task_id):
        print(lang,iterr)
        lang_data_dict[lang] = LanguageData(opt,lang,iterr,m,t_id)

    for (x,y) in data:
        iterr = get_iter(y)
        model.load_state_dict(torch.load(y))
        masker = MTL_Masker(model,x)
        masker.to(device)
        for lang in opt.langs:
            print('-'*18 + 'Start Validating on ' + lang + '-'*18)
            metrics[lang] = languagelog(opt, model, lang_data_dict[lang],iterr,loss, masker)
            tflogger.record(lang, metrics[lang], iterr)


def list_models(masks_dir):
    dump = []
    paths = os.listdir(masks_dir)
    for path in paths:
        if 'best' in path and not '100' in path:
            pathlist = path.split('_')
            pathlist.remove('best')
            mask_path = masks_dir+'_'.join(pathlist)
            dump1 = torch.load(mask_path, "cpu")
            assert "mask" in dump1 and "pruning_time" in dump1
            logger.info(
            "loading pruning_time {}, mask in {}".format(dump1["pruning_time"],mask_path)
            )
            weight_path = masks_dir+path
            dump.append((dump1['mask'],weight_path))
    return dump

#d= list_models('/home/azhar/TextRecogShip/Recognition/Experiments/Gen_Ban_Subnet/1/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_name',help='name of Config to be used')
    parser.add_argument('--masks_dir',help='path to mask directory')
    arg = parser.parse_args()
    train_config = getattr(M, arg.train_config_name)
    train_config.character = get_vocab()

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
        opt.batch_size = train_config.batch_size * train_config.num_gpu

    find_best_mask(train_config,arg.masks_dir)


    







