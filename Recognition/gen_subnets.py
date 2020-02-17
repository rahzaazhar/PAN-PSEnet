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
from train_utils import setup_model, setup_optimizer,setup_loss, printOptions
from utils import LanguageData, get_vocab
from sparse_sharing.src import prune
from sparse_sharing.src.prune import Pruning
import Config as M

'''def load_data(train_config):
    opt = train_config
    lang_data_dict = {}
    for lang,iterr,m in zip(opt.langs,opt.pli,opt.mode):
        print(lang,iterr)
        lang_data_dict[lang] = LanguageData(opt,lang,iterr,m)
    return lang_data_dict'''


def gen_task_subnet(train_config, prune_config):

    #task_data = load_data(train_config)
    model = setup_model(train_config)
    loss = setup_loss(train_config)
    optim = setup_optimizer(train_config, model)
    print(model)

    need_cut_names = list(set([s.strip() for s in prune_config.need_cut.split(",")]))
    prune_names = []

    for name, p in model.named_parameters():
        if not p.requires_grad or "bias" in name:
            continue
        for n in need_cut_names:
            if n in name:
                prune_names.append(name)
                break
    printOptions(train_config)
    pruner = Pruning(
        model, prune_names, final_rate=prune_config.final_rate, pruning_iter=prune_config.pruning_iter
    )
    if prune_config.init_masks is not None:
        pruner.load(prune_config.init_masks)
        pruner.apply_mask(pruner.remain_mask, pruner._model)

    train_config.save_path = f'{train_config.save_path}/{train_config.task_id[0]}'
    os.makedirs(train_config.save_path, exist_ok=True)

    for prune_step in range(pruner.pruning_iter + 1):
        # reset optimizer every time
        #optim_params = [p for p in model.parameters() if p.requires_grad]
        # utils.get_logger(__name__).debug(optim_params)
        #utils.get_logger(__name__).debug(len(optim_params))
        optim = setup_optimizer(train_config, model)
        # optimizer = TriOptim(optimizer, args.n_filters, args.warmup, args.decay)
        factor = pruner.cur_rate / 100.0
        factor = 1.0
        # print(factor, pruner.cur_rate)
        for pg in optim.param_groups:
            pg["lr"] = factor * pg["lr"]
        #utils.get_logger(__name__).info(optimizer)
        train_time_start = time.time()
        train(opt=train_config,model=model,optimizer=optim,criterion=loss,pruner=pruner)
        train_time_end = time.time()
        torch.save(
            model.state_dict(),
            os.path.join(
                train_config.save_path,
                "best_{}_{}.th".format(pruner.prune_times, pruner.cur_rate),
            ),
        )
        prune_time_start = time.time()
        pruner.pruning_model()
        prune_time_end = time.time()
        #summary_writer.add_scalar("remain_rate", pruner.cur_rate, prune_step + 1)
        #summary_writer.add_scalar("cutoff", pruner.last_cutoff, prune_step + 1)

        print('Train Time:{} Prune time:{}'.format(train_time_end-train_time_start,prune_time_end- prune_time_start))

        pruner.save(
            os.path.join(
                train_config.save_path, "{}_{}.th".format(pruner.prune_times, pruner.cur_rate)
            )
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_name',help='name of Config to be used')
    parser.add_argument('--prune_config_name',help='name of pruner Config')
    arg = parser.parse_args()
    train_config = getattr(M, arg.train_config_name)
    train_config.character = get_vocab()
    prune_config = getattr(M, arg.prune_config_name)

    os.makedirs(train_config.save_path, exist_ok=True)

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

    gen_task_subnet(train_config, prune_config)

