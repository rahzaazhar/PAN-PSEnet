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

from utils import CTCLabelConverter, AttnLabelConverter, Averager, tensorlog, Scheduler, LanguageData
#from modelv1 import Model, SharedLSTMModel, SLSLstm
from trainv2 import validation, languagelog
import train_utils
from train_utils import setup, save_best_model, log_best_metrics
from test import validation
import Config as M
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def filter(files,lastCp):
    filtered = []
    
    for file in files:
        print(file)
        if file == 'best_accuracy.pth' or file == 'best_norm_ED.pth' or file == 'opt.txt' or file == 'ABH(frozenCNN_2)_log.txt':
            continue
        iterr = getiter(file)
        if iterr > lastCp:
            filtered.append(file)
    
    #filtered.sort()
    return filtered


def getiter(filename):
    return int(filename.split('_')[-1].split('.')[0])

def freeze_head(model,head):
    for name,param in model.named_parameters():
        if head in name:
            param.requires_grad = False


def monitor(opt, model, criterion, lang_data_dict, checkpoint_path, tflogger):#initialise state of the program
    numCp = 0
    lastCp = 0
    best_acc = 0
    best_ED = 0
    metrics = {}
    
    while True:
        files = os.listdir(checkpoint_path)
        # if a new checkpoint is created 
        if (len(files) <= numCp):
            continue
        numCp = numCp + len(files) - numCp
        checkpoints = filter(files, lastCp)
        print(checkpoints)
            
        for checkpoint in checkpoints:
            model.load_state_dict(torch.load(checkpoint_path+checkpoint))
            iterr = getiter(checkpoint)

            #validating and recording metrics on all languages
            for lang in opt.langs:
                print('-'*18 + 'Start Validating on ' + lang + '-'*18)
                metrics[lang] = languagelog(opt, model, lang_data_dict[lang], iterr, criterion)

            best_acc, best_ED = save_best_model(opt, model, best_acc, best_ED, metrics)
            log_best_metrics(opt, best_acc, best_ED)
            
            for lang in opt.langs:
                tflogger.record(lang, metrics[lang], iterr)
            lastCp = iterr
                    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', help='path to directory with checkpoint files')
    parser.add_argument('--config_name', help='give name of config')
    arg = parser.parse_args()
    opt = getattr(M, arg.config_name)
    tflogger = tensorlog(opt)
    char_dict = {}
    f = open('characters.txt','r')
    lines = f.readlines()
    for line in lines:#@azhar
        char_dict[line.split(',')[0]] = line.strip().split(',')[-1]
    opt.character = char_dict
    model, criterion, _ = setup(opt)
    lang_data_dict = {}
    
    for lang,iterr,m in zip(opt.langs, opt.pli, opt.mode):
        print(lang, iterr)
        lang_data_dict[lang] = LanguageData(opt, lang, iterr, m)
    
    model.eval()
    monitor(opt, model, criterion, lang_data_dict, arg.checkpoint_path, tflogger)