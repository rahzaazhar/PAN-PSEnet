import os
import sys
import time
import random
import string
import argparse
from collections import deque,OrderedDict 
from dataclasses import dataclass, replace

import copy 
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import train_utils

from loader import get_loaders
from utils import CTCLabelConverter, Averager, Scheduler, get_vocab
from train_utils import save_best_model, log_best_metrics, setup_model, setup_optimizer, setup_loss, printOptions
from test import validation
#from plot1 import plot_grad_sim, gradient_similarity, multiplot, average_grad, dot_product, set_zero
import Config as M
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_preds(preds,labels):
    for pred, gt in zip(preds[:10], labels[:10]):
        print(f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}')


def validate(model,criterion,loaders,converter,config,lang):
    metrics = {}
    for name,loader in loaders.items():
        if not name in ['train']:
            print('Validating on:',name)
            val_loss, acc, NED, preds, labels, infer_time, length_of_data = validation(model, criterion, loader, 
                converter, config, lang)
            print_preds(preds,labels)
            metrics[name+'loss'], metrics[name+'acc'], metrics[name+'NED'] = val_loss, acc, NED
            print('Avg. loss: {:.4f}, Accuracy: ({:.2f}%), NED:{:.2f}, infer-time:{:.2f}'.format(val_loss,acc,NED,infer_time))

    return metrics


def update_metrics(metrics,new_update):
    if not len(metrics) > 0:
        for name, value in new_update.items():
            metrics[name] = [value]
    else:
        for name, value in new_update.items():
            metrics[name].append(value)



def plot_metrics(metrics,x,lang):
    for name,val in metrics.items():
        plt.plot(x, val)
        plt.title(lang)
        plt.xlabel('iterations')
        plt.ylabel(name)
        plt.savefig('{}.png'.format(name))
        plt.close('all')


def train_single_lang(config,taskconfig,model,optimizer,criterion,loaders,converter):
    print('enter train')
    lang = taskconfig.lang
    start_time = time.time()
    metrics = {}
    x = []
    loss_avg = Averager()
    best_acc = 0
    best_ED = 0
    #logger = metricsLog(config)
    i = 1
    while i <= taskconfig.numiters:
        
        image_tensors, labels = loaders['train'].get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=config.batch_max_length)
        batch_size = image.size(0)

        if 'CTC' in config.Prediction:
            batch_time_start = time.time()
            preds = model(image, text, lang).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2)  # to use CTCLoss format

            # (ctc_a) To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
            # https://github.com/jpuigcerver/PyLaia/issues/16
            torch.backends.cudnn.enabled = False
            cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
            torch.backends.cudnn.enabled = True

            # # (ctc_b) To reproduce our pretrained model / paper, use our previous code (below code) instead of (ctc_a).
            # # With PyTorch 1.2.0, the below code occurs NAN, so you may use PyTorch 1.1.0.
            # # Thus, the result of CTCLoss is different in PyTorch 1.1.0 and PyTorch 1.2.0.
            # # See https://github.com/clovaai/deep-text-recognition-benchmark/issues/56#issuecomment-526490707
            # cost = criterion(preds, text, preds_size, length)

        else:
            preds = model(image, text[:, :-1]) # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()
        loss_avg.add(cost)

        if i%100 == 0:
            print('iter:',i,'loss:',float(loss_avg.val()))
            loss_avg.reset()

        if i%config.valInterval == 0:
            model.eval()
            new_update = validate(model,criterion,loaders,converter,config,lang)
            #print(new_update)
            x.append(i)
            update_metrics(metrics,new_update)
            plot_metrics(metrics,x,lang)
            model.train()

            #metrics[current_lang] = languagelog(config, model, lang_data_dict[lang], i, criterion)
        i+=1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name',help='name of Config to be used')
    parser.add_argument('--task_config_name',help='name of Config to be used')
    arg = parser.parse_args()
    config = getattr(M, arg.config_name)
    taskconfig = getattr(M, arg.task_config_name)

    config.character = get_vocab()
    print(config.character)
    model = setup_model(config)
    print(model)
    loss = setup_loss(config)
    optimizer = setup_optimizer(config, model)
    
    if not config.experiment_name:
        config.experiment_name = f'{config.Transformation}-{config.FeatureExtraction}-{config.SequenceModeling}-{config.Prediction}'
        config.experiment_name += f'-Seed{config.manualSeed}'

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.save_path+'/saved_models', exist_ok=True)
    printOptions(config)

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(config.manualSeed)
    np.random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    torch.cuda.manual_seed(config.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    config.num_gpu = torch.cuda.device_count()
    # print('device count', config.num_gpu)
    if config.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        config.workers = config.workers * config.num_gpu
        config.batch_size = config.batch_size * config.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', config.batch_size)
        config.batch_size = config.batch_size * config.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        config.num_iter = int(config.num_iter / config.num_gpu)
        """

    loaders, converter = get_loaders(config,taskconfig)
    train_single_lang(config,taskconfig,model,optimizer,loss,loaders,converter)
