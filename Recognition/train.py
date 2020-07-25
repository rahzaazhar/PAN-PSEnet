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

from loader import get_multiple_loaders
from utils import CTCLabelConverter, Averager, Scheduler, get_vocab
from train_utils import save_best_model, log_best_metrics, setup_model, setup_optimizer, setup_loss, printOptions, load_clova_model
from test import validation
import Config as M
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_preds(preds,labels):
    for pred, gt in zip(preds[:10], labels[:10]):
        print(f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}')


def validate(model,criterion,loaders,converter,config,lang):
    metrics = {}
    for name,loader in loaders.items():
        if not name in ['train','eval']:
            print('Validating on:',name,'of',lang)
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



def plot_metrics(taskconfig,metrics,x,lang):
    for name,val in metrics.items():
        plt.plot(x, val)
        plt.title(lang)
        plt.xlabel('iterations')
        plt.ylabel(name)
        plt.savefig(taskconfig.hp.save_path+'/{}_{}.png'.format(lang,name))
        plt.close('all')



def train(taskconfig,model,optimizer,criterion,multi_loader):
    print('enter train')
    start_time = time.time()
    lang_metrics = {lang:{} for lang in multi_loader.keys()}
    x = []
    loss_avg = Averager()
    best_acc = 0
    best_ED = 0 
    total_iters = taskconfig.hp.num_iter 
    iterrs = 0
    while iterrs<total_iters:
        for lang, num_iters in taskconfig.schedule:
            loaders, converter = multi_loader[lang]
            for i in range(num_iters):
                image_tensors, labels = loaders['train'].get_batch()
                image = image_tensors.to(device)
                text, length = converter.encode(labels, batch_max_length=taskconfig.hp.batch_max_length)
                batch_size = image.size(0)

                if 'CTC' in taskconfig.hp.Prediction:
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), taskconfig.hp.grad_clip)  # gradient clipping with 5 (Default)
                optimizer.step()
                loss_avg.add(cost)

                iterrs+=1
                
                if iterrs%taskconfig.hp.print_iter == 0:
                    print('iter:',iterrs,'loss:',float(loss_avg.val()))
                    loss_avg.reset()

                if iterrs%taskconfig.hp.valInterval == 0:
                    model.eval()
                    x.append(iterrs)
                    for lang, (loaders, converter) in multi_loader.items():
                        print('-'*80)
                        new_update = validate(model,criterion,loaders,converter,taskconfig.hp,lang)
                        update_metrics(lang_metrics[lang],new_update)
                        plot_metrics(taskconfig,lang_metrics[lang],x,lang)
                    print('-'*80)
                    best_acc, best_ED = save_best_model(taskconfig.hp, model, best_acc, best_ED, lang_metrics)
                    model.train()

                if iterrs%taskconfig.hp.save_iter == 0:
                    save_path = taskconfig.hp.save_path+'/saved_models/'+str(iterrs)+'_iters.pth'
                    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--config_name',help='name of Config to be used')
    parser.add_argument('--task_config_name',help='name of Config to be used')
    arg = parser.parse_args()
    #config = getattr(M, arg.config_name)
    taskconfig = getattr(M, arg.task_config_name)

    taskconfig.hp.character = get_vocab()
    print(taskconfig.hp.character)
    model = setup_model(taskconfig.hp)
    if not taskconfig.hp.clova_pretrained_model_path == '':
    	load_clova_model(model,taskconfig.hp.clova_pretrained_model_path)
    if not taskconfig.hp.pretrained_model_path == '':
    	model.load_state_dict(torch.load(taskconfig.hp.pretrained_model_path))

    print(model)
    loss = setup_loss(taskconfig.hp)
    optimizer = setup_optimizer(taskconfig.hp, model)
    
    if not taskconfig.hp.experiment_name:
        taskconfig.hp.experiment_name = f'{taskconfig.hp.Transformation}-{taskconfig.hp.FeatureExtraction}-{taskconfig.hp.SequenceModeling}-{taskconfig.hp.Prediction}'
        taskconfig.hp.experiment_name += f'-Seed{taskconfig.hp.manualSeed}'

    os.makedirs(taskconfig.hp.save_path, exist_ok=True)
    os.makedirs(taskconfig.hp.save_path+'/saved_models', exist_ok=True)
    printOptions(taskconfig.hp)

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(taskconfig.hp.manualSeed)
    np.random.seed(taskconfig.hp.manualSeed)
    torch.manual_seed(taskconfig.hp.manualSeed)
    torch.cuda.manual_seed(taskconfig.hp.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    taskconfig.hp.num_gpu = torch.cuda.device_count()
    # print('device count', taskconfig.hp.num_gpu)
    if taskconfig.hp.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        taskconfig.hp.workers = taskconfig.hp.workers * taskconfig.hp.num_gpu
        taskconfig.hp.batch_size = taskconfig.hp.batch_size * taskconfig.hp.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', taskconfig.hp.batch_size)
        taskconfig.hp.batch_size = taskconfig.hp.batch_size * taskconfig.hp.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        taskconfig.hp.num_iter = int(taskconfig.hp.num_iter / taskconfig.hp.num_gpu)
        """
    multi_loader = get_multiple_loaders(taskconfig)
    #loaders, converter = get_loaders(config,taskconfig)
    train(taskconfig,model,optimizer,loss,multi_loader)
