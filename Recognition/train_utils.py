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
from modelv1 import sharedCNNModel, SharedLSTMModel, SLSLstm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

''' Settin Model, Loss and Optimizer'''
def setup_model(opt):
    if opt.rgb:
        opt.input_channel = 3

    #@azhar
    chardict = {}
    for lang,charlist in opt.character.items():
        chardict[lang] = len(charlist)
    
    #@azhar 
    if opt.share=='CNN':
        model = sharedCNNModel(opt,chardict)
    if opt.share=='CNN+LSTM':
        model = SharedLSTMModel(opt,chardict)
    if opt.share=='CNN+SLSTM':
        model = SLSLstm(opt,chardict)   

    # data parallel for multi-GPU
    #model = torch.nn.DataParallel(model).to(device)
    model = model.to(device)
    model = weight_innit(model)
    model.train()
    
    return model

def setup_loss(opt):
    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    return criterion


def setup_optimizer(opt, model):
    filtered_parameters = filter_params(model)

    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    
    return optimizer


def weight_innit(model):
    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue
    return model


def load_clova_model(model,clova_model_path):
    weights = torch.load(clova_model_path) 
    for name, para in model.named_parameters():
        if 'FeatureExtraction' in name:
            name = 'module.'+name
            para.data.copy_(weights[name].data)
        if 'rnn' in name:
            name.replace('rnn_lang','module.SequenceModeling')
    return model


def filter_params(model):
    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))

    return filtered_parameters


def setup(opt):

    model = setup_model(opt)
    optimizer = setup_optimizer(opt, model)
    criterion = setup_loss(opt)

    #print("Model:")
    #print(model)

    #print("Optimizer:")
    #print(optimizer)

    model = load(opt, model)
    #freezeCNN(model)

    '''for lang,mode in zip(opt.langs,opt.mode):
        if(mode!='train'):
            freeze_head(model,lang)'''

    return model, criterion, optimizer

def freezeCNN(model):
    for name, param in model.named_parameters():
        if 'FeatureExtraction' in name:
            param.requires_grad = False
''' Model Loading Strategies'''
def load(opt, model):

    if opt.spath != '':
        print(f'loading shared weiths from {opt.spath}')
        #model = LoadW(opt,model,'hin')
        model = new_load(opt,model)
    #loading shared model
    if opt.shared_model != '':
        print(f'loading shared part of the model')
        model = loadSharedModel(opt,model)
    #loading Full model
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))

    return model


#@azhar load weights from Shared CNN model to (sharedCNN,sharedCNN+LSTM,sharedCNN+SLSTM)
def loadSharedModel(opt,model,freeze=False):
    checkpoint = torch.load(opt.shared_model)
    for x, y in model.named_parameters():
        if x in checkpoint.keys():
            y.data.copy_(checkpoint[x].data)
            if freeze:
                y.requires_grad = False
    return model


#@azhar

def new_load(opt, model):
    checkpoint = torch.load(opt.spath)
    for x, y in model.named_parameters():
        if x in checkpoint.keys():
            y.data.copy_(checkpoint[x].data)
        if 'rnn_lang.hin' in x or 'rnn_lang.ban' in x:
            x = x.split('.')
            _ = x.pop(2)
            x = '.'.join(x)
            y.data.copy_(checkpoint[x])
    return model


def LoadW(opt,model,lang):
    #@azhar loading shared weights from one architecture to another
    checkpoint = torch.load(opt.spath)
    if opt.share == 'CNN':
        for x, y in model.named_parameters():
            if x in checkpoint.keys() and 'FeatureExtraction' in x:
                y.data.copy_(checkpoint[x].data)
    if opt.share =='CNN+LSTM':
        for x, y in model.named_parameters():
            if x in checkpoint.keys() and ('FeatureExtraction' in x or 'Predictions.'+lang in x):
                y.data.copy_(checkpoint[x].data)
            if ('rnn_lang' in x):
                s = x.split('.')
                s.insert(2,lang)
                s = '.'.join(s)
                y.data.copy_(checkpoint[s].data)
    if opt.share =='CNN+SLSTM':
        for x, y in model.named_parameters():
            if x in checkpoint.keys() and ('FeatureExtraction' in x or 'Predictions.'+lang in x):
                y.data.copy_(checkpoint[x].data)
            if ('rnn_lang.'+lang+'.0' in x):
                s = x.replace('0','1',1)
                y.data.copy_(checkpoint[s].data)
            if ('Srnn_lang' in x):
                s = x.replace('Srnn_lang','rnn_lang')
                s = s.split('.')
                s.insert(2,lang)
                s = '.'.join(s)
                y.data.copy_(checkpoint[s].data)

    return model

''' Validation Utilities'''
def save_best_model(opt, model, best_acc, best_ED, metrics):

    ED, word_acc = return_new_score(metrics,opt.calc_best_metric_on)

    if word_acc > best_acc:
        best_acc = word_acc
        torch.save(model.state_dict(), opt.save_path+'/saved_models/best_acc_model.pth')
    if ED > best_ED:
        best_ED = ED
        torch.save(model.state_dict(), opt.save_path+'/saved_models/best_NED_model.pth')

    return best_acc, best_ED


def return_new_score(metrics,dataset):
    word_acc = 0
    ED = 0
    
    langs = list(metrics.keys())
    for lang in langs:
        word_acc += metrics[lang][dataset+'acc'][-1]
        ED += metrics[lang][dataset+'NED'][-1]

    ED = ED/len(langs)
    word_acc = word_acc/len(langs)

    return ED, word_acc


def log_best_metrics(opt, best_acc, best_ED):
    best_model_log = f'best_accuracy: {best_acc:0.3f}, best_norm_ED: {best_ED:0.2f}'
    print(best_model_log)
    log = open(f'./{opt.exp_dir}/{opt.experiment_name}/{opt.experiment_name}_log.txt', 'a')
    log.write(best_model_log + '\n')


def printOptions(opt):
    """ final options """
    # print(opt)
    with open(f'./{opt.exp_dir}/{opt.experiment_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)