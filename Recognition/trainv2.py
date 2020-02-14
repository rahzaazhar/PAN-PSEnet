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
import train_utils
from train_utils import setup, save_best_model, log_best_metrics 
#from datasetv1 import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from modelv1 import sharedCNNModel, SharedLSTMModel, SLSLstm
from test import validation
import Config as M
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def languagelog(opt, model, LangData, globaliter, criterion):#@azhar modified 
    metrics = {}
    with open(f'./{opt.exp_dir}/{opt.experiment_name}/{opt.experiment_name}_log.txt', 'a') as log:
        log.write('#'*18+'Start Validating on '+LangData.lang+'#'*18+'\n')
        log.write('validating on Synthetic data\n')
        Synvalidloss, Syn_valid_acc, SynvalED = validate(opt,model,criterion, LangData.Synvalid_loader, LangData.labelconverter,log,globaliter,'Syn-val-loss',LangData.lang)
        metrics['Syn_validation_loss'], metrics['Syn_val_Wordaccuracy'], metrics['Syn_val_edit-dist'] = Synvalidloss, Syn_valid_acc, SynvalED
        log.write('validating on Real data\n')
        Real_valid_loss,Real_valid_accuracy,Real_valid_norm_ED = validate(opt,model,criterion, LangData.Rvalid_loader, LangData.labelconverter,log,globaliter,'Real-val-loss',LangData.lang)
        metrics['Real_validation_loss'], metrics['Real_val_Wordaccuracy'], metrics['Real_val_edit-dist'] = Real_valid_loss, Real_valid_accuracy, Real_valid_norm_ED
        log.write('Evaluating on Train data\n')
        train_loss,train_accuracy,_ = validate(opt,model,criterion, LangData.Tvalid_loader, LangData.labelconverter,log,globaliter,'train-loss',LangData.lang)
        metrics['train_loss'], metrics['train_Wordaccuracy'] = train_loss, train_accuracy

    return metrics


def validate(opt, model, criterion, loader, converter, log, i, lossname, lang):#@azhar
    #print('enter validate')
    with torch.no_grad():
        valid_loss, current_accuracy, current_norm_ED, preds, labels, infer_time, length_of_data = validation(
            model, criterion, loader, converter, opt, lang)
    for pred, gt in zip(preds[:10], labels[:10]):
        if 'Attn' in opt.Prediction:
            pred = pred[:pred.find('[s]')]
            gt = gt[:gt.find('[s]')]
        print(f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}')
        log.write(f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}\n')
    valid_log = f'[{i}/{opt.num_iter}] {lossname}: {valid_loss:0.5f}'
    valid_log += f' accuracy: {current_accuracy:0.3f}, norm_ED: {current_norm_ED:0.2f}'
    print(valid_log)
    log.write(valid_log + '\n')
    return valid_loss, current_accuracy, current_norm_ED


def freeze_head(model,head):
    for name,param in model.named_parameters():
        if head in name:
            param.requires_grad = False
    
#@azhar testing function    
def paramCheck(model):
    for name, param in model.named_parameters():
        print(name,param.requires_grad) 


def train(opt,pruner=None,masker=None):
    """ dataset preparation """
    #opt.select_data = opt.select_data.split('-')#default will be syn and real
    #opt.batch_ratio = opt.batch_ratio.split('-')#default will be Syn-0.8 Real-0.2
    langQ = []
    for lang,mode in zip(opt.langs, opt.mode):
        if mode=='train':
            langQ.append(lang)
    langQ = deque(langQ)
 
    print(opt.train_data)
    lang_data_dict = {}
    for lang,iterr,m,t_id in zip(opt.langs,opt.pli,opt.mode,opt.task_id):
        print(lang,iterr)
        lang_data_dict[lang] = LanguageData(opt,lang,iterr,m,t_id)

    print('-' * 80)

    model, criterion, optimizer = setup(opt)
    if not masker == None:
        del model
        model = masker.model
    #printOptions(opt)

    start_time = time.time()
    metrics = {}
    globaliter = 1
    loss_avg = Averager()
    if opt.mode == 'val' or opt.mode == 'trainVal':
        tflogger = tensorlog(dirr=f'{opt.exp_dir}/{opt.experiment_name}/runs')
    
    best_acc = 0
    best_ED = 0

    while(True):
        i = 1
        current_lang = langQ.popleft()
        langQ.append(current_lang)
        
        for lang in langQ:
            if lang != current_lang:
                freeze_head(model,lang)

        while(i < lang_data_dict[current_lang].numiters):
       
            if globaliter%100 == 0: 
                print('iter:',globaliter)
            image_tensors, labels = lang_data_dict[current_lang].train_dataset.get_batch()
            image = image_tensors.to(device)
            text, length = lang_data_dict[current_lang].labelconverter.encode(labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)

            if 'CTC' in opt.Prediction:
                if not masker == None:
                    masker.before_forward(lang_data_dict[current_lang].task_id)
                preds = model(image, text, current_lang).log_softmax(2)
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
                if not masker == None:
                    masker.before_forward(lang_data_dict[current_lang].task_id)
                preds = model(image, text[:, :-1]) # align with Attention.forward
                target = text[:, 1:]  # without [GO] Symbol
                cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

            model.zero_grad()
            cost.backward()
            if not masker == None:
                    masker.after_forward(lang_data_dict[current_lang].task_id)

            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
            optimizer.step()

            if not pruner == None:
                pruner.on_batch_end()


            loss_avg.add(cost)

            # validation part

            if opt.mode == 'val' or opt.mode == 'trainVal':
                if globaliter % opt.valInterval == 0:
                    elapsed_time = time.time() - start_time
                    print(f'[{globaliter}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f} batch_ratio(Syn-Real):{opt.batch_ratio} Training on {current_lang} now')
                    log = open(f'./{opt.exp_dir}/{opt.experiment_name}/{opt.experiment_name}_log.txt', 'a')
                    log.write(f'[{globaliter}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f} batch_ratio(Syn-Real):{opt.batch_ratio} Training on {current_lang} now\n')
                    model.eval()

                    #validating and recording metrics on all languages
                    for lang in opt.langs:
                        print('-'*18+'Start Validating on '+lang+'-'*18)
                        metrics[lang] = languagelog(opt, model, lang_data_dict[lang], globaliter, criterion)

                    #replace saving by taking the best average value
                    best_acc, best_ED = save_best_model(opt, model, best_acc, best_ED, metrics)
                    log_best_metrics(opt, best_acc, best_ED)

                    for lang in opt.langs:
                        tflogger.record(lang, metrics[lang], iterr)
                    loss_avg.reset()

                    model.train()
            # save model per 1e+3 iter.
            if (globaliter) % 1e+3 == 0:
                torch.save(
                model.state_dict(), f'./{opt.exp_dir}/{opt.experiment_name}/saved_models/iter_{globaliter}.pth')

            if globaliter == opt.num_iter:
                '''print('end the training')
                sys.exit()'''
                return
                
            i += 1
            globaliter += 1 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name',help='name of Config to be used')
    arg = parser.parse_args()
    opt = getattr(M, arg.config_name)
    
    if not opt.experiment_name:
        opt.experiment_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.experiment_name += f'-Seed{opt.manualSeed}'
        # print(opt.experiment_name)

    os.makedirs(f'./{opt.exp_dir}/{opt.experiment_name}', exist_ok=True)
    os.makedirs(f'./{opt.exp_dir}/{opt.experiment_name}/saved_models', exist_ok=True)
    """ vocab / character number configuration """
    char_dict = {}
    f = open('characters.txt','r')
    lines = f.readlines()
    line = lines[0]
    #if opt.sensitive:
    #    # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    #    opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    for line in lines:#@azhar
        char_dict[line.split(',')[0]] = line.strip().split(',')[-1]

    opt.character = char_dict#@azhar
    print(opt.character)

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)

'''""" start training """
    if opt.saved_model != '':
        start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
        print(f'continue to train, start_iter: {start_iter}')  commented this out as I am not saving models'''


