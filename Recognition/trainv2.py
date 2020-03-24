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

from utils import CTCLabelConverter, AttnLabelConverter, Averager, tensorlog, Scheduler, LanguageData
from train_utils import save_best_model, log_best_metrics, setup_model, setup_optimizer, setup_loss, printOptions
from utils import LanguageData, get_vocab
from modelv1 import sharedCNNModel, SharedLSTMModel, SLSLstm
from test import validation
import Config as M

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#@azhar
def plot_grad_sim(opt, x, sims, module, para):
    fig, ax = plt.subplots()
    #fig.suptitle(module+para)
    xp = np.array(x)
    for name,sim in sims.items():
        if module in name and para in name:
            sim = np.array(sim)
            name = name.replace(module,'')
            plt.scatter(xp, sim, label=name)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax.set_title(module+para)
    #ax.legend(loc='upper right')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('iters')
    ax.set_ylabel('Cosine similarity')
    ax.grid(True)
    fig.savefig(opt.save_path+'/{}{}_gradSim.png'.format(module,para))
    del fig 
    del ax


#@azhar
def gradient_similarity(sims,grad_arab,grad_ban):
    for name in sims:
        g_arab = grad_arab[name]
        g_ban = grad_ban[name]
        g_arab = g_arab.view(-1)
        g_ban = g_ban.view(-1)
        g_arab = g_arab.cpu().detach().numpy()
        g_ban = g_ban.cpu().detach().numpy()
        sims[name].append(dot_product(g_arab,g_ban))


#@azhar
def multiplot(opt,x,sims):
    for module in ['FeatureExtraction.','rnn_lang.']:
        for para in ['weight','bias']:
            plot_grad_sim(opt,x,sims,module,para)

def average_grad(opt,grad):
    for name,para in grad.items():
        grad[name] = 2*para/opt.valInterval

#@azhar
def dot_product(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def check_para_equal(pa,pb):
    res = True
    for name in pa.keys():
        if not torch.equal(pa[name],pb[name]):
            res = False
            break
    return res

#@azhar
def languagelog(opt, model, LangData, globaliter, criterion, masker=None):#@azhar modified 
    metrics = {}
    with open(f'./{opt.exp_dir}/{opt.experiment_name}/{opt.experiment_name}_log.txt', 'a') as log:
        log.write('#'*18+'Start Validating on '+LangData.lang+'#'*18+'\n')
        log.write('validating on Synthetic data\n')
        Synvalidloss, Syn_valid_acc, SynvalED = validate(opt,model,criterion, LangData.Synvalid_loader, LangData.labelconverter,log,globaliter,'Syn-val-loss',LangData.lang,masker)
        metrics['Syn_validation_loss'], metrics['Syn_val_Wordaccuracy'], metrics['Syn_val_edit-dist'] = Synvalidloss, Syn_valid_acc, SynvalED
        log.write('validating on Real data\n')
        Real_valid_loss,Real_valid_accuracy,Real_valid_norm_ED = validate(opt,model,criterion, LangData.Rvalid_loader, LangData.labelconverter,log,globaliter,'Real-val-loss',LangData.lang,masker)
        metrics['Real_validation_loss'], metrics['Real_val_Wordaccuracy'], metrics['Real_val_edit-dist'] = Real_valid_loss, Real_valid_accuracy, Real_valid_norm_ED
        log.write('Evaluating on Train data\n')
        train_loss,train_accuracy,_ = validate(opt,model,criterion, LangData.Tvalid_loader, LangData.labelconverter,log,globaliter,'train-loss',LangData.lang,masker)
        metrics['train_loss'], metrics['train_Wordaccuracy'] = train_loss, train_accuracy

    return metrics

#@azhar
def validate(opt, model, criterion, loader, converter, log, i, lossname, lang, masker=None):#@azhar
    #print('enter validate')
    with torch.no_grad():
        valid_loss, current_accuracy, current_norm_ED, preds, labels, infer_time, length_of_data = validation(
            model, criterion, loader, converter, opt, lang,masker)
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

#@azhar
def freeze_head(model,head):
    for name,param in model.named_parameters():
        if head in name:
            param.requires_grad = False

#@azhar
def unfreeze_head(model,head):
    for name,param in model.named_parameters():
        if head in name:
            param.requires_grad = True

def set_zero(grad):
    for name,para in grad.items():
        grad[name] = torch.zeros(para.size())

#@azhar testing function    
def paramCheck(model):
    for name, param in model.named_parameters():
        print(name,param.requires_grad) 


def train(opt,model,optimizer,criterion,pruner=None,masker=None):
    """ dataset preparation """
    langQ = []
    for lang,mode in zip(opt.langs, opt.mode):
        if mode=='train' or mode=='dev':
            langQ.append(lang)
    langQ = deque(langQ)
 
    print(opt.train_data)
    lang_data_dict = {}
    for lang, iterr, m, t_id in zip(opt.langs,opt.pli,opt.mode,opt.task_id):
        print(lang,iterr)
        lang_data_dict[lang] = LanguageData(opt, lang, iterr, m, t_id)
    print('-' * 80)

    start_time = time.time()
    metrics = {}
    globaliter = 1
    loss_avg = Averager()
    best_acc = 0
    best_ED = 0
    grad1 = OrderedDict()
    grad2 = OrderedDict()
    dump = OrderedDict()
    sims = {}
    x = []
    grad_list = []
    grad1_list = []
    grad2_list = []
    j = 1
    

    if opt.mode[0] == 'val' or opt.mode[0] == 'dev':
        tflogger = tensorlog(opt)
        
    freeze_head(model,'hin')
    for name,para in model.named_parameters():
        if para.requires_grad == True: 
            if 'Predictions' not in name:
                sims[name] = []
                grad1[name] = torch.zeros(para.size())
                grad2[name] = torch.zeros(para.size())
            elif 'arab' in name:
                grad1[name] = torch.zeros(para.size())
            elif 'ban' in name:
                grad2[name] = torch.zeros(para.size())
  
    while(True):
        i = 1
        current_lang = langQ.popleft()
        langQ.append(current_lang)
        while(i <= lang_data_dict[current_lang].numiters):
            
            if globaliter%100 == 0: 
                print('iter:',globaliter)
                train_time = batch_time_end-batch_time_start
                print('Batch time:{}'.format(train_time))
            
            image_tensors, labels = lang_data_dict[current_lang].train_dataset.get_batch()
            image = image_tensors.to(device)
            text, length = lang_data_dict[current_lang].labelconverter.encode(labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)

            if 'CTC' in opt.Prediction:
                if not masker == None:
                    masker.before_forward(lang_data_dict[current_lang].task_id)
                batch_time_start = time.time()
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
            #if globaliter % opt.valInterval or globaliter % (opt.valInterval-1):
            if current_lang == 'arab':
                for name,para in model.named_parameters():
                    if name in grad1.keys():
                        #grad_arab[name] = copy.deepcopy(para.grad.data)
                        grad1[name] = grad1[name].to(device)+para.grad.data
            if current_lang == 'ban':
                for name,para in model.named_parameters():
                    if name in grad2.keys():
                        #grad_hin[name] = copy.deepcopy(para.gard.data)
                        grad2[name] = grad2[name].to(device)+para.grad.data
            '''if globaliter % opt.valInterval == 0:
                grad = OrderedDict()
                for name,para in model.named_parameters():
                    if name in grad_lang[current_lang].keys():
                        grad[name] = copy.deepcopy(para.grad.data)
                x.append(globaliter)
                grad_list.append(grad)'''
                
            if not masker == None:
                masker.after_forward(lang_data_dict[current_lang].task_id)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
            optimizer.step()
            #gradient accumulation
            '''if globaliter%2 == 0:
                model.zero_grad()
                for name,para in model.named_parameters():
                    if para.requires_grad == True:
                        #print(type(grad_arab[name]),type(grad_ban[name]))
                        accgrad = grad_arab[name].to(device)+grad_ban[name].to(device)
                        para.grad.data.copy_(accgrad)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
                optimizer.step()'''
                
            batch_time_end = time.time()
            if not pruner == None:
                pruner.on_batch_end()

            loss_avg.add(cost)

            #validation
            if opt.mode[0] == 'val' or opt.mode[0] == 'dev':
                if globaliter % opt.valInterval == 0:
                    elapsed_time = time.time() - start_time
                    print(f'[{globaliter}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f} batch_ratio(Syn-Real):{opt.batch_ratio} Training on {current_lang} now')
                    log = open(f'./{opt.exp_dir}/{opt.experiment_name}/{opt.experiment_name}_log.txt', 'a')
                    log.write(f'[{globaliter}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f} batch_ratio(Syn-Real):{opt.batch_ratio} Training on {current_lang} now\n')
                    model.eval()

                    #validating and recording metrics on all languages
                    for lang in opt.langs:
                        print('-'*18+'Start Validating on '+lang+'-'*18)
                        if not masker == None: 
                            metrics[lang] = languagelog(opt, model, lang_data_dict[lang], globaliter, criterion, masker)
                        else:
                            metrics[lang] = languagelog(opt, model, lang_data_dict[lang], globaliter, criterion)
                        
                    #replace saving by taking the best average value
                    best_acc, best_ED = save_best_model(opt, model, best_acc, best_ED, metrics)
                    log_best_metrics(opt, best_acc, best_ED)

                    #log metrics for languages on tensorboard
                    for lang in opt.langs:
                        tflogger.record(lang, metrics[lang], globaliter)
                    
                    x.append(globaliter)
                    average_grad(opt,grad1)
                    average_grad(opt,grad2)
                    grad1_list.append(copy.deepcopy(grad1))
                    grad2_list.append(copy.deepcopy(grad2))
                    gradient_similarity(sims,grad1,grad2)
                    multiplot(opt,x,sims)
                    dump = OrderedDict()
                    dump = {'iter':x,'grad_sims':sims}
                    save_name = opt.experiment_name.split('_')
                    save_name = save_name[0]+save_name[-1]
                    torch.save(dump,opt.save_path+'/{}.pth'.format(save_name))
                    torch.save(grad1_list,opt.save_path+'/grads_arab.pth')
                    torch.save(grad2_list,opt.save_path+'/grads_ban.pth')
                    
                    #torch.save(dump,opt.save_path+'/lang_grads_ban.pth')
                    set_zero(grad1)
                    set_zero(grad2)
                    loss_avg.reset()
                    model.train()
            
            # save model per 1e+3 iter.
            if pruner == None:
                if (globaliter) % 2e+3 == 0:
                    print('Not saving model')
                    #torch.save(
                    #model.state_dict(), f'./{opt.exp_dir}/{opt.experiment_name}/saved_models/iter_{globaliter}.pth')

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
    opt.character = get_vocab()

    model = setup_model(opt)
    print(model)
    loss = setup_loss(opt)
    optim = setup_optimizer(opt, model)
    
    if not opt.experiment_name:
        opt.experiment_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.experiment_name += f'-Seed{opt.manualSeed}'

    os.makedirs(opt.save_path, exist_ok=True)
    os.makedirs(opt.save_path+'/saved_models', exist_ok=True)
    printOptions(opt)

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
    train(opt,model,optim,loss)




