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

from utils import CTCLabelConverter, AttnLabelConverter, Averager, tensorlog, Scheduler
from datasetv1 import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from modelv1 import Model, SharedLSTMModel
from test import validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class LanguageData(object):
    def __init__(self, opt, lang, numiters, useSyn=True):
        self.AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.lang = lang
        self.current_iter = 1
        self.best_accuracy = 0
        self.best_norm_ED = 0
        self.numiters = int(numiters)
        self.useSyn = useSyn
        self.num_classes = len(opt.character[lang])
        #loaders and datasets
        #put usesyn in config 
        if self.useSyn:
            self.train_dataset = Batch_Balanced_Dataset(opt, opt.train_data+'/train_'+self.lang, self.lang, batch_ratio=[0.5,0.5], select_data=['Syn','Real'])
            self.Tvalid_dataset = hierarchical_dataset(self.lang, root=opt.train_data+'/train_'+self.lang, opt=opt)
            self.Tvalid_loader = self.genLoader(opt,self.Tvalid_dataset)
            self.Rvalid_dataset = hierarchical_dataset(self.lang,root=opt.valid_data+'/val_'+self.lang+'/Real',opt=opt)
            self.Rvalid_loader = self.genLoader(opt,self.Rvalid_dataset)
            self.Synvalid_dataset = hierarchical_dataset(self.lang,root=opt.valid_data+'/val_'+self.lang+'/Syn', opt=opt)
            self.Synvalid_loader = self.genLoader(opt,self.Synvalid_dataset)
        else:
            self.train_dataset = Batch_Balanced_Dataset(opt,opt.train_data+'/train_'+self.lang,self.lang,batch_ratio=[1.0],select_data=['Real'])
            self.Tvalid_dataset = hierarchical_dataset(self.lang,root=opt.train_data+'/train_'+self.lang, opt=opt, select_data=['Real'])
            self.Tvalid_loader = self.genLoader(opt,self.Tvalid_dataset)
            self.Rvalid_dataset = hierarchical_dataset(self.lang,root=opt.valid_data+'/val_'+self.lang+'/Real',opt=opt)
            self.Rvalid_loader = self.genLoader(opt,self.Rvalid_dataset)

        if 'CTC' in opt.Prediction:
            self.labelconverter = CTCLabelConverter(opt.character[lang])
        else:
            self.labelconverter = AttnLabelConverter(opt.character[lang])

    def genLoader(self,opt,dataset):
        return torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size,
                shuffle=True,  # 'True' to check training progress with validation function.
                num_workers=int(opt.workers),
                collate_fn=self.AlignCollate_valid, pin_memory=True)

def train(opt):
    """ dataset preparation """
    opt.select_data = opt.select_data.split('-')#default will be syn and real
    opt.batch_ratio = opt.batch_ratio.split('-')#default will be Syn-0.8 Real-0.2
    #langs = deque(opt.langs)
    langQ = []
    for lang,mode in zip(opt.langs, opt.mode):
    	if mode=='train':
    		langQ.append(lang)
    langQ = deque(langQ)
 
    print(opt.train_data)
    LangDataDict = {}
    for lang,iterr in zip(opt.langs,opt.pli):
        print(lang,iterr)
        LangDataDict[lang] = LanguageData(opt,lang,iterr)



    print('-' * 80)

    model, criterion, optimizer = setup(opt)
    printOptions(opt)

    start_time = time.time()
    metrics = {}
    globaliter = 1
    loss_avg = Averager()
    tflogger = tensorlog(dirr=f'{opt.exp_dir}/{opt.experiment_name}/runs')

    while(True):
        i = 1
        # can replace this with for loop to provide customisable training schedules 
        current_lang = langQ.popleft()
        langQ.append(current_lang)

        while(i<LangDataDict[current_lang].numiters):
       
            print('iter:',globaliter)
            image_tensors, labels = LangDataDict[current_lang].train_dataset.get_batch()
            image = image_tensors.to(device)
            text, length = LangDataDict[current_lang].labelconverter.encode(labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)

            if 'CTC' in opt.Prediction:
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
                preds = model(image, text[:, :-1]) # align with Attention.forward
                target = text[:, 1:]  # without [GO] Symbol
                cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
            optimizer.step()

            loss_avg.add(cost)

            # validation part
            if globaliter % opt.valInterval == 0:
                elapsed_time = time.time() - start_time
                print(f'[{globaliter}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f} batch_ratio(Syn-Real):{opt.batch_ratio} Training on {current_lang} now')
                log = open(f'./{opt.exp_dir}/{opt.experiment_name}/{opt.experiment_name}_log.txt', 'a')
                log.write(f'[{globaliter}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f} batch_ratio(Syn-Real):{opt.batch_ratio} Training on {current_lang} now\n')
                model.eval()

                #validating and recording metrics on all languages
                for lang in opt.langs:
                    #Synvalidloss,Syn_valid_acc, Real_valid_loss, current_accuracy, current_norm_ED, train_accuracy = languagelog(opt,model,)
                    print('-'*18+'Start Validating on '+lang+'-'*18)
                    metrics[lang]=languagelog(opt,model,LangDataDict[lang],globaliter,criterion)

                model.train()
                #replace saving by taking the best average value
                save_accuracy_model_flag = True
                save_ED_model_flag = True
                for lang in langQ:
                    if not metrics[lang][3] > LangDataDict[lang].best_accuracy:
                       save_accuracy_model_flag = False
                    if not metrics[lang][4] > LangDataDict[lang].best_norm_ED:
                       save_ED_model_flag = False

                if save_accuracy_model_flag:
                    for lang in langQ:
                        LangDataDict[lang].best_accuracy = metrics[lang][3]
                    torch.save(model.state_dict(), f'./{opt.exp_dir}/{opt.experiment_name}/{opt.experiment_name}_best_accuracy.pth')

                if save_ED_model_flag:
                    for lang in langQ:
                        LangDataDict[lang].best_norm_ED = metrics[lang][4]
                    torch.save(model.state_dict(), f'./{opt.exp_dir}/{opt.experiment_name}/{opt.experiment_name}_best_norm_ED.pth')
                    
                #with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a') as log:
                best_model_log = f'best_accuracy: {LangDataDict[current_lang].best_accuracy:0.3f}, best_norm_ED: {LangDataDict[current_lang].best_norm_ED:0.2f}'
                print(best_model_log)
                log = open(f'./{opt.exp_dir}/{opt.experiment_name}/{opt.experiment_name}_log.txt', 'a')
                log.write(best_model_log + '\n')

                for lang in opt.langs:
                	tflogger.record(model,lang,loss_avg.val(),metrics[lang][2],metrics[lang][0],metrics[lang][5],metrics[lang][3],metrics[lang][1],metrics[lang][4],metrics[lang][6],globaliter)
                
                loss_avg.reset()
                #tflogger for all languages has to be implemented takes in metrics dataclass of language as input

            '''if i==2000:
                opt.batch_ratio = batch_ratio_schedule.nextele()
                train_dataset.change_batch_ratio(opt)
            if i==5000:
                opt.batch_ratio = batch_ratio_schedule.nextele()
                train_dataset.change_batch_ratio(opt)'''
            '''if (i%10)==0:
                opt.batch_ratio = batch_ratio_schedule.nextele()
                train_dataset.change_batch_ratio(opt)'''


            # save model per 1e+3 iter.
            if (globaliter) % 1e+3 == 0:
                torch.save(
                    model.state_dict(), f'./{opt.exp_dir}/{opt.experiment_name}/{current_lang}_iter_{globaliter}.pth')

            if globaliter == opt.num_iter:
                print('end the training')
                sys.exit()
            i += 1
            LangDataDict[current_lang].current_iter += 1
            globaliter += 1 

#@azhar
def languagelog(opt,model,LangData,globaliter,criterion):#@azhar modified 
    with open(f'./{opt.exp_dir}/{opt.experiment_name}/{opt.experiment_name}_log.txt', 'a') as log:
        log.write('#'*18+'Start Validating on '+LangData.lang+'#'*18+'\n')
        log.write('validating on Synthetic data\n')
        Synvalidloss, Syn_valid_acc, SynvalED = validate(opt,model,criterion, LangData.Synvalid_loader, LangData.labelconverter,log,globaliter,'Syn-val-loss',LangData.lang)
        log.write('validating on Real data\n')
        Real_valid_loss,Real_valid_accuracy,Real_valid_norm_ED = validate(opt,model,criterion, LangData.Rvalid_loader, LangData.labelconverter,log,globaliter,'Real-val-loss',LangData.lang)
        log.write('Evaluating on Train data\n')
        _,train_accuracy,_ = validate(opt,model,criterion, LangData.Tvalid_loader, LangData.labelconverter,log,globaliter,'train-loss',LangData.lang)

    return [Synvalidloss,Syn_valid_acc, Real_valid_loss, Real_valid_accuracy, Real_valid_norm_ED, train_accuracy, SynvalED]

def validate(opt,model,criterion,loader,converter,log,i,lossname,lang):#@azhar
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

def setup(opt):
    """ model configuration """

    if opt.rgb:
        opt.input_channel = 3

    #@azhar
    chardict = {}
    for lang,charlist in opt.character.items():
        chardict[lang] = len(charlist)
    
    #@azhar
    if opt.share=='CNN':
        model = Model(opt,chardict)
    if opt.share=='CNN+LSTM':
        model = SharedLSTMModel(opt,chardict)    	

    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)


    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model = weight_innit(model)

    model = load_sharedW(model,opt)

    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    
    print("Model:")
    print(model)

    for lang,mode in zip(opt.langs,opt.mode):
    	if(mode!='train'):
    		model = freeze_head(model,lang)


    
    

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    
    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    return model, criterion, optimizer

#@azhar
def load_sharedW(model,opt):

    checkpoint = torch.load(opt.spath)
    for x,y in model.named_parameters():
        if x in checkpoint.keys() and 'FeatureExtraction' in x:
            y.data.copy_(checkpoint[x].data)

        if x in checkpoint.keys() and 'hin' in x:
            y.data.copy_(checkpoint[x].data)

        if ('rnn_lang' in x):
            s = x.split('.')
            s.insert(2,'hin')
            s = '.'.join(s)
            y.data.copy_(checkpoint[s].data)
    
    '''modules = list(model.named_children())
    modules = list(modules[0][1].named_children())
    for x,y in modules:
        if x=='FeatureExtraction':
            print('loading CNN module:',x)
            y.load_state_dict(torch.load('CNN_hin_saved.pth'))#y.load_state_dict(torch.load(opt.spath))'''

    '''for name, para in model.named_parameters():
        print(name,para)'''

    return model

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

#@azhar to freeze weights of task head 
def freeze_head(model,head):
    for name,param in model.named_parameters():
        if head in name:
        	param.requires_grad = False
    return model

#@azhar testing function    
def paramCheck(model):
    for name, param in model.named_parameters():
        print(name,param.requires_grad) 





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', help='Directory for all experiments')#@azhar
    parser.add_argument('--experiment_name', help='Where to store logs and models')
    parser.add_argument('--train_data', help='list of sub-directories for  training dataset', required=True)
    parser.add_argument('--valid_data', help='list of sub-directories for  training dataset', required=True)
    parser.add_argument('--spath',help='path to shared weights')#@azhar
    #parser.add_argument('--train_data', required=True, help='list of sub-directories for  training dataset')
    #parser.add_argument('--valid_data', required=True, help='list of sub-directories for  validation dataset')
    #parser.add_argument('--Synvalid_data', required=True, help='path to validation dataset')
    parser.add_argument('--langs', required=True, nargs='+',help='list of languages to train on')#@azhar
    parser.add_argument('--pli', required=True, nargs='+',help='list of number of iteration per language')#@azhar
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='Syn-Real',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--mode',required=True,nargs='+',help='which mode train|validate|train&validate')
    parser.add_argument('--batch_max_length', type=int, default=30, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--share', type=str, default='CNN', help='Share CNN|CNN+LSTM|CNN+SLSTM',)#@azhar
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    if not opt.experiment_name:
        opt.experiment_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.experiment_name += f'-Seed{opt.manualSeed}'
        # print(opt.experiment_name)

    os.makedirs(f'./{opt.exp_dir}/{opt.experiment_name}', exist_ok=True)

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




         
'''class prettyprint:
	def __init__(self,opt):
		self.default_exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}-Seed{opt.manualSeed}'
		self.exp_name = f'./{opt.exp_dir}/{opt.experiment_name}'
		self.exp_log = self.exp_name+f'/{opt.experiment_name}_log.txt'
@dataclass
class metrics:
	Real_
""" start training """
    if opt.saved_model != '':
        start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
        print(f'continue to train, start_iter: {start_iter}')  commented this out as I am not saving models'''


