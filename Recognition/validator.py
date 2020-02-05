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
from modelv1 import Model, SharedLSTMModel, SLSLstm
from trainv2 import setup, validation, languagelog
from test import validation
import Config as M
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def filter(files,lastCp):
	filtered = []
	for file in files:
		if file =='best_accuracy.pth' or file == 'best_norm_ED.pth':
			continue
		iterr = file.split('_')[-1].split('.')[0]
		if int(iterr)>lastCp:
			filtered.append(file)
	return filtered



#initialise state of the program
def monitor(opt,model,criterion,LangDataDict,checkpoint_path):
	numCp = 0
	lastCp = 0
	combined_best_acc = 0
	combined_best_ED
	tflogger = tensorlog(dirr=f'{opt.exp_dir}/{opt.experiment_name}/runs')
	metrics = {}
	
	while 1:
		files = os.listdir(checkpoint_path)
		# if a new checkpoint is created 
		if (len(files)>numCp):
			numCp = numCp+(len(files)-numCp
			checkpoints = filter(files,lastCp)
			checkpoints.sort()
			for checkpoint in checkpoints:
				model.load_state_dict(torch.load(checkpoint))
				iterr = int(file.split('_')[-1].split('.')[0])
				#validating and recording metrics on all languages
                for lang in opt.langs:
                    print('-'*18+'Start Validating on '+lang+'-'*18)
                    metrics[lang]=languagelog(opt,model,LangDataDict[lang],iterr,criterion)

                #replace saving by taking the best average value
                word_acc = 0
                ED = 0
                for lang in opt.langs:
                    word_acc += metrics[lang][3]
                    ED += metrics[lang][4]
                ED = ED/len(opt.langs)
                word_acc = word_acc/len(opt.langs)
                #print(ED,word_acc)
                if word_acc>combined_best_acc:
                    combined_best_acc = word_acc
                    torch.save(model.state_dict(), f'./{opt.exp_dir}/{opt.experiment_name}/{opt.experiment_name}_best_accuracy.pth')
                if ED>combined_best_ED:
                    combined_best_ED = ED
                    torch.save(model.state_dict(), f'./{opt.exp_dir}/{opt.experiment_name}/{opt.experiment_name}_best_NED.pth')

                #with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a') as log:
                best_model_log = f'best_accuracy: {combined_best_acc:0.3f}, best_norm_ED: {combined_best_ED:0.2f}'
                print(best_model_log)
                log = open(f'./{opt.exp_dir}/{opt.experiment_name}/{opt.experiment_name}_log.txt', 'a')
                log.write(best_model_log + '\n')

                    for lang in opt.langs:
                        tflogger.record(model,lang,metrics[lang][7],metrics[lang][2],metrics[lang][0],metrics[lang][5],metrics[lang][3],metrics[lang][1],metrics[lang][4],metrics[lang][6],globaliter)
                        #process next checkpoint after last checkpoint


if __name__ == '__main__':

	parser.add_argument('--checkpoint_path',help='path to directory with checkpoint files')
	parser.add_argument('--config_name',help='give name of config')
	arg = parser.parse_args()
	opt = getattr(M, arg.config_name)
	model, criterion, _ = setup(opt)
	LangDataDict = {}
    for lang,iterr,m in zip(opt.langs,opt.pli,opt.mode):
        print(lang,iterr)
        LangDataDict[lang] = LanguageData(opt,lang,iterr,m)
	model.eval()
	monitor(opt,model,criterion,LangDataDict,arg.checkpoint_path)




