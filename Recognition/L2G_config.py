from dataclasses import dataclass, replace
import torch.nn as nn
from modules.sequence_modeling import BidirectionalLSTM





@dataclass
class L2BG_Config():

    #re: Resources
    
    exp_name: str
    config_name: str
    exp_dir: str
    datamode: str
    pretrain_path: str = ''
    single_task_name: str = 'aircraft'
    n_tasks: int = 5
    lr: float = 0.01
    batch_size: int = 64
    sim_strat: str = 'RSA'
    freeze_past: str = 'freeze'
    epochs: int = 5
    alpha: float = 0.5
    data_usage: float = 1.0
    run_single: bool = False
    #VDD data 
    data_dir: str = '/home/azhar/TextRecogShip/Recognition/decathlon/'
    imdb_dir: str = '/home/azhar/TextRecogShip/Recognition/decathlon/annotations'


#config1 To find out whwther a randomly grown network performs worse than RSA strategy 
config1 = L2BG_Config(exp_name='Random_growth_10T',exp_dir='L2G_graphs/',datamode='CIFAR100',sim_strat='random_growth',
						config_name='config2',n_tasks=10,epochs=5)
#config2 To train on CIFAR100 using RSA_sim
config2 = L2BG_Config(exp_name='RSA_sim',exp_dir='L2G_graphs/',datamode='CIFAR100',sim_strat='RSA',
						config_name='config2')
#config3 train task one for more iterations and subsequent tasks for lesser iterations select + most dissimilar task (10-5)
config3 = L2BG_Config(exp_name='RSA_sim_1',exp_dir='L2G_graphs/',datamode='CIFAR100',sim_strat='RSA',
						config_name='config3')
#config4 train task one for more iterations and subsequent tasks for lesser iterations + select most similar task (10-5)
config4 = L2BG_Config(exp_name='RSA_sim_2',exp_dir='L2G_graphs/',datamode='CIFAR100',sim_strat='RSA',
						config_name='config4')
#config5 To train on datasets by freezing weights of previous tasks
config5 = L2BG_Config(exp_name='RSA_sim_2_freeze',exp_dir='L2G_graphs/',datamode='CIFAR100',sim_strat='RSA',
						config_name='config5',freeze_past=True)
#config 6 To train 10 CIFAR Tasks each for 10 epochs wo freeze
config6 = L2BG_Config(exp_name='CIFAR100_10',exp_dir='L2G_graphs/',datamode='CIFAR100',sim_strat='RSA',
						config_name='config6',n_tasks=10)
#config 7 To train 10 CIFAR Tasks each for 10 epochs w freeze
config7 = L2BG_Config(exp_name='CIFAR100_10_freeze',exp_dir='L2G_graphs/',datamode='CIFAR100',sim_strat='RSA',
						config_name='config7',freeze_past=True,n_tasks=10)
#config_CIFAR100_alpha config_pmnist_alpha config_pmnist_alpha 
#above configs are to study the effect of alpha on model selection and accuracy
config_CIFAR100_alpha = {}
config_pmnist_alpha = {}
config_smnist_alpha = {}
config_VDD_alpha_5 = {}
alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for al in alphas:
  config_CIFAR100_alpha['alpha'+str(al)] = L2BG_Config(exp_name='alpha'+str(al)+'_CIFAR100',exp_dir='L2GB_exps/',datamode='CIFAR100',sim_strat='RSA',
                                                  config_name='config_CIFAR100_alpha',n_tasks=10,alpha=al,epochs=5)
  config_pmnist_alpha['alpha'+str(al)] = L2BG_Config(exp_name='alpha'+str(al)+'_pmnist',exp_dir='L2GB_exps/',datamode='pmnist',sim_strat='RSA',
                                                  config_name='config_pmnist_alpha',n_tasks=10,alpha=al,epochs=5)
  config_pmnist_alpha['alpha'+str(al)] = L2BG_Config(exp_name='alpha'+str(al)+'_smnist',exp_dir='L2GB_exps/',datamode='smnist',sim_strat='RSA',
                                                  config_name='config_smnist_alpha',n_tasks=10,alpha=al,epochs=5)
  config_VDD_alpha_5['alpha'+str(al)] = L2BG_Config(exp_name='alpha'+str(al)+'_VDD',exp_dir='L2GB_exps/VDD/alpha_exp_5ep/',datamode='VDD',sim_strat='RSA',
                                                  config_name='config_VDD_alpha',n_tasks=8,alpha=al,epochs=5,data_usage=0.4)





########################tests and sanity checks#################################### 
config_pmnist_test = L2BG_Config(exp_name='pmnist_test',exp_dir='L2G_graphs/',datamode='pmnist',sim_strat='RSA',
                        config_name='config_pmnist',freeze_past=False,n_tasks=2,epochs=1)

config_CIFAR100_test = L2BG_Config(exp_name='CIFAR100_test',exp_dir='L2G_graphs/',datamode='CIFAR100',sim_strat='RSA',
                        config_name='config_pmnist',freeze_past=False,n_tasks=2,epochs=1)

config_smnist_test = L2BG_Config(exp_name='smnist_test',exp_dir='L2G_graphs/',datamode='smnist',sim_strat='RSA',
                        config_name='config_pmnist',freeze_past=False,n_tasks=2,epochs=1)

#config 8 To train VDD_test 
config_VDD_test = L2BG_Config(exp_name='VDD_test_1',exp_dir='L2G_graphs/',datamode='VDD',sim_strat='RSA',
                        config_name='config8',freeze_past=False,n_tasks=6,epochs=2,data_usage=0.2)

#config_san_1 To check if freezing is working or not
config_san_1 = L2BG_Config(exp_name='freezing_check',exp_dir='L2G_graphs/',datamode='CIFAR100',sim_strat='RSA',
                        config_name='config_san_1',freeze_past=True,n_tasks=2,epochs=1)
 
#Configs to generate final graphs
#pmnist
final_config1 = L2BG_Config(exp_name='3ad_pmnist_test',exp_dir='L2G_graphs/',datamode='pmnist',sim_strat='RSA',
                        config_name='config_pmnist',freeze_past=False,n_tasks=5,epochs=1)

final_config2 = L2BG_Config(exp_name='3ad_Cifar_freeze_5ep',exp_dir='L2G_graphs/',datamode='CIFAR100',sim_strat='RSA',
                        config_name='CIFAR',freeze_past='freeze',n_tasks=10,epochs=5)

final_config3 = L2BG_Config(exp_name='Cifar_slowlr_5ep',exp_dir='L2G_graphs/',datamode='CIFAR100',sim_strat='RSA',
                        config_name='CIFAR',freeze_past='slow_lr',n_tasks=10,epochs=5)

final_config4 = L2BG_Config(exp_name='pmnist_freeze_5ep',exp_dir='L2G_graphs/',datamode='pmnist',sim_strat='RSA',
                        config_name='CIFAR',freeze_past='freeze',n_tasks=10,epochs=5)

single_task_config = L2BG_Config(exp_name='single_task_run_cifar',exp_dir='L2G_graphs/',datamode='CIFAR100',sim_strat='RSA',
                        config_name='CIFAR',freeze_past='slow_lr',n_tasks=10,epochs=5,single_task_name='aircraft',run_single=True)
CIFAR_base = L2BG_Config(exp_name='Random_growth',exp_dir='L2G_graphs/',datamode='CIFAR100',sim_strat='RSA',
            config_name='config2')
VDD_base = L2BG_Config(exp_name='Random_growth',exp_dir='L2G_graphs/',datamode='VDD',sim_strat='RSA',
            config_name='config2')

MLT_test_config = L2BG_Config(exp_name='MLTR_test',exp_dir='L2G_graphs/',datamode='mltr',sim_strat='RSA',
            config_name='config2',n_tasks=2,epochs=1)

cifar_with_pretrain = L2BG_Config(exp_name='CIFAR100_pretrain',exp_dir='L2G_graphs/',datamode='CIFAR100',sim_strat='RSA',
                        config_name='config_san_1',freeze_past=True,n_tasks=10,epochs=5,pretrain_path='L2G_graphs/pretrained/')