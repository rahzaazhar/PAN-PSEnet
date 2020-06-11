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
    grc: float = 0.55
    data_usage: float = 1.0
    run_single: bool = False
    #VDD data dirs
    data_dir: str = '/home/azhar/L2BG/decathlon/'
    imdb_dir: str = '/home/azhar/L2BG/decathlon/annotations'


 
#Configs to generate final graphs
#pmnist
final_config1 = L2BG_Config(exp_name='pmnist',exp_dir='L2G_graphs/',datamode='pmnist',sim_strat='RSA',
                        config_name='config_pmnist',freeze_past='freeze',n_tasks=10,epochs=4,lr=0.01)

final_config2 = L2BG_Config(exp_name='CIFAR100',exp_dir='L2G_graphs/',datamode='CIFAR100',sim_strat='RSA',
                        config_name='config_CIFAR',freeze_past='slow_lr',n_tasks=10,epochs=10,lr=0.01)

final_config3 = L2BG_Config(exp_name='VDD',exp_dir='L2G_graphs/',datamode='VDD',sim_strat='RSA',
                        config_name='VDD',freeze_past='freeze',n_tasks=8,epochs=10,lr=0.001)

final_config4 = L2BG_Config(exp_name='MLTR',exp_dir='L2G_graphs/',datamode='mltr',sim_strat='RSA',
                        config_name='MLTR',n_tasks=5,epochs=15,lr=1.0)
