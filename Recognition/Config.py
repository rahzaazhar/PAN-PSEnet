from dataclasses import dataclass, replace, field
from typing import List

'''@dataclass
class Resources:
    exp_dir: str
    train_data: str
    valid_data: str
    spath: str 
    shared_model: str

@dataclass
class HP:
    lr: float = 1.0
    beta1: float = 0.9
    rho: float = 0.95
    eps: float = 1e-8
    grad_clip: int = 5'''
@dataclass
class Task:
    lang: str
    #mode: str
    numiters: int #per language iteration
    useReal: bool = True
    useSyn: bool = True

    


@dataclass
class PruneConfig:
    
    final_rate: float = 0.1
    pruning_iter: int = 10
    init_masks: str = None
    need_cut: str = 'rnn_lang,FeatureExtraction'



@dataclass
class Config:

    #re: Resources
    
    experiment_name: str
    exp_dir: str
    train_data: str
    valid_data: str
    #langs: List[str]
    #pli: List[int]
    #mode: List[str]
    #task_id: List[int]
    save_path: str = field(init=False)
    select_data: List[int] = field(default_factory=list, init=False)
    batch_ratio: List[int] = field(default_factory=list, init=False)

    #hp: HP
    character: str = None
    shared_model: str = ''
    spath: str = ''
    Transformation: str = 'None'
    FeatureExtraction: str = 'VGG'
    SequenceModeling: str = 'BiLSTM'
    Prediction: str = 'CTC'
    share: str = 'CNN+LSTM'
    manualSeed: int = 1111
    workers: int = 4
    batch_size: int = 192
    num_iter: int = 300000
    valInterval: int = 1000
    saved_model: str = ''
    FT: bool = False
    adam: bool = False
    total_data_usage_ratio: str = '1.0'
    lr: float = 1.0
    beta1: float = 0.9
    rho: float = 0.95
    eps: float = 1e-8
    grad_clip: int = 5
    batch_max_length: int = 30
    imgH: int = 32
    imgW: int = 100
    rgb: bool = False
    sensitive: bool = False
    PAD: bool = False
    data_filtering_off: bool = False
    num_fiducial: int = 20
    input_channel: int = 1
    output_channel: int = 512
    hidden_size: int = 256

    def __post_init__(self):
        self.save_path = f'{self.exp_dir}/{self.experiment_name}'
        self.select_data = ['Syn','Real']
        self.batch_ratio = [0.5,0.5]



#re1 = Resources(exp_dir='Experiments', train_data='training/', valid_data='validation/', spath='lol.pth', shared_model='ha.pth')
#h1 = HP()

#C1 = Config(experiment_name = 'TestDataclass', exp_dir = 'Experiments', train_data = 'training/', valid_data = 'validation/', langs = ['ban','hin'], pli = [1000,1000], mode = ['train','train'])
#C2 = replace(C1, experiment_name = 'TestDataclass1', langs = ['hin','ban','arab'], pli = [1000,1000,1000], mode = ['train','train','train'])
#C3_test = replace(C2, experiment_name = 'ABH(CNN)', pli = [2,2,2], share = 'CNN', total_data_usage_ratio = '0.05')
#C3_tst_val = replace(C3_test, mode = ['val','val','val'])
#print(C2)

'''
#training arab + frozen CNN + CNN+LSTM pretrained on hindi bangla
C1 = Config(experiment_name = 'ABH(frozenCNN)', exp_dir = 'Experiments', train_data = '/content/drive/My Drive/data/training', valid_data = '/content/drive/My Drive/data/validation', langs = ['arab'], pli = [1000], mode = ['train'],spath='/content/drive/My Drive/SharedLSTM_AlternateHinBan_scratch_best_accuracy.pth',share='CNN')
C1_val = replace(C1, mode = ['val'])

#training baseline Bangla model CNN+LSTM
C4 = Config(experiment_name = 'Bangla_Baseline_Test', exp_dir = 'Experiments', train_data = '/content/drive/My Drive/data/training', valid_data = '/content/drive/My Drive/data/validation', langs = ['arab'], pli = [1000], mode = ['train'])
C4_val = replace(C4,mode='val')

C2 = replace(C1, experiment_name = 'TestDataclass1', langs = ['hin','ban','arab'], pli = [1000,1000,1000], mode = ['train','train','train'])
C3_test = replace(C2, experiment_name = 'ABH(CNN)', pli = [2,2,2], share = 'CNN', total_data_usage_ratio = '0.05')
C3_tst_val = replace(C3_test, mode = ['val','val','val'])
'''

'''C_subnet_ban = Config(experiment_name = 'Gen_Ban_Subnet1', exp_dir='Experiments', train_data = 'training', valid_data = 'validation', langs = ['ban'], pli = [1000], mode=['train'], num_iter = 100, task_id = [1])
C_subnet_ban_val = replace(C_subnet_ban,mode=['val'])
P_subnet_ban = PruneConfig()

C_subnet_ban = Config(experiment_name = 'Gen_Ban_Subnet', exp_dir='Experiments', train_data = '/content/drive/My Drive/data/training', valid_data = '/content/drive/My Drive/data/validation', langs = ['ban'], pli = [1000], mode=['train'], num_iter = 6000, task_id = [1])
C_subnet_ban_val = replace(C_subnet_ban,mode=['val'])

C_subnet_arab = Config(experiment_name = 'Gen_Arab_Subnet', exp_dir='Experiments', train_data = '/content/drive/My Drive/data/training', valid_data = '/content/drive/My Drive/data/validation', langs = ['arab'], pli = [1000], mode=['train'], num_iter = 6000, task_id = [0])
C_subnet_arab_val = replace(C_subnet_arab,mode=['val'])

C_subnet_hin = Config(experiment_name = 'Gen_Hin_Subnet', exp_dir='Experiments', train_data = '/content/drive/My Drive/data/training', valid_data = '/content/drive/My Drive/data/validation', langs = ['hin'], pli = [1000], mode=['train'], num_iter = 6000, task_id = [2])
P_subnet_ban = PruneConfig()

C_arab_subnet_train = Config(experiment_name = 'Arab_Subnet_train', exp_dir='Experiments', train_data = 'training', valid_data = 'validation', langs = ['arab'], pli = [1000], mode=['trainVal'], num_iter = 6000, task_id = [0],total_data_usage_ratio=0.05,valInterval=2)'''
path = '/path/to/data/'
test_new_train = Config(experiment_name='test_new_train_hin',exp_dir='tests',train_data=path+'training',valid_data=path+'validation')
taskconfig_hin = Task(lang='hin',numiters=10)