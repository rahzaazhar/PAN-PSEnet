import torch
from torch.utils.data import Subset
from datasetv1 import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from utils import CTCLabelConverter
from utils import get_vocab
import os

def genLoader(opt,dataset,shuffle=True):
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    return torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size,
            shuffle=shuffle, 
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_valid, pin_memory=True)


#def get_subset(loaders,data_usage):

def get_multiple_loaders(taskconfig):
    multiple_loaders = {}
    for lang_config in taskconfig.langs:
        multiple_loaders[lang_config.lang_name] = get_loaders(taskconfig.hp,lang_config)

    return multiple_loaders


def get_loaders(config,langconfig):
    loaders = {}
    lang = langconfig.lang_name
    train_string = os.path.join(langconfig.base_data_path,'training','train_'+lang)
    val_string = os.path.join(langconfig.base_data_path,'validation','val_'+lang)
    if langconfig.useReal and langconfig.useSyn:
        print('use Real and SYN')
        select = [langconfig.which_real_data,langconfig.which_syn_data]
        batch_ratio = [langconfig.real_percent,langconfig.syn_percent]
        synval_dataset = hierarchical_dataset(lang, root= val_string+'/Syn', opt=config)
        rval_dataset = hierarchical_dataset(lang, root=val_string+'/Real', opt=config)
        loaders['synval'] =  genLoader(config,synval_dataset,False)
        loaders['rval'] = genLoader(config,rval_dataset,False)
    elif langconfig.useReal:
        print('use Real only')
        select = [langconfig.which_real_data]
        batch_ratio = [1.0]
        rval_dataset = hierarchical_dataset(lang, root=val_string+'/Real', opt=config)
        loaders['rval'] = genLoader(config,rval_dataset,False)
    elif langconfig.useSyn:
        print('Use Syn only')
        select = [langconfig.which_syn_data]
        batch_ratio = [1.0]
        synval_dataset = hierarchical_dataset(lang, root=val_string+'/Syn', opt=config)
        loaders['synval'] =  genLoader(config,synval_dataset,False)
    loaders['train'] = Batch_Balanced_Dataset(config, train_string, lang, batch_ratio=batch_ratio, select_data=select)
    train_dataset_eval = hierarchical_dataset(lang, root=train_string, opt=config, select_data=select)
    loaders['eval'] = genLoader(config,train_dataset_eval,False)

    labelconverter = CTCLabelConverter(config.character[lang])

    return (loaders, labelconverter)


