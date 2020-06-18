import torch
from torch.utils.data import Subset
from datasetv1 import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from utils import CTCLabelConverter

def genLoader(opt,dataset,shuffle=True):
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    return torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size,
            shuffle=shuffle, 
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_valid, pin_memory=True)


#def get_subset(loaders,data_usage):


def get_loaders(config,taskconfig):
    loaders = {}
    lang = taskconfig.lang
    train_string = config.train_data+'/train_'+lang
    val_string = config.valid_data+'/val_'+lang
    if taskconfig.useReal and taskconfig.useSyn:
        print('use Real and SYN')
        select = ['Real','Syn']
        batch_ratio = [0.5,0.5]
        synval_dataset = hierarchical_dataset(lang, root= val_string+'/Syn', opt=config)
        rval_dataset = hierarchical_dataset(lang, root=val_string+'/Real', opt=config)
        loaders['synval'] =  genLoader(config,synval_dataset,False)
        loaders['rval'] = genLoader(config,rval_dataset,False)
    elif taskconfig.useReal:
        print('use Real only')
        select = ['Real']
        batch_ratio = [1.0]
        rval_dataset = hierarchical_dataset(lang, root=val_string+'/Real', opt=config)
        loaders['rval'] = genLoader(config,rval_dataset,False)
    elif taskconfig.useSyn:
        print('Use Syn only')
        select = ['Syn']
        batch_ratio = [1.0]
        synval_dataset = hierarchical_dataset(lang, root= val_string+'/Syn', opt=config)
        loaders['synval'] =  genLoader(config,synval_dataset,False)
    loaders['train'] = Batch_Balanced_Dataset(config, train_string, taskconfig.lang, batch_ratio=batch_ratio, select_data=select)
    train_dataset_eval = hierarchical_dataset(lang, root=train_string, opt=config, select_data=select)
    loaders['eval'] = genLoader(config,train_dataset_eval,False)

    labelconverter = CTCLabelConverter(config.character[lang])

    return loaders, labelconverter