import torch
import os
import copy
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from torch.utils.data import Subset
#from datasetv1 import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
#from fastNLP import logger
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)
        
    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.tensor(text,dtype=torch.long),torch.tensor(length,dtype=torch.long))#(torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

#@azhar
class metricsLog():
    def __init__(self, opt, lang_datas):
        syn = ['Syn_validation_loss','Syn_val_Wordaccuracy','Syn_val_edit-dist']
        real = ['Real_validation_loss','Real_val_Wordaccuracy','Real_val_edit-dist']
        self.steps = []
        self.lang_metrics = {}
        self.save_plots = opt.save_path+'/plots/'
        self.exp_name = opt.experiment_name
        os.makedirs(self.save_plots,exist_ok=True)

        for lang in lang_datas.keys():
            metrics = {}
            if lang_datas[lang].useSyn:
                for name in syn:
                  metrics[name] = []
            if lang_datas[lang].useReal:
                for name in real:
                  metrics[name] = []
            for name in ['train_loss','train_Wordaccuracy']:
                metrics[name] = []
            self.lang_metrics[lang] = copy.deepcopy(metrics)
    
    def update_steps(self,step):
        self.steps.append(step)

    def update_metrics(self, lang, metrics):
        for name, metric in metrics.items():
            self.lang_metrics[lang][name].append(metric)
        #print('Sanity check')
        #print(self.lang_metrics)

    def plot_metrics(self,lang):
        plot_path = self.save_plots+lang+'/'
        os.makedirs(plot_path,exist_ok=True)
        for name,val in self.lang_metrics[lang].items():
          plt.plot(self.steps, val)
          plt.title(lang)
          plt.xlabel('iterations')
          plt.ylabel(name)
          plt.savefig(plot_path+'{}.png'.format(name))
          plt.close('all')
    
    def save_metrics(self,path=None):
        exp_name = self.exp_name
        if path == None:
          torch.save(self.lang_metrics,self.save_plots+'metrics.pth')
        else:
          torch.save(self.lang_metrics,path+'{}.pth'.format(exp_name))


            



#@azhar
class tensorlog():
    def __init__(self,opt):
        self.writer = SummaryWriter(log_dir=f'{opt.exp_dir}/{opt.experiment_name}/',filename_suffix= opt.experiment_name)

    def record(self, lang, metric, step):

        for name, value in metric.items():
            self.writer.add_scalar(lang + '/' + name, value, step)


        '''self.writer.add_scalar(lang+'/train_loss',trainloss,step)
        self.writer.add_scalar(lang+'/Real_validation_loss',realvalloss,step)
        self.writer.add_scalar(lang+'/Syn_validation_loss', synvalloss,step)
        self.writer.add_scalar(lang+'/train_Wordaccuracy',trainacc,step)
        self.writer.add_scalar(lang+'/Real_val_Wordaccuracy',realvalaccuracy,step)
        self.writer.add_scalar(lang+'/Syn_val_Wordaccuracy',synvalaccuracy,step)
        self.writer.add_scalar(lang+'/Real_val_edit-dist',real_editdist,step)
        self.writer.add_scalar(lang+'/Syn_val_edit-dist',Syn_val_ED,step)'''

        '''for tag,value in model.named_parameters():
            tag = tag.replace('.', '/')
            #print(tag) to diagnose gradients
            self.writer.add_histogram('activation/'+tag,value.data.cpu().numpy(),step)
            if value.requires_grad==True:
                #print(tag,value.grad.data.cpu().numpy()) to diagnose gradients
                self.writer.add_histogram('gradients/'+tag,value.grad.data.cpu().numpy(),step)'''


        
#@azhar
class Scheduler():

    def __init__(self,sch_ele):
        self.queue = deque(sch_ele)
        #self.sche = sch
        #self.steps = 0

    def nextele(self):
        ele = self.queue.popleft()
        self.queue.append(ele)
        return ele

#@azhar
class LanguageData(object):
    def __init__(self, opt, lang, numiters, mode, task_id, useSyn=False,useReal=True):
        self.AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.lang = lang
        self.task_id = task_id
        self.numiters = int(numiters)
        self.useSyn = useSyn
        self.useReal = useReal
        self.num_classes = len(opt.character[lang])
        self.mode = mode
        #Setup Dataset and Loaders
        if useReal and useSyn:
            print('enter use Real and SYN')
            select = ['Real','Syn']
            batch_ratio = [0.5,0.5]  
        elif useReal:
            print('enter use Real ')
            select = ['Real']
            batch_ratio = [1.0]
        elif useSyn:
            select = ['Syn']
            batch_ratio = [1.0]

        val_string = opt.valid_data+'/val_'+lang
        train_string = opt.train_data+'/train_'+lang

        if not mode == 'val':
            self.train_dataset = Batch_Balanced_Dataset(opt, train_string, lang, batch_ratio=batch_ratio, select_data=select)
            
        if mode == 'val':
            if useSyn:
                Synvalid_dataset = hierarchical_dataset(lang, root= val_string+'/Syn', opt=opt)
                self.Synvalid_loader = self.genLoader(opt, Synvalid_dataset)
                
            train_dataset_eval = hierarchical_dataset(lang, root=train_string, opt=opt, select_data=select)
            self.Tvalid_loader = self.genLoader(opt, train_dataset_eval)
            Rvalid_dataset = hierarchical_dataset(lang, root=val_string+'/Real', opt=opt)
            self.Rvalid_loader = self.genLoader(opt, Rvalid_dataset)
        
        if mode == 'dev':
            if useSyn:
                if lang in ['arab','hin','ban']:
                  indices = list(range(2000))
                  Synvalid_dataset = hierarchical_dataset(lang, root= val_string+'/Syn', opt=opt)
                  Synvalid_subset = Subset(Synvalid_dataset,indices)
                  self.Synvalid_loader = self.genLoader(opt, Synvalid_subset)
                else:
                  Synvalid_dataset = hierarchical_dataset(lang, root= val_string+'/Syn', opt=opt)
                  self.Synvalid_loader = self.genLoader(opt, Synvalid_dataset)
            if useReal:
                Rvalid_dataset = hierarchical_dataset(lang, root=val_string+'/Real', opt=opt)
                self.Rvalid_loader = self.genLoader(opt, Rvalid_dataset)
            train_dataset_eval = hierarchical_dataset(lang, root=train_string, opt=opt, select_data=select)
            indices = list(range(int(1.0*len(train_dataset_eval))))
            if lang in ['arab','hin','ban']:
              indices = list(range(7701))
            train_subset = Subset(train_dataset_eval,indices)
            print(len(train_subset))
            self.Tvalid_loader = self.genLoader(opt, train_subset)
        
        if mode == 'test':
            if useSyn:
                Synvalid_dataset = hierarchical_dataset(lang, root= val_string+'/Syn', opt=opt)
                indices = list(range(int(0.2*len(Synvalid_dataset))))
                Synvalid_subset = Subset(Synvalid_dataset,indices)
                self.Synvalid_loader = self.genLoader(opt, Synvalid_subset)
            if useReal:
                Rvalid_dataset = hierarchical_dataset(lang, root=val_string+'/Real', opt=opt)
                indices = list(range(int(0.2*len(Rvalid_dataset))))
                Rvalid_subset = Subset(Rvalid_dataset,indices)
                self.Rvalid_loader = self.genLoader(opt, Rvalid_subset)
            train_dataset_eval = hierarchical_dataset(lang, root=train_string, opt=opt, select_data=select)
            indices = list(range(int(0.02*len(train_dataset_eval))))
            train_subset = Subset(train_dataset_eval,indices)
            self.Tvalid_loader = self.genLoader(opt, train_subset)
        
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

#@azhar
def get_vocab(vocab_file='characters.txt'):
    vocab_dict = {}
    f = open(vocab_file,'r')
    lines = f.readlines()
    for line in lines:#@azhar
        vocab_dict[line.split(',')[0]] = line.strip().split(',')[-1]

    return vocab_dict

class logger():
    def __init__(self,file_path):
        self.file_path = file_path

    def write(self,string):
        with open(file_path,'w+') as f:
            f.write(string)
