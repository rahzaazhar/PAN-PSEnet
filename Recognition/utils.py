import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
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
        print(self.dict)

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

class tensorlog():

    def __init__(self,dirr,inc):
        self.writer = SummaryWriter(log_dir=dirr)
        self.step = inc
        self.inc = inc

    def record(self,model,lang,trainloss,realvalloss,synvalloss,trainacc,realvalaccuracy,synvalaccuracy,real_editdist):

        self.writer.add_scalar(lang+'/train_loss',trainloss,self.step)
        self.writer.add_scalar(lang+'/Real_validation_loss',realvalloss,self.step)
        self.writer.add_scalar(lang+'/Syn_validation_loss', synvalloss,self.step)
        self.writer.add_scalar(lang+'/train_Wordaccuracy',trainacc,self.step)
        self.writer.add_scalar(lang+'/Real_val_Wordaccuracy',realvalaccuracy,self.step)
        self.writer.add_scalar(lang+'/Syn_val_Wordaccuracy',synvalaccuracy,self.step)
        self.writer.add_scalar(lang+'/Real_val_edit-dist',real_editdist,self.step)

        for tag,value in model.named_parameters():
            tag = tag.replace('.', '/')
            #print(tag) to diagnose gradients
            self.writer.add_histogram('activation/'+tag,value.data.cpu().numpy(),self.step)
            if value.requires_grad==True:
                #print(tag,value.grad.data.cpu().numpy()) to diagnose gradients
                self.writer.add_histogram('gradients/'+tag,value.grad.data.cpu().numpy(),self.step)


        self.step = self.step+self.inc

class Scheduler():

    def __init__(self,sch_ele):
        self.queue = deque(sch_ele)
        #self.sche = sch
        #self.steps = 0

    def nextele(self):
        ele = self.queue.popleft()
        self.queue.append(ele)
        return ele

        



