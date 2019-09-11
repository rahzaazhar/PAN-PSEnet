from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.nn.functional import log_softmax
import numpy as np
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss
import os
import utils
import dataset_new

import models.crnn as crnn
import params

alpha='.:ँंःअआइईउऊऋएऐऑओऔकखगघचछजझञटठडढणतथदधनपफबभमयरलळवशषसहािीुूृॅेैॉोौ्ॐड़ढ़०१२३४५६७८९\u200c\u200d()'
parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', required=True, help='path to train dataset')
parser.add_argument('--valroot', required=True, help='path to val dataset')
args = parser.parse_args()
'/home/azhar/crnn-pytorch/data/train_gt.txt'
'/home/azhar/crnn-pytorch/data/validate_gt.txt'
if not os.path.exists(params.expr_dir):
    os.makedirs(params.expr_dir)

random.seed(params.manualSeed)
np.random.seed(params.manualSeed)
torch.manual_seed(params.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not params.cuda:
    print("WARNING: You have a CUDA device, so you should probably set cuda in params.py to True")

# -------------------------------------------------------------------------------------------------
# dealwith train and test data
train_dataset = dataset_new.RecogDataset(data_file=args.trainroot)
assert train_dataset
if not params.random_sample:
    sampler = dataset_new.randomSequentialSampler(train_dataset, params.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batchSize, 
        shuffle=True, sampler=sampler, num_workers=int(params.workers))
test_dataset = dataset_new.RecogDataset(data_file=args.valroot)
#train_dataset1 = dataset.lmdbDataset(root=args.trainroot, transform=dataset.resizeNormalize((params.imgW, params.imgH)))

# -------------------------------------------------------------------------------------------------
# net init
# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

nclass = len(alpha) + 1
print(nclass)
crnn = crnn.CRNN(params.imgH, params.nc, nclass, params.nh)
crnn.apply(weights_init)
if params.pretrained != '':
    print('loading pretrained model from %s' % params.pretrained)
    if params.multi_gpu:
        crnn = torch.nn.DataParallel(crnn)
    crnn.load_state_dict(torch.load(params.pretrained))
print(crnn)

# -------------------------------------------------------------------------------------------------
converter = utils.strLabelConverter(alpha)
criterion = CTCLoss()

image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
text = torch.IntTensor(params.batchSize * 5)
length = torch.IntTensor(params.batchSize)
if params.cuda and torch.cuda.is_available():
    crnn.cuda()
    if params.multi_gpu:
        crnn = torch.nn.DataParallel(crnn, device_ids=range(params.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()
image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if params.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
elif params.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)


def val(net, dataset, criterion, max_iter=100):
    #print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=params.batchSize, num_workers=int(params.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length)
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        #cpu_texts_decode = []
        #for i in cpu_texts:
        #    cpu_texts_decode.append(i.decode('utf-8', 'strict'))
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target:
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * params.batchSize)
    print('loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    return loss_avg.val(), accuracy



def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    
    optimizer.zero_grad()
    preds = crnn(image)
    preds = preds.log_softmax(2)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length)
    # crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


if __name__ == "__main__":
    with open(params.expr_dir+"/log.txt","w") as f:  
        prevvalloss = -1
        for epoch in range(params.nepoch):
            train_iter = iter(train_loader)
            i = 0
            while i < len(train_loader):
                for p in crnn.parameters():
                    p.requires_grad = True
                crnn.train()

                cost = trainBatch(crnn, criterion, optimizer)
                loss_avg.add(cost)
                i += 1

                if i % params.displayInterval == 0:
                    print('[%d/%d][%d/%d] Loss: %f' %
                          (epoch+1, params.nepoch, i, len(train_loader), loss_avg.val()))
                    loss_avg.reset()
            print("end of epoch [%d/%d]" %(epoch+1,params.nepoch))
            print("start testing on val set")
            valloss, valaccuracy = val(crnn, test_dataset, criterion)
            print("start testing on train set to check for overfitting")
            #trainloss, trainaccuracy = val(crnn, train_dataset, criterion)
            line = str(valloss)+"\t"+str(valaccuracy)+"\n"#"\t"+str(trainloss)+"\t"+str(trainaccuracy)+"\n"
            print(line)
            f.write(line)
            f.flush()
            if valloss<prevvalloss or prevvalloss==-1:
                # do checkpointing
                torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(params.expr_dir, epoch,valloss))
                prevvalloss = valloss
