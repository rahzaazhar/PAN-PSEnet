''' Adapted from https://github.com/whai362/PSENet '''
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import shutil
import tensorflow as tf
#from tensorflow import summary 
import datetime
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
import os
#import tensorboardX
#from tensorboardX import SummaryWriter
from dataset import IC19Loader
from metrics import runningScore
import models
from util.logger import Logger
from util.tflogger import tfLogger 
from util.misc import AverageMeter
import time
import util

globalcounter=1
def ohem_single(score, gt_text, training_mask):
    pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))
    
    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask
    
    neg_num = (int)(np.sum(gt_text <= 0.5))
    neg_num = (int)(min(pos_num * 3, neg_num))
    
    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    neg_score_sorted = np.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
    return selected_mask

def ohem_batch(scores, gt_texts, training_masks):
    scores = scores.data.cpu().numpy()
    gt_texts = gt_texts.data.cpu().numpy()
    training_masks = training_masks.data.cpu().numpy()

    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = np.concatenate(selected_masks, 0)
    selected_masks = torch.from_numpy(selected_masks).float()

    return selected_masks

def dice_loss(input, target, mask):
    input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)
    mask = mask.contiguous().view(mask.size()[0], -1)
    
    input = input * mask
    target = target * mask

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss

def cal_text_score(texts, gt_texts, training_masks, running_metric_text):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = torch.sigmoid(texts).data.cpu().numpy() * training_masks
    pred_text[pred_text <= 0.5] = 0
    pred_text[pred_text >  0.5] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text

def cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel):
    mask = (gt_texts * training_masks).data.cpu().numpy()
    kernel = kernels[:, -1, :, :]
    gt_kernel = gt_kernels[:, -1, :, :]
    pred_kernel = torch.sigmoid(kernel).data.cpu().numpy()
    pred_kernel[pred_kernel <= 0.5] = 0
    pred_kernel[pred_kernel >  0.5] = 1
    pred_kernel = (pred_kernel * mask).astype(np.int32)
    gt_kernel = gt_kernel.data.cpu().numpy()
    gt_kernel = (gt_kernel * mask).astype(np.int32)
    running_metric_kernel.update(gt_kernel, pred_kernel)
    score_kernel, _ = running_metric_kernel.get_scores()
    return score_kernel

def validate(val_loader,model,criterion):
    with torch.no_grad():    
        model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        running_metric_text = runningScore(2)
        running_metric_kernel = runningScore(2)

        end = time.time()
        for batch_idx, (imgs, gt_texts, gt_kernels, training_masks) in enumerate(val_loader):
            data_time.update(time.time() - end)

            imgs = Variable(imgs.cuda())
            gt_texts = Variable(gt_texts.cuda())
            gt_kernels = Variable(gt_kernels.cuda())
            training_masks = Variable(training_masks.cuda())

            outputs = model(imgs)
            texts = outputs[:, 0, :, :]
            kernels = outputs[:, 1:, :, :]

            selected_masks = ohem_batch(texts, gt_texts, training_masks)
            selected_masks = Variable(selected_masks.cuda())

            loss_text = criterion(texts, gt_texts, selected_masks)
        
            loss_kernels = []
            mask0 = torch.sigmoid(texts).data.cpu().numpy()
            mask1 = training_masks.data.cpu().numpy()
            selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
            selected_masks = torch.from_numpy(selected_masks).float()
            selected_masks = Variable(selected_masks.cuda())
            for i in range(6):
                kernel_i = kernels[:, i, :, :]
                gt_kernel_i = gt_kernels[:, i, :, :]
                loss_kernel_i = criterion(kernel_i, gt_kernel_i, selected_masks)
                loss_kernels.append(loss_kernel_i)
            loss_kernel = sum(loss_kernels) / len(loss_kernels)
        
            loss = 0.7 * loss_text + 0.3 * loss_kernel
            losses.update(loss.item(), imgs.size(0))

            score_text = cal_text_score(texts, gt_texts, training_masks, running_metric_text)
            score_kernel = cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel)

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 5 == 0:
                output_log  = '({batch}/{size}) Batch: {bt:.3f}s | TOTAL: {total:.0f}min | ETA: {eta:.0f}min '.format(batch=batch_idx + 1,
                    size=len(val_loader),
                    bt=batch_time.avg,
                    total=batch_time.avg * batch_idx / 60.0,
                    eta=batch_time.avg * (len(val_loader) - batch_idx) / 60.0)
                print(output_log)
                sys.stdout.flush()

    return (float(losses.avg), float(score_text['Mean Acc']), float(score_kernel['Mean Acc']), float(score_text['Mean IoU']), float(score_kernel['Mean IoU']))


def train(train_loader, model, criterion, optimizer, epoch, tflogger):
    model.train()
    #taglist = ['module.conv1.weight','module.bn1.weight','module.bn1.bias','module.conv2.weight','module.conv2.bias','module.bn2.weight','module.bn2.bias','module.conv3.weight','module.conv3.bias','module.bn3.weight','module.bn3.bia','module.conv4.weight','module.conv4.bias']
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    running_metric_text = runningScore(2)
    running_metric_kernel = runningScore(2)
    global globalcounter

    end = time.time()
    for batch_idx, (imgs, gt_texts, gt_kernels, training_masks) in enumerate(train_loader):
        data_time.update(time.time() - end)

        imgs = Variable(imgs.cuda())
        gt_texts = Variable(gt_texts.cuda())
        gt_kernels = Variable(gt_kernels.cuda())
        training_masks = Variable(training_masks.cuda())

        outputs = model(imgs)
        texts = outputs[:, 0, :, :]
        kernels = outputs[:, 1:, :, :]

        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        selected_masks = Variable(selected_masks.cuda())

        loss_text = criterion(texts, gt_texts, selected_masks)
        
        loss_kernels = []
        mask0 = torch.sigmoid(texts).data.cpu().numpy()
        mask1 = training_masks.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float()
        selected_masks = Variable(selected_masks.cuda())
        for i in range(6):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = criterion(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernel = sum(loss_kernels) / len(loss_kernels)
        
        loss = 0.7 * loss_text + 0.3 * loss_kernel
        losses.update(loss.item(), imgs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        score_text = cal_text_score(texts, gt_texts, training_masks, running_metric_text)
        score_kernel = cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel)

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 20 == 0:
            output_log  = '({batch}/{size}) Batch: {bt:.3f}s | TOTAL: {total:.0f}min | ETA: {eta:.0f}min | Loss: {loss:.4f} | Acc_t: {acc: .4f} | IOU_t: {iou_t: .4f} | IOU_k: {iou_k: .4f}'.format(
                batch=batch_idx + 1,
                size=len(train_loader),
                bt=batch_time.avg,
                total=batch_time.avg * batch_idx / 60.0,
                eta=batch_time.avg * (len(train_loader) - batch_idx) / 60.0,
                loss=losses.avg,
                acc=score_text['Mean Acc'],
                iou_t=score_text['Mean IoU'],
                iou_k=score_kernel['Mean IoU'])
            print(output_log)
            sys.stdout.flush()
            
        if batch_idx %100 == 0:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                tflogger.histo_summary(tag, value.data.detach().cpu().numpy(), globalcounter)
                tflogger.histo_summary(tag+'/grad', value.grad.data.detach().cpu().numpy(), globalcounter)
            globalcounter += 1


    return (float(losses.avg), float(score_text['Mean Acc']), float(score_kernel['Mean Acc']), float(score_text['Mean IoU']), float(score_kernel['Mean IoU']))

def adjust_learning_rate(args, optimizer, epoch):
    global state
    if epoch in args.schedule:
        args.lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def main(args):

    #initial setup
    if args.checkpoint == '':
        args.checkpoint = "checkpoints1/ic19val_%s_bs_%d_ep_%d"%(args.arch, args.batch_size, args.n_epoch)
    if args.pretrain:
        if 'synth' in args.pretrain:
            args.checkpoint += "_pretrain_synth"
        else:
            args.checkpoint += "_pretrain_ic17"

    print ('checkpoint path: %s'%args.checkpoint)
    print ('init lr: %.8f'%args.lr)
    print ('schedule: ', args.schedule)
    sys.stdout.flush()

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    kernel_num = 7
    min_scale = 0.4
    start_epoch = 0
    validation_split = 0.1
    random_seed = 42
    prev_val_loss = -1
    val_loss_list = []
    loggertf = tfLogger('./log/'+args.arch)
    #end

    #setup data loaders
    data_loader = IC19Loader(is_transform=True, img_size=args.img_size, kernel_num=kernel_num, min_scale=min_scale)
    dataset_size = len(data_loader)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_incidies, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_incidies)
    validate_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=args.batch_size,
        num_workers=3,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler)

    validate_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=args.batch_size,
        num_workers=3,
        drop_last=True,
        pin_memory=True,
        sampler=validate_sampler)
    #end

    #Setup architecture and optimizer
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=kernel_num)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=kernel_num)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=kernel_num)
    elif args.arch == "resPAnet50":
        model = models.resPAnet50(pretrained=True, num_classes=kernel_num)
    elif args.arch == "resPAnet101":
        model = models.resPAnet101(pretrained=True, num_classes=kernel_num)
    elif args.arch == "resPAnet152":
        model = models.resPAnet152(pretrained=True, num_classes=kernel_num)

    model = torch.nn.DataParallel(model).cuda()
    
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    #end

    #options to resume/use pretrained model/train from scratch
    title = 'icdar2019MLT'
    if args.pretrain:
        print('Using pretrained model.')
        assert os.path.isfile(args.pretrain), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['state_dict'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss','Train Acc.', 'Train IOU.','Validate Loss','Validate Acc','Validate IOU'])
    elif args.resume:
        print('Resuming from checkpoint.')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        print('Training from scratch.')
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss','Train Acc.', 'Train IOU.','Validate Loss','Validate Acc','Validate IOU'])
    #end

    #start training model
    for epoch in range(start_epoch, args.n_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.n_epoch, optimizer.param_groups[0]['lr']))

        train_loss, train_te_acc, train_ke_acc, train_te_iou, train_ke_iou = train(train_loader, model, dice_loss, optimizer, epoch, loggertf)
        val_loss, val_te_acc, val_ke_acc, val_te_iou, val_ke_iou = validate(validate_loader, model, dice_loss)
        
        #logging on tensorboard
        loggertf.scalar_summary('Training/Accuracy', train_te_acc, epoch+1)
        loggertf.scalar_summary('Training/Loss', train_loss, epoch+1)
        loggertf.scalar_summary('Training/IoU', train_te_iou, epoch+1)
        loggertf.scalar_summary('Validation/Accuracy', val_te_acc, epoch+1)
        loggertf.scalar_summary('Validation/Loss', val_loss, epoch+1)
        loggertf.scalar_summary('Validation/IoU', val_te_iou, epoch+1)
        #end

        #Boring Book Keeping 
        print("End of Epoch %d",epoch+1)
        print("Train Loss: {loss:.4f} | Train Acc: {acc: .4f} | Train IOU: {iou_t: .4f}".format(loss=train_loss,acc=train_te_acc,iou_t=train_te_iou))
        print("Validation Loss: {loss:.4f} | Validation Acc: {acc: .4f} | Validation IOU: {iou_t: .4f}".format(loss=val_loss,acc=val_te_acc,iou_t=val_te_iou))
        #end

        #Saving improving and Best Models 
        val_loss_list.append(val_loss)
        if(val_loss<prev_val_loss or prev_val_loss==-1):
            checkpointname = "{loss:.3f}".format(loss=val_loss)+"_epoch"+str(epoch+1)+"_checkpoint.pth.tar"
            save_checkpoint({
                     'epoch': epoch + 1,
                     'state_dict': model.state_dict(),
                     'lr': args.lr,
                     'optimizer' : optimizer.state_dict(),
                     }, checkpoint=args.checkpoint,filename=checkpointname)
        if(val_loss<min(val_loss_list)):
            checkpointname = "best_checkpoint.pth.tar"
            save_checkpoint({
                     'epoch': epoch + 1,
                     'state_dict': model.state_dict(),
                     'lr': args.lr,
                     'optimizer' : optimizer.state_dict(),
                     }, checkpoint=args.checkpoint,filename=checkpointname)
        #end
        prev_val_loss = val_loss
        logger.append([optimizer.param_groups[0]['lr'], train_loss, train_te_acc, train_te_iou, val_loss,val_te_acc, val_te_iou])
    #end traing model
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--img_size', nargs='?', type=int, default=640, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=600, 
                        help='# of the epochs')
    parser.add_argument('--schedule', type=int, nargs='+', default=[200, 400],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16, 
                        help='Batch Size')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--pretrain', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
    args = parser.parse_args()

    main(args)
