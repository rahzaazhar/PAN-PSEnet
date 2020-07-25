import os
import cv2
import sys
import time
import collections
import torch
import argparse
import numpy as np
#import torch.nn as nn
#import torch.nn.functional as F
import params
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.utils import data

from Detection.PSEnet import models as psenet_models
from Detection.PSEnet import util
from Detection.PSEnet.pypse import pse as pypse
from PIL import Image

from Recognition.utils import get_vocab, CTCLabelConverter
from Recognition.train_utils import setup_model
import params as M 


def scaleimg(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def crop(img,bbox):
    #img = cv2.imread(imgpath)
    bbox = bbox.reshape(4,2)
    topleft_x = np.min(bbox[:,0])
    topleft_y = np.min(bbox[:,1])
    bot_right_x = np.max(bbox[:,0])
    bot_right_y = np.max(bbox[:,1])
    cropped_img = img[topleft_y:bot_right_y, topleft_x:bot_right_x]
    cropped_img = cv2.resize(cropped_img,(100,32))
    cropped_img = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2GRAY)
    cropped_img = Image.fromarray(cropped_img)
    #cropped_img = cropped_img.convert('RGB')
    cropped_img = transforms.ToTensor()(cropped_img)
    return cropped_img

def drawBBox(bboxs,img):
    for bbox in bboxs:
        bbox = np.reshape(bbox,(4,2))
        cv2.drawContours(img, [bbox],-1, (0, 255, 0), 2)
    cv2.imwrite('result_new.jpg',img)


def get_recognition_model(e2e_config):
    e2e_config.recog_hp.character = get_vocab('./Recognition/characters.txt')
    model = setup_model(e2e_config.recog_hp)
    if e2e_config.recog_model_path is not None:
        model.load_state_dict(torch.load(e2e_config.recog_model_path))
    else:
            print("No checkpoint found at '{}'".format(e2e_config.recog_model_path))
            sys.stdout.flush()
    return model


def get_detection_model(e2e_config):
    if e2e_config.arch == "resnet50":
        model = psenet_models.resnet50(pretrained=False, num_classes=7, scale=e2e_config.scale)
    elif e2e_config.arch == "resnet101":
        model = psenet_models.resnet101(pretrained=False, num_classes=7, scale=e2e_config.scale)
    elif e2e_config.arch == "resnet152":
        model = psenet_models.resnet152(pretrained=False, num_classes=7, scale=e2e_config.scale)
    for param in model.parameters():
        param.requires_grad = False

    model = model.cuda()

    if e2e_config.det_model_path is not None:                                         
        if os.path.isfile(e2e_config.det_model_path):
            print("Loading model and optimizer from checkpoint '{}'".format(e2e_config.det_model_path))
            checkpoint = torch.load(e2e_config.det_model_path)
            
            # model.load_state_dict(checkpoint['state_dict'])
            d = collections.OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)

            print("Loaded checkpoint '{}' (epoch {})"
                  .format(e2e_config.det_model_path, checkpoint['epoch']))
            sys.stdout.flush()
        else:
            print("No checkpoint found at '{}'".format(e2e_config.det_model_path))
            sys.stdout.flush()

    model.eval()
    return model


def detect(e2e_config,model,org_img):
    s = time.time()
    scaled_img = scaleimg(org_img[:,:,[2,1,0]])
    scaled_img = Image.fromarray(scaled_img)
    scaled_img = scaled_img.convert('RGB')
    scaled_img = transforms.ToTensor()(scaled_img)
    scaled_img = transforms.Normalize(mean=[0.0618, 0.1206, 0.2677], std=[1.0214, 1.0212, 1.0242])(scaled_img)
    scaled_img = torch.unsqueeze(scaled_img,0)
    scaled_img = Variable(scaled_img.cuda())

    outputs = model(scaled_img)

    score = torch.sigmoid(outputs[:, 0, :, :])
    outputs = (torch.sign(outputs - e2e_config.binary_th) + 1) / 2

    text = outputs[:, 0, :, :]
    kernels = outputs[:, 0:e2e_config.kernel_num, :, :] * text

    score = score.data.cpu().numpy()[0].astype(np.float32)
    text = text.data.cpu().numpy()[0].astype(np.uint8)
    kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
    pred = pypse(kernels, e2e_config.min_kernel_area / (e2e_config.scale * e2e_config.scale))

    scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])
    label = pred
    label_num = np.max(label) + 1
    bboxes = []
    for i in range(1, label_num):
        points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

        if points.shape[0] < e2e_config.min_area / (e2e_config.scale * e2e_config.scale):
            continue

        score_i = np.mean(score[label == i])
        if score_i < e2e_config.min_score:
            continue

        rect = cv2.minAreaRect(points)
        bbox = cv2.boxPoints(rect) * scale
        bbox = bbox.astype('int32')
        bboxes.append(bbox.reshape(-1))
    drawBBox(bboxes,org_img)
    e = time.time()
    print('Detection Time taken:',e-s)
    return bboxes


def recognise(e2e_config,model,org_img,bboxes):
    model.eval()
    lang = e2e_config.lang
    converter = CTCLabelConverter(e2e_config.recog_hp.character[lang])
    print('PREDICTION:')
    for bbox in bboxes:
        cropped_img = crop(org_img,bbox)
        if torch.cuda.is_available():
            image = cropped_img.cuda()
        image = image.view(1, *image.size())
        #image = Variable(image)
        preds = model(image, None, lang).log_softmax(2)
        preds_size = torch.IntTensor([preds.size(1)] * 1)
        preds = preds.permute(1, 0, 2)
        # Select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)
        print(f'{preds_str[0]:20s}')


def main(e2e_config, image_path):
    print ('reading image..')
    image = cv2.imread(image_path)
    det_model = get_detection_model(e2e_config)
    recog_model = get_recognition_model(e2e_config)
    print ('detecting text')
    bboxes = detect(e2e_config, det_model, image)
    print ('recognizing text')
    recognise(e2e_config,recog_model,image,bboxes)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='image path')
    parser.add_argument('--img', nargs='?', type=str, default='demo/result.jpg',    
                        help='Path to test image')
    parser.add_argument('--e2e_config_name', type=str,
                        help = 'end to end config')
    args = parser.parse_args()
    e2e_config = getattr(M, args.e2e_config_name)
    image_path = args.img
    main(e2e_config, image_path)











