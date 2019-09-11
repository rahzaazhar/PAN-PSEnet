# dataloader add 3.0 scale
# dataloader add filer text
import numpy as np
from PIL import Image
from torch.utils import data
import util
import cv2
import random
import torchvision.transforms as transforms
import torch

'''ic19_root_dir = '/content/drive/My Drive/IC19MLT_root/' 
ic19_train_data_dir = ic19_root_dir + 'train_IC19MLT/'
ic19_train_gt_dir = ic19_root_dir + 'train_IC19MLT_gt/'''

ic19_root_dir = '/home/azhar/summerResearch/data/IC19MLT_root/' 
ic19_train_data_dir = ic19_root_dir + 'train_IC19MLT/'
ic19_test_data_dir = ic19_root_dir + 'test_IC19MLT/'


random.seed(123456)

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print (img_path)
        raise
    return img

def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

class IC19TestLoader(data.Dataset):
    def __init__(self, part_id=0, part_num=1, long_size=2240,indic=False,data_dirs=None):

        if(indic==True):
            data_dirs = [ic19_train_data_dir]
        else:
            data_dirs = [data_dirs]

        self.img_paths = []
        
        for data_dir in data_dirs:
            img_names = util.io_.ls(data_dir, '.jpg')
            img_names.extend(util.io_.ls(data_dir, '.png'))
            img_names.extend(util.io_.ls(data_dir, '.jpeg'))

            img_paths = []
            if(indic==True):
                for idx, img_name in enumerate(img_names):
                    k=int(img_name.split('.')[0].split('_')[-1])
                    if(8001<=k<=10000):
                        img_path = data_dir + img_name
                        img_paths.append(img_path)
            else:
                for idx, img_name in enumerate(img_names):
                    img_path = data_dir + img_name
                    img_paths.append(img_path)
            self.img_paths.extend(img_paths)

        part_size = len(self.img_paths) / part_num
        l = int(part_id * part_size)
        r = int((part_id + 1) * part_size)
        #self.img_paths = self.img_paths[l:r]
        self.long_size = long_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path)

        scaled_img = scale(img, self.long_size)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        #scaled_img = transforms.Normalize(mean=[0.0618, 0.1206, 0.2677], std=[1.0214, 1.0212, 1.0242])(scaled_img)
        scaled_img = transforms.Normalize(mean=[126.1446, 126.0139, 125.8503], std=[11.1815, 11.0344, 11.1635])(scaled_img)
        return  img[:, :, [2, 1, 0]], scaled_img 
