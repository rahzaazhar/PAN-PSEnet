import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import shutil

from torch.autograd import Variable
from torch.utils import data
import os

from dataset import IC19Loader
from util.misc import get_mean_and_std

data_loader = IC19Loader(is_transform=False, img_size=640, kernel_num=7, min_scale=0.4)
mean, std = get_mean_and_std(data_loader)
print(mean,std)