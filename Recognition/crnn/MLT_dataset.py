#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import cv2
import sys
import numpy as np
from PIL import Image


def get_img(img_path):
	try:
		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#img = img[:, :, [2, 1, 0]]
	except Exception as e:
		print (img_path)
		raise
	return img

class TrainDataset(Dataset):
	def __init__(self,data_file,imgW=100,imgH=32,lang='eng'):
		self.img_paths = []
		self.labels = []
		self.imgW = 100
		self.imgH = 32
		self.active_paths = []
		with open(data_file,'r') as f:
			lines = f.readlines()
			for line in lines:
				path, label = line.split(',')
				label = label.strip()
				self.img_paths.append(path)
				self.labels.append(label)
		self.active_paths = self.img_paths[0:6729]
		self.active_labels = self.labels[0:6729]

	def __len__(self):
		return len(self.active_paths)

	def __getitem__(self,index):
		img_path = self.active_paths[index]
		label = self.active_labels[index]
		label = label.strip()
		img = get_img(img_path)
		img = cv2.resize(img,dsize=(self.imgW,self.imgH))
		#img = img[:,:,[2,1,0]]
		img = Image.fromarray(img)
		#img = img.convert('RGB')
		img = transforms.ToTensor()(img)
		return img/255,label
	
	def setlang(self,lang):
		if(lang=='eng'):
			self.active_paths = self.img_paths[0:47081]
			self.active_labels = self.labels[0:47081]
		if(lang=='bangla'):
			self.active_paths = self.img_paths[47081:50229]
			self.active_labels = self.labels[47081:50229]
		if(lang=='hin'):
			self.active_paths = self.img_paths[50229:53373]
			self.active_labels = self.labels[50229:53373]

class ValDataset(Dataset):
	def __init__(self,data_file,imgW=100,imgH=32,lang='eng'):
		self.img_paths = []
		self.labels = []
		self.imgW = 100
		self.imgH = 32
		self.active_paths = []
		with open(data_file,'r') as f:
			lines = f.readlines()
			for line in lines:
				path, label = line.split(',')
				label = label.strip()
				self.img_paths.append(path)
				self.labels.append(label)
		self.active_paths = self.img_paths[0:1170]
		self.active_labels = self.labels[0:1170]

	def __len__(self):
		return len(self.active_paths)

	def __getitem__(self,index):
		img_path = self.active_paths[index]
		label = self.active_labels[index]
		label = label.strip()
		img = get_img(img_path)
		img = cv2.resize(img,dsize=(self.imgW,self.imgH))
		#img = img[:,:,[2,1,0]]
		img = Image.fromarray(img)
		#img = img.convert('RGB')
		img = transforms.ToTensor()(img)
		return img/255,label
	
	def setlang(self,lang):
		if(lang=='eng'):
			self.active_paths = self.img_paths[0:11770]
			self.active_labels = self.labels[0:11770]
		if(lang=='bangla'):
			self.active_paths = self.img_paths[11770:12557]
			self.active_labels = self.labels[11770:12557]
		if(lang=='hin'):
			self.active_paths = self.img_paths[12557:13344]
			self.active_labels = self.labels[12557:13344]

class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples
				

#loader = RecogDataset(data_file='/home/azhar/crnn-pytorch/data/train_gt.txt')
