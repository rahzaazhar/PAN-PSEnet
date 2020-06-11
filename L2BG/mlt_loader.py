from dataclasses import dataclass, replace
from datasetv1 import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from utils import get_vocab
import torch
import random
from torch.utils.data import Subset
@dataclass
class langconfig():
	valid_data: str
	train_data: str
	character: str = None
	data_filtering_off: bool = False
	batch_max_length: int = 30
	rgb: bool = False
	sensitive: bool = False
	batch_size: int = 64
	workers: int = 0
	imgH:int = 32
	imgW:int = 100
	PAD:bool = False

opt = langconfig(valid_data='validation/',train_data='training/')
opt.character = get_vocab()


def get_MLT_loaders(tasks,batch_size):
	opt.batch_size = batch_size
	AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
	train_loaders = {}
	val_loaders = {}
	num_classes = {}
	for task in tasks:
		num_classes[task] = len(opt.character[task])
		val_string = opt.valid_data+'/val_'+task
		train_string = opt.train_data+'/train_'+task
		train_dataset = hierarchical_dataset(task,root=train_string+'/Syn',opt=opt)
		val_dataset = hierarchical_dataset(task, root= val_string+'/Syn', opt=opt)
		if task in ['hin','ban','arab']:
			total_train_data_points = len(train_dataset)
			total_val_data_points = len(val_dataset)
			train_indices = random.sample(range(0,total_train_data_points),1000)
			val_indices = random.sample(range(0,total_val_data_points),100)
			train_dataset = Subset(train_dataset,train_indices)
			val_dataset = Subset(val_dataset,val_indices)
		train_loaders[task] = torch.utils.data.DataLoader(
										train_dataset, batch_size=opt.batch_size,
										shuffle=True,  # 'True' to check training progress with validation function.
										num_workers=int(opt.workers),
										collate_fn=AlignCollate_valid, pin_memory=True)
		val_loaders[task] = torch.utils.data.DataLoader(
										val_dataset, batch_size=opt.batch_size,
										shuffle=True,  # 'True' to check training progress with validation function.
										num_workers=int(opt.workers),
										collate_fn=AlignCollate_valid, pin_memory=True)

	return train_loaders, val_loaders, num_classes







