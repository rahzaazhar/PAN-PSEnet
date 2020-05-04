import torch
import torch.nn as nn
import train_pmnist
from modelv1 import GradCL
from collections import OrderedDict, deque
from train_pmnist import train_single_task, train_task_pair
from pmnist_dataset import get_Pmnist_tasks
import torch.optim as optim 
import copy
template = {'linear1_input':nn.Linear(32*32,200),'relu1':nn.ReLU(),'linear2':nn.Linear(200,200),'relu2':nn.ReLU(),
				'linear3_output':nn.Linear(200,10),'softmax':nn.Softmax(dim=0)}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1111)
def weight_innit(model):
	# weight initialization
	for name, param in model.named_parameters():
		if 'localization_fc2' in name:
			print(f'Skip {name} as it is already initialized')
			continue
		try:
			if 'bias' in name:
				init.constant_(param, 0.0)
			elif 'weight' in name:
				init.kaiming_normal_(param)
				#init.constant_(param, 0.0)
		except Exception as e:  # for batchnorm.
			if 'weight' in name:
				param.data.fill_(1)
			continue
	return model

def test1():
	model = GradCL(template)
	model.init_subgraph('task1')
	print('model after adding task1')
	print(model)
	model.init_subgraph('task2')
	model.init_subgraph('task3')
	model.add_node('task2','layer1')
	print('model after adding task3,task2')
	print(model)
	print('printing model parameters')
	for name, parameters in model.named_parameters():
		print(name)
	model.point_to_node('task2','task1','layer1')
	print('change pointer of layer1 of task2 to layer1 of task1')
	for name, parameters in model.named_parameters():
		print(name)

def test2():
	n_tasks = 3
	batch_size = 32
	train_loaders, val_loaders = get_Pmnist_tasks(n_tasks,batch_size)
	task_names = list(train_loaders.keys())
	model = GradCL(template,0.5)
	#model.to(device)
	criterion = nn.CrossEntropyLoss()
	
	for idx, task_name in enumerate(task_names):
		model.init_subgraph(task_name)
		if idx == 0:
			optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
			#weight_innit(model)
			model.to(device)
			train_single_task(model,criterion,optimizer,train_loaders['task_0'],val_loaders['task_0'],task_name)
		else:
			model.init_subgraph(task_name)
			model.add_node(task_name,'layer1')
			print('Weights after adding node to:',task_name)
			for name,_ in model.named_parameters():
				print(name)
			optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
			#weight_innit(model)
			model.to(device)
			train_single_task(model,criterion,optimizer,train_loaders['task_0'],val_loaders['task_0'],task_name)

def test3():
	#test to check pairwise training and gradient collection
	n_tasks = 2
	batch_size = 32
	train_loaders, val_loaders = get_Pmnist_tasks(n_tasks,batch_size)
	task_names = list(train_loaders.keys())
	model = GradCL(template,0.5)
	model.init_subgraph(task_names[0])
	model.init_subgraph(task_names[1])
	for name,_ in model.named_parameters():
		print(name)
	model.to(device)
	sims,x,metrics = train_task_pair(model,task_names[1],task_names[0],train_loaders,val_loaders)

	dump = OrderedDict()
	dump = {'sims':sims,'x':x,'metrics':metrics}
	torch.save(dump,'grads_collect_test.pth')
	

test2()
#test3()







#test1()
#torch.manual_seed(1111)
#template = OrderedDict()
#template = {'input_layer':nn.Linear(32*32,200),'relu1':nn.ReLU(),'output_layer':nn.Linear(200,10),'relu2':nn.ReLU()}
#template = [('input_layer',nn.Linear(32*32,200)),('relu1',nn.ReLU()),('output_layer',nn.Linear(200,10)),('relu2',nn.ReLU())]
#tasks = ['task0','task1','task3']
#model = GradCL(template)
#model.init_subgraph('task0')
#print(model)
#model.init_subgraph('task1')
#print(model)
#for name, para in model.named_parameters():
#	print(name)
#x = torch.randn(1,32*32)
#out = model(x,'task1')
#print(out)
#print(list(model.super_network['input_layer']['task1'].named_parameters()))
#model.add_node('task1','input_layer')
#print(model.super_network['output_layer'])
#for name, para in model.named_parameters():
#	print(name)
#torch.save(model.super_network,'super_graph.pth')
#sp = torch.load('super_graph.pth')
#model.super_network = sp
#print(list(model.super_network['input_layer']['task1'].named_parameters()))
#out = model(x,'task1')
#print(out)
'''
out = model(x,'task1')
print('after adding node')
model.add_node('task1','output_layer')
for key in model.super_network.keys():
	print(list(model.super_network[key]['task1'].named_parameters())[0],'->', end="")
'''
