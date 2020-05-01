import torch
import torch.nn as nn
from modelv1 import GradCL
from collections import OrderedDict
import copy

def test1():
	template = {'input_layer':nn.Linear(32*32,200),'relu1':nn.ReLU(),'layer1':nn.Linear(200,200),'relu2':nn.ReLU(),
				'output_layer':nn.Linear(200,10),'relu2':nn.Softmax()}
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

test1()
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
