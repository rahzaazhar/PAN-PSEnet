import torch
from modelv1 import GradCL
import pmnist_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import copy
from RSA import RSA
from collections import OrderedDict, deque
from torch.utils.data import Subset
from pmnist_dataset import get_Pmnist_tasks
from plot import plot_grad_sim, gradient_similarity, multiplot, average_grad, dot_product, set_zero
from plot import compute_per_layer_pearsonc, norm2
from utils import Averager
import plot
import random
import numpy as np
random.seed(1111)
torch.manual_seed(1111)
np.random.seed(1111)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def get_layer_names(p1):
	layer_set = set()
	for key in p1.keys():
		key = key.split('.')[0]
		layer_set.add(key)

	return list(layer_set)

def compute_stats(gradsims):
	gs = np.array(gradsims)
	mean = np.mean(gs)
	var = np.var(gs)
	max_val = np.max(gs)
	min_val = np.min(gs)
	max_diff = max_val-min_val
	return {'mean':mean,'variance':var,'max_val':max_val,'min_val':min_val,'max_diff':max_diff}

def some_metric(gradsims):
	pairwise_weight_score = {}
	for name, layer_gradsims in gradsims.items():
		pairwise_weight_score[name] = compute_something(layer_gradsims) 

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

def test(model,criterion,test_loader,task):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('testing task:',task)
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data in test_loader:
			#print(x.size())
			bs = data[0].size(0)
			x = data[0]
			y = data[1]
			x = x.view(bs,32*32*1)
			x = x.to(device)
			#print(type(x))
			y = y.to(device)
			#print(type(y))
			output = model(x,task)
			test_loss += criterion(output, y).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(y.data.view_as(pred)).sum()
		test_loss /= len(test_loader.dataset)
		#test_losses.append(test_loss)
	print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
			test_loss, correct, len(test_loader.dataset),
			100. * correct / len(test_loader.dataset)))
	return float(((100.*correct)/len(test_loader.dataset))),test_loss

def train_single_task(model,criterion,optimizer,trainloader,valloader,current_task,epochs=1):
	#epochs = 10
	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	for epoch in range(epochs):
		for batch_idx, (x, y) in enumerate(trainloader):
			#x, y = next(iter(trainloader[current_task]))
			x = x.view(32,32*32*1)
			x = x.to(device)
			y = y.to(device)
			output = model(x,current_task)
			loss = criterion(output,y)
			model.zero_grad()
			loss.backward()
			optimizer.step()
			if batch_idx % 200 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(x), len(trainloader.dataset),
				100. * batch_idx / len(trainloader), loss.item()))
		_, _ = test(model,criterion,valloader,current_task)

### task_sim_scores dict {task_<name>:sim_score with new task} ###
### task_layerwise_sims dict {task_<name>:{layer_<name>:layer_sim_score}}
def learn_to_grow(model,criterion,trainloaders,valloaders,task_names,epochs,similarity='pearson'):
	#task_groups = {'group1':[]}
	tasks = []
	avg_acc = []
	task_counter = []
	start_time = 0
	end_time = 0
	for task_id, new_task in enumerate(task_names):
		model.init_subgraph(new_task)
		if task_id == 0:
			optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
			model.to(device)
			start_time = time.time()
			train_single_task(model,criterion,optimizer,trainloaders[current_task],valloaders[current_task],current_task,epochs)
			end_time = time.time()
			train_time = round(end_time-start_time,2)
			print("training time:",train_time)
			#task_groups['group1'].append(current_task)
			tasks.append(current_task)
			continue
		#group_sim_scores = {}
		task_sim_scores = {}
		task_layerwise_sims = {}
		p_1 = {}
		p_2 = {}
		if similarity == 'pearson':
			for task in tasks:
				task_sim_scores[task], task_layerwise_sims[task] = get_task_sim_score(model,current_task,task,trainloaders,valloaders)
		
		if similarity == 'RSA':
			task_layerwise_scores, task_sim_scores = RSA_Sim(model,new_task,tasks,train_loaders,val_loaders)

		selected_tasks = find_similar_tasks(task_sim_scores,n=1)#returns list containing k similar tasks

		'''if create_new_group:
			task_groups[group] = [current_task]
			model.clone(current_task)
			train_single_task(model,criterion,optimizer,trainloaders,current_task)
			tasks.append(current_task)
			continue
			task_groups[group].append(current_task)
			model.grow(group_sim_scores[group])
		'''
		model.grow_graph(current_task,selected_tasks,task_layerwise_sims)
		print('Model after adding ',current_task)
		for name,_ in model.named_parameters():
			print(name)
		#account for new parameters added
		optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
		train_single_task(model,criterion,optimizer,trainloaders[current_task],valloaders[current_task],current_task)
		tasks.append(current_task)
		print('Testing on previous tasks after training task:',current_task)
		avg_accuracy = 0
		for task in tasks:
			avg_acc,_ = test(model,criterion,valloaders[task],task)
			avg_accuracy += avg_acc
		print('average Accuracy:',avg_accuracy/len(tasks))


def get_task_sim_score(model,new_task,task,trainloaders,valloaders):
	grad_sims, x, metrics = train_task_pair(model,new_task,task,trainloaders,valloaders)#used to collect gradients and computing pearson
	acc1 = metrics[new_task]['val_loss']
	acc2 = metrics[task]['val_loss']
	p_1, p_2 = compute_per_layer_pearsonc(grad_sims,acc1,acc2)
	task_score = norm2(list(p_1.values()),list(p_2.values()))
	print('##################Similarity Metric Scores between ',new_task,'and ',task,' ##################')
	print(task_score)
	print(p_1)
	print(p_2)

	pairwise_weight_score = {}
	norm_const = 0
	for key in p_1.keys():
		res = np.sqrt((p_1[key]-p_2[key])**2)
		norm_const += res
		pairwise_weight_score[key] = res
  
	for key in p1.keys():
		pairwise_weight_score[key] = pairwise_weight_score[key]/norm_const
  
	pairwise_layer_score = {}
	layer_names = get_layer_names(p_1)
	for layer in layer_names:
		pairwise_layer_score[layer] = pairwise_weight_score[layer+'.weight'] + pairwise_weight_score[layer+'.bias']

	print(pairwise_layer_score)
	print('#############################################################################')
	return task_score, pairwise_layer_score, p_1, p_2

def score_to_distribution(layer_scores):
	norm_const = sum(list(layer_scores.values()))
	for key in layer_scores.keys():
		layer_scores[key] = layer_scores[key]/norm_const
	return layer_scores



def train_task_pair(model,new_task,task,trainloaders,valloaders):
	weights = copy.deepcopy(model.state_dict())
	model = weight_innit(model)
	print('entered train task pair')
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	start_collect_iter = 0
	valInterval = 100
	avg_steps = 50
	num_iter = 1000
	loss_avg = Averager()
	taskQ = deque([new_task,task])
	model.change_subgraph_pointer(new_task,task)
	metrics = {}
	grads_list = {}
	grads_collect = {}
	sims = {}
	iters = []
	for task in taskQ:
		metrics[task]={}
		metrics[task]['val_acc'] = []
		metrics[task]['val_loss'] = []
		grads_list[task] = []
		grads_collect[task] = OrderedDict()
		for layer_name in model.super_network:
				if 'linear' in layer_name:
					for wb in ['.weight','.bias']:
						sims[layer_name+wb] = []
						if wb == '.weight':
							grads_collect[task][layer_name+wb] = torch.zeros(model.super_network[layer_name][task].weight.size())	
						elif wb == '.bias':
							grads_collect[task][layer_name+wb] = torch.zeros(model.super_network[layer_name][task].bias.size())

		'''for name,para in model.named_parameters():
			if para.requires_grad == True: 
				if 'task_head' not in name:
					sims[name] = []
					grads_collect[task][name] = torch.zeros(para.size())'''
	#print(grads_collect)
	globaliter = 1
	collect_flag = False
	grad_collect_checkpoints = list(range(start_collect_iter+(valInterval-2*avg_steps)+1,num_iter,valInterval))
	while globaliter<=num_iter:

		c_task = taskQ.popleft()
		taskQ.append(c_task)
		x, y = next(iter(trainloaders[c_task]))
		x = x.view(32,32*32*1)
		x = x.to(device)
		y = y.to(device)
		output = model(x,c_task)
		loss = criterion(output,y)
		#loss_avg.add(loss.item())
		model.zero_grad()
		loss.backward()
		#optimizer.step()
		if globaliter in grad_collect_checkpoints:
			collect_flag = True
		if collect_flag:
			for layer_name in model.super_network:
				if 'linear' in layer_name:
					for wb in ['.weight','.bias']:
						if wb == '.weight':
							grads_collect[c_task][layer_name+wb] = copy.deepcopy(grads_collect[c_task][layer_name+wb].to(device)+model.super_network[layer_name][c_task].weight.grad.data)	
						elif wb == '.bias':
							grads_collect[c_task][layer_name+wb] = copy.deepcopy(grads_collect[c_task][layer_name+wb].to(device)+model.super_network[layer_name][c_task].bias.grad.data)

			'''for name,para in model.named_parameters():
				if c_task in name:
					grads_collect[c_task][name] = copy.deepcopy(grads_collect[c_task][name].to(device)+para.grad.data)'''
		if globaliter % 20 == 0:
			print('Training Loss at iteration:',globaliter,'is ',loss.item())
		optimizer.step()
		if globaliter > start_collect_iter and  (globaliter-start_collect_iter) % valInterval == 0:
			print("Start Validating")
			iters.append(globaliter)

			for task in taskQ:
				val_acc, val_loss = test(model,criterion,valloaders[task],task)
				metrics[task]['val_acc'].append(val_acc)
				metrics[task]['val_loss'].append(val_loss)
				average_grad(grads_collect[task],avg_steps)
			gradient_similarity(sims,grads_collect[new_task],grads_collect[task])
			set_zero(grads_collect[new_task])
			set_zero(grads_collect[task])
			collect_flag = False
						
			#loss_avg.reset()
			model.train()
		globaliter += 1
	model.load_state_dict(weights)
	del weights
	return sims,iters,metrics


def find_similar_tasks(task_sim_scores,n=3):
	
	result = {k: v for k, v in sorted(task_sim_scores.items(), key=lambda item: item[1])}
	selected = list(result.keys())
	selected = selected[0:n]
	return selected


def run_learn_to_grow(n_tasks):
	batch_size = 32
	train_loaders, val_loaders = get_Pmnist_tasks(n_tasks,batch_size)
	task_names = list(train_loaders.keys())
	template = {'linear1_input':nn.Linear(32*32,200),'relu1':nn.ReLU(),'linear2':nn.Linear(200,200),'relu2':nn.ReLU(),
				'linear3_output':nn.Linear(200,10),'softmax':nn.Softmax(dim=0)}
	model = GradCL(template,0.5)
	model.to(device)
	print(model)
	#for name,para in model.named_parameters():
	#   print(name,para)
	criterion = nn.CrossEntropyLoss()
	#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	#train(model,optimizer,criterion,train_loaders['task_0'],val_loaders['task_0'],task_names)
	#learn_to_grow(model,criterion,train_loaders,val_loaders,task_names)
	model.save_model('super_network.pth')

#run_learn_to_grow(5)


def get_500_images_loader(train_loader):
	indices = random.sample(range(0,5000),10)
	dataset_500 = Subset(train_loader.dataset,indices)
	return torch.utils.data.DataLoader(dataset_500, batch_size=2, shuffle=False, num_workers=1)


def get_activations(model,image_loader_500,task):
	accumulate_activations = {}
	with torch.no_grad():
		for idx, data in enumerate(image_loader_500):
			print(idx)
			x = data[0]
			print(x.size())
			x = x.view(2,32*32*1)
			activations = model.get_layer_activations(data[0],task)
			for name,layer_activations in activations.items():
				print(name,activations.size())
				if len(accumulate_activations) == 0:
					pass
					#accumulate_activations[name] = layer_activations
				else:
					pass
					#accumulate_activations[name] = torch.cat((accumulate_activations[name],layer_activations),0)
				#print(accumulate_activations[name].size())
	return 1 #accumulate_activations


def RSA_Sim(model,new_task,tasks,train_loaders,val_loaders):
	start_time = time.time()
	weights = copy.deepcopy(model.state_dict())
	model = weight_innit(model)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	train_single_task(model,criterion,optimizer,train_loaders[new_task],val_loaders[new_task],new_task,epochs)

	image_loader_500 = get_500_images_loader(train_loaders[new_task])

	activations_new_task = get_activations(model,image_loader_500,new_task)
	model.load_state_dict(weights)
	del weights
	pairwise_layer_scores = {}
	pairwise_task_scores = {}
	for task in tasks:
		activations_task = get_activations(model,image_loader_500,task)
		pairwise_layer_scores[task] = {}
		for name in activations_task:
			rsa = RSA(activations_new_task[name],activations_task[name])
			rsa.create_RDMs()
			rsa.compute_RDM_similarity()
			pairwise_layer_scores[task][name] = rsa.similarity
		pairwise_task_scores[task] = np.norm(list(pairwise_layer_scores.values()))
		pairwise_layer_scores[task] = score_to_distribution(pairwise_layer_scores[task])
		

	task_score = 0
	end_time = time.time()
	function_time = end_time - start_time
	#print('time taken for RSA_Sim execution', function_time)
	return pairwise_layer_scores, pairwise_task_scores









