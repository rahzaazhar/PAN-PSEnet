import torch
from modelv1 import GradCL
import pmnist_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from pmnist_dataset import get_Pmnist_tasks
from plot import plot_grad_sim, gradient_similarity, multiplot, average_grad, dot_product, set_zero
from plot compute_per_layer_pearsonc, norm2
from utils import Averager
import plot
torch.manual_seed(1111)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#1)get data loaders for each task
#2)

#issues
#1) How to compare with exsiting models what is the procedure
#2) gradient collection with trained model or randomly initialise all models
#   and do comparisom
#3) method train_single(trainloader,valloader,taskname)

# To compute similarity with previous models 
# need a template network, build the templa
# need two dataloaders one for the new task data and one for old task data corresponding to that model
# step 1: find similar model
# how? How do you collect gradients for this model?  
#save template and parameters 
#combined batch how will the network know which example belongs to which batch
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
				#init.kaiming_normal_(param)
				init.constant_(param, 0.0)
		except Exception as e:  # for batchnorm.
			if 'weight' in name:
				param.data.fill_(1)
			continue
	return model

def test(model,criterion,test_loader,task):
	print('testing task:',task)
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for x, y in test_loader:
			#print(x.size())
			bs = x.size(0)
			x = x.view(bs,32*32*1)
			x = x.to(device)
			y = y.to(device)
			output = model(x,task)
			test_loss += criterion(output, y).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(y.data.view_as(pred)).sum()
		test_loss /= len(test_loader.dataset)
		#test_losses.append(test_loss)
		print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, len(test_loader.dataset),
			100. * correct / len(test_loader.dataset)))
		return ((100.*correct)/len(test_loader.dataset)),test_loss

def train(model,optimizer,criterion,trainloaders,valloaders,tasks):
	taskQ = tasks
	epochs = 1

	for current_task in taskQ:
		#model = weight_innit(model)
		print('training task:',current_task)
		for epoch in range(epochs):
			for batch_idx, (x, y) in enumerate(trainloaders):
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
					epoch, batch_idx * len(x), len(trainloaders.dataset),
					100. * batch_idx / len(trainloaders), loss.item()))
		test(model,criterion,valloaders,current_task)

def run(n_tasks):
	batch_size = 32
	train_loaders, val_loaders = get_Pmnist_tasks(n_tasks,batch_size)
	task_names = list(train_loaders.keys())
	template = [('input_layer',nn.Linear(32*32,200)),('relu1',nn.ReLU()),('layer1',nn.Linear(200,200)),('relu2',nn.ReLU()),
				('output_layer',nn.Linear(200,10)),('relu2',nn.Softmax())]
	model = GradCL(template,task_names)
	model.to(device)
	print(model)
	#for name,para in model.named_parameters():
	#   print(name,para)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	train(model,optimizer,criterion,train_loaders['task_0'],val_loaders['task_0'],task_names)

run(3)




#get_group_sim_score : returns overall group sim score, per layer sim score for each group task
def learn_to_grow(model,criterion,optimizer,trainloaders,valloaders,tasks):
	#task_groups = {'group1':[]}
	tasks = []
	for task_id, current_task in enumerate(tasks):
		model.init_subgraph(current_task)
		if task_id == 0:
			train_single_task(model,criterion,optimizer,trainloaders,current_task)
			#task_groups['group1'].append(current_task)
			tasks.append(current_task)
			continue
		#get architecture for new task then estimate its parameters
		#group_sim_scores = {}
		task_sim_scores = {}
		task_layerwise_sims = {}
		#for group_name, group_tasks in task_groups.items():
		for task in tasks:
			#group_sim_scores[group_name] = get_group_sim_score(model,current_task,group_tasks,trainloaders,valloaders)
			### task_sim_scores dict {task_<name>:sim_score with new task} ###
			### task_layerwise_sims dict {task_<name>:{layer_<name>:layer_sim_score}}
			task_sim_scores[task], task_layerwise_sims[task] = get_task_sim_score(model,current_task,task,trainloaders,valloaders)
		#group,create_new_group = assign_group(group_sim_scores)
		### find_similar_tasks returns list containing k similar tasks
		selected_tasks = find_similar_tasks(task_sim_scores,k=3)

		'''if create_new_group:
			task_groups[group] = [current_task]
			model.clone(current_task)
			train_single_task(model,criterion,optimizer,trainloaders,current_task)
			tasks.append(current_task)
			continue'''
			#task_groups[group].append(current_task)
		
		#model.grow(group_sim_scores[group])
		model.grow_graph(current_task,selected_tasks,task_layerwise_sims)
		#account for new parameters added
		optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
		train_single_task(model,criterion,optimizer,trainloaders,current_task)
		tasks.append(current_task)


def train_single_task(model,criterion,optimizer,trainloaders,valloaders,current_task):
	epochs = 100
	for epoch in range(epochs):
		for batch_idx, (x, y) in enumerate(trainloaders[current_task]):
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
				epoch, batch_idx * len(x), len(trainloaders[current_task].dataset),
				100. * batch_idx / len(trainloaders[current_task]), loss.item()))
			test(model,criterion,valloaders[current_task],current_task)

def get_task_sim_score(model,new_task,task,trainloaders,valloaders):
	#group_trainloader, group_valloader = get_group_loaders(group_tasks,trainloaders,valloaders)
	#pairwise_scores = {}
	#pairwise_layer_scores = {}
	#for task in group_tasks:
	grad_sims, x, metrics = train_task_pair(model,new_task,task,trainloaders,valloaders)#used to collect gradients and computing pearson
	acc1 = metrics[new_task]['val_loss']
	acc2 = metrics[task]['val_loss']
	p_1, p_2 = compute_per_layer_pearsonc(grad_sim,acc1,acc2)
	task_score = norm2(list(p_1.items()),list(p_2.items()))
	for key in p_1.keys():
		pairwise_layer_score[p_1] = p_1[key] - p_2[key]


	#pairwise_scores[task]= sim_score
	#pairwise_layer_scores[task] = layer_sim_score

	#group_score = average(pairwise_scores)
	return task_score, pairwise_layer_scores

def train_task_pair(model,new_task,task,trainloaders,valloaders):
	start_collect_iter = 0
	valInterval = 200
	avg_steps = 50
	num_iter = 2000
	loss_avg = Averager()
	taskQ = deque([new_task,task])
	model.change_subgraph_pointer(new_task,task)
	metrics = {}
	grads_list = {}
	grads_collect = {}
	sims = {}
	x = []
	for task in taskQ:
		metrics[task]={}
		metrics[task]['val_acc'] = []
		metrics[task]['val_loss'] = []
		grads_list[task] = []
		grads_collect[task] = OrderedDict()
		for name,para in model.named_parameters():
			if para.requires_grad == True: 
				if 'task_head' not in name:
					sims[name] = []
					grads_collect[task][name] = torch.zeros(para.size())

	globaliter = 1
	collect_flag = False
	grad_collect_checkpoints = list(range(start_collect_iter+(valInterval-2*avg_steps)+1,num_iter,valInterval))
	while globaliter<num_iter:

		c_task = taskQ.popleft()
		taskQ.append(c_task)
		x, y = next(iter(trainloaders[c_task]))
		x = x.view(32,32*32*1)
		x = x.to(device)
		y = y.to(device)
		output = model(x,c_task)
		loss = criterion(output,y)
		model.zero_grad()
		loss.backward()
		optimizer.step()
		if globaliter in grad_collect_checkpoints:
			collect_flag = True
		if collect_flag:
			for name,para in model.named_parameters():
				if c_task in name:
					grads_collect[c_task][name] = copy.deepcopy(grads_collect[c_task][name].to(device)+para.grad.data)

		optimizer.step()
		if globaliter > start_collect_iter and  (globaliter-start_collect_iter) % opt.valInterval == 0:
			print("Start Validating")

			for task in taskQ:
				metrics[task]['val_acc'], metrics[task]['val_loss'] = test(model,criterion,valloaders[task],task)
			x.append(globaliter)
			for task in taskQ:
				average_grad(grads_collect[task],avg_steps)
				gradient_similarity(sims,grads_collect[new_task],grads_collect[task])
				#dump = OrderedDict()
				#dump = {'iter':x,'grad_sims':sims}
				#torch.save(dump,opt.exp_dir+'Gradsims/'+'{}.pth'.format(opt.experiment_name))
			set_zero(grads_collect[new_task])
			set_zero(grads_collect[task])
			collect_flag = False
						
			#loss_avg.reset()
			model.train()
	return sims,x,metrics


def find_similar_tasks(task_sim_scores,n=3):
	
	result = {k: v for k, v in sorted(task_sim_scores.items(), key=lambda item: item[1])}
	selected = list(result.keys())
	selected = selected[0:3]
	return selected


	
'''class LitModel(pl.LightningModule):
	def __init__(self):
		super().__init__(template,task_names)
		self.model = GradCL(template,task_names)

	def forward(x,task):
		return self.model(x,task)

	def cross_entropy_loss(preds,targets):
		return nn.CrossEntropyLoss(preds,targets)

	def configure_optimizers(self):
		optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
		return optimizer'''
		




