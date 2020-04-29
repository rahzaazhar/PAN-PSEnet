import torch
from modelv1 import GradCL
import pmnist_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from pmnist_dataset import get_Pmnist_tasks
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

def train(model,optimizer,criterion,trainloaders,valloaders,tasks):
	taskQ = tasks
	epochs = 1

	for current_task in taskQ:
		model = weight_innit(model)
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



