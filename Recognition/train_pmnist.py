import torch
from modelsv1 import GradCL

#1)get data loaders for each task
#2)

#issues
#1) How to compare with exsiting models what is the procedure
#2) gradient collection with trained model or randomly initialise all models
#	and do comparisom
#3) method train_single(trainloader,valloader,taskname)

# To compute similarity with previous models 
# need a template network, build the templa
# need two dataloaders one for the new task data and one for old task data corresponding to that model
# step 1: find similar model
# how? How do you collect gradients for this model?  
#save template and parameters 