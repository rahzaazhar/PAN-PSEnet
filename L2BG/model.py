import torch
import logging
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import torchvision.models as vision_models
from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class visual_context(nn.Module):
    def __init__(self):
        super(visual_context, self).__init__()
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
    def forward(self,visual_feature):
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        return visual_feature

def get_alexnet_template(pretrained):
    alexnet = vision_models.alexnet(pretrained=pretrained)
    template = {}
    feature_remapping = {'features.0':'conv2d_input1','features.1':'relu1','features.2':'maxpool1','features.3':'conv2d_2',
            'features.4':'relu2','features.5':'maxpool2','features.6':'conv2d_3','features.7':'relu3',
            'features.8':'conv2d_4','features.9':'relu4','features.10':'conv2d_5','features.11':'relu5',
            'features.12':'maxpool3'}

    linear_remapping = {'classifier.0':'dropout1','classifier.1':'linear1','classifier.2':'relu6','classifier.3':'dropout2',
                'classifier.4':'linear2','classifier.5':'relu7'}
    for name, module in alexnet.named_modules():
        if 'features' in name and name in feature_remapping.keys():
            template[feature_remapping[name]] = module
    template['AdaptiveAvgPool2d'] = nn.AdaptiveAvgPool2d((6, 6))
    template['flatten'] = Flatten()
    for name, module in alexnet.named_modules():
        if 'classifier' in name and name in linear_remapping.keys():
            template[linear_remapping[name]] = module
    return template


class GradCL(nn.Module):

    def __init__(self,template,sim_thres):
        super(GradCL,self).__init__()
        self.template = template # template of the first task with the output head
        self.super_network = nn.ModuleDict()
        self.task_count = 0
        self.tasks = []
        self.sim_thres = sim_thres
        self.sub_graphs = {}
        #for name, layer in template:
        for name, layer in template.items():
            self.super_network[name] = nn.ModuleDict()
        self.super_network['task_heads'] = nn.ModuleDict()
        #self.super_network['softmax'] = nn.ModuleDict()

    def  init_subgraph(self,new_task_name,datamode,new_head=None, point_to_task=0):
        #for layer_name, layer in self.template:
        if not new_head == None:
            if datamode == 'smnist' or datamode == 'pmnist':
                #128 for RCL 300 for L2G
                self.super_network['task_heads'][new_task_name] = nn.Linear(300,new_head)
            if datamode == 'CIFAR100':
                self.super_network['task_heads'][new_task_name] = nn.Linear(2048,new_head)
            if datamode == 'VDD':
                self.super_network['task_heads'][new_task_name] = nn.Linear(4096,new_head)
            if datamode == 'mltr':
                self.super_network['task_heads'][new_task_name] = nn.Linear(256,new_head)

            #self.super_network['softmax'][new_task_name] = nn.Softmax(dim=1)
        else:
            self.super_network['task_heads'][new_task_name] = self.super_network['task_heads'][self.tasks[point_to_task]]
            #self.super_network['softmax'][new_task_name] = copy.copy(self.super_network['softmax'][self.tasks[point_to_task]])

        #self.super_network['task_head'][new_task_name] = nn.Linear(200,nclasses) 
        #self.sub_graphs[new_task_name] = {}
        for layer_name, layer in self.template.items():
            if self.task_count == 0:
                self.super_network[layer_name][new_task_name] = layer
                #self.sub_graphs[new_task_name][layer_name] = new_task_name

            else:
                self.super_network[layer_name][new_task_name] = self.super_network[layer_name][self.tasks[point_to_task]]
                #self.sub_graphs[new_task_name][layer_name] = self.tasks[point_to_task]
        self.task_count = self.task_count + 1
        self.tasks.append(new_task_name)

    def change_subgraph_pointer(self,source_task,dest_task):
        dest_task_index = self.tasks.index(dest_task)
        self.init_subgraph(source_task,dest_task_index)

    def save_model(self,path):
        print('saving model at',path)
        torch.save(self.super_network,path)

    def load_model(self,path):
        self.super_network = torch.load(path)

    def random_growth(self,new_task,selected_tasks):
        self.sub_graphs[new_task] = {}
        most_similar_task = selected_tasks[0]
        selected_layers = []
        layer_names = [name for name in self.template if 'linear' in name or 'conv2d' in name]
        total_no_layers = len(layer_names)
        no_to_expand = np.random.randint(0,total_no_layers)
        selected_layers = np.random.choice(layer_names,no_to_expand,replace=False)
        for layer_name in self.template.keys():
            if 'linear' in layer_name or 'conv2d' in layer_name:
                if layer_name in selected_layers:
                    self.add_node(new_task,layer_name)
                    print(layer_name,' added')
                    logging.debug('%s added',layer_name)
                    self.sub_graphs[new_task][layer_name] = 'new'
                else:
                    self.point_to_node(new_task,most_similar_task,layer_name)
                    print('binding to ',layer_name,' of',most_similar_task)
                    logging.debug('binding to %s of %s',layer_name,most_similar_task)
                    self.sub_graphs[new_task][layer_name] = layer_name+' of '+most_similar_task

    def grow_graph(self,new_task,selected_tasks,task_layerwise_sims):
        #print('Growing Task',new_task)
        self.sub_graphs[new_task] = {}
        most_similar_task = selected_tasks[0]
        selected_layers = []
        for task in selected_tasks:
            task_layerwise_sims[task] = {k: v for k, v in sorted(task_layerwise_sims[task].items(), key=lambda item: item[1], reverse=True)}
            #print('In grow_graph')
            #print('distribution over layers')
            logging.debug('distribution over layers')
            logging.debug('%s',task_layerwise_sims[task])
            for name, score in task_layerwise_sims[task].items():
                print(name+':'+str(score),end=' ')
            
            #print(task_layerwise_sims[task])
            print()
            cumulative_sim = 0
            for layer_name, sim in task_layerwise_sims[task].items():
                if cumulative_sim < self.sim_thres:
                    cumulative_sim += sim
                    selected_layers.append(layer_name)
            for layer_name in self.template.keys():
                if 'linear' in layer_name or 'conv2d' in layer_name or 'BiLstm' in layer_name or 'bn' in layer_name:
                    if layer_name in selected_layers:
                        self.add_node(new_task,layer_name)
                        print(layer_name,' added')
                        logging.debug('%s added',layer_name)
                        self.sub_graphs[new_task][layer_name] = 'new'
                    else:
                        self.point_to_node(new_task,most_similar_task,layer_name)
                        print('binding to ',layer_name,' of',most_similar_task)
                        logging.debug('binding to %s of %s',layer_name,most_similar_task)
                        self.sub_graphs[new_task][layer_name] = layer_name+' of '+most_similar_task

    def forward(self,x,task,get_layer_activations=False):
        if not get_layer_activations:
            for layer in self.super_network:
                x = self.super_network[layer][task](x)
                #print(layer)
            return x
        else:
            out = {}
            for layer in self.super_network:
                x = self.super_network[layer][task](x)
                if 'linear' in layer or 'conv2d' in layer or 'bn' in layer:
                    out[layer] = F.relu(x)#.view(x.size(0),-1)
                if 'task_heads' in layer or 'BiLstm' in layer:
                    out[layer] = x#.view(x.size(0),-1)
            return out

    def add_node(self,task,layer):
        #self.super_network[layer][task] = self.super_network[layer]['task0']
        self.super_network[layer][task] = copy.deepcopy(self.template[layer])

    def point_to_node(self,source_task,dest_task,layer_name):
        self.super_network[layer_name][source_task] = self.super_network[layer_name][dest_task]


