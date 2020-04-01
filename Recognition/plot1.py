import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr
from scipy.spatial import distance
import seaborn
import cv2

def plot_grad_sim(x, sims, module, para, ylabel,plot=True):
    fig, ax = plt.subplots()
    #fig.suptitle(module+para)
    xp = np.array(x)
    for name,sim in sims.items():
        if module in name and para in name:
            sim = np.array(sim)
            name = name.replace(module,'')
            plt.scatter(xp, sim, label=name)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax.set_title(module+para)
    #ax.legend(loc='upper right')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('iters')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if plot:
    	plt.show()
    else:
    	fig.savefig(opt.save_path+'/{}{}_gradSim.png'.format(module,para))
    plt.close('all')


def gradient_similarity(sims,grad1,grad2):
    for name in sims:
        g_1 = grad1[name]
        g_2 = grad2[name]
        g_1 = g_1.view(-1)
        g_2 = g_2.view(-1)
        g_1 = g_1.cpu().detach().numpy()
        g_2 = g_2.cpu().detach().numpy()
        sims[name].append(dot_product(g_1,g_2))


def compute_per_layer_pearsonc(grad_sim,acc1,acc2):
    p_task1 = {}
    p_task2 = {}
    for name, sim_value in grad_sim.items():
        p_task1[name]=pearsonr(np.array(acc1),np.array(sim_value))[0]
        p_task2[name]=pearsonr(np.array(acc2),np.array(sim_value))[0]
    return p_task1, p_task2


def multiplot(x,sims,ylabel):
    for module in ['FeatureExtraction.','rnn_lang.']:
        for para in ['weight','bias']:
            plot_grad_sim(x,sims,module,para,ylabel)

def plot_pearsonc(p_task1,lang1,p_task2,lang2):
    labels = p_task1.keys()
    #values = p_arabacc.values()
    fig, axs = plt.subplots(figsize=(20,8))
    axs.tick_params(axis='x', rotation=90)
    box = axs.get_position()
    axs.set_title('Comparing '+lang1+'-'+lang2+' per layer pearson correlation')
    axs.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    plt.scatter(list(labels), list(p_task1.values()),label=lang1)
    plt.scatter(list(labels),list(p_task2.values()),label=lang2)
    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs.set_xlabel('layers')
    axs.set_ylabel('pearson co-relation between gradient similarity and lang accurcay')
    axs.grid(True)
    plt.show()


def norm2(a,b,dist='l2'):
    a = np.array(a)
    a = a/np.linalg.norm(a)
    b = np.array(b)
    b = b/np.linalg.norm(b)
    if dist == 'l2':
        #return np.sqrt(np.sum((a - b) ** 2))
        return distance.euclidean(a,b)
        #return distance.correlation(a,b)
    if dist == 'al2':
        return np.sqrt(np.sum((np.abs(a) - np.abs(b)) ** 2))
    if dist == 'cosine':
        return (1-np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))*100
    #return np.linalg.norm(a-b)

def dot_product(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    #return np.dot(a,b)


def check_para_equal(pa,pb):
    res = True
    for name in pa.keys():
        if not torch.equal(pa[name],pb[name]):
            res = False
            break
    return res

def get_subset(l,start,nele):
    if isinstance(l,dict):
        d = {}
        for name,values in l.items():
            d[name] = values[start:nele]
        return d 
    else: 
        return l[start:nele]

def find_sims(sims,grads1,grads2):
    for grad1,grad2 in zip(grads1,grads2):
        gradient_similarity(sims,grad1,grad2)


def execute_pearson_comp(grads1, metrics1, lang1, grads2, metrics2, lang2, plot=False, dist='l2'):
    sims_g1_g2 = {}
    for name in grads1[0].keys():
        if 'Predictions' not in name:
            sims_g1_g2[name] = []
    find_sims(sims_g1_g2, grads1, grads2)
    p_g1, p_g2 = compute_per_layer_pearsonc(sims_g1_g2, metrics1, metrics2)
    similarity_score = 1-norm2(list(p_g1.values()),list(p_g2.values()),dist)
    
    if plot:
        plot_pearsonc(p_g1,lang1,p_g2,lang2)
    print('The similarity_score between '+lang1+' and '+lang2+' is:',similarity_score)
    return similarity_score

def pearson_vs_iters(grads1,metrics1,lang1,grads2,metrics2,lang2,dist):
    length = len(metrics1)
    scores = []
    for i in range(length):
        grads1_subset = get_subset(grads1,i+1)
        grads2_subset = get_subset(grads2,i+1)
        metrics1_subset = get_subset(metrics1,i+1)
        metrics2_subset = get_subset(metrics2,i+1)
        score = execute_pearson_comp(grads1_subset,metrics1_subset,lang1,grads2_subset,metrics2_subset,lang2,dist)
        scores.append(score)
    return scores

def simple_plot(x,y,xlabel,ylabel,title):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.close('all')

def average_grad(opt,grad,avg_steps):
    for name,para in grad.items():
        grad[name] = para/avg_steps

def set_zero(grad):
    for name,para in grad.items():
        grad[name] = torch.zeros(para.size())

#@azhar
def unfreeze_head(model,head):
    for name,param in model.named_parameters():
        if head in name:
            param.requires_grad = True

#@azhar testing function    
def paramCheck(model):
    for name, param in model.named_parameters():
        print(name,param.requires_grad)

def all_gradsim_to_score(data,metric_names,distances,plot=False):
    """
    parameters:
    data : dictionary of tuples with gradsimilarity and metrics for language pair {'AB':(gradsim,metrics)}
    metrics_names: list of metrics to be used for similarity score evaluation

    output:
    similarity scores for all language pairs

    """

    encoding = {'A':'arab','B':'ban','H':'hin','M':'mar'}
    for dist in distances:
        for metric_name in metric_names:
            print('Evaluating similarity_score using',metric_name,'with',dist,'distance')
            for langpair, (gradsim,metrics) in data.items():
                langpair = list(langpair)
                lang1 = encoding[langpair[0]]
                lang2 = encoding[langpair[1]]
                metrics1= metrics[lang1][metric_name]
                metrics2= metrics[lang2][metric_name]

                p_g1, p_g2 = compute_per_layer_pearsonc(gradsim, metrics1, metrics2)
                similarity_score = norm2(list(p_g1.values()),list(p_g2.values()),dist)
                if plot:
                    plot_pearsonc(p_g1,lang1,p_g2,lang2)
                print('The similarity_score between '+lang1+' and '+lang2+' is:',similarity_score)
            print()

#plot variation of similarity score with iters for all language pairs for a given metric and distance
def all_simscore_vs_iters(x,data,dist,metric_name,start):
    length = len(x)
    encoding = {'A':'arab','B':'ban','H':'hin','M':'mar'}
    indx = x.index(start)
    x = x[indx+1:]
    for langpair, (gradsim,metrics) in data.items():
        langpair = list(langpair)
        lang1 = encoding[langpair[0]]
        lang2 = encoding[langpair[1]]
        metrics1= metrics[lang1][metric_name]
        metrics2= metrics[lang2][metric_name]
        scores = []
        for i in range(indx,length-1):
            gradsim_subset = get_subset(gradsim,indx,i+2)
            metrics1_subset = get_subset(metrics1,indx,i+2)
            metrics2_subset = get_subset(metrics2,indx,i+2)
            p_g1, p_g2 = compute_per_layer_pearsonc(gradsim_subset, metrics1_subset, metrics2_subset)
            similarity_score = norm2(list(p_g1.values()),list(p_g2.values()),dist)
            scores.append(similarity_score)
        simple_plot(x,scores,'iters','similarity score','similarity vs iters({}-{})'.format(lang1,lang2))


