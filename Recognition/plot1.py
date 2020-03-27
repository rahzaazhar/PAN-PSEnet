import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr
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


def norm2(a,b):
    a = np.array(a)
    a = a/np.linalg.norm(a)
    b = np.array(b)
    b = b/np.linalg.norm(b)
    return np.sqrt(np.sum((a - b) ** 2))
    #return (1-np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))*100
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

def get_subset(l,nele):
    return l[0:nele]

def find_sims(sims,grads1,grads2):
    for grad1,grad2 in zip(grads1,grads2):
        gradient_similarity(sims,grad1,grad2)


def execute_pearson_comp(grads1, metrics1, lang1, grads2, metrics2, lang2, plot=False):
    sims_g1_g2 = {}
    for name in grads1[0].keys():
        if 'Predictions' not in name:
            sims_g1_g2[name] = []
    find_sims(sims_g1_g2, grads1, grads2)
    p_g1, p_g2 = compute_per_layer_pearsonc(sims_g1_g2, metrics1, metrics2)
    similarity_score = 1-norm2(list(p_g1.values()),list(p_g2.values()))
    
    if plot:
        print('The similarity_score between '+lang1+' and '+lang2+' is:',similarity_score)
        plot_pearsonc(p_g1,lang1,p_g2,lang2)
    
    return similarity_score

def pearson_vs_iters(grads1,metrics1,lang1,grads2,metrics2,lang2):
    length = len(metrics1)
    scores = []
    for i in range(lenght):
        grads1_subset = gen_subset(grads1,i+1)
        grads2_subset = gen_subset(grads2,i+1)
        metrics1_subset = gen_subset(metrics1,i+1)
        metrics2_subset = gen_subset(metrics2,i+1)
        score = execute_pearson_comp(grads1_subset,metrics1_subset,lang1,grads2_subset,metrics2_subset,lang2)
        scores.append(score)
    return scores

def simple_plot(x,y,xlabel,ylabel,title):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.close('all')




